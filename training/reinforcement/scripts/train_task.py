import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import os
import sys
import random
import time
import math
import numpy as np

# Ensure project root is in path for imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

# Assuming these imports are correct based on your file structure
from src_shared.model import ChessAIModel
from src_shared.data_loader import ChessDataset, _worker_init_fn


class TrainTask:
    def __init__(self, output_dir: str, best_model_path: str, model_config: dict, training_config: dict, state_config: dict, global_config: dict, hdf5_path: str, cycle_number: int):
        self.training_config = training_config
        self.best_model_path = best_model_path
        self.model_config = model_config
        self.state_config = state_config
        self.global_config = global_config
        self.output_dir = output_dir
        self.hdf5_path = hdf5_path
        self.cycle_number = cycle_number

        self.logger = self._setup_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Training task initialized. Using device: {self.device}")

    def _setup_logger(self):
        logger = logging.getLogger("TrainTask")
        logger.setLevel(self.training_config['main_logging_level'])
        
        if logger.hasHandlers():
            logger.handlers.clear()
        
        log_file_path = os.path.join(self.output_dir, f"training_run.log")

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger


    def _get_shuffled_sample_indices(self):
        """
        Calculates a list of absolute sample indices from the original HDF5 file.
        """
        sampling_ratio = self.training_config['sampling_ratio']
        total_positions_source = self.state_config['buffer']['count']
        batch_size = self.training_config['batch_size']

        # Total pool of individual sample indices (positions) to sample FROM
        all_sample_indices = np.arange(total_positions_source)
        
        # Train for dynamic number of steps based on a fixed sampling ratio
        fixed_steps = int(sampling_ratio * (self.global_config['data_generation_positions_per_cycle'] / batch_size))
        
        self.logger.info(f"Target samples to sample: {fixed_steps}")
        self.logger.info("Using uniform sampling from all available HDF5 samples.")
        
        final_indices = np.random.choice(
            all_sample_indices, 
            size=fixed_steps, 
            replace=self.training_config['replacement']
        ).tolist()

        
        random.shuffle(final_indices)
        self.logger.info(f"Successfully generated {len(final_indices)} globally shuffled, sample-prioritized indices.")
        
        return final_indices


    def _get_dataloaders(self):
        
        # 1. Generate the absolute, shuffled indices
        self.training_indices = self._get_shuffled_sample_indices() 
        batch_size = self.training_config['batch_size']
        
        # 2. Instantiate the random-access dataset using the ORIGINAL hdf5_path
        full_dataset = ChessDataset(hdf5_path=self.hdf5_path, indices=self.training_indices) 

        num_workers = self.training_config['data_loader_workers']

        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

        # Calculate the total steps based on the number of samples
        total_steps = (len(self.training_indices) + batch_size - 1) // batch_size
        self.logger.info(f"Training set size: {len(full_dataset)} samples ({total_steps} batches)")

        return train_loader, full_dataset


    def run_training_loop(self):
        final_model_path = None
        training_steps_completed = 0
        full_dataset = None 
        
        try:
            train_loader, full_dataset = self._get_dataloaders()
            
            total_training_steps_this_cycle = len(train_loader) 
            self.logger.info(f"Total training steps for this run: {total_training_steps_this_cycle}")
            
            model = ChessAIModel(
                num_input_planes=self.model_config['input_planes'],
                num_residual_blocks=self.model_config['resblocks'],
                num_filters=self.model_config['filters'],
                dropout_rate_conv=self.training_config['dropout_rate_conv'],
                dropout_rate_fc=self.training_config['dropout_rate_fc'],
                dropout_conv_start_block=self.training_config['dropout_conv_start_block']
            ).to(self.device)

            # Loading model state
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Model weights loaded from checkpoint dictionary.")
            
            self.logger.info("Model initialized and weights loaded.")

            policy_criterion = nn.KLDivLoss(reduction='batchmean')
            value_criterion = nn.MSELoss()

            global_step = self.state_config['lifetime']['training_steps']

            # Get LR based on step count
            if global_step < self.training_config['lr_high_cutoff']:
                target_lr = float(self.training_config['lr_high'])
            elif global_step < self.training_config['lr_mid_cutoff']:
                target_lr = float(self.training_config['lr_mid'])
            else:
                target_lr = float(self.training_config['lr_low'])

            momentum_rate = self.training_config['momentum_rate']

            
            # 2. Setup Optimizer with that fixed LR
            optimizer = optim.SGD(
                model.parameters(), 
                lr=target_lr,
                momentum=momentum_rate,
                weight_decay=float(self.training_config['weight_decay'])
            )

            scaler = GradScaler('cuda')

            # 3. Load Optimizer State (Preserve Momentum)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = target_lr
                
                self.logger.info(f"Optimizer state loaded, but Learning Rate forced to static value: {target_lr}")

            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.logger.info(
                f"Optimizer (SGD with Momentum={momentum_rate}) initialized. "
                f"Using static Learning Rate: {target_lr}"
            )
            
            # Training Phase: Single Pass
            model.train()
            running_policy_loss = 0.0
            running_value_loss = 0.0
            running_entropy_loss = 0.0
            running_total_loss = 0.0
            
            for batch_idx, (board_tensors, policy_target, value_targets) in enumerate(train_loader):
                current_step = global_step + batch_idx + 1
                
                batch_start_time = time.perf_counter()
                
                transfer_to_gpu_start = time.perf_counter()
                board_tensors = board_tensors.to(self.device, non_blocking=True)
                policy_target = policy_target.to(self.device, non_blocking=True)
                value_targets = value_targets.to(self.device, non_blocking=True)
                torch.cuda.synchronize()
                transfer_to_gpu_end = time.perf_counter()
                
                optimizer.zero_grad()
                
                forward_pass_start = time.perf_counter()

                with autocast('cuda'):
                    policy_logits, value_outputs = model(board_tensors)

                    # Mask illegal moves before softmax
                    legal_mask = policy_target > 0.0
                    mask_value = torch.finfo(policy_logits.dtype).min
                    policy_logits = policy_logits.masked_fill(~legal_mask, mask_value)

                    policy_log_softmax = F.log_softmax(policy_logits, dim=1)
                    policy_probs = torch.exp(policy_log_softmax)
                    value_outputs = value_outputs.squeeze(1)
                    torch.cuda.synchronize()
                    forward_pass_end = time.perf_counter()

                    batch_entropy = -torch.sum(policy_probs * policy_log_softmax, dim=1).mean()
                    running_entropy_loss += batch_entropy.item()

                    policy_loss = policy_criterion(policy_log_softmax, policy_target)
                    value_loss = value_criterion(value_outputs, value_targets)

                    running_policy_loss += policy_loss.item() * self.training_config['policy_loss_weight']
                    running_value_loss += value_loss.item() * self.training_config['value_loss_weight']
                    

                    total_loss = (policy_loss * self.training_config['policy_loss_weight']) + \
                                 (value_loss * self.training_config['value_loss_weight']) - \
                                 (batch_entropy * self.training_config['entropy_loss_weight'])

                                        
                backward_pass_start = time.perf_counter()
                scaler.scale(total_loss).backward()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                backward_pass_end = time.perf_counter()

                scaler.step(optimizer)
                scaler.update()

                running_total_loss += total_loss.item()
                                                                
                batch_end_time = time.perf_counter()
                
                # Log detailed info to file at intervals
                if current_step % self.training_config['log_interval'] == 0:
                    self.logger.info(f"Training Step {current_step}/{total_training_steps_this_cycle + global_step}: "
                                     f"P_Loss={(policy_loss.item() * self.training_config['policy_loss_weight']):.4f}, "
                                     f"V_Loss={(value_loss.item() * self.training_config['value_loss_weight']):.4f}, "
                                     f"E_Loss={(batch_entropy.item() * self.training_config['entropy_loss_weight']):.4f}, "
                                     f"T_Loss={total_loss.item():.4f}, "
                                     f"LR={optimizer.param_groups[0]['lr']:.6f}, "
                                     f"GPU Xfer: {(transfer_to_gpu_end - transfer_to_gpu_start)*1000:.2f}ms, "
                                     f"FW: {(forward_pass_end - forward_pass_start)*1000:.2f}ms, "
                                     f"BW: {(backward_pass_end - backward_pass_start)*1000:.2f}ms, "
                                     f"Batch Total: {(batch_end_time - batch_start_time)*1000:.2f}ms")
                    
            avg_entropy_train = running_entropy_loss / len(train_loader)
            training_steps_completed = len(train_loader)
            
            self.logger.info(f"--- Training Run Summary ---")
            self.logger.info(f"Total Steps Completed This Cycle: {training_steps_completed} "
                            f"Average Policy Loss: {running_policy_loss / len(train_loader):.4f}, "
                            f"Average Value Loss: {running_value_loss / len(train_loader):.4f}, "
                            f"Average entropy Loss: {avg_entropy_train}, "
                            f"Average Total Loss: {running_total_loss / len(train_loader):.4f}")
                            
            
            #  Save the final model (weights AND optimizer state)
            final_model_path = os.path.join(self.output_dir, f"model_iter_{self.cycle_number}.pth")
            
            # Create the checkpoint dictionary
            model_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            
            torch.save(model_dict, final_model_path)
            self.logger.info(f"Final model (with optimizer state) saved to {final_model_path}")
            self.logger.info("Training complete for this task!")
            
        finally:
            if full_dataset:
                self.logger.info("Closing HDF5 file handle(s) in ChessDataset.")
                full_dataset.close()
            
        return final_model_path, training_steps_completed, avg_entropy_train