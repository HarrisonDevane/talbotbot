import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import os
import sys
from datetime import datetime
import time

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

from src_shared.model import ChessAIModel
from src_shared.data_loader import ChessDataset, _worker_init_fn


class TrainTask:
    def __init__(self, output_dir: str, model_config: dict, training_config: dict, hdf5_path: str, cycle_number: int):
        self.training_config = training_config
        self.model_config = model_config
        self.output_dir = output_dir
        self.hdf5_path = hdf5_path
        self.cycle_number = cycle_number

        self.log_dir = os.path.join(self.output_dir, "logs")

        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = self._setup_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Training task initialized. Using device: {self.device}")

    def _setup_logger(self):
        logger = logging.getLogger("TrainTask")
        logger.setLevel(self.training_config['main_logging_level'])
        
        if logger.hasHandlers():
            logger.handlers.clear()
        
        log_file_path = os.path.join(self.log_dir, f"training_run.log")

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(log_file_path, mode = 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        
        return logger

    def _get_dataloaders(self):
        self.logger.info(f"Loading data from HDF5 file: {self.hdf5_path}")
        full_dataset = ChessDataset(hdf5_path=self.hdf5_path)

        total_samples = len(full_dataset)
        validation_split = self.training_config['validation_split']
        val_samples = int(total_samples * validation_split)
        train_samples = total_samples - val_samples

        train_dataset, val_dataset = random_split(full_dataset, [train_samples, val_samples],
                                                  generator=torch.Generator().manual_seed(42))

        num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None
        )
        
        self.logger.info(f"Total dataset size: {total_samples} samples")
        self.logger.info(f"Training set size: {len(train_dataset)} samples ({len(train_loader)} batches)")
        self.logger.info(f"Validation set size: {len(val_dataset)} samples ({len(val_loader)} batches)")
        
        return train_loader, val_loader

    def run_training_loop(self):
        train_loader, val_loader = self._get_dataloaders()
        
        model = ChessAIModel(
            num_input_planes=self.model_config['input_planes'],
            num_residual_blocks=self.model_config['resblocks'],
            num_filters=self.model_config['filters'],
            dropout_rate_conv=self.training_config['dropout_rate_conv'],
            dropout_rate_fc=self.training_config['dropout_rate_fc'],
            dropout_conv_start_block=self.training_config['dropout_conv_start_block']
        ).to(self.device)

        self.logger.info(f"Loading previous model from: {self.model_config['best_model_path']}")

        model.load_state_dict(torch.load(self.model_config['best_model_path'], map_location=self.device, weights_only=True))        
        self.logger.info("Model initialized.")

        policy_criterion = nn.KLDivLoss(reduction='batchmean')
        value_criterion = nn.MSELoss()
        
        optimizer = optim.AdamW(model.parameters(), lr=float(self.training_config['cosine_eta_max']), weight_decay=float(self.training_config['weight_decay']))
        total_training_steps = self.training_config['epochs'] * len(train_loader)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps, 
                                     eta_min=float(self.training_config['cosine_eta_min']))
        
        scaler = GradScaler('cuda')

        best_val_loss = float('inf')
        best_model_state_dict = None
        best_epoch = -1

        for epoch in range(self.training_config['epochs']):
            # --- Training Phase ---
            model.train()
            running_total_loss = 0.0
            
            for batch_idx, (board_tensors, policy_target, value_targets) in enumerate(train_loader):
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
                    policy_log_softmax = F.log_softmax(policy_logits, dim=1)
                    value_outputs = value_outputs.squeeze(1)
                    torch.cuda.synchronize()
                    forward_pass_end = time.perf_counter()

                    policy_loss = policy_criterion(policy_log_softmax, policy_target)
                    value_loss = value_criterion(value_outputs, value_targets)
                    
                    total_loss = (policy_loss * self.training_config['policy_loss_weight']) + \
                                (value_loss * self.training_config['value_loss_weight'])

                backward_pass_start = time.perf_counter()
                scaler.scale(total_loss).backward()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                backward_pass_end = time.perf_counter()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                running_total_loss += total_loss.item()
                
                batch_end_time = time.perf_counter()
                
                # Log detailed info to file at intervals
                if (batch_idx + 1) % self.training_config['log_interval'] == 0:
                    self.logger.info(f"Training Epoch {epoch+1}, Batch {batch_idx+1}: "
                                     f"P_Loss={policy_loss.item():.4f}, "
                                     f"V_Loss={value_loss.item():.4f}, "
                                     f"T_Loss={total_loss.item():.4f}, "
                                     f"LR={optimizer.param_groups[0]['lr']:.6f}, "
                                     f"GPU Xfer: {(transfer_to_gpu_end - transfer_to_gpu_start)*1000:.2f}ms, "
                                     f"FW: {(forward_pass_end - forward_pass_start)*1000:.2f}ms, "
                                     f"BW: {(backward_pass_end - backward_pass_start)*1000:.2f}ms, "
                                     f"Batch Total: {(batch_end_time - batch_start_time)*1000:.2f}ms")

            avg_total_loss_train = running_total_loss / len(train_loader)
            self.logger.info(f"--- Epoch {epoch+1} Train Summary ---")
            self.logger.info(f"Average Total Loss: {avg_total_loss_train:.4f}")
            
            # --- Validation Phase ---
            model.eval()
            running_total_loss_val = 0.0

            # Replace tqdm(val_loader, ...) with val_loader
            with torch.no_grad():
                for batch_idx, (board_tensors, policy_target, value_targets) in enumerate(val_loader):
                    board_tensors = board_tensors.to(self.device, non_blocking=True)
                    policy_target = policy_target.to(self.device, non_blocking=True)
                    value_targets = value_targets.to(self.device, non_blocking=True)

                    with autocast('cuda'):
                        policy_logits, value_outputs = model(board_tensors)
                        value_outputs = value_outputs.squeeze(1)
                        
                        policy_log_softmax = F.log_softmax(policy_logits, dim=1)
                        policy_loss = policy_criterion(policy_log_softmax, policy_target)
                        value_loss = value_criterion(value_outputs, value_targets)
                        
                        total_loss = (policy_loss * self.training_config['policy_loss_weight']) + \
                                    (value_loss * self.training_config['value_loss_weight'])
                        
                    running_total_loss_val += total_loss.item()
                    
                    # Log detailed info to file at intervals
                    if (batch_idx + 1) % self.training_config['log_interval'] == 0:
                        self.logger.info(f"Validation Epoch {epoch+1}, Batch {batch_idx+1}: "
                                         f"P_Loss={policy_loss.item():.4f}, "
                                         f"V_Loss={value_loss.item():.4f}, "
                                         f"T_Loss={total_loss.item():.4f}")

            avg_total_loss_val = running_total_loss_val / len(val_loader)
            self.logger.info(f"--- Epoch {epoch+1} Validation Summary ---")
            self.logger.info(f"Average Total Loss: {avg_total_loss_val:.4f}")

            if avg_total_loss_val < best_val_loss:
                best_val_loss = avg_total_loss_val
                best_model_state_dict = model.state_dict()
                best_epoch = epoch
                self.logger.info(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}. Will save at the end.")

        best_model_path = os.path.join(self.output_dir, f"model_iter_{self.cycle_number}.pth")
        
        torch.save(best_model_state_dict, best_model_path)
        self.logger.info(f"Best model from epoch {best_epoch+1} saved to {best_model_path}")

        self.logger.info("Training complete for this task!")
        training_steps = len(train_loader) * best_epoch
        
        return best_model_path, training_steps  