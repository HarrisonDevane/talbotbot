import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import os
import sys
import random
import numpy as np

# Ensure project root is in path for imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

# Assuming these imports are correct based on your file structure
from src_shared.model import ChessAIModel
from src_shared.data_loader import ChessDataset


class TrainTask:
    def __init__(self, latest_model_path: str, best_model_path: str, model_config: dict, 
                 training_config: dict, state_config: dict, global_config: dict, hdf5_path: str):
        
        self.training_config = training_config
        self.latest_model_path = latest_model_path
        self.best_model_path = best_model_path
        self.model_config = model_config
        self.state_config = state_config
        self.global_config = global_config
        self.output_dir = None
        self.logger = None
        self.hdf5_path = hdf5_path
        
        self.last_log_dir = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration for Resets and EMA
        self.reset_interval = self.training_config.get('reset_interval_steps', 40000)
        self.ema_tau = self.training_config.get('ema_tau', 0.005)
        self.shrink_alpha = self.training_config.get('shrink_alpha', 0.8)

        # 1. Initialize Active Training Model
        self.model = ChessAIModel(
            num_input_planes=self.model_config['input_planes'],
            num_residual_blocks=self.model_config['resblocks'],
            num_filters=self.model_config['filters'],
            bottleneck_channels=self.model_config['bottleneck_channels'],
            broadcast_reduction_ratio=self.model_config['broadcast_reduction_ratio'],
            broadcast_interval=self.model_config['broadcast_interval']
        ).to(self.device)

        # 2. Initialize EMA Target Model
        self.ema_model = ChessAIModel(
            num_input_planes=self.model_config['input_planes'],
            num_residual_blocks=self.model_config['resblocks'],
            num_filters=self.model_config['filters'],
            bottleneck_channels=self.model_config['bottleneck_channels'],
            broadcast_reduction_ratio=self.model_config['broadcast_reduction_ratio'],
            broadcast_interval=self.model_config['broadcast_interval']
        ).to(self.device)

        # 3. Initialize static Adam Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=float(self.training_config['learning_rate']),
            weight_decay=float(self.training_config['weight_decay'])
        )

        self.scaler = GradScaler('cuda')

        # 4. Criteria
        self.policy_criterion = nn.KLDivLoss(reduction='batchmean')
        self.value_criterion = nn.MSELoss()

        checkpoint = torch.load(self.latest_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        ema_checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.ema_model.load_state_dict(ema_checkpoint['model_state_dict'])
        self.ema_model.eval()


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
        total_positions_source = self.state_config['buffer']['count']
        all_sample_indices = np.arange(total_positions_source)
        total_samples = self.training_config['batch_size']
        
        self.logger.debug(f"Target samples to sample: {total_samples}")
        self.logger.debug("Using uniform sampling from all available HDF5 samples.")
        
        final_indices = np.random.choice(
            all_sample_indices, 
            size=total_samples, 
            replace=self.training_config['replacement']
        ).tolist()
        
        random.shuffle(final_indices)
        self.logger.debug(f"Successfully generated {len(final_indices)} globally shuffled, sample-prioritized indices.")
        
        return final_indices


    def _get_dataloaders(self):
        self.training_indices = self._get_shuffled_sample_indices()

        full_dataset = ChessDataset(
            hdf5_path=self.hdf5_path, 
            indices=self.training_indices
        ) 

        batch_size = self.training_config['batch_size']

        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        return train_loader, full_dataset


    def _apply_network_resets(self):
        """
        Applies Shrink and Perturb to the ENTIRE network.
        Wipes the Adam moment estimates for all parameters to prevent immediate corruption.
        """
        self.logger.warning(f"Applying global network soft resets (Shrink and Perturb alpha={self.shrink_alpha}).")
        
        dummy_model = ChessAIModel(
            num_input_planes=self.model_config['input_planes'],
            num_residual_blocks=self.model_config['resblocks'],
            num_filters=self.model_config['filters'],
            bottleneck_channels=self.model_config['bottleneck_channels'],
            broadcast_reduction_ratio=self.model_config['broadcast_reduction_ratio'],
            broadcast_interval=self.model_config['broadcast_interval']
        ).to(self.device)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                dummy_param = dict(dummy_model.named_parameters())[name]
                
                # Apply Shrink and Perturb to ALL parameters, including heads
                param.data.copy_(self.shrink_alpha * param.data + (1.0 - self.shrink_alpha) * dummy_param.data)

                if param in self.optimizer.state:
                    if 'exp_avg' in self.optimizer.state[param]:
                        self.optimizer.state[param]['exp_avg'].zero_()
                    if 'exp_avg_sq' in self.optimizer.state[param]:
                        self.optimizer.state[param]['exp_avg_sq'].zero_()

    def _update_ema_network(self):
        """
        Updates the EMA target network weights towards the active training model.
        """
        with torch.no_grad():
            for ema_param, train_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.copy_(self.ema_tau * train_param.data + (1.0 - self.ema_tau) * ema_param.data)


    def run_single_step(self, current_log_dir, state_config):
        """
        Performs exactly one gradient update. 
        Rebuilds DataLoader each time to ensure perfectly fresh sampling.
        """
        self.state_config = state_config['state']

        # 1. Logging Rotation
        if current_log_dir != self.last_log_dir:
            self.output_dir = current_log_dir
            self.logger = self._setup_logger()
            self.last_log_dir = current_log_dir

        # 2. Get Fresh Data
        train_loader, _ = self._get_dataloaders()
        batch = next(iter(train_loader))
        board_tensors, policy_target, value_targets = [t.to(self.device, non_blocking=True) for t in batch]

        # 3. Static LR 
        global_step = self.state_config['lifetime']['training_steps']
        target_lr = float(self.training_config['learning_rate'])
            
        # Force the loaded optimizer to use the static config learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = target_lr
        
        self.logger.debug(f"Optimizer state loaded. Learning Rate: {target_lr}")

        # --- APPLY NETWORK RESETS ---
        if global_step > 0 and global_step % self.reset_interval == 0:
            self._apply_network_resets()

        # 5. Training Step
        self.model.train()
        self.optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            policy_logits, value_outputs = self.model(board_tensors)
            
            legal_mask = policy_target > 0.0
            policy_logits = policy_logits.masked_fill(~legal_mask, torch.finfo(policy_logits.dtype).min)
            policy_log_softmax = F.log_softmax(policy_logits, dim=1)
            
            policy_loss = self.policy_criterion(policy_log_softmax, policy_target)
            value_loss = self.value_criterion(value_outputs.squeeze(1), value_targets)
            total_loss = (policy_loss * self.training_config['policy_loss_weight']) + \
                         (value_loss * self.training_config['value_loss_weight'])
            
            self.logger.info(f"Training Step {global_step+1}: "
                    f"P_Loss={(policy_loss.item() * self.training_config['policy_loss_weight']):.4f}, "
                    f"V_Loss={(value_loss.item() * self.training_config['value_loss_weight']):.4f}, "
                    f"T_Loss={total_loss.item():.4f}, "
                    f"LR={self.optimizer.param_groups[0]['lr']:.4f}")

        # 6. Backprop
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # --- UPDATE EMA TARGET NETWORK ---
        self._update_ema_network()

        updated_latest_path = os.path.join(self.output_dir, "updated_latest.pth")
        updated_best_path = os.path.join(self.output_dir, "updated_best.pth")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }, updated_latest_path)

        if global_step % self.training_config['sync_interval'] == 0:
            torch.save({'model_state_dict': self.ema_model.state_dict()}, updated_best_path)

            return updated_latest_path, updated_best_path

        return updated_latest_path, None