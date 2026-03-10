import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import os
import sys
import numpy as np
import time

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

from src_shared.model import ChessAIModel
from src_shared.data_loader import ChessDataset

class TrainTask:
    def __init__(self, latest_model_path: str, best_model_path: str, model_config: dict, 
                 training_config: dict, state_config: dict, global_config: dict, buffer_dir: str):
        
        self.training_config = training_config
        self.latest_model_path = latest_model_path
        self.best_model_path = best_model_path
        self.model_config = model_config
        self.state_config = state_config
        self.global_config = global_config
        self.output_dir = None
        self.logger = None
        self.buffer_dir = buffer_dir
        
        self.last_log_dir = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reset_interval = self.training_config.get('reset_interval_steps', 40000)
        self.ema_tau = self.training_config.get('ema_tau', 0.005)
        self.shrink_alpha = self.training_config.get('shrink_alpha', 0.8)

        self.model = ChessAIModel(
            num_input_planes=self.model_config['input_planes'],
            num_residual_blocks=self.model_config['resblocks'],
            num_filters=self.model_config['filters'],
            bottleneck_channels=self.model_config['bottleneck_channels'],
            broadcast_reduction_ratio=self.model_config['broadcast_reduction_ratio'],
            broadcast_interval=self.model_config['broadcast_interval']
        ).to(self.device)

        self.ema_model = ChessAIModel(
            num_input_planes=self.model_config['input_planes'],
            num_residual_blocks=self.model_config['resblocks'],
            num_filters=self.model_config['filters'],
            bottleneck_channels=self.model_config['bottleneck_channels'],
            broadcast_reduction_ratio=self.model_config['broadcast_reduction_ratio'],
            broadcast_interval=self.model_config['broadcast_interval']
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=float(self.training_config['learning_rate']),
            weight_decay=float(self.training_config['weight_decay'])
        )
        self.scaler = GradScaler('cuda')
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

    def _get_dataloaders(self):
        total_positions = self.state_config['buffer']['count']
        batch_size = self.training_config['batch_size']
        
        # Grab exactly 1 batch worth of random indices
        indices = np.random.choice(total_positions, size=batch_size, replace=self.training_config['replacement']).tolist()

        full_dataset = ChessDataset(
            buffer_dir=self.buffer_dir, 
            indices=indices,
            max_capacity=self.global_config['buffer_size']
        ) 

        # We can safely use num_workers=0 because memmap random reads are instantaneous.
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader

    def _apply_network_resets(self):
        # (Keep your existing reset code here, omitted for brevity but remains unchanged)
        pass

    def _update_ema_network(self):
        with torch.no_grad():
            for ema_param, train_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.copy_(self.ema_tau * train_param.data + (1.0 - self.ema_tau) * ema_param.data)

    def run_single_step(self, current_log_dir, state_config):
        self.state_config = state_config['state']

        if current_log_dir != self.last_log_dir:
            self.output_dir = current_log_dir
            self.logger = self._setup_logger()
            self.last_log_dir = current_log_dir

        data_start = time.perf_counter()
        train_loader = self._get_dataloaders()
        batch = next(iter(train_loader))
        board_tensors, policy_target, value_targets = [t.to(self.device, non_blocking=True) for t in batch]
        data_time = (time.perf_counter() - data_start) * 1000

        global_step = self.state_config['lifetime']['training_steps']
        target_lr = float(self.training_config['learning_rate'])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = target_lr

        self.model.train()
        self.optimizer.zero_grad()
        
        fw_start = time.perf_counter()
        with torch.amp.autocast('cuda'):
            policy_logits, value_outputs = self.model(board_tensors)
            legal_mask = policy_target > 0.0
            policy_logits = policy_logits.masked_fill(~legal_mask, torch.finfo(policy_logits.dtype).min)
            policy_log_softmax = F.log_softmax(policy_logits, dim=1)
            policy_loss = self.policy_criterion(policy_log_softmax, policy_target)
            value_loss = self.value_criterion(value_outputs.squeeze(1), value_targets)
            total_loss = (policy_loss * self.training_config['policy_loss_weight']) + (value_loss * self.training_config['value_loss_weight'])
        fw_time = (time.perf_counter() - fw_start) * 1000

        bw_start = time.perf_counter()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        bw_time = (time.perf_counter() - bw_start) * 1000

        self.logger.info(f"Training Step {global_step+1}: P_Loss={(policy_loss.item() * self.training_config['policy_loss_weight']):.4f} | V_Loss={(value_loss.item() * self.training_config['value_loss_weight']):.4f} | T_Loss={total_loss.item():.4f} | LR={target_lr:.4f} | FW: {fw_time:.1f}ms | BW: {bw_time:.1f}ms | Data: {data_time:.1f}ms")

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