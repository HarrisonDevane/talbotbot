import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import time
import numpy as np
import h5py

# Ensure project root is in path for imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

from src_shared.model import ChessAIModel


class TrainTask:
    def __init__(self, best_model_path: str, model_config: dict, 
                 training_config: dict, state_config: dict, global_config: dict, hdf5_path: str):
        
        self.training_config = training_config
        self.best_model_path = best_model_path
        self.model_config = model_config
        self.state_config = state_config
        self.global_config = global_config
        self.output_dir = None
        self.logger = None
        self.hdf5_path = hdf5_path
        
        self.last_log_dir = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Initialize Model Architecture once
        self.model = ChessAIModel(
            num_input_planes=self.model_config['input_planes'],
            num_residual_blocks=self.model_config['resblocks'],
            num_filters=self.model_config['filters'],
            bottleneck_channels=self.model_config['bottleneck_channels'],
            broadcast_reduction_ratio=self.model_config['broadcast_reduction_ratio'],
            broadcast_interval=self.model_config['broadcast_interval']
        ).to(self.device)

        # 2. Get static LR for initialization (overridden during step)
        initial_lr = float(self.training_config['learning_rate'])

        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=initial_lr,
            momentum=self.training_config['momentum_rate'],
            weight_decay=float(self.training_config['weight_decay'])
        )

        self.scaler = GradScaler('cuda')

        # 3. Criteria
        self.policy_criterion = nn.KLDivLoss(reduction='batchmean')
        self.value_criterion = nn.MSELoss()
                
        # 4. LOAD WEIGHTS ONCE
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])


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


    def run_single_step(self, current_log_dir, state_config):
        """
        Performs exactly one gradient update using high-speed HDF5 bulk reading.
        """
        self.state_config = state_config['state']

        # 1. Logging Rotation
        if current_log_dir != self.last_log_dir:
            self.output_dir = current_log_dir
            self.logger = self._setup_logger()

            if hasattr(self, 'tb_writer') and self.tb_writer is not None:
                self.tb_writer.close()
            tb_dir = os.path.join(current_log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

            self.last_log_dir = current_log_dir

        # 2. Get Fresh Data (Bulk Read + Replacement)
        data_start = time.perf_counter()
        
        batch_size = self.training_config['batch_size']
        total_positions = self.state_config['buffer']['count']
        
        if total_positions == 0:
            raise ValueError("Replay buffer is empty. Cannot perform training step.")

        # True random sample WITH replacement
        raw_indices = np.random.choice(total_positions, size=batch_size, replace=True)
        
        # Extract unique indices for HDF5, and get the inverse map to rebuild the duplicates
        unique_indices, inverse_indices = np.unique(raw_indices, return_inverse=True)

        # Do ONE massive disk read using only the unique, sorted indices
        with h5py.File(self.hdf5_path, 'r') as hf:
            boards_unique = hf['inputs'][unique_indices]
            policies_unique = hf['policies'][unique_indices]
            values_unique = hf['values'][unique_indices]
            masks_unique = hf['legal_masks'][unique_indices]

        # Expand the unique data back to the full batch size (restoring duplicates & random order)
        boards_np = boards_unique[inverse_indices]
        policies_np = policies_unique[inverse_indices]
        values_np = values_unique[inverse_indices]
        masks_np = masks_unique[inverse_indices]

        # Convert to PyTorch and send directly to VRAM
        board_tensors = torch.from_numpy(boards_np).float().to(self.device, non_blocking=True)
        policy_target = torch.from_numpy(policies_np).float().to(self.device, non_blocking=True)
        value_targets = torch.tensor(values_np, dtype=torch.float32).to(self.device, non_blocking=True)
        true_legal_masks = torch.from_numpy(masks_np).bool().to(self.device, non_blocking=True)
        
        data_time = (time.perf_counter() - data_start) * 1000

        # 3. Dynamic LR Update
        global_step = self.state_config['lifetime']['training_steps']
        target_lr = float(self.training_config['learning_rate'])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = target_lr

        # 4. Training Step
        self.model.train()
        self.optimizer.zero_grad()
        
        fw_start = time.perf_counter()
        with torch.amp.autocast('cuda'):
            policy_logits, value_outputs = self.model(board_tensors)

        # Step outside autocast for numerically sensitive ops
        policy_logits = policy_logits.float()  # promote to FP32
        policy_logits = policy_logits.masked_fill(~true_legal_masks, -1e9)
        policy_log_softmax = F.log_softmax(policy_logits, dim=1)

        policy_loss = self.policy_criterion(policy_log_softmax, policy_target.float())
        value_loss = self.value_criterion(value_outputs.squeeze(1).float(), value_targets.float())

        total_loss = (policy_loss * self.training_config['policy_loss_weight']) + \
                    (value_loss * self.training_config['value_loss_weight'])
            
        # Additional Stats Calculation (No Grad)
        with torch.no_grad():
            # Entropy: sum(-p * log(p))
            policy_probs = torch.exp(policy_log_softmax)
            policy_entropy = -(policy_probs * policy_log_softmax).sum(dim=1).mean()
            
            # Value Means
            v_out_mean = value_outputs.mean()
            v_tar_mean = value_targets.mean()
            
        fw_time = (time.perf_counter() - fw_start) * 1000

        # 5. Backprop
        bw_start = time.perf_counter()
        self.scaler.scale(total_loss).backward()
        
        # Unscale before computing grad norm
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        bw_time = (time.perf_counter() - bw_start) * 1000

        self.logger.info(
            f"Step {global_step+1} | "
            f"Loss: T={total_loss.item():.4f} (P={(policy_loss.item() * self.training_config['policy_loss_weight']):.4f}, V={(value_loss.item() * self.training_config['value_loss_weight']):.4f}) | "
            f"P_Ent={policy_entropy.item():.4f} | "
            f"V_Out(mean)={v_out_mean.item():.4f} | "
            f"V_Tar(mean)={v_tar_mean.item():.4f} | "
            f"GradNorm={grad_norm.item():.2f} | "
            f"LR={self.optimizer.param_groups[0]['lr']:.4f} | "
            f"ms: Data={data_time:.1f} FW={fw_time:.1f} BW={bw_time:.1f}"
        )

        if self.tb_writer:
            # Losses
            self.tb_writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            self.tb_writer.add_scalar('Loss/Policy', (policy_loss.item() * self.training_config['policy_loss_weight']), global_step)
            self.tb_writer.add_scalar('Loss/Value', (value_loss.item() * self.training_config['value_loss_weight']), global_step)
            
            # Key AlphaZero Metrics
            self.tb_writer.add_scalar('Metrics/Policy_Entropy', policy_entropy.item(), global_step)
            self.tb_writer.add_scalar('Metrics/Value_Target_Mean', v_tar_mean.item(), global_step)
            self.tb_writer.add_scalar('Metrics/Value_Output_Mean', v_out_mean.item(), global_step)
            
            # System / Optimizer Health
            self.tb_writer.add_scalar('System/GradNorm', grad_norm.item(), global_step)
            self.tb_writer.add_scalar('System/LearningRate', self.optimizer.param_groups[0]['lr'], global_step)
            
            # Hardware Bottleneck Tracking
            self.tb_writer.add_scalar('Hardware_MS/Data_Load', data_time, global_step)
            self.tb_writer.add_scalar('Hardware_MS/Forward_Pass', fw_time, global_step)
            self.tb_writer.add_scalar('Hardware_MS/Backward_Pass', bw_time, global_step)

        # 6. Save Weights directly from VRAM
        updated_model_path = os.path.join(self.output_dir, "updated_weights.pth")
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict()}, updated_model_path)

        return updated_model_path