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
import lz4.frame

# Ensure project root is in path for imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

from src_shared.model import ChessAIModel
import src_shared.utils as utils


class TrainTask:
    def __init__(self, best_model_path: str, model_config: dict, 
                 training_config: dict, state_config: dict, global_config: dict, lmdb_path: str, env):
        
        self.training_config = training_config
        self.best_model_path = best_model_path
        self.model_config = model_config
        self.state_config = state_config
        self.global_config = global_config
        self.output_dir = None
        self.logger = None
        self.lmdb_path = lmdb_path
        
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

        self.env = env

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
        Performs exactly one gradient update using high-speed LMDB + LZ4 random access.
        """
        self.state_config = state_config['state']

        # 1. Logging Rotation (Handles TensorBoard and Folder logic)
        if current_log_dir != self.last_log_dir:
            self.output_dir = current_log_dir
            self.logger = self._setup_logger()

            if hasattr(self, 'tb_writer') and self.tb_writer is not None:
                self.tb_writer.close()
            tb_dir = os.path.join(current_log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

            self.last_log_dir = current_log_dir

        # 2. Get Fresh Data (Random Key Access)
        data_start = time.perf_counter()
        
        batch_size = self.training_config['batch_size']
        total_positions = self.state_config['buffer']['count']

        # Generate batch_size random indices within the current valid buffer range
        indices = np.random.randint(0, total_positions, size=batch_size)
        
        boards, policies, values, masks = [], [], [], []

        with self.env.begin(write=False, buffers=True) as txn:
            for idx in indices:
                key = f"{idx}".encode('ascii')
                compressed_blob = txn.get(key)
                
                # FIX: Re-sample until we get a valid key to maintain exact batch size
                while not compressed_blob:
                    new_idx = np.random.randint(0, total_positions)
                    compressed_blob = txn.get(f"{new_idx}".encode('ascii'))

                # Decompress to raw buffer
                buf = lz4.frame.decompress(compressed_blob)
                
                # 1. Read Header and set offsets correctly
                num_moves = np.frombuffer(buf[0:2], dtype=np.uint16)[0]

                off_board = 2 + utils.BOARD_BYTES
                off_mask  = off_board + utils.MASK_BYTES
                off_idx   = off_mask  + (2 * num_moves)
                off_val   = off_idx   + (2 * num_moves)

                # 2. Reconstruct Board (Now buf[2:554] will correctly pull the board)
                board_bits = np.frombuffer(buf[2:off_board], dtype=np.uint8)
                board = np.unpackbits(board_bits)[:utils.TOTAL_INPUT_SIZE]
                boards.append(board.reshape(utils.INPUT_CHANNELS, 8, 8).astype(np.float32))

                # 3. Reconstruct Mask
                mask_bits = np.frombuffer(buf[off_board:off_mask], dtype=np.uint8)
                masks.append(np.unpackbits(mask_bits)[:utils.TOTAL_POLICY_MOVES].astype(np.bool_))

                # 4. Reconstruct Policy (Scatter)
                pi_vec = np.zeros(utils.TOTAL_POLICY_MOVES, dtype=np.float32)
                pi_vec[np.frombuffer(buf[off_mask:off_idx], dtype=np.uint16)] = \
                    np.frombuffer(buf[off_idx:off_val], dtype=np.float16)
                policies.append(pi_vec)

                # 5. Value
                values.append(np.frombuffer(buf[off_val:off_val+2], dtype=np.float16))


        # Stack lists into batch tensors for the GPU
        board_tensors = torch.from_numpy(np.stack(boards)).to(self.device, non_blocking=True)
        policy_target = torch.from_numpy(np.stack(policies)).to(self.device, non_blocking=True)
        value_targets = torch.from_numpy(np.stack(values)).float().to(self.device, non_blocking=True).flatten()
        true_legal_masks = torch.from_numpy(np.stack(masks)).to(self.device, non_blocking=True)
        
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

        # Numerical Stability: Step outside autocast and promote to FP32
        policy_logits = policy_logits.float() 
        policy_logits = policy_logits.masked_fill(~true_legal_masks, -1e9)
        policy_log_softmax = F.log_softmax(policy_logits, dim=1)

        policy_loss = self.policy_criterion(policy_log_softmax, policy_target.float())
        value_loss = self.value_criterion(value_outputs.squeeze(1).float(), value_targets.float())

        total_loss = (policy_loss * self.training_config['policy_loss_weight']) + \
                    (value_loss * self.training_config['value_loss_weight'])
            
        # Stats Calculation
        with torch.no_grad():
            policy_probs = torch.exp(policy_log_softmax)
            policy_entropy = -(policy_probs * policy_log_softmax).sum(dim=1).mean()
            v_out_mean = value_outputs.mean()
            v_tar_mean = value_targets.mean()
            
        fw_time = (time.perf_counter() - fw_start) * 1000

        # 5. Backprop
        bw_start = time.perf_counter()
        self.scaler.scale(total_loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        bw_time = (time.perf_counter() - bw_start) * 1000

        self.logger.info(
            f"Step {global_step+1} | "
            f"Loss: T={total_loss.item():.4f} (P={(policy_loss.item() * self.training_config['policy_loss_weight']):.4f}, V={(value_loss.item() * self.training_config['value_loss_weight']):.4f}) | "
            f"P_Ent={policy_entropy.item():.4f} | "
            f"V_Out={v_out_mean.item():.4f} | V_Tar={v_tar_mean.item():.4f} | "
            f"Grad={grad_norm.item():.2f} | LR={self.optimizer.param_groups[0]['lr']:.4f} | "
            f"ms: Data={data_time:.1f} FW={fw_time:.1f} BW={bw_time:.1f}"
        )

        if self.tb_writer:
            self.tb_writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            self.tb_writer.add_scalar('Metrics/Policy_Entropy', policy_entropy.item(), global_step)
            self.tb_writer.add_scalar('Hardware_MS/Data_Load', data_time, global_step)

        # 6. Save Weights (Final step)
        updated_model_path = os.path.join(self.output_dir, "updated_weights.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }, updated_model_path)

        return updated_model_path