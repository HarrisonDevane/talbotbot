import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import time
import numpy as np
import lmdb

import zstandard as zstd

from model import ChessAIModel

class TrainTask:
    def __init__(self, model_path: str, model_config: dict, training_config: dict,
                 state_config: dict, global_config: dict, lmdb_path: str, env):
        self.training_config = training_config
        self.model_path = model_path
        self.model_config = model_config
        self.state_config = state_config
        self.global_config = global_config
        self.lmdb_path = lmdb_path
        self.env = env 
        
        self.output_dir = None
        self.logger = None
        self.last_log_dir = None
        self.tb_writer = None
        self.device = torch.device("cuda")
        
        self.decompressor = zstd.ZstdDecompressor()

        m_cfg = self.model_config['model']
        c_cfg = self.model_config['chess']

        self.input_planes = m_cfg['input_planes']
        self.board_dim = c_cfg['board_dim']
        self.total_input_size = self.input_planes * self.board_dim * self.board_dim
        self.total_policy_moves = c_cfg['total_policy_moves']

        self.board_bytes = (self.total_input_size + 7) // 8
        self.mask_bytes = (self.total_policy_moves + 7) // 8

        self.model = ChessAIModel(self.model_config).to(self.device)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=float(self.training_config['learning_rate']),
            momentum=self.training_config['momentum_rate'],
            weight_decay=float(self.training_config['weight_decay'])
        )

        self.scaler = GradScaler('cuda')

        self.policy_criterion = nn.KLDivLoss(reduction='batchmean')
        self.value_criterion = nn.MSELoss()

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

    def _setup_logger(self):
        logger = logging.getLogger("TrainTask")
        logger.setLevel(self.training_config['log_level'])
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        log_file_path = os.path.join(self.output_dir, "training_run.log")
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }, path)

    def run_single_step(self, current_log_dir: str, state_config: dict):
        self.state_config = state_config

        if current_log_dir != self.last_log_dir:
            self.output_dir = current_log_dir
            self.logger = self._setup_logger()

            if self.tb_writer is not None:
                self.tb_writer.close()
            tb_dir = os.path.join(current_log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

            self.last_log_dir = current_log_dir

        data_start = time.perf_counter()
        batch_size = self.training_config['batch_size']

        boards, policies, values, masks = [], [], [], []

        # Uses the shared environment. read-only transaction, so it doesn't block C++ writes.
        with self.env.begin(write=False, buffers=True) as txn:
            actual_count = txn.stat()['entries']
            indices = np.random.randint(0, actual_count, size=batch_size)

            for idx in indices:
                compressed_blob = txn.get(f"{idx}".encode('ascii'))

                if compressed_blob is None:
                    for _ in range(10):
                        new_idx = np.random.randint(0, actual_count)
                        compressed_blob = txn.get(f"{new_idx}".encode('ascii'))
                        if compressed_blob is not None:
                            break
                    if compressed_blob is None:
                        continue

                buf = self.decompressor.decompress(compressed_blob)

                num_moves = np.frombuffer(buf[0:2], dtype=np.uint16)[0]

                off_board = 2 + self.board_bytes
                off_mask  = off_board + self.mask_bytes
                off_idx   = off_mask  + (2 * num_moves)
                off_val   = off_idx   + (2 * num_moves)

                board_bits = np.frombuffer(buf[2:off_board], dtype=np.uint8)
                board = np.unpackbits(board_bits)[:self.total_input_size]
                boards.append(board.reshape(self.input_planes, self.board_dim, self.board_dim).astype(np.float32))

                mask_bits = np.frombuffer(buf[off_board:off_mask], dtype=np.uint8)
                masks.append(np.unpackbits(mask_bits)[:self.total_policy_moves].astype(np.bool_))

                pi_vec = np.zeros(self.total_policy_moves, dtype=np.float32)
                pi_vec[np.frombuffer(buf[off_mask:off_idx], dtype=np.uint16)] = \
                    np.frombuffer(buf[off_idx:off_val], dtype=np.float16)
                policies.append(pi_vec)

                values.append(np.frombuffer(buf[off_val:off_val + 2], dtype=np.float16))

        board_tensors    = torch.from_numpy(np.stack(boards)).to(self.device, non_blocking=True)
        policy_target    = torch.from_numpy(np.stack(policies)).to(self.device, non_blocking=True)
        value_targets    = torch.from_numpy(np.stack(values)).float().to(self.device, non_blocking=True).flatten()
        true_legal_masks = torch.from_numpy(np.stack(masks)).to(self.device, non_blocking=True)

        data_time = (time.perf_counter() - data_start) * 1000

        global_step = self.state_config['lifetime']['training_steps']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(self.training_config['learning_rate'])

        self.model.train()
        self.optimizer.zero_grad()

        fw_start = time.perf_counter()
        with torch.amp.autocast('cuda'):
            policy_logits, value_outputs = self.model(board_tensors)

        policy_logits = policy_logits.float().masked_fill(~true_legal_masks, -1e9)
        policy_log_softmax = F.log_softmax(policy_logits, dim=1)

        policy_loss = self.policy_criterion(policy_log_softmax, policy_target.float())
        value_loss  = self.value_criterion(value_outputs.squeeze(1).float(), value_targets.float())

        total_loss = (policy_loss * self.training_config['policy_loss_weight']) + \
                     (value_loss  * self.training_config['value_loss_weight'])

        with torch.no_grad():
            policy_probs   = torch.exp(policy_log_softmax)
            policy_entropy = -(policy_probs * policy_log_softmax).sum(dim=1).mean()
            v_out_mean     = value_outputs.mean()
            v_tar_mean     = value_targets.mean()

        fw_time = (time.perf_counter() - fw_start) * 1000

        bw_start = time.perf_counter()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        bw_time = (time.perf_counter() - bw_start) * 1000

        self.logger.info(
            f"Step {global_step + 1} | "
            f"Loss: T={total_loss.item():.4f} "
            f"(P={(policy_loss.item() * self.training_config['policy_loss_weight']):.4f}, "
            f"V={(value_loss.item() * self.training_config['value_loss_weight']):.4f}) | "
            f"P_Ent={policy_entropy.item():.4f} | "
            f"V_Out={v_out_mean.item():.4f} | V_Tar={v_tar_mean.item():.4f} | "
            f"Grad={grad_norm.item():.2f} | LR={self.optimizer.param_groups[0]['lr']:.6f} | "
            f"ms: Data={data_time:.1f} FW={fw_time:.1f} BW={bw_time:.1f}"
        )

        if self.tb_writer is not None:
            self.tb_writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            self.tb_writer.add_scalar('Loss/Policy', policy_loss.item() * self.training_config['policy_loss_weight'], global_step)
            self.tb_writer.add_scalar('Loss/Value', value_loss.item() * self.training_config['value_loss_weight'], global_step)
            self.tb_writer.add_scalar('Metrics/Policy_Entropy', policy_entropy.item(), global_step)
            self.tb_writer.add_scalar('Metrics/Value_Output_Mean', v_out_mean.item(), global_step)
            self.tb_writer.add_scalar('Metrics/Value_Target_Mean', v_tar_mean.item(), global_step)
            self.tb_writer.add_scalar('Metrics/Grad_Norm', grad_norm.item(), global_step)
            self.tb_writer.add_scalar('Hardware_MS/Data_Load', data_time, global_step)
            self.tb_writer.add_scalar('Hardware_MS/Forward', fw_time, global_step)
            self.tb_writer.add_scalar('Hardware_MS/Backward', bw_time, global_step)