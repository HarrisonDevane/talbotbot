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
import torch.multiprocessing as mp 
import concurrent.futures
import psutil
import struct
import traceback
import threading
import ctypes

from model import ChessAIModel

class AsyncBatchPrefetcher:
    def __init__(self, db_path, batch_size, input_planes, board_dim, policy_moves, core_ids, prefetch_workers, rl_dir, log_level, rotation_interval):
        self.ready_queue = mp.Queue(maxsize=3) 
        self.free_queue = mp.Queue(maxsize=3)
        for i in range(3):
            self.free_queue.put(i)

        self.db_path = db_path
        self.batch_size = batch_size
        self.input_planes = input_planes
        self.board_dim = board_dim
        self.policy_moves = policy_moves
        self.core_ids = core_ids
        self.prefetch_workers = prefetch_workers
        self.rl_dir = rl_dir
        self.log_level = log_level
        self.rotation_interval = rotation_interval
        
        # Allocate exactly 3 static buffers in Shared Memory. This locks the RAM usage permanently.
        self.shm_b = torch.zeros((3, batch_size, input_planes, board_dim, board_dim), dtype=torch.float32).share_memory_()
        self.shm_p = torch.zeros((3, batch_size, policy_moves), dtype=torch.float32).share_memory_()
        self.shm_v = torch.zeros((3, batch_size), dtype=torch.float16).share_memory_()
        self.shm_m = torch.zeros((3, batch_size, policy_moves), dtype=torch.bool).share_memory_()
        self.shm_valid = torch.zeros(3, dtype=torch.int32).share_memory_()

        self.worker = mp.Process(target=self._worker_loop)
        self.worker.daemon = True
        self.worker.start()

    def _worker_loop(self):
        logger = logging.getLogger("Prefetcher")
        logger.setLevel(self.log_level)
        logger.propagate = False

        current_log_dir = None

        def setup_or_rotate_logger(step):
            nonlocal current_log_dir
            target_folder = (step // self.rotation_interval) * self.rotation_interval
            new_log_dir = os.path.join(self.rl_dir, f"run_step_{target_folder:06d}")

            if new_log_dir != current_log_dir:
                os.makedirs(new_log_dir, exist_ok=True)
                current_log_dir = new_log_dir

                if logger.hasHandlers():
                    logger.handlers.clear()

                formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [PREFETCH] %(message)s")
                log_file_path = os.path.join(current_log_dir, "prefetcher.log")
                file_handler = logging.FileHandler(log_file_path, mode='a')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                logger.info(f"Prefetcher logger rotated to {current_log_dir}")

        try:
            psutil.Process().cpu_affinity(self.core_ids)

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetProcessWorkingSetSize(handle, ctypes.c_size_t(4 * 1024 * 1024 * 1024), ctypes.c_size_t(8 * 1024 * 1024 * 1024))

            env = lmdb.open(
                self.db_path, 
                map_size=1024 * 1024 * 1024 * 16,
                readonly=True, 
                lock=False, 
                readahead=False
            )

            board_bytes = ((self.input_planes * self.board_dim * self.board_dim) + 7) // 8
            mask_bytes = (self.policy_moves + 7) // 8
            total_input_size = self.input_planes * self.board_dim * self.board_dim

            thread_local = threading.local()

            def fetch_single(idx):
                if not hasattr(thread_local, "decompressor"):
                    thread_local.decompressor = zstd.ZstdDecompressor()
                with env.begin(write=False, buffers=True) as txn:
                    compressed_blob = txn.get(f"{idx}".encode('ascii'))
                    if compressed_blob is None:
                        return None
                    return thread_local.decompressor.decompress(compressed_blob)

            setup_or_rotate_logger(0)
            logger.info("Core affinity set successfully. Connected to LMDB.")

            b_arr = np.empty((self.batch_size, self.input_planes, self.board_dim, self.board_dim), dtype=np.float32)
            p_arr = np.empty((self.batch_size, self.policy_moves), dtype=np.float32)
            v_arr = np.empty(self.batch_size, dtype=np.float16)
            m_arr = np.empty((self.batch_size, self.policy_moves), dtype=np.bool_)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.prefetch_workers) as executor:
                while True:
                    # Wait for the PyTorch thread to finish with a buffer
                    slot = self.free_queue.get()

                    with env.begin(write=False) as txn:
                        cpp_blob = txn.get(b"__CPP_STATE")
                        py_blob  = txn.get(b"__PY_STATE")

                        if not cpp_blob or not py_blob:
                            time.sleep(1)
                            self.free_queue.put(slot)
                            continue

                        actual_count  = struct.unpack('QQdQQQ', cpp_blob)[3]
                        current_step  = struct.unpack('Qd', py_blob)[0]

                    setup_or_rotate_logger(current_step)

                    if actual_count < self.batch_size:
                        logger.debug(f"Buffer size ({actual_count}) < Batch Size ({self.batch_size}). Waiting...")
                        time.sleep(1)
                        self.free_queue.put(slot)
                        continue

                    t_start = time.perf_counter()
                    indices = np.random.randint(0, actual_count, size=self.batch_size)
                    results = list(executor.map(fetch_single, indices))

                    t_fetch = (time.perf_counter() - t_start) * 1000

                    p_arr.fill(0.0)

                    valid_idx = 0
                    for buf in results:
                        if buf is None:
                            continue

                        num_moves = np.frombuffer(buf[0:2], dtype=np.uint16)[0]
                        off_board = 2 + board_bytes
                        off_mask  = off_board + mask_bytes
                        off_idx   = off_mask  + (2 * num_moves)
                        off_val   = off_idx   + (2 * num_moves)

                        board_bits = np.frombuffer(buf[2:off_board], dtype=np.uint8)
                        b_arr[valid_idx] = np.unpackbits(board_bits)[:total_input_size].reshape(self.input_planes, self.board_dim, self.board_dim)

                        mask_bits = np.frombuffer(buf[off_board:off_mask], dtype=np.uint8)
                        m_arr[valid_idx] = np.unpackbits(mask_bits)[:self.policy_moves]

                        move_indices = np.frombuffer(buf[off_mask:off_idx], dtype=np.uint16)
                        move_probs   = np.frombuffer(buf[off_idx:off_val], dtype=np.float16)
                        p_arr[valid_idx, move_indices] = move_probs

                        v_arr[valid_idx] = np.frombuffer(buf[off_val:off_val + 2], dtype=np.float16)[0]
                        valid_idx += 1

                    t_unpack = (time.perf_counter() - (t_start + (t_fetch / 1000))) * 1000

                    # Copy directly into the pre-allocated Shared Memory slot
                    self.shm_b[slot][:valid_idx].copy_(torch.from_numpy(b_arr[:valid_idx]))
                    self.shm_p[slot][:valid_idx].copy_(torch.from_numpy(p_arr[:valid_idx]))
                    self.shm_v[slot][:valid_idx].copy_(torch.from_numpy(v_arr[:valid_idx]))
                    self.shm_m[slot][:valid_idx].copy_(torch.from_numpy(m_arr[:valid_idx]))
                    self.shm_valid[slot] = valid_idx

                    logger.debug(f"Assembled batch of {valid_idx}. Fetch: {t_fetch:.1f}ms | Unpack: {t_unpack:.1f}ms. Pushing slot {slot} to queue...")

                    self.ready_queue.put(slot)

        except Exception as e:
            if logger:
                logger.critical(f"FATAL PREFETCHER ERROR: {str(e)}")
                logger.critical(traceback.format_exc())
            raise

class TrainTask:
    def __init__(self, model_path: str, model_config: dict, training_config: dict,
                 state_config: dict, global_config: dict, env, db_path: str):
        self.training_config = training_config
        self.model_path = model_path
        self.model_config = model_config
        self.state_config = state_config
        self.global_config = global_config
        self.env = env 
        
        self.output_dir = None
        self.logger = None
        self.last_log_dir = None
        self.tb_writer = None
        self.device = torch.device("cuda")

        m_cfg = self.model_config['model']
        c_cfg = self.model_config['chess']

        self.input_planes = m_cfg['input_planes']
        self.board_dim = c_cfg['board_dim']
        self.total_input_size = self.input_planes * self.board_dim * self.board_dim
        self.total_policy_moves = c_cfg['total_policy_moves']

        rl_dir = os.path.dirname(db_path)
        log_level = self.training_config.get('logging_level', 20)
        rotation_interval = self.global_config['logging_rotation_steps']

        # Pass the rotation interval so the prefetcher knows when to switch folders
        self.prefetcher = AsyncBatchPrefetcher(
            db_path=db_path,
            batch_size=self.training_config['batch_size'],
            input_planes=self.input_planes,
            board_dim=self.board_dim,
            policy_moves=self.total_policy_moves,
            core_ids=self.training_config.get('io_read_cores', [3, 4]),
            prefetch_workers=self.training_config.get('prefetch_workers', 16),
            rl_dir=rl_dir,
            log_level=log_level,
            rotation_interval=rotation_interval
        )

        self.model = ChessAIModel(self.model_config).to(self.device)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=float(self.training_config['learning_rate']),
            momentum=self.training_config['momentum_rate'],
            weight_decay=float(self.training_config['weight_decay'])
        )

        self.scaler = GradScaler('cuda')
        self.policy_criterion = nn.KLDivLoss(reduction='batchmean')
        self.value_criterion = nn.KLDivLoss(reduction='batchmean')

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

    def _setup_logger(self):
        logger = logging.getLogger("TrainTask")
        logger.setLevel(self.training_config['logging_level'])
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
        
        # Get the index of the ready buffer
        slot = self.prefetcher.ready_queue.get()
        valid_idx = self.prefetcher.shm_valid[slot].item()

        # Slice the shared memory directly
        t_b = self.prefetcher.shm_b[slot][:valid_idx]
        t_p = self.prefetcher.shm_p[slot][:valid_idx]
        t_v = self.prefetcher.shm_v[slot][:valid_idx]
        t_m = self.prefetcher.shm_m[slot][:valid_idx]

        shuffle_idx = torch.randperm(valid_idx)

        board_tensors    = t_b[shuffle_idx].to(self.device, non_blocking=True)
        policy_target    = t_p[shuffle_idx].to(self.device, non_blocking=True)
        value_targets    = t_v[shuffle_idx].float().to(self.device, non_blocking=True).flatten()
        true_legal_masks = t_m[shuffle_idx].to(self.device, non_blocking=True)

        # CRITICAL: Tell the worker it can overwrite this buffer for the next batch
        self.prefetcher.free_queue.put(slot)

        data_time = (time.perf_counter() - data_start) * 1000

        global_step = self.state_config['lifetime']['training_steps']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(self.training_config['learning_rate'])

        value_targets_wdl = torch.zeros((value_targets.size(0), 3), device=self.device, dtype=torch.float32)
        value_targets_wdl[:, 0] = (value_targets == 1.0).float()  # Win
        value_targets_wdl[:, 1] = (value_targets == 0.0).float()  # Draw
        value_targets_wdl[:, 2] = (value_targets == -1.0).float() # Loss

        with torch.no_grad():
            total_v = len(value_targets)
            wins = (value_targets == 1.0).sum().item()
            draws = (value_targets == 0.0).sum().item()
            losses = (value_targets == -1.0).sum().item()
            pct_w = (wins / total_v) * 100
            pct_d = (draws / total_v) * 100
            pct_l = (losses / total_v) * 100

        self.model.train()
        self.optimizer.zero_grad()

        fw_start = time.perf_counter()
        with torch.amp.autocast('cuda'):
            policy_logits, value_outputs = self.model(board_tensors)

        policy_logits = policy_logits.float().masked_fill(~true_legal_masks, -1e9)
        policy_log_softmax = F.log_softmax(policy_logits, dim=1)

        policy_loss = self.policy_criterion(policy_log_softmax, policy_target.float())
        value_log_probs = torch.log(value_outputs.clamp(min=1e-8))
        value_loss = self.value_criterion(value_log_probs, value_targets_wdl)

        total_loss = (policy_loss * self.training_config['policy_loss_weight']) + \
                     (value_loss  * self.training_config['value_loss_weight'])

        with torch.no_grad():
            policy_probs   = torch.exp(policy_log_softmax)
            policy_entropy = -(policy_probs * policy_log_softmax).sum(dim=1).mean()
            v_out_mean     = (value_outputs[:, 0] - value_outputs[:, 2]).mean()
            v_tar_mean     = value_targets.mean()

            pred_w_mean = value_outputs[:, 0].mean().item() * 100.0
            pred_d_mean = value_outputs[:, 1].mean().item() * 100.0
            pred_l_mean = value_outputs[:, 2].mean().item() * 100.0

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
            f"Tar (W/D/L): {pct_w:.1f}% / {pct_d:.1f}% / {pct_l:.1f}% | "
            f"Pred: {pred_w_mean:.1f}% / {pred_d_mean:.1f}% / {pred_l_mean:.1f}% | "
            f"P_Ent={policy_entropy.item():.4f} | "
            f"Grad={grad_norm.item():.2f} | LR={self.optimizer.param_groups[0]['lr']:.6f}"
        )

        if self.tb_writer is not None:
            self.tb_writer.add_scalar('Loss/Total', total_loss.item(), global_step)
            self.tb_writer.add_scalar('Loss/Policy', policy_loss.item() * self.training_config['policy_loss_weight'], global_step)
            self.tb_writer.add_scalar('Loss/Value', value_loss.item() * self.training_config['value_loss_weight'], global_step)
            self.tb_writer.add_scalar('Metrics/Policy_Entropy', policy_entropy.item(), global_step)
            self.tb_writer.add_scalar('Metrics/Value_Output_Mean', v_out_mean.item(), global_step)
            self.tb_writer.add_scalar('Metrics/Value_Target_Mean', v_tar_mean.item(), global_step)
            self.tb_writer.add_scalar('Metrics/Grad_Norm', grad_norm.item(), global_step)
            self.tb_writer.add_scalar('Batch_Composition/Wins_Pct', pct_w, global_step)
            self.tb_writer.add_scalar('Batch_Composition/Draws_Pct', pct_d, global_step)
            self.tb_writer.add_scalar('Batch_Composition/Losses_Pct', pct_l, global_step)
            self.tb_writer.add_scalar('Predictions/Predicted_Win_Pct', pred_w_mean, global_step)
            self.tb_writer.add_scalar('Predictions/Predicted_Draw_Pct', pred_d_mean, global_step)
            self.tb_writer.add_scalar('Predictions/Predicted_Loss_Pct', pred_l_mean, global_step)
            self.tb_writer.add_scalar('Hardware_MS/Data_Load', data_time, global_step)
            self.tb_writer.add_scalar('Hardware_MS/Forward', fw_time, global_step)
            self.tb_writer.add_scalar('Hardware_MS/Backward', bw_time, global_step)


    def cleanup(self):
        if hasattr(self, 'prefetcher'):
            self.prefetcher.worker.terminate()
            self.prefetcher.worker.join()
            self.prefetcher.ready_queue.close()
            self.prefetcher.free_queue.close()
        if self.tb_writer is not None:
            self.tb_writer.close()