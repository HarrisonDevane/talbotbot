import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import os
import sys
import random
import time
import h5py
import numpy as np
import warnings

# Ensure project root is in path for imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

# Assuming these imports are correct based on your file structure
from src_shared.model import ChessAIModel
from src_shared.data_loader import ChessDataset, _worker_init_fn

def unwrap_single_batch(batch):
    """
    Custom collate function for DataLoader to unwrap batches
    from a Dataset that returns a full batch per __getitem__.
    """
    return batch[0]


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
        self.shuffled_hdf5_path = None
        self.io_chunk_size = self.training_config['io_chunk_size']
        # New attribute to store the actual number of batches (steps) processed
        self.num_batches_processed = 0 

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


    def _duplicate_and_shuffle_hdf5(self):
        """
        Creates a globally and locally shuffled HDF5 file from the source.
        The destination size is clamped to min(config_size, source_size).
        Implements Recency-based PER by ensuring a fixed ratio 
        of the final training file's batches are sampled from the newest cycle's data,
        and the remainder are sampled strictly from the older data.
        """
        fixed_steps = self.training_config['training_steps']
        
        # Required state variables from the circular buffer
        write_head_position = self.state_config['buffer_write_head']
        cycle_positions = self.state_config['data_generation_positions_current'] 
        
        self.logger.info("Starting HDF5 duplication and shuffle: processing ALL source batches, writing LIMITED steps.")
        self.logger.info(f"Using Recency PER to ensure {self.training_config['recency_ratio']*100}% of the final file comes from the newest cycle's data.")

        base_name, ext = os.path.splitext(os.path.basename(self.hdf5_path))

        temp_file_name = f"{base_name}_shuffled_temp_{os.getpid()}{ext}"
        self.shuffled_hdf5_path = os.path.join(os.path.dirname(self.hdf5_path), temp_file_name)
        temp_path_writing = f"{self.shuffled_hdf5_path}.tmp"
        
        current_write_position = 0 
        
        try:
            # Step 1: Keep the source file open for the duration of reading
            with h5py.File(self.hdf5_path, 'r') as hf_source:
                
                total_positions_source = hf_source['inputs'].shape[0]
                batch_size = self.training_config['batch_size']
                io_chunk_size = self.training_config['io_chunk_size']
                
                num_batches_total_source = total_positions_source // batch_size
                batches_per_io_chunk = io_chunk_size // batch_size

                # Determine the size of the destination file (Clamped)
                total_positions_to_write = min(fixed_steps * batch_size, total_positions_source)
                num_batches_to_write = total_positions_to_write // batch_size
                
                self.logger.info(f"Target positions to write: {total_positions_to_write} ({num_batches_to_write} batches)")

                 # A. Calculate required samples from Newest and Older data
                num_new_batches_to_sample = int(num_batches_to_write * self.training_config['recency_ratio'])
                num_older_batches_to_sample = num_batches_to_write - num_new_batches_to_sample
                
                # B. Identify the indices of the Newest Batches (the current cycle's data)
                num_new_cycle_positions = min(cycle_positions, total_positions_source)
                
                # The newest data starts *before* the write head, counting backwards
                new_data_start_position = (write_head_position - num_new_cycle_positions) % total_positions_source
                
                all_source_batch_indices = list(range(num_batches_total_source))
                new_cycle_batch_indices = []
                older_batch_indices = []
                
                # C. Separate the New Cycle Batches from the Older Batches (Handle wrap-around)
                for batch_i in all_source_batch_indices:
                    batch_pos = batch_i * batch_size
                    is_new_cycle = False
                    
                    if write_head_position > new_data_start_position:
                        # Scenario 1: No wrap-around. Newest data is contiguous.
                        if new_data_start_position <= batch_pos < write_head_position:
                            is_new_cycle = True
                    else:
                        # Scenario 2: Wrap-around occurred. Newest data is split.
                        if batch_pos >= new_data_start_position or batch_pos < write_head_position:
                            is_new_cycle = True
                            
                    if is_new_cycle:
                        new_cycle_batch_indices.append(batch_i)
                    else:
                        older_batch_indices.append(batch_i)
                
                # D. Sample from the two pools with compensation for small buffer.
                # 1. Sample what is available from the Older pool, capped by the 70% quota.
                num_older_to_sample = min(num_older_batches_to_sample, len(older_batch_indices))
                older_sample_indices = random.sample(older_batch_indices, num_older_to_sample)
                
                # 2. Determine the number of batches still needed to hit the target size.
                batches_needed_after_older = num_batches_to_write - len(older_sample_indices)
                
                # The required sample size for the new data is the max of:
                num_new_to_sample = max(num_new_batches_to_sample, batches_needed_after_older)
                
                # 3. Sample from the newest cycle's data (P_new)
                #    Clamp the final sample size to the amount actually available in the new pool
                if num_new_to_sample > len(new_cycle_batch_indices):
                    self.logger.warning(f"Not enough total data available. New sample size reduced from {num_new_to_sample} to available {len(new_cycle_batch_indices)}.")

                num_new_to_sample = min(num_new_to_sample, len(new_cycle_batch_indices))
                recency_sample_indices = random.sample(new_cycle_batch_indices, num_new_to_sample)
                
                # 4. Combine and Global Shuffle
                all_batch_indices = recency_sample_indices + older_sample_indices
                random.shuffle(all_batch_indices)

                # E. Logging (for verification)
                self.logger.info(f"New batches ({self.training_config['recency_ratio']*100}% target): {len(recency_sample_indices)}")
                self.logger.info(f"Older batches ({round((1 - self.training_config['recency_ratio'])*100)}% target): {len(older_sample_indices)}")
                self.logger.info(f"Total batches for training file: {len(all_batch_indices)}")

                # Step 2b: Create destination HDF5 datasets and close immediately
                with h5py.File(temp_path_writing, 'w') as hf_dest:
                    board_shape = hf_source['inputs'].shape[1:]
                    policy_shape = hf_source['policies'].shape[1:]
                    
                    # Datasets are sized to the clamped total_positions_to_write
                    hdf5_chunk_size = self.training_config['hdf5_chunk_size']
                    hf_dest.create_dataset('inputs', shape=(total_positions_to_write, *board_shape), dtype=np.float16, chunks=(hdf5_chunk_size, *board_shape), compression='gzip')
                    hf_dest.create_dataset('policies', shape=(total_positions_to_write, *policy_shape), dtype=np.float16, chunks=(hdf5_chunk_size, *policy_shape), compression='gzip')
                    hf_dest.create_dataset('values', shape=(total_positions_to_write,), dtype=np.float16, chunks=(hdf5_chunk_size,), compression='gzip')

                # Step 3: Global shuffle is complete (all_batch_indices is ready)
                
                # Step 4: Open destination file for R/W and process chunks
                with h5py.File(temp_path_writing, 'r+') as hf_dest:
                    
                    # The iteration uses the length of the selected batch list
                    for io_chunk_idx, start_batch_idx in enumerate(range(0, len(all_batch_indices), batches_per_io_chunk)):

                        self.logger.info(f"Processing IO chunk {io_chunk_idx + 1}")
                        
                        if current_write_position >= total_positions_to_write:
                            break

                        end_batch_idx = min(start_batch_idx + batches_per_io_chunk, len(all_batch_indices))
                        current_batches_source_indices = all_batch_indices[start_batch_idx:end_batch_idx]

                        # Read from hf_source (still open)
                        inputs_chunks, policies_chunks, values_chunks = [], [], []
                        for batch_i in current_batches_source_indices:
                            batch_start = batch_i * batch_size
                            batch_end = batch_start + batch_size
                            inputs_chunks.append(hf_source['inputs'][batch_start:batch_end])
                            policies_chunks.append(hf_source['policies'][batch_start:batch_end])
                            values_chunks.append(hf_source['values'][batch_start:batch_end])

                        # Concatenate and local shuffle
                        chunk_inputs = np.concatenate(inputs_chunks, axis=0)
                        chunk_policies = np.concatenate(policies_chunks, axis=0)
                        chunk_values = np.concatenate(values_chunks, axis=0)
                        
                        chunk_size = chunk_inputs.shape[0]
                        local_indices = np.arange(chunk_size)
                        np.random.shuffle(local_indices)
                        chunk_inputs = chunk_inputs[local_indices]
                        chunk_policies = chunk_policies[local_indices]
                        chunk_values = chunk_values[local_indices]

                        # Write Limit Logic
                        remaining_to_write = total_positions_to_write - current_write_position
                        write_count = min(chunk_size, remaining_to_write) 
                        
                        # Write the limited portion of the shuffled chunk back to destination
                        write_start = current_write_position
                        write_end = current_write_position + write_count

                        hf_dest['inputs'][write_start:write_end] = chunk_inputs[:write_count]
                        hf_dest['policies'][write_start:write_end] = chunk_policies[:write_count]
                        hf_dest['values'][write_start:write_end] = chunk_values[:write_count]

                        current_write_position += write_count
                        
                        if write_count < chunk_size:
                            self.logger.info(f"Write limit reached. Wrote final partial chunk of size {write_count}.")
                            break

                os.rename(temp_path_writing, self.shuffled_hdf5_path)
                self.logger.info(f"Successfully created shuffled HDF5 file with {current_write_position} entries at: {self.shuffled_hdf5_path}")

        except Exception as e:
            self.logger.error(f"Failed to create shuffled HDF5 file: {e}")
            if os.path.exists(temp_path_writing):
                os.remove(temp_path_writing)
            self.shuffled_hdf5_path = None
            raise


    def _clean_up_shuffled_file(self):
        """Deletes the temporary shuffled HDF5 file."""
        if self.shuffled_hdf5_path and os.path.exists(self.shuffled_hdf5_path):
            self.logger.info(f"Deleting temporary shuffled HDF5 file: {self.shuffled_hdf5_path}")
            os.remove(self.shuffled_hdf5_path)


    def _get_dataloaders(self):
        # We now load data from the temporary shuffled HDF5 file
        self.logger.info(f"Loading data from shuffled HDF5 file: {self.shuffled_hdf5_path}")
        
        chunk_size = self.training_config['batch_size']
        full_dataset = ChessDataset(hdf5_path=self.shuffled_hdf5_path, chunk_size=chunk_size)

        num_workers = 4

        train_loader = DataLoader(
            full_dataset,
            batch_size=1,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            collate_fn=unwrap_single_batch
        )

        self.logger.info(f"Training set size: {len(full_dataset)} chunks ({len(train_loader)} batches)")

        return train_loader



    def run_training_loop(self):
        best_model_path = None
        
        try:
            self._duplicate_and_shuffle_hdf5()
            train_loader = self._get_dataloaders()
            
            # The number of steps in the current training file/dataloader
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

            model.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))      
            self.logger.info("Model initialized and weights loaded.")

            policy_criterion = nn.KLDivLoss(reduction='batchmean')
            value_criterion = nn.MSELoss()
            
            # Use self.training_config directly for optimizer LR
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=float(self.training_config['cosine_eta_max']), 
                weight_decay=float(self.training_config['weight_decay'])
            )
                    
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.global_config['total_steps'], 
                eta_min=float(self.training_config['cosine_eta_min'])
            )
            
            # Set the scheduler's initial state for resume training
            steps_completed = self.state_config['total_training_steps']
            if steps_completed > 0:
                self.logger.info(f"Advancing scheduler by {steps_completed} steps.")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    for _ in range(steps_completed):
                        scheduler.step()
            
            self.logger.info(f"Initial LR after state restoration: {optimizer.param_groups[0]['lr']:.6f}")
            
            scaler = GradScaler('cuda')

            # --- Training Phase: Single Pass ---
            model.train()
            running_total_loss = 0.0
            
            # The loop runs for the exact number of batches defined by the temporary HDF5 file
            for batch_idx, (board_tensors, policy_target, value_targets) in enumerate(train_loader):
                # Calculate the global step count for logging
                global_step = steps_completed + batch_idx + 1
                
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
                scheduler.step() 
                scaler.update()
                            
                running_total_loss += total_loss.item()
                
                batch_end_time = time.perf_counter()
                
                # Log detailed info to file at intervals
                if (batch_idx + 1) % self.training_config['log_interval'] == 0:
                    self.logger.info(f"Training Step {batch_idx+1}/{total_training_steps_this_cycle} (Global Step {global_step}/{self.global_config['total_steps']}): "
                                     f"P_Loss={policy_loss.item():.4f}, "
                                     f"V_Loss={value_loss.item():.4f}, "
                                     f"T_Loss={total_loss.item():.4f}, "
                                     f"LR={optimizer.param_groups[0]['lr']:.6f}, "
                                     f"GPU Xfer: {(transfer_to_gpu_end - transfer_to_gpu_start)*1000:.2f}ms, "
                                     f"FW: {(forward_pass_end - forward_pass_start)*1000:.2f}ms, "
                                     f"BW: {(backward_pass_end - backward_pass_start)*1000:.2f}ms, "
                                     f"Batch Total: {(batch_end_time - batch_start_time)*1000:.2f}ms")

            avg_total_loss_train = running_total_loss / len(train_loader)
            training_steps_completed = len(train_loader)
            
            self.logger.info(f"--- Training Run Summary ---")
            self.logger.info(f"Total Steps Completed This Cycle: {training_steps_completed}, Average Total Loss: {avg_total_loss_train:.4f}")
            
            # Save the final model after all steps are complete
            final_model_path = os.path.join(self.output_dir, f"model_iter_{self.cycle_number}.pth")
            torch.save(model.state_dict(), final_model_path)
            self.logger.info(f"Final model after {training_steps_completed} steps saved to {final_model_path}")
            
            best_model_path = final_model_path
            
            self.logger.info("Training complete for this task!")
            
        finally:
            self._clean_up_shuffled_file()
            
        return best_model_path, training_steps_completed