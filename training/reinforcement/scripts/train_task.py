import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import os
import sys
import random
import time
import h5py
import numpy as np

# Ensure project root is in path for imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

from src_shared.model import ChessAIModel
from src_shared.data_loader import ChessDataset, _worker_init_fn

def unwrap_single_batch(batch):
    """
    Custom collate function for DataLoader to unwrap batches
    from a Dataset that returns a full batch per __getitem__.
    """
    return batch[0]


class TrainTask:
    def __init__(self, output_dir: str, model_config: dict, training_config: dict, buffer_size: int, hdf5_path: str, best_model_cycle: int, cycle_number: int):
        self.training_config = training_config
        self.model_config = model_config
        self.buffer_size = buffer_size
        self.output_dir = output_dir
        self.hdf5_path = hdf5_path
        self.best_model_cycle = best_model_cycle
        self.cycle_number = cycle_number
        self.shuffled_hdf5_path = None
        self.batch_size = self.training_config['batch_size']
        self.io_chunk_size = self.training_config['io_chunk_size']
        
        # Calculate total training steps dynamically based on effective epochs and new data size
        self.total_training_steps = self.training_config['effective_epochs'] * (self.buffer_size // self.training_config['batch_size'])

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
        Create a shuffled copy of the buffer with weighted sampling,
        operating at the HDF5 chunk level (chunks of size self.batch_size).
        Handles partial final chunks gracefully.
        """
        self.logger.info("Starting HDF5 duplication and weighted sampling at chunk level.")

        base_name, ext = os.path.splitext(os.path.basename(self.hdf5_path))
        temp_file_name = f"{base_name}_shuffled_temp_{os.getpid()}{ext}"
        self.shuffled_hdf5_path = os.path.join(os.path.dirname(self.hdf5_path), temp_file_name)
        temp_path_writing = f"{self.shuffled_hdf5_path}.tmp"

        try:
            with h5py.File(self.hdf5_path, 'r') as hf_source:
                total_positions = hf_source['inputs'].shape[0]
                chunk_size = self.batch_size
                num_chunks = (total_positions + chunk_size - 1) // chunk_size  # ceil division

                # Build list of valid chunk ranges
                chunk_ranges = []
                chunk_cycle_ids = []
                for i in range(0, total_positions, chunk_size):
                    end = min(i + chunk_size, total_positions)
                    chunk_ranges.append((i, end))
                    chunk_cycle_ids.append(hf_source['iterations'][i])  # take first value

                chunk_cycle_ids = np.array(chunk_cycle_ids)
                chunk_ids = np.arange(len(chunk_ranges))

                # Identify new and old chunk indices
                new_chunk_ids = chunk_ids[chunk_cycle_ids == self.best_model_cycle]
                old_chunk_ids = chunk_ids[chunk_cycle_ids < self.cycle_number]

                self.logger.info(f"New data chunks (cycle {self.cycle_number}): {len(new_chunk_ids)}")
                self.logger.info(f"Old data chunks (cycles < {self.cycle_number}): {len(old_chunk_ids)}")

                new_data_ratio = self.training_config['new_data_ratio']
                if self.cycle_number == 1 or len(old_chunk_ids) == 0:
                    self.logger.info("Initial cycle or no old data. Sampling only from new data.")
                    new_data_ratio = 1.0
                    old_data_ratio = 0.0
                else:
                    old_data_ratio = 1.0 - new_data_ratio

                total_training_positions = self.total_training_steps * chunk_size
                total_training_chunks = (total_training_positions + chunk_size - 1) // chunk_size

                num_new_chunks = int(total_training_chunks * new_data_ratio)
                num_old_chunks = int(total_training_chunks * old_data_ratio)

                new_sampled_chunks = np.random.choice(new_chunk_ids, size=num_new_chunks, replace=True)
                old_sampled_chunks = np.random.choice(old_chunk_ids, size=num_old_chunks, replace=True)

                all_sampled_chunks = np.concatenate([new_sampled_chunks, old_sampled_chunks])
                np.random.shuffle(all_sampled_chunks)

                with h5py.File(temp_path_writing, 'w') as hf_dest:
                    board_shape = hf_source['inputs'].shape[1:]
                    policy_shape = hf_source['policies'].shape[1:]

                    estimated_total_positions = 0
                    for chunk_id in all_sampled_chunks:
                        start, end = chunk_ranges[chunk_id]
                        estimated_total_positions += end - start

                    hf_dest.create_dataset('inputs',
                        shape=(estimated_total_positions, *board_shape),
                        dtype=np.float16,
                        chunks=(chunk_size, *board_shape),
                        compression='gzip'
                    )
                    hf_dest.create_dataset('policies',
                        shape=(estimated_total_positions, *policy_shape),
                        dtype=np.float16,
                        chunks=(chunk_size, *policy_shape),
                        compression='gzip'
                    )
                    hf_dest.create_dataset('values',
                        shape=(estimated_total_positions,),
                        dtype=np.float16,
                        chunks=(chunk_size,),
                        compression='gzip'
                    )

                    write_index = 0
                    chunks_per_io = self.io_chunk_size // chunk_size

                    for i in range(0, len(all_sampled_chunks), chunks_per_io):
                        end_i = min(i + chunks_per_io, len(all_sampled_chunks))
                        chunk_batch_ids = all_sampled_chunks[i:end_i]

                        # Sort chunk indices
                        sorted_chunk_ids = np.sort(chunk_batch_ids)
                        self.logger.info(f"Sorted chunk ID's: {sorted_chunk_ids}")
                        unique_chunk_ids, counts = np.unique(chunk_batch_ids, return_counts=True)

                        all_inputs = []
                        all_policies = []
                        all_values = []
                        for chunk_id, count in zip(unique_chunk_ids, counts):
                            start, end = chunk_ranges[chunk_id]
                            
                            # Read the data for this one chunk
                            inputs_chunk = hf_source['inputs'][start:end]
                            policies_chunk = hf_source['policies'][start:end]
                            values_chunk = hf_source['values'][start:end]

                            # Append this data to our list 'count' number of times
                            for _ in range(count):
                                all_inputs.append(inputs_chunk)
                                all_policies.append(policies_chunk)
                                all_values.append(values_chunk)

                        # Concatenate all the chunks into a single NumPy array
                        inputs_chunk = np.concatenate(all_inputs, axis=0)
                        policies_chunk = np.concatenate(all_policies, axis=0)
                        values_chunk = np.concatenate(all_values, axis=0)

                        # Shuffle all positions in memory
                        self.logger.info(f"Shuffling positions")
                        perm = np.random.permutation(len(inputs_chunk))
                        inputs_chunk = inputs_chunk[perm]
                        policies_chunk = policies_chunk[perm]
                        values_chunk = values_chunk[perm]

                        # Write back in chunks of `chunk_size` (may include a final partial chunk)
                        self.logger.info(f"Writing back to file")
                        for j in range(0, len(inputs_chunk), chunk_size):
                            end_j = min(j + chunk_size, len(inputs_chunk))
                            hf_dest['inputs'][write_index:write_index + (end_j - j)] = inputs_chunk[j:end_j]
                            hf_dest['policies'][write_index:write_index + (end_j - j)] = policies_chunk[j:end_j]
                            hf_dest['values'][write_index:write_index + (end_j - j)] = values_chunk[j:end_j]
                            write_index += end_j - j

            os.rename(temp_path_writing, self.shuffled_hdf5_path)
            self.logger.info(f"Successfully created shuffled HDF5 file at: {self.shuffled_hdf5_path}")

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
        self.logger.info(f"Loading data from shuffled HDF5 file: {self.shuffled_hdf5_path}")
        
        # The data in the HDF5 file is already pre-shuffled, so we just need to iterate sequentially.
        chunk_size = self.training_config['batch_size']
        full_dataset = ChessDataset(hdf5_path=self.shuffled_hdf5_path, chunk_size=chunk_size)

        num_workers = 4

        train_loader = DataLoader(
            full_dataset,
            batch_size=1,
            sampler=SequentialSampler(full_dataset),
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            collate_fn=unwrap_single_batch
        )

        self.logger.info(f"Total dataset size: {len(full_dataset)} batches")
        self.logger.info(f"Training set size: {len(train_loader.sampler)} batches")

        return train_loader
    
    def run_training_loop(self):
        final_model_path = None
        training_steps = 0
        
        try:
            self._duplicate_and_shuffle_hdf5()
            train_loader = self._get_dataloaders()
            
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
            
            # Use the number of batches in the DataLoader for T_max
            scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), 
                                          eta_min=float(self.training_config['cosine_eta_min']))
            
            scaler = GradScaler('cuda')

            model.train() # Set to training mode once

            # The DataLoader with SequentialSampler will handle the step count, so we just iterate
            for batch_idx, (board_tensors, policy_target, value_targets) in enumerate(train_loader):
                training_steps = batch_idx + 1 # Update step count based on batch index

                batch_start_time = time.perf_counter()
                
                transfer_to_gpu_start = time.perf_counter()
                board_tensors = board_tensors.to(self.device, non_blocking=True)
                policy_target = policy_target.to(self.device, non_blocking=True)
                value_targets = value_targets.to(self.device, non_blocking=True)
                torch.cuda.synchronize()
                transfer_to_gpu_end = time.perf_counter()
                
                optimizer.zero_grad()
                
                forward_pass_start = time.perf_counter()
                with autocast(device_type="cuda"):
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
                
                batch_end_time = time.perf_counter()
                
                # Log detailed info to file at intervals
                if training_steps % self.training_config['log_interval'] == 0 or training_steps == len(train_loader):
                    self.logger.info(f"Training Step {training_steps}/{len(train_loader)}: "
                                     f"P_Loss={policy_loss.item():.4f}, "
                                     f"V_Loss={value_loss.item():.4f}, "
                                     f"T_Loss={total_loss.item():.4f}, "
                                     f"LR={optimizer.param_groups[0]['lr']:.6f}, "
                                     f"GPU Xfer: {(transfer_to_gpu_end - transfer_to_gpu_start)*1000:.2f}ms, "
                                     f"FW: {(forward_pass_end - forward_pass_start)*1000:.2f}ms, "
                                     f"BW: {(backward_pass_end - backward_pass_start)*1000:.2f}ms, "
                                     f"Batch Total: {(batch_end_time - batch_start_time)*1000:.2f}ms")
            
            final_model_path = os.path.join(self.output_dir, f"model_iter_{self.cycle_number}.pth")
            torch.save(model.state_dict(), final_model_path)
            self.logger.info(f"Final model after {training_steps} steps saved to {final_model_path}")

            self.logger.info("Training complete for this task!")
            
        finally:
            self._clean_up_shuffled_file()
            
        return final_model_path, training_steps