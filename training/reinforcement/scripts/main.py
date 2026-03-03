import os
import time
import yaml
import logging
import numpy as np
import h5py
import shutil
import random
import torch
import textwrap
import multiprocessing as mp

from data_generation_task import DataGenerationTask
from train_task import TrainTask

current_script_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

from src_shared.model import ChessAIModel

RL_DIR = os.path.abspath(os.path.join(rl_dir, "rl_dir"))
CONFIG_RL_STATE_FILE = os.path.abspath(os.path.join(rl_dir, "config", "rl_state.yaml"))
CONFIG_RL_PARAMS_FILE = os.path.abspath(os.path.join(rl_dir, "config", "rl_config.yaml"))


class RLOrchestrator:
    def __init__(self):
        self.logger = None

        with open(CONFIG_RL_PARAMS_FILE, 'r') as f:
            self.params_config = yaml.safe_load(f)

        # Create state file (is missing)
        if not os.path.exists(CONFIG_RL_STATE_FILE):            
            default_state_yaml = textwrap.dedent("""
            state:
                # Replay Buffer State
                buffer:
                    count: 0                    # Total valid positions currently in the buffer
                    head_ptr: 0                 # Circular buffer index (where next sample goes)
                    wraps: 0                    # Number of buffer wraps

                # Lifetime Statistics (Global accumulators)
                lifetime:
                    training_steps: 0           # Tracks global gradient updates (batches trained)
                    games_played: 0             # Total games played since inception
                    samples_generated: 0        # Total positions generated (including overwritten ones)
                    hours_generating: 0         # Total hours spent in self-play
                    hours_training: 0           # Total hours spent updating weights
                    self_play_entropy: 0.0
                
                # Statistics from current save interval                                  
                current_interval:
                    samples_generated: 0        # Positions generated in the current specific cycle
                    games_played: 0             # Number of games played this cycle
                    self_play_entropy: 0.0      # Entropy  from self play games
                """).strip()
            
            with open(CONFIG_RL_STATE_FILE, 'w') as f:
                f.write(default_state_yaml)

        with open(CONFIG_RL_STATE_FILE, 'r') as f:
            self.state_config = yaml.safe_load(f)

        self.current_steps = mp.Value('i', self.state_config['state']['lifetime']['training_steps'])
        self.total_steps = self.params_config['global']['total_training_steps']
        self.save_interval = self.params_config['global']['save_interval_steps']

        # Calculate the correct initial folder based on loaded state
        rotation_interval = self.params_config['global']['logging_rotation_steps']
        target_folder_step = (self.current_steps.value // rotation_interval) * rotation_interval
        initial_log_dir = os.path.join(RL_DIR, f"run_step_{target_folder_step:06d}")
        
        os.makedirs(initial_log_dir, exist_ok=True)
        
        # Logger now points to the correct versioned folder immediately
        self.logger = self._setup_persistent_logger(initial_log_dir)

        self.weight_update_event = mp.Event()
        self.best_model_path = os.path.abspath(os.path.join(RL_DIR, "best_models", "best_model.pth"))

        if not os.path.exists(self.best_model_path):
            self._create_new_model()

        os.makedirs(RL_DIR, exist_ok=True)

        # Get buffer path from the global config file and make it absolute
        buffer_file_name = os.path.abspath(os.path.join(RL_DIR, "replay_memory.hdf5"))
        self.buffer_file_path = buffer_file_name

    def _create_new_model(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Create a new, seeded instance of the model
        model = ChessAIModel(
            num_input_planes=self.params_config['model']['input_planes'],
            num_residual_blocks=self.params_config['model']['resblocks'],
            num_filters=self.params_config['model']['filters']
        )

        model_dict = {
            'model_state_dict': model.state_dict(),
        }

        os.makedirs(os.path.join(RL_DIR, 'best_models'),  exist_ok=True)

        torch.save(model_dict, os.path.join(RL_DIR, 'best_models', 'initial_model.pth'))
        torch.save(model_dict, os.path.join(RL_DIR, 'best_models', 'best_model.pth'))


    def _setup_persistent_logger(self, log_dir):
        """
        Configures a persistent logger for the current 1k-step rotation directory.
        """
        logger = logging.getLogger("RLOrchestrator")
        logger.setLevel(logging.INFO)
        
        # Clear old handlers to avoid duplicate logging when rotating folders
        if logger.hasHandlers():
            logger.handlers.clear()
            
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        log_filepath = os.path.join(log_dir, "orchestrator.log")
        
        # 'a' mode is critical for persisting logs if the script restarts
        file_handler = logging.FileHandler(log_filepath, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    

    def _save_state(self):
        with open(CONFIG_RL_STATE_FILE, 'w') as f:
            yaml.safe_dump(self.state_config, f)



    def get_dynamic_buffer_limit(self):
        """
        Calculates the target buffer capacity for the current training step.
        Returns a linear ramp from min_size to max_size.
        """

        if self.current_steps.value >= self.params_config['global']['buffer_ramp_steps']:
            return self.params_config['global']['max_buffer_size']
        
        # Linear Interpolation Formula
        # Capacity = Min + (Progress % * (Max - Min))
        progress = self.current_steps.value / self.params_config['global']['buffer_ramp_steps']
        growth_range = self.params_config['global']['max_buffer_size'] - self.params_config['global']['min_buffer_size']
        
        current_capacity = self.params_config['global']['min_buffer_size'] + (progress * growth_range)
        
        return int(current_capacity)


    def _update_circular_buffer(self, new_data):
        max_positions = self.get_dynamic_buffer_limit()
        
        boards = np.array([item['board_state'] for item in new_data], dtype=np.float16)
        policies = np.array([item['policy'] for item in new_data], dtype=np.float16)
        values = np.array([item['value_target'] for item in new_data], dtype=np.float16)

        self.logger.info(f"Appending new data ({len(new_data)} positions) to the circular buffer: {self.buffer_file_path}")

        # Proceed with the original, efficient in-place modification
        current_size = self.state_config['state']['buffer']['count']
        current_write_head = self.state_config['state']['buffer']['head_ptr']
        
        with h5py.File(self.buffer_file_path, 'a') as hf:
            if 'inputs' in hf and 'policies' in hf and 'values' in hf:
                self.logger.debug(f"Existing datasets (headers) are: {list(hf.keys())}.")

                inputs_dset = hf['inputs']
                policies_dset = hf['policies']
                values_dset = hf['values']

                if inputs_dset.shape[0] < max_positions:
                    self.logger.info(f"Resizing HDF5 datasets to {max_positions}")
                    inputs_dset.resize(max_positions, axis=0)
                    policies_dset.resize(max_positions, axis=0)
                    values_dset.resize(max_positions, axis=0)

                num_remaining = len(boards)
                
                # Check if the new data fits contiguously without wrapping
                # (e.g. Ptr=395k, Data=10k, Max=410k -> 405k <= 410k -> Fits!)
                if current_write_head + num_remaining <= max_positions:
                    # Case A: No Wrap (or flows into new space)
                    inputs_dset[current_write_head : current_write_head + num_remaining] = boards
                    policies_dset[current_write_head : current_write_head + num_remaining] = policies
                    values_dset[current_write_head : current_write_head + num_remaining] = values

                    # Pointer simply moves forward
                    current_write_head += num_remaining
                    new_count = max(current_size, current_write_head)
                
                else:
                    # Case B: Wrap Around
                    # (e.g. Ptr=405k, Data=10k, Max=410k -> 415k > 410k -> Split)
                    first_part_len = max_positions - current_write_head
                    second_part_len = num_remaining - first_part_len

                    # Write to the end (filling the new space)
                    inputs_dset[current_write_head:] = boards[:first_part_len]
                    policies_dset[current_write_head:] = policies[:first_part_len]
                    values_dset[current_write_head:] = values[:first_part_len]

                    # Wrap the rest to 0 (overwriting oldest data)
                    inputs_dset[:second_part_len] = boards[first_part_len:]
                    policies_dset[:second_part_len] = policies[first_part_len:]
                    values_dset[:second_part_len] = values[first_part_len:]

                    # Pointer wraps to the end of the second part
                    current_write_head = second_part_len
                    new_count = max_positions
                    self.state_config['state']['buffer']['wraps'] += 1

                # --- 3. UPDATE STATE ---
                # Ensure the pointer is strictly modulo the size (safety)
                current_write_head = current_write_head % max_positions
                
                self.state_config['state']['buffer']['head_ptr'] = current_write_head
                self.state_config['state']['buffer']['count'] = new_count

            else:
                # File is brand new or was just created
                self.logger.info("Creating new HDF5 datasets for the buffer with explicit chunking.")
                hdf5_chunk_size = self.params_config['global']['hdf5_chunk_size']
                
                # Use the shape of the first data chunk to define the rest of the dimensions
                board_shape = boards.shape[1:]
                policy_shape = policies.shape[1:]
                
                # Create datasets with an explicit chunk shape
                hf.create_dataset('inputs', data=boards, maxshape=(None, *board_shape), dtype=np.float16, compression='gzip', chunks=(hdf5_chunk_size, *board_shape))
                hf.create_dataset('policies', data=policies, maxshape=(None, *policy_shape), dtype=np.float16, compression='gzip', chunks=(hdf5_chunk_size, *policy_shape))
                hf.create_dataset('values', data=values, maxshape=(None,),dtype=np.float16, compression='gzip', chunks=(hdf5_chunk_size,))
                
                # Update the new state variables
                self.state_config['state']['buffer']['count'] = len(new_data)
                self.state_config['state']['buffer']['head_ptr'] = len(new_data) % max_positions


    def run(self):
        """
        Main persistent loop.
        """
        # 1. Initialize persistent generation ONCE
        self.data_gen_task = DataGenerationTask(
            output_dir=RL_DIR,
            current_steps=self.current_steps,
            rotation_interval=self.params_config['global']['logging_rotation_steps'],
            best_model_path=self.best_model_path,
            model_config=self.params_config['model'],
            data_generation_config=self.params_config['data_generation'],
            state_config=self.state_config,
            weight_update_event=self.weight_update_event
        )

        self.train_task = TrainTask(
            best_model_path=self.best_model_path,
            model_config=self.params_config['model'],
            training_config=self.params_config['training'],
            state_config=self.state_config['state'],
            global_config=self.params_config['global'],
            hdf5_path=self.buffer_file_path,
        )

        # 2. Start the generator
        data_iterator = self.data_gen_task.run_persistently(chunk_size=int(self.params_config['training']['batch_size'] / self.params_config['training']['sampling_ratio']))
        current_log_dir = None

        while self.current_steps.value < self.total_steps:
            rotation_interval = self.params_config['global']['logging_rotation_steps']
            target_folder_step = ((self.current_steps.value // rotation_interval) * rotation_interval)
            new_log_dir = os.path.join(RL_DIR, f"run_step_{target_folder_step:06d}")
            
            if new_log_dir != current_log_dir:
                os.makedirs(new_log_dir, exist_ok=True)
                current_log_dir = new_log_dir
                self.logger = self._setup_persistent_logger(current_log_dir)
                self.logger.info(f"New logging phase started at step {self.current_steps.value}")

            # --- Step 1: Collect Data (1 Step worth) ---
            start_data_gen_time = time.time()
            new_data_chunk, games_in_chunk, chunk_entropy = next(data_iterator)
            total_data_gen_time = time.time() - start_data_gen_time
            self.state_config['state']['lifetime']['hours_generating'] = round(self.state_config['state']['lifetime']['hours_generating']  + (total_data_gen_time / 3600), 4)

            # --- Step 2: Update Buffer (SWMR) ---
            self._update_circular_buffer(new_data_chunk)

            start_train_time = time.time()
            test_model_path = self.train_task.run_single_step(current_log_dir, self.state_config)
            total_train_time = time.time() - start_train_time
            self.state_config['state']['lifetime']['hours_training'] = round(self.state_config['state']['lifetime']['hours_training']  + (total_train_time / 3600), 4)

            shutil.copy(test_model_path, self.best_model_path)
            os.remove(test_model_path)
            
            # Signal InferenceBatchers to reload from disk
            self.weight_update_event.set()

            # --- Step 5: Update State ---
            with self.current_steps.get_lock():
                self.current_steps.value += 1

            self.state_config['state']['lifetime']['training_steps'] = int(self.current_steps.value)
            self.state_config['state']['lifetime']['samples_generated'] += int(len(new_data_chunk))
            self.state_config['state']['lifetime']['games_played'] += int(games_in_chunk)
            self.state_config['state']['lifetime']['self_play_entropy'] += int(round(chunk_entropy, 4))

            self.state_config['state']['current_interval']['samples_generated'] += int(len(new_data_chunk))
            self.state_config['state']['current_interval']['games_played'] += int(games_in_chunk)
            self.state_config['state']['current_interval']['self_play_entropy'] += int(round(chunk_entropy, 4))
            
            self._save_state()

            if self.current_steps.value // self.save_interval > (self.current_steps.value - 1) // self.save_interval:
                backup_dir = os.path.join(RL_DIR, 'backup')
                os.makedirs(backup_dir, exist_ok=True)

                # Save current best model with step metadata
                model_backup_path = os.path.join(
                    backup_dir, 
                    f'step_{self.current_steps.value:06d}_model.pth'
                )
                shutil.copy(self.best_model_path, model_backup_path)

                buffer_backup_path = os.path.join(
                    backup_dir, 
                    f'step_{self.current_steps.value:06d}_replay_memory.hdf5'
                )
                
                self.logger.info(f"Creating periodic backup at step {self.current_steps.value}...")
                shutil.copy(self.buffer_file_path, buffer_backup_path)

                # Backup the state and config files as well
                shutil.copy(CONFIG_RL_STATE_FILE, os.path.join(current_log_dir, f'step_{self.current_steps.value:06d}_state.yaml'))
                shutil.copy(CONFIG_RL_PARAMS_FILE, os.path.join(current_log_dir, f'step_{self.current_steps.value:06d}_config.yaml'))

                self.state_config['state']['current_interval']['samples_generated'] = 0
                self.state_config['state']['current_interval']['games_played'] = 0
                self.state_config['state']['current_interval']['self_play_entropy'] = 0

        # Training loop is finished
        self.logger.info("Total training steps reached. Shutting down workers...")
        
        # 1. Signal everyone to stop
        self.data_gen_task.stop_event.set()
        
        # 2. Cleanup
        self.data_gen_task.terminate_all()
        
        self.logger.info("All persistent tasks terminated. Orchestrator exiting.")
                    

if __name__ == "__main__":
    orchestrator = RLOrchestrator()
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\nManual interruption detected. Shutting down...")
    except Exception as e:
        # Use the orchestrator logger if available, otherwise print
        if orchestrator.logger:
            orchestrator.logger.error(f"Orchestrator encountered an error: {e}", exc_info=True)
        else:
            print(f"Orchestrator encountered an error: {e}")
    finally:
            # Check if the task was initialized inside run()
            if hasattr(orchestrator, 'self.data_gen_task'):
                print("Cleaning up background workers...")
                orchestrator.self.data_gen_task.terminate_all()
            print("Shutdown complete.")