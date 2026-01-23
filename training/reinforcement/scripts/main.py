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

from data_generation_task import DataGenerationTask
from train_task import TrainTask

current_script_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

from src_shared.model import ChessAIModel

RL_CYCLES_DIR = os.path.abspath(os.path.join(rl_dir, "rl_cycles"))
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
                    cycle_idx: 1                # Tracks the macro loop (Generate -> Train -> Eval)
                    training_steps: 0           # Tracks global gradient updates (batches trained)
                    games_played: 0             # Total games played since inception
                    samples_generated: 0        # Total positions generated (including overwritten ones)
                    hours_generating: 0         # Total hours spent in self-play
                    hours_training: 0           # Total hours spent updating weights

                # Ephemeral / Current Cycle Stats
                current_cycle:
                    samples_collected: 0        # Positions generated in the current specific cycle
                    games_played: 0             # Number of games played this cycle
                    self_play_entropy: 0.0      # Entropy  from self play games
                    training_avg_entropy: 0.0   # Entropy from training
                                                 
                """).strip()
            
            with open(CONFIG_RL_STATE_FILE, 'w') as f:
                f.write(default_state_yaml)

        with open(CONFIG_RL_STATE_FILE, 'r') as f:
            self.state_config = yaml.safe_load(f)

        self.current_steps = self.state_config['state']['lifetime']['training_steps']
        self.current_cycle = self.state_config['state']['lifetime']['cycle_idx']

        self.total_steps = self.params_config['global']['total_training_steps']
        self.save_interval = self.params_config['global']['save_interval_steps']


        self.best_model_path = os.path.abspath(os.path.join(RL_CYCLES_DIR, "best_models", "best_model.pth"))

        if not os.path.exists(self.best_model_path):
            self._create_new_model()

        os.makedirs(RL_CYCLES_DIR, exist_ok=True)

        # Get buffer path from the global config file and make it absolute
        buffer_file_name = os.path.abspath(os.path.join(RL_CYCLES_DIR, "replay_memory.hdf5"))
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

        os.makedirs(os.path.join(RL_CYCLES_DIR, 'best_models'),  exist_ok=True)

        torch.save(model_dict, os.path.join(RL_CYCLES_DIR, 'best_models', 'initial_model.pth'))
        torch.save(model_dict, os.path.join(RL_CYCLES_DIR, 'best_models', 'best_model.pth'))


    def _setup_cycle_logger(self, cycle_dir):
        logger = logging.getLogger(f"RLOrchestrator_Cycle_{self.current_cycle}")
        logger.setLevel(logging.INFO)
        
        if logger.hasHandlers():
            logger.handlers.clear()
            
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        log_filename = "orchestrator.log"
        file_handler = logging.FileHandler(os.path.join(cycle_dir, log_filename), mode='a')
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

        if self.current_steps >= self.params_config['global']['buffer_ramp_steps']:
            return self.params_config['global']['max_buffer_size']
        
        # Linear Interpolation Formula
        # Capacity = Min + (Progress % * (Max - Min))
        progress = self.current_steps / self.params_config['global']['buffer_ramp_steps']
        growth_range = self.params_config['global']['max_buffer_size'] - self.params_config['global']['min_buffer_size']
        
        current_capacity = self.params_config['global']['min_buffer_size'] + (progress * growth_range)
        
        return int(current_capacity)


    def _update_circular_buffer(self, new_data):
        max_positions = self.get_dynamic_buffer_limit()
        
        boards = np.array([item['board_state'] for item in new_data], dtype=np.float16)
        policies = np.array([item['policy'] for item in new_data], dtype=np.float16)
        values = np.array([item['value_target'] for item in new_data], dtype=np.float16)

        self.logger.info(f"Appending new data ({len(new_data)} positions) to the circular buffer: {self.buffer_file_path}")
        
        # Check for file corruption *before* opening for modification
        if os.path.exists(self.buffer_file_path):
            try:
                with h5py.File(self.buffer_file_path, 'r') as hf_check:
                    if not all(dset in hf_check for dset in ['inputs', 'policies', 'values']):
                        raise Exception(
                            f"Buffer file is logically corrupted. Missing one or more datasets: {list(hf_check.keys())}."
                        )
            except Exception as e:
                self.logger.critical(f"FATAL: Buffer file is corrupted and cannot be used: {e}")
                raise Exception("Buffer file is corrupted. Cannot proceed.") from e

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
        The main orchestration loop.
        """
        while self.current_steps <= self.total_steps:
            # --- Setup cycle-specific directories and logger ---
            cycle_dir = os.path.join(
                RL_CYCLES_DIR, 
                f"iter_{self.current_cycle:04d}_step_{self.current_steps}"
            )            
            os.makedirs(cycle_dir, exist_ok=True)
            
            # --- Create a NEW logger for this cycle ---
            self.logger = self._setup_cycle_logger(cycle_dir)


            self.logger.info(f"Orchestrator initialized. Reading state from: {CONFIG_RL_STATE_FILE}")
            self.logger.info(f"Last saved self-play positions: {self.state_config['state']['current_cycle']['samples_collected']}")
            self.logger.info(f"Current buffer size: {self.state_config['state']['buffer']['count']}")
            
            self.logger.info(f"Cycle-specific logs will be stored in: {cycle_dir}")

            # Step 1: Run self-play data generation in chunks
            self.logger.info("1. Generating self-play data...")
            
            total_positions_for_cycle = self.params_config['global']['data_generation_positions_per_cycle']
            save_interval = self.params_config['global']['data_generation_save_interval']
            
            # Get the starting point from the state config
            positions_generated_current_cycle = self.state_config['state']['current_cycle']['samples_collected']
            remaining_positions_current_cycle = total_positions_for_cycle - positions_generated_current_cycle
            self.state_config['state']['lifetime']['cycle_idx'] = self.current_cycle
            
            # Instantiate the DataGenerationTask once for the cycle
            if remaining_positions_current_cycle > 0:
                data_generation_task = DataGenerationTask(
                    output_dir=cycle_dir,
                    current_steps=self.current_steps,
                    best_model_path=self.best_model_path,
                    model_config=self.params_config['model'],
                    data_generation_config=self.params_config['data_generation'],
                    state_config=self.state_config
                )

                start_data_gen_time = time.time()
                for new_data_chunk, games_in_chunk, chunk_entropy in data_generation_task.run_for_n_positions(remaining_positions_current_cycle, save_interval):
                             
                    # Process the new chunk of data
                    self._update_circular_buffer(new_data_chunk)
                    
                    self.state_config['state']['current_cycle']['samples_collected'] +=  len(new_data_chunk)
                    self.state_config['state']['current_cycle']['games_played'] += games_in_chunk
                    self.state_config['state']['current_cycle']['self_play_entropy'] = round(float(self.state_config['state']['current_cycle']['self_play_entropy'] + chunk_entropy), 2)
                    
                    self.state_config['state']['lifetime']['samples_generated'] += len(new_data_chunk)
                    self.state_config['state']['lifetime']['games_played'] += games_in_chunk

                    total_data_gen_time = time.time() - start_data_gen_time
                    self.state_config['state']['lifetime']['hours_generating'] = round(self.state_config['state']['lifetime']['hours_generating']  + (total_data_gen_time / 3600), 2)
                    start_data_gen_time = time.time()


                    # Periodically save the state to the YAML file
                    self.logger.info(
                        f"Chunk saved. Total positions for cycle {self.current_cycle}: "
                        f"{ len(new_data_chunk)}/{remaining_positions_current_cycle}. Saving state..."
                    )
                    self._save_state()

                self.logger.info(f"--- Cycle {self.current_cycle} self-play completed successfully! ---")
                self._save_state()
                data_generation_task = None
                            
                            
            # Step 2. Training
            self.logger.info("2. Training a new model on the updated data buffer...")
            start_train_time = time.time()

            train_task = TrainTask(
                output_dir=cycle_dir,
                best_model_path=self.best_model_path,
                model_config=self.params_config['model'],
                training_config=self.params_config['training'],
                state_config=self.state_config['state'],
                global_config=self.params_config['global'],
                hdf5_path=self.buffer_file_path,
                cycle_number=self.current_cycle,
            )
            test_model_path, steps, train_entropy = train_task.run_training_loop()
            total_train_time = time.time() - start_train_time

            self.state_config['state']['lifetime']['hours_training'] = round(self.state_config['state']['lifetime']['hours_training']  + (total_train_time / 3600), 2)
            train_task = None

            self.logger.info(f"2. Model has trained successfully for {steps} steps.")
            self.current_steps += steps
            self.state_config['state']['lifetime']['training_steps'] = self.current_steps
            self.state_config['state']['current_cycle']['training_avg_entropy'] = round(train_entropy, 2)
                        
            # Override the best model with the new model
            self.logger.info(f"Saving new model from {test_model_path} to {self.best_model_path}...")
            shutil.copy(test_model_path, self.best_model_path)
            os.remove(test_model_path)
                        
            self.logger.info("Saving state for the next cycle...")
            self._save_state()

            # Save config and state each cycle
            self.logger.info(f"Saving current state as backup")
            shutil.copy(CONFIG_RL_STATE_FILE, os.path.join(cycle_dir, f'iter_{self.current_cycle:04d}_step_{self.current_steps}_state.yaml'))

            self.logger.info(f"Saving current config as backup")
            shutil.copy(CONFIG_RL_PARAMS_FILE, os.path.join(cycle_dir, f'iter_{self.current_cycle:04d}_step_{self.current_steps}_config.yaml'))

            self.state_config['state']['current_cycle']['samples_collected'] = 0
            self.state_config['state']['current_cycle']['games_played'] = 0
            self.state_config['state']['current_cycle']['self_play_entropy'] = 0
            self.state_config['state']['current_cycle']['training_avg_entropy'] = 0

            self._save_state()

            # Save copy of best model and replay buffer every save interval
            if self.current_steps // self.save_interval > (self.current_steps - steps) // self.save_interval:
                os.makedirs(os.path.join(RL_CYCLES_DIR, 'backup'),  exist_ok=True)

                current_best_model = os.path.join(RL_CYCLES_DIR, f'best_models/iter_{self.current_cycle:04d}_step_{self.current_steps}_model.pth')

                shutil.copy(self.best_model_path, current_best_model)

                self.logger.info(f"Saving current circular buffer as backup")
                shutil.copy(self.buffer_file_path, os.path.join(RL_CYCLES_DIR, f'backup/iter_{self.current_cycle:04d}_step_{self.current_steps}_replay_memory.hdf5'))

            self.current_cycle += 1

            # Increment loop
            if self.current_steps <= self.total_steps:
                self.logger.info(f"Sleeping for 10 seconds before starting cycle {self.current_cycle}...")
                time.sleep(10)
        

if __name__ == "__main__":
    orchestrator = RLOrchestrator()
    orchestrator.run()