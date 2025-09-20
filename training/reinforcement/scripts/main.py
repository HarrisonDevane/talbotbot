import os
import time
import yaml
import logging
import numpy as np
import h5py
import shutil

# Assuming this import is in your project structure
from data_generation_task import DataGenerationTask
from train_task import TrainTask
from evaluation_task import EvaluationTask

# --- Configuration Paths ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

RL_CYCLES_DIR = os.path.abspath(os.path.join(rl_dir, "rl_cycles"))
CONFIG_RL_STATE_FILE = os.path.abspath(os.path.join(rl_dir, "config", "rl_state.yaml"))
CONFIG_RL_PARAMS_FILE = os.path.abspath(os.path.join(rl_dir, "config", "rl_config.yaml"))


class RLOrchestrator:
    def __init__(self):
        # Initialize logger as None. It will be set for the first time inside the run() method.
        self.logger = None

        self.params_config, self.state_config = self._load_configs()
        self.current_cycle = self.state_config['state']['current_cycle']
        self.total_cycles = self.params_config['global']['total_cycles']

        os.makedirs(RL_CYCLES_DIR, exist_ok=True)

        # Get buffer path from the global config file and make it absolute
        buffer_file_name = self.params_config['global']['buffer_file_path']
        self.buffer_file_path = buffer_file_name

    def _setup_cycle_logger(self, cycle_dir):
        # Create a 'logs' subdirectory within the cycle directory
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


    def _load_configs(self):
        with open(CONFIG_RL_PARAMS_FILE, 'r') as f:
            params_config = yaml.safe_load(f)

        with open(CONFIG_RL_STATE_FILE, 'r') as f:
            state_config = yaml.safe_load(f)
        
        return params_config, state_config
    

    def _save_state(self):
        with open(CONFIG_RL_STATE_FILE, 'w') as f:
            yaml.safe_dump(self.state_config, f)


    def _update_circular_buffer(self, new_data):
        max_positions = self.params_config['global']['buffer_positions_total']
        
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
        current_size = self.state_config['state']['buffer_positions_current']
        current_write_head = self.state_config['state']['buffer_write_head']
        
        with h5py.File(self.buffer_file_path, 'a') as hf:
            if 'inputs' in hf and 'policies' in hf and 'values' in hf:
                self.logger.debug(f"Existing datasets (headers) are: {list(hf.keys())}.")

                inputs_dset = hf['inputs']
                policies_dset = hf['policies']
                values_dset = hf['values']

                boards_remaining = boards
                policies_remaining = policies
                values_remaining = values

                ### STEP 1: APPEND to unfilled part of buffer ###
                if current_size < max_positions:
                    space_left = max_positions - current_size
                    append_count = min(len(boards_remaining), space_left)

                    boards_to_append = boards_remaining[:append_count]
                    policies_to_append = policies_remaining[:append_count]
                    values_to_append = values_remaining[:append_count]

                    # Resize datasets to accommodate append
                    new_size = current_size + append_count
                    inputs_dset.resize(new_size, axis=0)
                    policies_dset.resize(new_size, axis=0)
                    values_dset.resize(new_size, axis=0)

                    inputs_dset[current_size : current_size + append_count] = boards_to_append
                    policies_dset[current_size : current_size + append_count] = policies_to_append
                    values_dset[current_size : current_size + append_count] = values_to_append

                    self.state_config['state']['buffer_positions_current'] = new_size

                    # Remove appended data from the remaining pool
                    boards_remaining = boards_remaining[append_count:]
                    policies_remaining = policies_remaining[append_count:]
                    values_remaining = values_remaining[append_count:]

                    current_write_head = new_size % max_positions  # Important: update write head if it wrapped to max

                ### STEP 2: CIRCULAR OVERWRITE for remaining data ###
                num_remaining = len(boards_remaining)
                if num_remaining > 0:
                    if current_write_head + num_remaining <= max_positions:
                        # fits in one go
                        inputs_dset[current_write_head : current_write_head + num_remaining] = boards_remaining
                        policies_dset[current_write_head : current_write_head + num_remaining] = policies_remaining
                        values_dset[current_write_head : current_write_head + num_remaining] = values_remaining
                    else:
                        # wraparound
                        first_part_len = max_positions - current_write_head
                        second_part_len = num_remaining - first_part_len

                        inputs_dset[current_write_head:] = boards_remaining[:first_part_len]
                        policies_dset[current_write_head:] = policies_remaining[:first_part_len]
                        values_dset[current_write_head:] = values_remaining[:first_part_len]

                        inputs_dset[:second_part_len] = boards_remaining[first_part_len:]
                        policies_dset[:second_part_len] = policies_remaining[first_part_len:]
                        values_dset[:second_part_len] = values_remaining[first_part_len:]

                    # Write head moves forward circularly
                    new_write_head = (current_write_head + num_remaining) % max_positions
                    self.state_config['state']['buffer_write_head'] = new_write_head

                # final size is always clamped to max_positions
                self.state_config['state']['buffer_positions_current'] = min(
                    self.state_config['state']['buffer_positions_current'] + num_remaining, max_positions
                )
                final_size = self.state_config['state']['buffer_positions_current']

            else:
                # File is brand new or was just created
                self.logger.info("Creating new HDF5 datasets for the buffer with explicit chunking.")
                
                # Get the batch size from your config
                batch_size = self.params_config['training']['batch_size']
                
                # Use the shape of the first data chunk to define the rest of the dimensions
                board_shape = boards.shape[1:]
                policy_shape = policies.shape[1:]
                
                # Create datasets with an explicit chunk shape
                hf.create_dataset('inputs', 
                                data=boards, 
                                maxshape=(None, *board_shape), 
                                dtype=np.float16, 
                                compression='gzip', 
                                chunks=(batch_size, *board_shape))
                
                # Policies are a single dimension
                hf.create_dataset('policies', 
                                data=policies, 
                                maxshape=(None, *policy_shape), 
                                dtype=np.float16, 
                                compression='gzip', 
                                chunks=(batch_size, *policy_shape))
                
                # Values are a single dimension
                hf.create_dataset('values', 
                                data=values, 
                                maxshape=(None,),
                                dtype=np.float16, 
                                compression='gzip', 
                                chunks=(batch_size,))
                
                # Update the new state variables
                self.state_config['state']['buffer_positions_current'] = len(new_data)
                final_size = len(new_data)

        self.logger.info(
            f"Circular buffer updated. It now contains {final_size} of "
            f"{max_positions} total positions.")


    def run(self):
        """
        The main orchestration loop.
        """
        while self.current_cycle <= self.total_cycles:
            # --- Setup cycle-specific directories and logger ---
            cycle_dir = os.path.join(RL_CYCLES_DIR, f"iteration_{self.current_cycle}")
            os.makedirs(cycle_dir, exist_ok=True)
            
            # --- Create a NEW logger for this cycle ---
            self.logger = self._setup_cycle_logger(cycle_dir)

            # Initial messages are now logged here, inside the first cycle's log
            if self.current_cycle == self.state_config['state']['current_cycle']:
                self.logger.info(f"Orchestrator initialized. Reading state from: {CONFIG_RL_STATE_FILE}")
                self.logger.info(f"Last completed cycle: {self.current_cycle}. Total cycles to run: {self.total_cycles - self.current_cycle}.")
                self.logger.info(f"Last saved self-play positions: {self.state_config['state']['data_generation_positions_current']}")
                self.logger.info(f"Current buffer size: {self.state_config['state']['buffer_positions_current']}")
            
            self.logger.info(f"\n--- Starting RL Cycle {self.current_cycle} (run {self.current_cycle} of {self.total_cycles}) ---")
            self.logger.info(f"Cycle-specific logs will be stored in: {cycle_dir}")

            # Step 1: Run self-play data generation in chunks
            self.logger.info("1. Generating self-play data...")
            
            total_positions_for_cycle = self.params_config['global']['data_generation_positions_per_cycle']
            save_interval = self.params_config['global']['data_generation_save_interval']
            best_model_path = self.params_config['model']['best_model_path']
            
            # Get the starting point from the state config
            positions_generated_this_cycle = self.state_config['state']['data_generation_positions_current']
            remaining_positions_this_cycle = total_positions_for_cycle - positions_generated_this_cycle
            
            # Instantiate the DataGenerationTask once for the cycle
            if remaining_positions_this_cycle > 0:
                data_generation_task = DataGenerationTask(
                    output_dir=cycle_dir,
                    model_config=self.params_config['model'],
                    data_generation_config=self.params_config['data_generation'],
                    best_iter = self.state_config['state']['best_model_cycle']
                )

                start_data_gen_time = time.time()
                for new_data_chunk, games_in_chunk in data_generation_task.run_for_n_positions(remaining_positions_this_cycle, save_interval):
                    
                    # Process the new chunk of data
                    self._update_circular_buffer(new_data_chunk)
                    
                    positions_generated_this_cycle += len(new_data_chunk)
                    self.state_config['state']['data_generation_positions_current'] = positions_generated_this_cycle
                    self.state_config['state']['total_positions'] += len(new_data_chunk)
                    self.state_config['state']['total_games'] += games_in_chunk

                    total_data_gen_time = time.time() - start_data_gen_time
                    self.state_config['state']['total_hours_data_generation'] += (total_data_gen_time / 3600)
                    start_data_gen_time = time.time()


                    # Periodically save the state to the YAML file
                    self.logger.info(
                        f"Chunk saved. Total positions for cycle {self.current_cycle}: "
                        f"{positions_generated_this_cycle}/{remaining_positions_this_cycle}. Saving state..."
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
                model_config=self.params_config['model'],
                training_config=self.params_config['training'],
                hdf5_path=self.buffer_file_path,
                cycle_number=self.current_cycle
            )
            model_path, steps = train_task.run_training_loop()
            total_train_time = time.time() - start_train_time
            self.state_config['state']['total_hours_training'] += (total_train_time / 3600)

            train_task = None
            self.logger.info(f"2. Model has trained successfully for {steps} steps.")
            
            # # Step 3. Evaluation
            self.logger.info(f"3. Initiating evaluation by self play for {self.params_config['global']['eval_games']} games against current best model...")
            start_eval_time = time.time()

            evaluation_task = EvaluationTask(
                output_dir=cycle_dir,
                test_model=model_path,
                model_config=self.params_config['model'],
                evaluation_config=self.params_config['evaluation'],
                current_iter = self.current_cycle,
                best_iter = self.state_config['state']['best_model_cycle']
            )
            test_score, best_score = evaluation_task.run_for_n_games(self.params_config['global']['eval_games'])

            total_eval_time = time.time() - start_eval_time
            self.state_config['state']['total_hours_evaluation'] += (total_eval_time / 3600)
            self.state_config['state']['total_hours'] = self.state_config['state']['total_hours_data_generation'] + self.state_config['state']['total_hours_training'] + self.state_config['state']['total_hours_evaluation']

            evaluation_task = None
            self.logger.info(f"3. Evaluation finished with result: {test_score}-{best_score}")

            # Step 4. Save best model
            win_rate = test_score / self.params_config['global']['eval_games']
            if win_rate > self.params_config['global']['eval_cutoff']:
                self.logger.info(f"New model has a win rate of {win_rate:.3f} (> {self.params_config['global']['eval_cutoff']}), accepting it as the new best model.")
                                
                
                # Override the best model with the new, better model
                self.logger.info(f"Saving new model from {model_path} to {best_model_path}...")
                shutil.copy(model_path, best_model_path)

                self.state_config['state']['best_model_updates'] += 1
                self.state_config['state']['total_training_steps'] += steps
                self.state_config['state']['best_model_cycle'] = self.current_cycle
                
            self.logger.info(f"Current best cycle: {self.state_config['state']['best_model_cycle']}...")

            # After the loop is complete, update the state for the next cycle
            self.state_config['state']['data_generation_positions_current'] = 0
            self.state_config['state']['current_cycle'] = self.current_cycle+1
            
            self.logger.info("Saving state for the next cycle...")
            self._save_state()

            # Save copy of best model and replay buffer every save interval
            if self.current_cycle > 0 and self.current_cycle % self.params_config['global']['best_model_save_interval'] == 0:
                current_best_model = os.path.join(rl_dir, f'rl_cycles/best_models/best_model_iter_{self.current_cycle}.pth')

                self.logger.info(f"Saving current best model to after {self.params_config['global']['best_model_save_interval']} to {current_best_model}")
                shutil.copy(best_model_path, current_best_model)

                self.logger.info(f"Saving current circular buffer as backup")
                shutil.copy(self.buffer_file_path, os.path.join(rl_dir, f'rl_cycles/backup/circular_buffer_cycle_{self.current_cycle}.hdf5'))

                self.logger.info(f"Saving current state as backup")
                shutil.copy(CONFIG_RL_STATE_FILE, os.path.join(rl_dir, f'rl_cycles/backup/state_cycle_{self.current_cycle}.yaml'))

            self.current_cycle += 1
            os.remove(model_path)

            # Increment loop
            if self.current_cycle < self.total_cycles:
                self.logger.info(f"Sleeping for 10 seconds before starting cycle {self.current_cycle}...")
                time.sleep(10)
        

if __name__ == "__main__":
    orchestrator = RLOrchestrator()
    orchestrator.run()