import os
import time
import yaml
import logging
import numpy as np
import h5py
import shutil
from datetime import datetime

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
        log_dir = os.path.join(cycle_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger(f"RLOrchestrator_Cycle_{self.current_cycle}")
        logger.setLevel(logging.INFO)
        
        if logger.hasHandlers():
            logger.handlers.clear()
            
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        log_filename = "orchestrator.log"
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), mode='a')
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
        
        with h5py.File(self.buffer_file_path, 'a') as hf:
            if 'inputs' in hf and 'policies' in hf and 'values' in hf:
                inputs_dset = hf['inputs']
                policies_dset = hf['policies']
                values_dset = hf['values']

                current_size = inputs_dset.shape[0]
                new_size = current_size + len(new_data)
                
                inputs_dset.resize(new_size, axis=0)
                policies_dset.resize(new_size, axis=0)
                values_dset.resize(new_size, axis=0)
                
                inputs_dset[current_size:] = boards
                policies_dset[current_size:] = policies
                values_dset[current_size:] = values
                
                self.state_config['state']['buffer_positions_current'] = new_size

                if new_size > max_positions:
                    trim_start = new_size - max_positions
                    temp_inputs = inputs_dset[trim_start:]
                    temp_policies = policies_dset[trim_start:]
                    temp_values = values_dset[trim_start:]

                    del hf['inputs']
                    del hf['policies']
                    del hf['values']
                    
                    hf.create_dataset('inputs', data=temp_inputs, maxshape=(None, *temp_inputs.shape[1:]), dtype=np.float16, compression='gzip', chunks=True)
                    hf.create_dataset('policies', data=temp_policies, maxshape=(None, *temp_policies.shape[1:]), dtype=np.float16, compression='gzip', chunks=True)
                    hf.create_dataset('values', data=temp_values, maxshape=(None, *temp_values.shape[1:]), dtype=np.float16, compression='gzip', chunks=True)
                    
                    final_size = max_positions
                    self.state_config['state']['buffer_positions_current'] = final_size
                else:
                    final_size = new_size
            else:
                hf.create_dataset('inputs', data=boards, maxshape=(None, *boards.shape[1:]), dtype=np.float16, compression='gzip', chunks=True)
                hf.create_dataset('policies', data=policies, maxshape=(None, *policies.shape[1:]), dtype=np.float16, compression='gzip', chunks=True)
                hf.create_dataset('values', data=values, maxshape=(None, *values.shape[1:]), dtype=np.float16, compression='gzip', chunks=True)
                final_size = len(new_data)
                self.state_config['state']['buffer_positions_current'] = final_size

            self.logger.info(
                f"Circular buffer updated. It now contains {final_size} of "
                f"{max_positions} total positions.")
    

    def run(self):
        """
        The main orchestration loop.
        """
        while self.current_cycle < self.total_cycles:
            next_cycle = self.current_cycle + 1

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
            training_steps = self.state_config['state']['training_steps']
            remaining_positions_this_cycle = total_positions_for_cycle - positions_generated_this_cycle
            
            # Instantiate the DataGenerationTask once for the cycle
            data_generation_task = DataGenerationTask(
                output_dir=cycle_dir,
                model_config=self.params_config['model'],
                data_generation_config=self.params_config['data_generation'],
                best_iter = self.state_config['state']['best_model_cycle']
            )

            for new_data_chunk, games_in_chunk in data_generation_task.run_for_n_positions(remaining_positions_this_cycle, save_interval):
                
                # Process the new chunk of data
                self._update_circular_buffer(new_data_chunk)
                
                positions_generated_this_cycle += len(new_data_chunk)
                self.state_config['state']['data_generation_positions_current'] = positions_generated_this_cycle
                self.state_config['state']['total_positions'] += len(new_data_chunk)
                self.state_config['state']['total_games'] += games_in_chunk

                # Periodically save the state to the YAML file
                self.logger.info(
                    f"Chunk saved. Total positions for cycle {self.current_cycle}: "
                    f"{positions_generated_this_cycle}/{remaining_positions_this_cycle}. Saving state..."
                )
                self._save_state()

            self.logger.info(f"--- Cycle {self.current_cycle} self-play completed successfully! ---")
            self._save_state()
            
            # Step 2. Training
            self.logger.info("2. Training a new model on the updated data buffer...")

            train_task = TrainTask(
                output_dir=cycle_dir,
                model_config=self.params_config['model'],
                training_config=self.params_config['training'],
                hdf5_path=self.buffer_file_path,
                cycle_number=self.current_cycle
            )
            model_path, steps = train_task.run_training_loop()

            self.logger.info(f"2. Model has trained successfully for {steps} steps.")

            # Save first non-random model
            if self.current_cycle == 1:
                shutil.copy(model_path, best_model_path)

                self.state_config['state']['training_steps'] = training_steps + steps
                self.state_config['state']['best_model_cycle'] = self.current_cycle
            else: 

                # # Step 3. Evaluation
                self.logger.info(f"3. Initiating evaluation by self play for {self.params_config['global']['eval_games']} games against current best model...")

                evaluation_task = EvaluationTask(
                    output_dir=cycle_dir,
                    test_model=model_path,
                    model_config=self.params_config['model'],
                    evaluation_config=self.params_config['evaluation'],
                    current_iter = self.current_cycle,
                    best_iter = self.state_config['state']['best_model_cycle']
                )
                test_score, best_score = evaluation_task.run_for_n_games(self.params_config['global']['eval_games'])

                self.logger.info(f"3. Evaluation finished with result: {test_score}-{best_score}")

                # Step 4. Save best model
                win_rate = test_score / self.params_config['global']['eval_games']
                if win_rate > self.params_config['global']['eval_cutoff']:
                    self.logger.info(f"New model has a win rate of {win_rate:.2f} (> {self.params_config['global']['eval_cutoff']}), accepting it as the new best model.")
                                    
                    
                    # Override the best model with the new, better model
                    self.logger.info(f"Saving new model from {model_path} to {best_model_path}...")
                    shutil.copy(model_path, best_model_path)

                    self.state_config['state']['training_steps'] = training_steps + steps
                    self.state_config['state']['best_model_cycle'] = self.current_cycle
                    
                self.logger.info(f"Current best cycle: {self.state_config['state']['best_model_cycle']}...")

            # Save copy of best model every save interval
            if self.current_cycle > 0 and self.current_cycle % self.params_config['global']['best_model_save_interval'] == 0:
                current_best_model = os.path.join(rl_dir, f'best_models/best_model_iter_{self.current_cycle}.pth')

                self.logger.info(f"Saving current best model to after {self.params_config['global']['best_model_save_interval']} to {current_best_model}")
                shutil.copy(best_model_path, current_best_model)


            # After the loop is complete, update the state for the next cycle
            self.current_cycle = next_cycle
            self.state_config['state']['data_generation_positions_current'] = 0
            self.state_config['state']['current_cycle'] = self.current_cycle
            
            self.logger.info("Saving state for the next cycle...")
            self._save_state()

            # Increment loop
            if self.current_cycle < self.total_cycles:
                self.logger.info(f"Sleeping for 10 seconds before starting cycle {self.current_cycle}...")
                time.sleep(10)
        
        self.logger.info("\n--- All requested RL cycles have been completed. The orchestrator will now shut down. ---")


if __name__ == "__main__":
    orchestrator = RLOrchestrator()
    orchestrator.run()