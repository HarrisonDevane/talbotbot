# main.py

import os
import time
import yaml
import logging
import numpy as np
import h5py
from datetime import datetime

# Assuming this import is in your project structure
from self_play_task import SelfPlayTask

# --- Configuration Paths ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

RL_CYCLES_DIR = os.path.abspath(os.path.join(rl_dir, "rl_cycles"))
RL_ORCHESTRATOR_LOG_DIR = os.path.abspath(os.path.join(RL_CYCLES_DIR, "rl_logs"))
CONFIG_STATE_FILE = os.path.abspath(os.path.join(rl_dir, "config", "rl_state.yaml"))
CONFIG_SELF_PLAY_FILE = os.path.abspath(os.path.join(rl_dir, "config", "rl_config.yaml"))


class RLOrchestrator:
    def __init__(self):
        self.logger = self._setup_global_logger()
        self.logger.info(f"Orchestrator initialized. Reading state from: {CONFIG_STATE_FILE}")

        self.current_cycle, self.total_cycles = self._load_state()
        self.logger.info(f"Last completed cycle: {self.current_cycle}. Total cycles to run: {self.total_cycles}.")

        os.makedirs(RL_CYCLES_DIR, exist_ok=True)

        # Load global config to get buffer size and file path
        with open(CONFIG_SELF_PLAY_FILE, 'r') as f:
            self.global_config = yaml.safe_load(f)
        
        # New: Get buffer path from the config file and make it absolute
        buffer_file_name = self.global_config['stored_data']['buffer_file_path']
        self.buffer_file_path = os.path.join(rl_dir, buffer_file_name)


    def _setup_global_logger(self):
        """
        Sets up a main logger for the orchestrator, logging only to a single file.
        """
        os.makedirs(RL_ORCHESTRATOR_LOG_DIR, exist_ok=True)
        
        logger = logging.getLogger("RLOrchestrator")
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to prevent duplicate logs
        if logger.hasHandlers():
            logger.handlers.clear()
        
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        
        # File handler for a persistent log of the entire run
        log_filename = f"orchestrator_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        file_handler = logging.FileHandler(os.path.join(RL_ORCHESTRATOR_LOG_DIR, log_filename))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
            
        return logger

    def _load_state(self):
        if os.path.exists(CONFIG_STATE_FILE):
            with open(CONFIG_STATE_FILE, 'r') as f:
                state = yaml.safe_load(f)
                return state.get('current_cycle', 0), state.get('total_cycles', 1)


    def _save_state(self, cycle_number):
        with open(CONFIG_STATE_FILE, 'w') as f:
            yaml.safe_dump({'current_cycle': cycle_number, 'total_cycles': self.total_cycles}, f)
    

    def _update_circular_buffer(self, new_data, cycle_dir):
        """
        First, saves the data to a cycle-specific HDF5 file.
        Second, appends this new data to the main circular buffer HDF5 file,
        trimming it to the maximum size.
        """
        max_positions = self.global_config['stored_data']['positions_total']
        
        # Step 1: Save data to the cycle-specific HDF5 file
        iteration_data_dir = os.path.join(cycle_dir, "data")
        os.makedirs(iteration_data_dir, exist_ok=True)
        iteration_data_path = os.path.join(iteration_data_dir, "iteration_data.hdf5")
        
        self.logger.info(f"Saving cycle data to: {iteration_data_path}")

        # Extract data from list of dictionaries into separate arrays
        boards = np.array([item['board_state'] for item in new_data], dtype=np.float16)
        policies = np.array([item['policy'] for item in new_data], dtype=np.int32)
        values = np.array([item['value_target'] for item in new_data], dtype=np.float16)

        # Create HDF5 file and datasets for the current iteration
        with h5py.File(iteration_data_path, 'w') as hf:
            hf.create_dataset(
                'inputs',
                data=boards,
                dtype=np.float16,
                compression='gzip',
                chunks=True
            )
            hf.create_dataset(
                'policies',
                data=policies,
                dtype=np.int32,
                compression='gzip',
                chunks=True
            )
            hf.create_dataset(
                'values',
                data=values,
                dtype=np.float16,
                compression='gzip',
                chunks=True
            )
        self.logger.info("Cycle data saved successfully.")

        # Step 2: Append new data to the main circular buffer
        self.logger.info(f"Appending new data to the main circular buffer: {self.buffer_file_path}")
        with h5py.File(self.buffer_file_path, 'a') as hf:
            if 'inputs' in hf and 'policies' in hf and 'values' in hf:
                # Append to existing datasets
                inputs_dset = hf['inputs']
                policies_dset = hf['policies']
                values_dset = hf['values']

                # Resize datasets to accommodate new data
                current_size = inputs_dset.shape[0]
                new_size = current_size + len(new_data)
                
                inputs_dset.resize(new_size, axis=0)
                policies_dset.resize(new_size, axis=0)
                values_dset.resize(new_size, axis=0)
                
                # Append new data
                inputs_dset[current_size:] = boards
                policies_dset[current_size:] = policies
                values_dset[current_size:] = values
                
                # Trim if over max size
                if new_size > max_positions:
                    trim_start = new_size - max_positions
                    
                    # Create temporary datasets for the trimmed data
                    temp_inputs = inputs_dset[trim_start:]
                    temp_policies = policies_dset[trim_start:]
                    temp_values = values_dset[trim_start:]

                    # Delete original datasets and create new ones with trimmed data
                    del hf['inputs']
                    del hf['policies']
                    del hf['values']
                    
                    hf.create_dataset(
                        'inputs',
                        data=temp_inputs,
                        maxshape=(None, *temp_inputs.shape[1:]),
                        dtype=np.float16,
                        compression='gzip',
                        chunks=True
                    )
                    hf.create_dataset(
                        'policies',
                        data=temp_policies,
                        maxshape=(None, *temp_policies.shape[1:]),
                        dtype=np.int32,
                        compression='gzip',
                        chunks=True
                    )
                    hf.create_dataset(
                        'values',
                        data=temp_values,
                        maxshape=(None, *temp_values.shape[1:]),
                        dtype=np.float16,
                        compression='gzip',
                        chunks=True
                    )
                    
                    final_size = max_positions
                else:
                    final_size = new_size
            else:
                hf.create_dataset(
                    'inputs',
                    data=boards,
                    maxshape=(None, *boards.shape[1:]),
                    dtype=np.float16,
                    compression='gzip',
                    chunks=True
                )
                hf.create_dataset(
                    'policies',
                    data=policies,
                    maxshape=(None, *policies.shape[1:]),
                    dtype=np.int32,
                    compression='gzip',
                    chunks=True
                )
                hf.create_dataset(
                    'values',
                    data=values,
                    maxshape=(None, *values.shape[1:]),
                    dtype=np.float16,
                    compression='gzip',
                    chunks=True
                )
                final_size = len(new_data)

            self.logger.info(
                f"Circular buffer updated. It now contains {final_size} of "
                f"{max_positions} total positions.")

    def run(self):
        """
        The main orchestration loop.
        """
        for i in range(self.total_cycles):
            next_cycle = self.current_cycle + 1

            # --- Setup cycle-specific directories ---
            cycle_dir = os.path.join(RL_CYCLES_DIR, f"iteration_{next_cycle}")
            os.makedirs(cycle_dir, exist_ok=True)
            
            self.logger.info(f"\n--- Starting RL Cycle {next_cycle} (run {i + 1} of {self.total_cycles}) ---")
            self.logger.info(f"Cycle-specific logs will be stored in: {cycle_dir}")
            
            # Step 1: Run Self-Play
            self.logger.info("1. Generating self-play data...")
            self_play_task = SelfPlayTask(
                config_path=CONFIG_SELF_PLAY_FILE,
                output_dir=cycle_dir,
            )
            all_training_data = self_play_task.run()
            
            # Step 2: Update circular buffer
            self.logger.info("2. Updating the circular buffer with new data...")
            # Pass the cycle_dir to the update method
            self._update_circular_buffer(all_training_data, cycle_dir)

            # After successful completion, update the state
            self.logger.info(f"--- Cycle {next_cycle} self-play completed successfully! ---")
            self.current_cycle = next_cycle
            self._save_state(self.current_cycle)

            if i < self.total_cycles - 1:
                self.logger.info(f"Sleeping for 10 seconds before starting cycle {next_cycle + 1}...")
                time.sleep(10)
        
        self.logger.info("\n--- All requested RL cycles have been completed. The orchestrator will now shut down. ---")


if __name__ == "__main__":
    orchestrator = RLOrchestrator()
    orchestrator.run()
