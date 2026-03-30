import os
import time
import yaml
import logging
import numpy as np
import shutil
import random
import torch
import textwrap
import multiprocessing as mp
import lmdb
import lz4.frame

from data_generation_task import DataGenerationTask
from train_task import TrainTask

current_script_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

from src_shared.model import ChessAIModel
import src_shared.utils as utils

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
        self.best_model_path = os.path.abspath(os.path.join(RL_DIR, "best_models", "best_model.pth"))

        if not os.path.exists(self.best_model_path):
            self._create_new_model()

        # Get buffer path from the global config file and make it absolute
        self.buffer_file_path = os.path.abspath(os.path.join(RL_DIR, "replay_memory.lmdb"))
        self.env = lmdb.open(
            self.buffer_file_path,
            map_size=1024 * 1024 * 1024 * 128,
            writemap=True,
            map_async=True,
            max_dbs=1
        )

    def _create_new_model(self):
        random.seed(self.params_config['training']['seed'])
        np.random.seed(self.params_config['training']['seed'])
        torch.manual_seed(self.params_config['training']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.params_config['training']['seed'])

        # Create a new, seeded instance of the model
        model = ChessAIModel(
            num_input_planes=self.params_config['model']['input_planes'],
            num_residual_blocks=self.params_config['model']['resblocks'],
            num_filters=self.params_config['model']['filters'],
            bottleneck_channels=self.params_config['model']['bottleneck_channels'],
            broadcast_reduction_ratio=self.params_config['model']['broadcast_reduction_ratio'],
            broadcast_interval=self.params_config['model']['broadcast_interval']
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
        logger.setLevel(self.params_config['data_generation']['main_logging_level'])
        
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
        current_write_head = self.state_config['state']['buffer']['head_ptr']
        current_size = self.state_config['state']['buffer']['count']

        with self.env.begin(write=True) as txn:
            for item in new_data:
                p_board = np.packbits(item['board_state'].astype(np.bool_)).tobytes()
                p_mask = np.packbits(item['legal_mask'].astype(np.bool_)).tobytes()
                
                policy = item['policy'].astype(np.float32)
                idx = np.where(policy > 0)[0].astype(np.uint16)
                val = policy[idx].astype(np.float16)
                
                raw_payload = (
                    np.uint16(len(idx)).tobytes() + p_board + p_mask + 
                    idx.tobytes() + val.tobytes() + np.float16(item['value_target']).tobytes()
                )
                compressed_blob = lz4.frame.compress(raw_payload)

                key = f"{current_write_head}".encode('ascii')
                txn.put(key, compressed_blob)

                if current_write_head == current_size:
                    current_size += 1

                current_write_head = (current_write_head + 1) % max_positions
                
                if current_write_head == 0:
                    self.state_config['state']['buffer']['wraps'] += 1
                    

        self.state_config['state']['buffer']['head_ptr'] = current_write_head
        self.state_config['state']['buffer']['count'] = current_size


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
        )

        self.train_task = TrainTask(
            best_model_path=self.best_model_path,
            model_config=self.params_config['model'],
            training_config=self.params_config['training'],
            state_config=self.state_config['state'],
            global_config=self.params_config['global'],
            lmdb_path=self.buffer_file_path,
            env=self.env
        )

        # 2. Start the generator
        data_iterator = self.data_gen_task.run_persistently()
        current_log_dir = None

        while self.current_steps.value < self.total_steps:
            start_time = time.time()
            rotation_interval = self.params_config['global']['logging_rotation_steps']
            target_folder_step = ((self.current_steps.value // rotation_interval) * rotation_interval)
            new_log_dir = os.path.join(RL_DIR, f"run_step_{target_folder_step:06d}")
            
            if new_log_dir != current_log_dir:
                os.makedirs(new_log_dir, exist_ok=True)
                current_log_dir = new_log_dir
                self.logger = self._setup_persistent_logger(current_log_dir)
                self.logger.info(f"New logging phase started at step {self.current_steps.value}")

            # --- Step 1: Collect Data ---
            local_chunk = []
            positions_collected = 0
            games_in_chunk = 0
            chunk_entropy = 0.0
            chunk_size = self.params_config['training']['batch_size'] / self.params_config['training']['sampling_ratio']

            while positions_collected < chunk_size:
                # Get one game from worker (training_data, game_entropy)
                new_game_data, game_entropy = next(data_iterator)
                
                local_chunk.extend(new_game_data)
                positions_collected += len(new_game_data)
                games_in_chunk += 1
                chunk_entropy += game_entropy

            # --- Step 2: Update Buffer (SWMR) ---
            self._update_circular_buffer(local_chunk)

            test_model_path = self.train_task.run_single_step(current_log_dir, self.state_config)

            shutil.copy(test_model_path, self.best_model_path)
            os.remove(test_model_path)

            # --- Step 5: Update State ---
            with self.current_steps.get_lock():
                self.current_steps.value += 1

            self.state_config['state']['lifetime']['training_steps'] = int(self.current_steps.value)
            self.state_config['state']['lifetime']['samples_generated'] += int(len(local_chunk))
            self.state_config['state']['lifetime']['games_played'] += int(games_in_chunk)
            self.state_config['state']['lifetime']['self_play_entropy'] = round(
                self.state_config['state']['lifetime']['self_play_entropy'] + float(chunk_entropy), 4)

            self.state_config['state']['current_interval']['samples_generated'] += int(len(local_chunk))
            self.state_config['state']['current_interval']['games_played'] += int(games_in_chunk)

            self.state_config['state']['current_interval']['self_play_entropy'] = round(
                self.state_config['state']['current_interval']['self_play_entropy'] + float(chunk_entropy), 4)

            self._save_state()

            total_time = time.time() - start_time
            self.state_config['state']['lifetime']['hours_training'] = round(self.state_config['state']['lifetime']['hours_training']  + (total_time / 3600), 4)

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
                    f'step_{self.current_steps.value:06d}_replay_memory.lmdb'
                )
                
                self.logger.info(f"Creating periodic backup at step {self.current_steps.value}...")

                # Backup the state and config files as well
                shutil.copy(CONFIG_RL_STATE_FILE, os.path.join(current_log_dir, f'step_{self.current_steps.value:06d}_state.yaml'))
                shutil.copy(CONFIG_RL_PARAMS_FILE, os.path.join(current_log_dir, f'step_{self.current_steps.value:06d}_config.yaml'))

                self.state_config['state']['current_interval']['samples_generated'] = 0
                self.state_config['state']['current_interval']['games_played'] = 0
                self.state_config['state']['current_interval']['self_play_entropy'] = 0

                self._save_state()

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
            if hasattr(orchestrator, 'data_gen_task'):
                print("Cleaning up background workers...")
                orchestrator.data_gen_task.terminate_all()
            print("Shutdown complete.")