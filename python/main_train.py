import os
import time
import yaml
import logging
import subprocess
import shutil
import random
import psutil
import torch
import textwrap
import lmdb
import warnings

from trainer import TrainTask
from model import ChessAIModel, fuse_bn_for_export

current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

RL_DIR = os.path.abspath(os.path.join(root_dir, "rl_dir"))
RL_STATE_FILE = os.path.abspath(os.path.join(root_dir, "config", "rl_state.yaml"))
RL_PARAMS_FILE = os.path.abspath(os.path.join(root_dir, "config", "rl_training.yaml"))
MODEL_FILE = os.path.abspath(os.path.join(root_dir, "config", "model.yaml"))

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.onnx")

class RLOrchestrator:
    def __init__(self):
        self.logger = None

        # Load configurations
        with open(RL_PARAMS_FILE, 'r') as f:
            self.params_config = yaml.safe_load(f)
        with open(MODEL_FILE, 'r') as f:
            self.model_config = yaml.safe_load(f)

        if not os.path.exists(RL_STATE_FILE):            
            default_state_yaml = textwrap.dedent("""
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
            with open(RL_STATE_FILE, 'w') as f:
                f.write(default_state_yaml)

        with open(RL_STATE_FILE, 'r') as f:
            self.state_config = yaml.safe_load(f)

        self.current_step = self.state_config['lifetime']['training_steps']
        self.total_steps = self.params_config['global']['total_training_steps']
        self.save_interval = self.params_config['global']['backup_interval_steps']

        rotation_interval = self.params_config['global']['logging_rotation_steps']
        target_folder_step = (self.current_step // rotation_interval) * rotation_interval
        initial_log_dir = os.path.join(RL_DIR, f"run_step_{target_folder_step:06d}")

        os.makedirs(initial_log_dir, exist_ok=True)
        self.logger = self._setup_persistent_logger(initial_log_dir)

        base_path = os.path.join(root_dir, self.params_config['global']['model_path'])
        self.best_model_pth = os.path.abspath(base_path + ".pth")
        
        if not os.path.exists(self.best_model_pth):
            self._create_seed_models()

        self.buffer_file_path = os.path.abspath(os.path.join(RL_DIR, "replay_memory.lmdb"))

        if not os.path.exists(self.buffer_file_path):
            tmp_env = lmdb.open(self.buffer_file_path, map_size=1024*1024*1024*128)
            tmp_env.close()

        self.env = lmdb.open(
            self.buffer_file_path,
            map_size=1024 * 1024 * 1024 * 128,
            readonly=True,
            lock=False,
            readahead=False
        )

    def _create_seed_models(self):
        random.seed(self.params_config['training']['seed'])
        torch.manual_seed(self.params_config['training']['seed'])
        model = ChessAIModel(self.model_config)
        os.makedirs(os.path.join(RL_DIR, 'best_models'), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, self.best_model_pth)
        self._export_to_cpp()
        self.logger.info("Seed models created successfully.")
    
    def _setup_persistent_logger(self, log_dir):
        logger = logging.getLogger("RLOrchestrator")
        logger.setLevel(self.params_config['data_generation']['main_logging_level'])
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()
            
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        log_filepath = os.path.join(log_dir, "orchestrator_py.log")
        
        file_handler = logging.FileHandler(log_filepath, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def _get_lmdb_entry_count(self):
        return self.env.stat()['entries']

    def _save_state(self):
        with open(RL_STATE_FILE, 'w') as f:
            yaml.safe_dump(self.state_config, f)

    def run(self):
        # 1. Threading & Library Optimization
        training_cores = self.params_config['training']['training_cores']
        num_threads = len(training_cores)
        
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
            os.environ[var] = str(num_threads)
            
        torch.set_num_threads(num_threads)
        
        # 2. Briefly unlock all cores so subprocess can inherit full visibility
        proc = psutil.Process()
        all_cores = list(range(psutil.cpu_count()))
        proc.cpu_affinity(all_cores)
        
        self.train_task = TrainTask(
            best_model_path=self.best_model_pth,
            model_config=self.model_config,
            training_config=self.params_config['training'],
            state_config=self.state_config,
            global_config=self.params_config['global'],
            lmdb_path=self.buffer_file_path,
            env=self.env
        )

        engine_exe = os.path.abspath(os.path.join(root_dir, "build", "talbot_engine.exe"))
        self.logger.info(f"Launching C++ Engine: {engine_exe}")
        cmd = [
            engine_exe,
            "--rl_dir", RL_DIR,
            "--state_file", RL_STATE_FILE,
            "--config_file", RL_PARAMS_FILE,
            "--model_file", MODEL_FILE,
            "--db_path", self.buffer_file_path
        ]
        
        # 3. Launch Engine
        engine_process = subprocess.Popen(cmd)

        # 4. Pin Python to training silo and elevate priority
        proc.cpu_affinity(training_cores)
        proc.nice(psutil.HIGH_PRIORITY_CLASS)
            
        self.logger.info(f"Python Orchestrator pinned to cores {training_cores}.")

        rotation_interval = self.params_config['global']['logging_rotation_steps']
        target_folder_step = (self.current_step // rotation_interval) * rotation_interval
        current_log_dir = os.path.join(RL_DIR, f"run_step_{target_folder_step:06d}")

        try:
            while self.current_step < self.total_steps:
                step_start_time = time.time()

                target_folder_step = (self.current_step // rotation_interval) * rotation_interval
                new_log_dir = os.path.join(RL_DIR, f"run_step_{target_folder_step:06d}")

                if new_log_dir != current_log_dir:
                    os.makedirs(new_log_dir, exist_ok=True)
                    current_log_dir = new_log_dir
                    self.logger = self._setup_persistent_logger(current_log_dir)
                    self.logger.info(f"New logging phase started at step {self.current_step}")

                buffer_size = self._get_lmdb_entry_count()
                min_required = self.params_config['training']['batch_size']

                if buffer_size < min_required:
                    self.logger.debug(f"Waiting for data... ({buffer_size}/{min_required} in LMDB)")
                    time.sleep(1)
                    continue

                self.logger.info(f"Starting Step {self.current_step + 1}. Buffer size: {buffer_size}")

                self.train_task.run_single_step(current_log_dir, self.state_config)
                self.current_step += 1
                self.state_config['lifetime']['training_steps'] = self.current_step

                if self.current_step % self.params_config['global']['new_model_interval_steps'] == 0:
                    self.train_task.save_checkpoint(self.best_model_pth)
                    self._export_to_cpp()

                total_time = time.time() - step_start_time
                self.state_config['lifetime']['hours_training'] = round(
                    self.state_config['lifetime'].get('hours_training', 0.0) + (total_time / 3600), 6)

                self.logger.info(
                    f"Step {self.current_step} complete | "
                    f"Buffer: {buffer_size} | "
                    f"Step time: {total_time:.2f}s"
                )

                if self.current_step // self.save_interval > (self.current_step - 1) // self.save_interval:
                    backup_dir = os.path.join(RL_DIR, 'backup')
                    os.makedirs(backup_dir, exist_ok=True)

                    model_backup_path = os.path.join(backup_dir, f'step_{self.current_step:06d}_model.pth')
                    shutil.copy(self.best_model_pth, model_backup_path)

                    shutil.copy(RL_STATE_FILE, os.path.join(current_log_dir, f'step_{self.current_step:06d}_state.yaml'))
                    shutil.copy(RL_PARAMS_FILE, os.path.join(current_log_dir, f'step_{self.current_step:06d}_config.yaml'))

                    self.state_config['current_interval']['samples_generated'] = 0
                    self.state_config['current_interval']['games_played'] = 0
                    self.state_config['current_interval']['self_play_entropy'] = 0.0

                    self.logger.info(f"Periodic backup saved at step {self.current_step}.")

                self._save_state()

        finally:
            self.logger.info("Shutting down C++ engine...")
            engine_process.terminate()
            engine_process.wait()

    def _export_to_cpp(self):
        # 1. Standard loading sequence
        checkpoint = torch.load(self.best_model_pth, map_location='cpu', weights_only=True)
        model = ChessAIModel(self.model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 2. Fuse and cast to FP16
        model = fuse_bn_for_export(model)
        model = model.cuda().half() 

        # 3. Create FP16 dummy input
        dummy_input = torch.zeros(
            1, 
            self.model_config['model']['input_planes'], 
            self.model_config['chess']['board_dim'], 
            self.model_config['chess']['board_dim'], 
            dtype=torch.float16
        ).cuda()

        onnx_path = self.best_model_pth.replace(".pth", ".onnx")
        
        # 4. Use the stable legacy exporter
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['policy', 'value'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'policy': {0: 'batch_size'},
                'value': {0: 'batch_size'}
            }
        )
        self.logger.info(f"Stable FP16 ONNX export successful: {onnx_path}")

if __name__ == "__main__":
    orchestrator = RLOrchestrator()
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\nManual interruption detected. Shutting down...")
    except Exception as e:
        if orchestrator.logger:
            orchestrator.logger.error(f"Orchestrator encountered an error: {e}", exc_info=True)
        else:
            print(f"Orchestrator encountered an error: {e}")