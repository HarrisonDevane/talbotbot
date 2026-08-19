import os
import time
import yaml
import logging
import subprocess
import shutil
import random
import psutil
import torch
import struct
import lmdb
import warnings
import json
import onnx

from trainer import TrainTask
from model import ChessAIModel, fuse_bn_for_export

current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_script_dir, ".."))

RL_PARAMS_FILE = os.path.abspath(os.path.join(root_dir, "config", "train.yaml"))
MODEL_FILE = os.path.abspath(os.path.join(root_dir, "config", "model.yaml"))

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch.onnx.*")

# Format matching the C++ structs
CPP_STATE_FMT = 'QQdQQQ'  
PY_STATE_FMT = 'Qd'       

class RLOrchestrator:
    def __init__(self):
        self.logger = None
        self.state_logger = None

        with open(RL_PARAMS_FILE, 'r') as f:
            self.params_config = yaml.safe_load(f)
        with open(MODEL_FILE, 'r') as f:
            self.model_config = yaml.safe_load(f)

        # Resolve train_dir. Relative paths are anchored at project root so
        # config files remain portable; absolute paths (e.g. a scratch disk)
        # are respected as-is.
        train_dir_cfg = self.params_config['global']['train_dir']
        if os.path.isabs(train_dir_cfg):
            self.train_dir = train_dir_cfg
        else:
            self.train_dir = os.path.abspath(os.path.join(root_dir, train_dir_cfg))
        os.makedirs(os.path.join(self.train_dir, "models"), exist_ok=True)

        self.buffer_file_path = os.path.join(self.train_dir, "replay_memory.lmdb")
        
        db_is_new = not os.path.exists(self.buffer_file_path)
        
        if db_is_new:
            os.makedirs(self.buffer_file_path, exist_ok=True)

        self.env = lmdb.open(
            self.buffer_file_path,
            map_size=1024 * 1024 * 1024 * self.params_config['global']['buffer_size_gb'], 
            readonly=False,
            lock=True,
            readahead=False,
        )

        if db_is_new:
            self._initialize_empty_lmdb()

        self.state_config = self._read_state_from_lmdb()

        self.current_step = self.state_config['lifetime']['training_steps']
        self.total_hours_accumulator = self.state_config['lifetime']['hours_training'] 
        
        self.total_steps = self.params_config['global']['total_training_steps']

        rotation_interval = self.params_config['global']['logging_rotation_steps']
        target_folder_step = (self.current_step // rotation_interval) * rotation_interval
        initial_log_dir = os.path.join(self.train_dir, f"run_step_{target_folder_step:06d}")

        os.makedirs(initial_log_dir, exist_ok=True)
        
        self.logger = self._setup_logger(
            log_dir=initial_log_dir, 
            name="RLOrchestrator", 
            filename="orchestrator_py.log", 
            level=self.params_config['data_generation']['main_logging_level'], 
            fmt="[%(asctime)s] [%(levelname)s] %(message)s"
        )
        
        self.state_logger = self._setup_logger(
            log_dir=initial_log_dir, 
            name="StateLogger", 
            filename="state_metrics.log", 
            level=logging.INFO, 
            fmt="[%(asctime)s] %(message)s"
        )

        self.model_pth = os.path.join(self.train_dir, "models", "model.pth")
        self.train_task = None
        self.next_build_step = self._calculate_next_build_step(self.current_step)
        
        if self._get_lmdb_signal(b"__TRT_EXPORT_SIGNAL") is None:
            if not os.path.exists(self.model_pth):
                self._create_seed_models()

        self._export_to_cpp()

    def _initialize_empty_lmdb(self):
        cpp_blob = struct.pack(CPP_STATE_FMT, 0, 0, 0.0, 0, 0, 0)
        py_blob = struct.pack(PY_STATE_FMT, 0, 0.0)
        
        with self.env.begin(write=True) as txn:
            txn.put(b"__CPP_STATE", cpp_blob)
            txn.put(b"__PY_STATE", py_blob)

    def _get_lmdb_signal(self, key: bytes):
        with self.env.begin(write=False) as txn:
            blob = txn.get(key)
            if blob:
                return struct.unpack('Q', blob)[0]
            return None
        
    def _calculate_next_build_step(self, current_target_step):
        build_steps = self.params_config['global']['build_steps']
        return ((current_target_step // build_steps) + 1) * build_steps
                
    def _wait_for_trt_engine(self):
        target_step = self._get_lmdb_signal(b"__TRT_EXPORT_SIGNAL")

        ready_step = self._get_lmdb_signal(b"__TRT_ENGINE_READY")
        
        if ready_step is None or ready_step < target_step:
            self.logger.info(f"Waiting for C++ to build TRT Engine for step {target_step}...")

            while True:
                ready_step = self._get_lmdb_signal(b"__TRT_ENGINE_READY")
                if ready_step is not None and ready_step >= target_step:
                    break
                time.sleep(1.0)
            
            self.logger.info(f"TRT Engine for step {target_step} is ready.")

    def _read_state_from_lmdb(self):
        with self.env.begin(write=False) as txn:
            cpp_blob = txn.get(b"__CPP_STATE")
            py_blob = txn.get(b"__PY_STATE")

        if not cpp_blob or not py_blob:
            return {
                'buffer': {'count': 0, 'head_ptr': 0, 'wraps': 0},
                'lifetime': {
                    'training_steps': 0, 
                    'games_played': 0, 
                    'samples_generated': 0, 
                    'hours_training': 0.0, 
                    'self_play_entropy': 0.0
                }
            }

        cpp_state = struct.unpack(CPP_STATE_FMT, cpp_blob)
        py_state = struct.unpack(PY_STATE_FMT, py_blob)

        return {
            'buffer': {
                'count': cpp_state[3],
                'head_ptr': cpp_state[4],
                'wraps': cpp_state[5]
            },
            'lifetime': {
                'training_steps': py_state[0],
                'games_played': cpp_state[0],
                'samples_generated': cpp_state[1],
                'hours_training': py_state[1],
                'self_play_entropy': cpp_state[2]
            }
        }

    def _write_py_state_to_lmdb(self):
        blob = struct.pack(
            PY_STATE_FMT, 
            self.state_config['lifetime']['training_steps'], 
            self.state_config['lifetime']['hours_training']
        )
        with self.env.begin(write=True) as txn:
            txn.put(b"__PY_STATE", blob)

    def _create_seed_models(self):
        random.seed(self.params_config['training']['seed'])
        torch.manual_seed(self.params_config['training']['seed'])
        model = ChessAIModel(self.model_config)
        
        # Ensure base models directory exists
        os.makedirs(os.path.join(self.train_dir, 'models'), exist_ok=True)
        
        # 1. Save the active model for the C++ engine
        save_dict = {'model_state_dict': model.state_dict()}
        torch.save(save_dict, self.model_pth)
        
        # 2. Save the permanent seed snapshot to the correct subfolder
        backup_dir = os.path.join(self.train_dir, 'models', 'checkpoints')
        os.makedirs(backup_dir, exist_ok=True)
        
        seed_backup_path = os.path.join(backup_dir, 'step_000000_model.pth')
        torch.save(save_dict, seed_backup_path)
        
        self._export_to_cpp()
    
    def _setup_logger(self, log_dir, name, filename, level, fmt):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter(fmt)
        log_filepath = os.path.join(log_dir, filename)
        file_handler = logging.FileHandler(log_filepath, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def run(self):
        training_cores = self.params_config['training']['training_cores']
        num_threads = len(training_cores)
        
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
            os.environ[var] = str(num_threads)
            
        torch.set_num_threads(num_threads)
        
        proc = psutil.Process()
        all_cores = list(range(psutil.cpu_count()))
        proc.cpu_affinity(all_cores)

        engine_exe = os.path.abspath(os.path.join(root_dir, "build", "Release", "talbot_engine.exe"))
        self.logger.info(f"Launching C++ Engine: {engine_exe}")
        cmd = [
            engine_exe,
            "--train_dir", self.train_dir,
            "--config_file", RL_PARAMS_FILE,
            "--model_file", MODEL_FILE,
            "--db_path", self.buffer_file_path
        ]
        
        engine_process = subprocess.Popen(cmd)

        self._wait_for_trt_engine()

        self.logger.info("Initializing PyTorch TrainTask...")
        self.train_task = TrainTask(
            model_path=self.model_pth,
            model_config=self.model_config,
            training_config=self.params_config['training'],
            state_config=self.state_config,
            global_config=self.params_config['global'],
            db_path=self.buffer_file_path 
        )

        proc.cpu_affinity(training_cores)
        proc.nice(psutil.HIGH_PRIORITY_CLASS)
        self.logger.info(f"Python Orchestrator pinned to cores {training_cores}.")

        rotation_interval = self.params_config['global']['logging_rotation_steps']
        target_folder_step = (self.current_step // rotation_interval) * rotation_interval
        current_log_dir = os.path.join(self.train_dir, f"run_step_{target_folder_step:06d}")

        last_time_check = time.time()

        try:
            while self.current_step < self.total_steps:
                
                current_time = time.time()
                self.total_hours_accumulator += (current_time - last_time_check) / 3600.0
                last_time_check = current_time

                self.state_config = self._read_state_from_lmdb()
                self.state_config['lifetime']['hours_training'] = round(self.total_hours_accumulator, 6)

                buffer_size = self.state_config['buffer']['count']
                samples_generated = self.state_config['lifetime']['samples_generated']
                
                samples_per_step = self.params_config['training']['batch_size'] / self.params_config['training']['sampling_ratio']
                if samples_per_step <= 0:
                    samples_per_step = 1 
                    
                target_steps = int(samples_generated // samples_per_step)

                if self.current_step >= target_steps:
                    self._write_py_state_to_lmdb()
                    time.sleep(0.5)
                    continue

                min_required = self.params_config['training']['batch_size']
                if buffer_size < min_required:
                    self._write_py_state_to_lmdb()
                    time.sleep(0.5)
                    continue

                step_start_time = time.time()

                target_folder_step = (self.current_step // rotation_interval) * rotation_interval
                new_log_dir = os.path.join(self.train_dir, f"run_step_{target_folder_step:06d}")

                if new_log_dir != current_log_dir:
                    os.makedirs(new_log_dir, exist_ok=True)
                    current_log_dir = new_log_dir
                    
                    self.logger = self._setup_logger(
                        log_dir=current_log_dir, 
                        name="RLOrchestrator", 
                        filename="orchestrator_py.log", 
                        level=self.params_config['data_generation']['main_logging_level'], 
                        fmt="[%(asctime)s] [%(levelname)s] %(message)s"
                    )
                    
                    self.state_logger = self._setup_logger(
                        log_dir=current_log_dir, 
                        name="StateLogger", 
                        filename="state_metrics.log", 
                        level=logging.INFO, 
                        fmt="[%(asctime)s] %(message)s"
                    )
                    
                    self.logger.info(f"New logging phase started at step {self.current_step}")

                self.train_task.run_single_step(current_log_dir, self.state_config)
                self.current_step += 1
                self.state_config['lifetime']['training_steps'] = self.current_step

                if self.current_step >= self.next_build_step:
                    self.train_task.save_checkpoint(self.model_pth)

                    # Free cached activations so the C++ builder has headroom, but
                    # keep model/optimizer/prefetcher resident (warm resume).
                    self.train_task.pause_for_build()

                    self._export_to_cpp()
                    self._wait_for_trt_engine()

                    # Store backup and state accurately corresponding to the build step
                    backup_dir = os.path.join(self.train_dir, 'models', 'checkpoints')
                    os.makedirs(backup_dir, exist_ok=True)

                    model_backup_path = os.path.join(backup_dir, f'step_{self.current_step:06d}_model.pth')
                    shutil.copy(self.model_pth, model_backup_path)
                    shutil.copy(RL_PARAMS_FILE, os.path.join(current_log_dir, f'step_{self.current_step:06d}_config.yaml'))
                    self.logger.info(f"Periodic backup saved at step {self.current_step}.")

                    self.train_task.resume_after_build()
                    self.logger.info("Resumed PyTorch TrainTask after build (warm, no rebuild).")

                    self.next_build_step = self._calculate_next_build_step(self.current_step)

                total_time = time.time() - step_start_time

                self.logger.debug(
                    f"Step {self.current_step} complete | "
                    f"Buffer: {buffer_size} | "
                    f"Step time: {total_time:.2f}s"
                )

                self._write_py_state_to_lmdb()
                self.state_logger.info(json.dumps(self.state_config))

        finally:
            self.logger.info("Shutting down C++ engine...")
            engine_process.terminate()
            engine_process.wait()
            

    def _export_to_cpp(self):
        checkpoint = torch.load(self.model_pth, map_location='cpu', weights_only=True)
        model = ChessAIModel(self.model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        model = fuse_bn_for_export(model)
        model = model.cuda().half() 

        dummy_input = torch.zeros(
            1, 
            self.model_config['model']['input_planes'], 
            self.model_config['model']['board_dim'], 
            self.model_config['model']['board_dim'], 
            dtype=torch.float16
        ).cuda()

        onnx_path = self.model_pth.replace(".pth", ".onnx")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            torch.onnx.export(
                model, 
                dummy_input, 
                onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['policy', 'value'],
                dynamic_axes={'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}},
                dynamo=False
            )
        
        self.logger.info(f"Stable FP16 ONNX export successful: {onnx_path}")

        del model
        del dummy_input
        torch.cuda.empty_cache()

        with self.env.begin(write=True) as txn:
            current = txn.get(b"__TRT_EXPORT_SIGNAL")
            prev = struct.unpack('Q', current)[0] if current else 0
            txn.put(b"__TRT_EXPORT_SIGNAL", struct.pack('Q', prev + 1))

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