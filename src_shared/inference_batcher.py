import os
import time
import logging
import queue
import torch
import torch.nn.functional as F

from src_shared.model import ChessAIModel

class InferenceBatcher:
    """
    A dedicated class to handle batched model inference in a separate process,
    writing results directly to shared memory buffers.
    """
    def __init__(self, name, model_path, model_config, batch_size, batch_timeout, log_dir, logging_level, training):
        self.name = name
        self.model_path = model_path
        self.model_config = model_config
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.log_dir = log_dir
        self.logging_level = logging_level
        self.model = None
        self.logger = None
        self.training = training

        self.inference_queue = None
        self.result_queues = None
        self.shared_input_buffer = None
        self.shared_value_buffer = None
        self.shared_policy_buffer = None
        self.stop_event = None
        self.local_model_step = None

        torch.set_num_threads(1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device.type == 'cuda'


    def _setup_logger(self):
        log_file = os.path.join(self.log_dir, f"inference_batcher_{self.name}.log")

        logger = logging.getLogger(f"InferenceBatcher_{self.name}") 
        logger.setLevel(self.logging_level)
        
        if logger.hasHandlers():
            logger.handlers.clear()
            
        formatter = logging.Formatter("[%(asctime)s][%(name)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.logger = logger
        return logger
    

    def _load_model(self):
        """Load the model and compile it via TorchScript for Windows compatibility."""
        checkpoint = torch.load(
            self.model_path, 
            map_location=self.device,
        )

        if self.model is None:
            raw_model = ChessAIModel(
                num_input_planes=self.model_config['input_planes'],
                num_residual_blocks=self.model_config['resblocks'],
                num_filters=self.model_config['filters'],
                bottleneck_channels=self.model_config['bottleneck_channels'],
                broadcast_reduction_ratio=self.model_config['broadcast_reduction_ratio'],
                broadcast_interval=self.model_config['broadcast_interval']
            )
            
            raw_model.to(self.device)
            if self.use_fp16:
                raw_model.half()
                self.logger.debug("Model converted to FP16 (half-precision).")
            
            raw_model.eval()

            # WINDOWS FIX: Use TorchScript instead of Triton/torch.compile
            self.model = torch.jit.script(raw_model)
            self.logger.debug("Model compiled via torch.jit.script.")

        # In-place weight update bypasses recompilation overhead
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.debug(f"Model weights loaded successfully from {self.model_path}")


    @staticmethod
    def _apply_process_settings(core_id):
        import os
        import psutil
        import torch

        os.environ["OMP_NUM_THREADS"] = str(len(core_id))
        os.environ["MKL_NUM_THREADS"] = str(len(core_id))
        os.environ["OPENBLAS_NUM_THREADS"] = str(len(core_id))
        os.environ["NUMEXPR_NUM_THREADS"] = str(len(core_id))
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(len(core_id))
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(len(core_id))
        os.environ["TF_NUM_INTEROP_THREADS"] = str(len(core_id))
        torch.set_num_threads(1)
        
        psutil.Process().cpu_affinity(core_id)


    def run(self, output_dir, inference_queue, result_queues, core_id, shared_input_buffer, shared_policy_buffer, shared_value_buffer, stop_event, current_steps, rotation_interval):
        self.output_dir = output_dir
        self.inference_queue = inference_queue
        self.result_queues = result_queues
        self.shared_input_buffer = shared_input_buffer
        self.shared_policy_buffer = shared_policy_buffer
        self.shared_value_buffer = shared_value_buffer
        self.stop_event = stop_event
        self.current_steps = current_steps
        self.rotation_interval = rotation_interval
        self.num_workers = len(result_queues)

        self._apply_process_settings(core_id)
        self._run_loop()

    def _run_loop(self):
        """The main loop for the batcher process."""
        
        # 0. CRITICAL: Prevent cuDNN from thrashing on dynamic batch sizes
        torch.backends.cudnn.benchmark = False
        
        if self.training:
            target_folder_step = (self.current_steps.value // self.rotation_interval) * self.rotation_interval
            self.log_dir = os.path.join(self.output_dir, f"run_step_{target_folder_step:06d}")
            os.makedirs(self.log_dir, exist_ok=True)
            self._setup_logger()
            self.local_model_step = self.current_steps.value
        else:
            self.log_dir = os.path.join(self.output_dir, "inference")
            os.makedirs(self.log_dir, exist_ok=True)
            self._setup_logger()

        self._load_model()
        
        stream = None
        if self.device.type == 'cuda':
            stream = torch.cuda.Stream()
        
        requests = []
                
        last_report_time = time.monotonic()
        
        interval_batches_processed = 0
        interval_total_processing_duration = 0.0
        interval_total_inferences = 0
        
        # Cache to avoid repeated lookups in the tight loop
        num_workers = self.num_workers 
        
        while not self.stop_event.is_set():
            if self.training:
                global_step = self.current_steps.value
                
                if self.local_model_step < global_step:
                    self.logger.info(f"Syncing model at step {global_step}...")
                    self._load_model() 
                    self.local_model_step = global_step

            # INSTANT QUEUE DRAINING
            try:
                requests.extend(self.inference_queue.get(timeout=self.batch_timeout))
                while len(requests) < self.batch_size:
                    try:
                        requests.extend(self.inference_queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            if requests:
                batch_process_start_time = time.monotonic()
                
                # 1. Fast C-level Unpacking
                worker_ids, slot_indices = zip(*requests)
                slot_indices_list = list(slot_indices) 
                
                # Slice shared memory (Already FP16 from the workers)
                states_to_process = self.shared_input_buffer[slot_indices_list]
                
                # 2. Fire the GPU asynchronously (No redundant dtype checks)
                with torch.cuda.stream(stream):          
                    states_gpu = states_to_process.to(self.device, non_blocking=True)
                    with torch.no_grad():
                        policy_gpu, value_gpu = self.model(states_gpu)

                # 3. --- CPU DOES WORK WHILE GPU COMPUTES ---
                # O(1) Array Routing (replaces the slow dictionary)
                worker_notifications = [[] for _ in range(num_workers)]
                for w_id, s_idx in requests:
                    worker_notifications[w_id].append(s_idx)

                # 4. Now sync the CPU, ensuring GPU math is done
                if stream is not None:
                    stream.synchronize()
                
                # 5. SAFE VECTORIZED SCATTER 
                # Transfer to CPU/dtype, then use direct assignment to trigger in-place scatter
                policy_cpu = policy_gpu.to(device='cpu', dtype=self.shared_policy_buffer.dtype)
                value_cpu = value_gpu.to(device='cpu', dtype=self.shared_value_buffer.dtype)

                self.shared_policy_buffer[slot_indices_list] = policy_cpu
                self.shared_value_buffer[slot_indices_list] = value_cpu

                # 6. BATCHED IPC NOTIFICATION
                for w_id in range(num_workers):
                    indices_list = worker_notifications[w_id]
                    if indices_list:
                        self.result_queues[w_id].put_nowait(indices_list)

                batch_total_duration = time.monotonic() - batch_process_start_time
                
                interval_batches_processed += 1
                interval_total_processing_duration += batch_total_duration
                interval_total_inferences += len(requests)
                
                requests = []

            # --- Periodic Performance Logging ---
            current_monotonic_time = time.monotonic()
            if current_monotonic_time - last_report_time >= 60.0: 
                elapsed_interval_time = current_monotonic_time - last_report_time

                if interval_batches_processed > 0:
                    avg_batch_process_time = interval_total_processing_duration / interval_batches_processed
                    active_processing_percentage = (interval_total_processing_duration / elapsed_interval_time) * 100
                    total_idle_in_interval = elapsed_interval_time - interval_total_processing_duration
                    inferences_per_second_overall = interval_total_inferences / elapsed_interval_time

                    self.logger.info(f"\n--- Inference Batcher Performance Report ({elapsed_interval_time:.2f}s interval) ---")
                    self.logger.info(f"Batches processed: {interval_batches_processed}")
                    self.logger.info(f"Total inferences: {interval_total_inferences}")
                    self.logger.info(f"Avg batch processing time: {avg_batch_process_time:.4f}s")
                    self.logger.info(f"Total active processing time: {interval_total_processing_duration:.4f}s")
                    self.logger.info(f"Total idle time (waiting for requests/batches): {total_idle_in_interval:.4f}s")
                    self.logger.info(f"Batcher active utilization: {active_processing_percentage:.2f}%")
                    self.logger.info(f"Overall Inferences/Second: {inferences_per_second_overall:.2f}")
                    self.logger.info("-----------------------------------\n")

                else:
                    self.logger.info(f"\n--- Inference Batcher Performance Report ({elapsed_interval_time:.2f}s interval) ---")
                    self.logger.info("No batches processed in this interval.")
                    self.logger.info("-----------------------------------\n")

                last_report_time = current_monotonic_time
                interval_batches_processed = 0
                interval_total_processing_duration = 0.0
                interval_total_inferences = 0
                
                if self.training:
                    target_folder_step = (self.current_steps.value // self.rotation_interval) * self.rotation_interval
                    new_log_dir = os.path.join(self.output_dir, f"run_step_{target_folder_step:06d}")

                    if new_log_dir != self.log_dir:
                        os.makedirs(new_log_dir, exist_ok=True)
                        self.log_dir = new_log_dir
                        self._setup_logger()
                        self.logger.info(f"Inference Batcher rotated to new log directory: {new_log_dir}")