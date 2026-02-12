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
    def __init__(self, name, model_path, model_config, batch_size, batch_timeout, log_dir, logging_level):
        self.name = name
        self.model_path = model_path
        self.model_config = model_config
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.log_dir = log_dir
        self.logging_level = logging_level
        self.model = None
        self.logger = None

        # These attributes will be set in the run method
        self.inference_queue = None
        self.result_queues = None
        self.shared_input_buffer = None
        self.shared_value_buffer = None
        self.shared_policy_buffer = None

        # Set the number of threads for internal PyTorch CPU operations.
        torch.set_num_threads(1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device.type == 'cuda'


    def _setup_logger(self):
        """Sets up the logger for this specific process."""
        log_file = os.path.join(self.log_dir, f"inference_batcher_{self.name}.log")

        logger = logging.getLogger("InferenceBatcher")
        logger.setLevel(self.logging_level)
        if not logger.handlers:
            formatter = logging.Formatter("[%(asctime)s][%(name)s] [%(levelname)s] %(message)s")
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        self.logger = logger
        self.logger.info("Inference batcher logger initialized.")


    def load_model(self):
        """Load the model. This should be called inside the process."""
        self.model = ChessAIModel(num_input_planes=self.model_config['input_planes'], 
                                  num_residual_blocks=self.model_config['resblocks'], 
                                  num_filters=self.model_config['filters'])
        
        checkpoint = torch.load(
            self.model_path, 
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Model loaded successfully from {self.model_path}")

        self.model.to(self.device)
        # Convert model to half-precision if CUDA is available
        if self.use_fp16:
            self.model.half()
            self.logger.info("Model converted to FP16 (half-precision).")
        else:
            self.logger.info("Running model in FP32 (full-precision) on CPU or non-FP16 compatible GPU.")

        self.model.eval()


    @staticmethod
    def _apply_process_settings(core_id):
        """Applies CPU affinity and thread limits to the current process."""
        import os
        import psutil
        import torch

        # Limit threads for various libraries
        os.environ["OMP_NUM_THREADS"] = str(len(core_id))
        os.environ["MKL_NUM_THREADS"] = str(len(core_id))
        os.environ["OPENBLAS_NUM_THREADS"] = str(len(core_id))
        os.environ["NUMEXPR_NUM_THREADS"] = str(len(core_id))
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(len(core_id))
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(len(core_id))
        os.environ["TF_NUM_INTEROP_THREADS"] = str(len(core_id))
        torch.set_num_threads(1)
        
        # Pin to one CPU core
        psutil.Process().cpu_affinity(core_id)


    def run(self, inference_queue, result_queues, core_id, shared_input_buffer, shared_policy_buffer, shared_value_buffer):
        """
        The main entry point for the batcher process.
        It sets up the process environment and then starts the main loop.
        """
        # Store queues and shared buffers as instance attributes
        self.inference_queue = inference_queue
        self.result_queues = result_queues
        self.shared_input_buffer = shared_input_buffer
        self.shared_policy_buffer = shared_policy_buffer
        self.shared_value_buffer = shared_value_buffer

        self._apply_process_settings(core_id)
        self._run_loop()


    def _run_loop(self):
        """The main loop for the batcher process."""
        self._setup_logger()
        self.load_model()
        
        stream = None
        if self.device.type == 'cuda':
            stream = torch.cuda.Stream()
        
        requests = []
                
        # --- Performance Tracking Variables ---
        last_report_time = time.monotonic()
        
        # Cumulative stats for the current logging interval
        interval_batches_processed = 0
        interval_total_processing_duration = 0.0
        interval_total_inferences = 0
        
        while True:
            # Collect requests from the inference queue.
            while len(requests) < self.batch_size:
                try:
                    # new_requests_batch is a list of (worker_id, buffer_index) tuples
                    new_requests_batch = self.inference_queue.get(timeout=self.batch_timeout)
                    requests.extend(new_requests_batch)
                    self.logger.debug(f"The current length of the queue is: {self.inference_queue.qsize()}")
                except queue.Empty:
                    break

            # 3. Process the batch if it's full OR if the timeout has elapsed AND there are requests
            if requests:
                batch_process_start_time = time.monotonic()
                self.logger.debug(f"Processing a batch of size {len(requests)}...")

                # A. Data Preparation (CPU-bound)
                start_data_prep = time.monotonic()
                
                # Extract worker_ids, the shared buffer indices (which were req[1]), and input tensors
                worker_ids = [req[0] for req in requests]
                slot_indices = [req[1] for req in requests] # Now correctly extracts the shared buffer index
                states_to_process = torch.stack([
                    self.shared_input_buffer[idx] for idx in slot_indices
                ])
                
                data_prep_duration = time.monotonic() - start_data_prep
                self.logger.debug(f"Time for data preparation: {data_prep_duration:.4f} seconds.")

                # B. GPU Transfer and Inference
                with torch.cuda.stream(stream):
                    start_gpu_transfer = time.monotonic()
                    # Convert input tensor to FP16 if enabled, otherwise keep as FP32
                    states_gpu = states_to_process.to(self.device, non_blocking=True)
                    if self.use_fp16 and states_gpu.dtype != torch.float16:
                            states_gpu = states_gpu.half()
                    
                    gpu_transfer_duration = time.monotonic() - start_gpu_transfer
                    self.logger.debug(f"Time to initiate GPU transfer: {gpu_transfer_duration:.4f} seconds.")

                    with torch.no_grad():
                        start_inference = time.monotonic()
                        policy_gpu, value_gpu = self.model(states_gpu)
                        inference_duration = time.monotonic() - start_inference
                        self.logger.debug(f"Time to initiate inference: {inference_duration:.4f} seconds.")

                    start_cpu_transfer = time.monotonic()
                    # Move results back to CPU. Use non_blocking=False or synchronize immediately after for safety
                    # when writing to shared memory. Using non_blocking=False here ensures the data is ready.
                    policy_cpu = policy_gpu.to('cpu', non_blocking=False)
                    value_cpu = value_gpu.to('cpu', non_blocking=False)
                    cpu_transfer_duration = time.monotonic() - start_cpu_transfer
                    self.logger.debug(f"Time to ensure CPU transfer sync: {cpu_transfer_duration:.4f} seconds.")

                    # Synchronize the stream to ensure all asynchronous operations are complete
                    # This is mostly redundant if non_blocking=False was used above, but good practice if not.
                    stream.synchronize()

                # C. WRITE DIRECTLY TO SHARED MEMORY AND NOTIFY WORKER 🚀
                start_write_and_notify = time.monotonic()
                for i, slot_index in enumerate(slot_indices):
                    # 1. Write policy results to the shared buffer
                    self.shared_policy_buffer[slot_index].copy_(policy_cpu[i])
                    
                    # 2. Write value results to the shared buffer
                    self.shared_value_buffer[slot_index].copy_(value_cpu[i])

                    # 3. Notify the worker that the result is ready by sending back the index
                    worker_id = worker_ids[i]
                    self.result_queues[worker_id].put_nowait(slot_index) 

                write_and_notify_duration = time.monotonic() - start_write_and_notify
                self.logger.debug(f"Time for shared memory write and notify: {write_and_notify_duration:.4f} seconds.")

                batch_total_duration = time.monotonic() - batch_process_start_time
                self.logger.debug(f"Total batch processing loop time: {batch_total_duration:.4f} seconds. (Batch size: {len(requests)})")
                
                # Update interval statistics
                interval_batches_processed += 1
                interval_total_processing_duration += batch_total_duration
                interval_total_inferences += len(requests)
                
                # Reset requests list and batch collection start time
                requests = []

            # --- Periodic Performance Logging ---
            current_monotonic_time = time.monotonic()
            if current_monotonic_time - last_report_time >= 60.0: # Log every 60 seconds
                elapsed_interval_time = current_monotonic_time - last_report_time

                if interval_batches_processed > 0:
                    avg_batch_process_time = interval_total_processing_duration / interval_batches_processed
                    
                    # Total time the batcher was "active" (processing batches)
                    active_processing_percentage = (interval_total_processing_duration / elapsed_interval_time) * 100
                    
                    # Total time the batcher was "idle" (waiting for requests or next batch)
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

                # Reset counters for the next interval
                last_report_time = current_monotonic_time
                interval_batches_processed = 0
                interval_total_processing_duration = 0.0
                interval_total_inferences = 0