import os
import time
import logging
import numpy as np
import multiprocessing as mp
import os
import psutil
import torch
from datetime import datetime

from self_play_agent import SelfPlayAgent
from data_generation_game_worker import DataGenerationGameWorker
from inference_batcher import InferenceBatcher


class DataGenerationTask:
    """
    Main class to orchestrate a multi-process AlphaZero-style data generation pipeline.
    It manages the creation of inference and worker processes, and handles inter-process
    communication via queues.
    """
    def __init__(self, output_dir, model_config, data_generation_config, best_iter):
        self.output_dir = output_dir
        self.model_config = model_config
        self.data_generation_config = data_generation_config
        self.best_iter = best_iter
        self.num_workers = data_generation_config['game_workers']
        # Number of inference batchers to create
        self.num_inference_batchers = data_generation_config['inference_workers']

        self.main_logger = self._setup_logger(
            "SelfPlayManager", 
            self.data_generation_config['main_logging_level'],
            os.path.join(self.output_dir, f"data_generation_manager.log")
        )

        # Multi-processing components
        # A list of queues, one for each inference batcher
        self.inference_queues = [mp.Queue() for _ in range(self.num_inference_batchers)]
        # A single queue for each worker process
        self.result_queues = [mp.Queue() for _ in range(self.num_workers)]
        self.data_queue = mp.Queue()
        
        self.inference_processes = []
        self.worker_processes = []
        self.game_number_counter = mp.Value('i', 0) 


    @staticmethod
    def _setup_logger(name: str, level: str, log_file: str):
        """Helper to set up a logger for a specific process."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if logger.hasHandlers():
            logger.handlers.clear()
        
        formatter = logging.Formatter("[%(asctime)s][%(name)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger


    @staticmethod
    def _worker_main(worker_id, core_id, output_dir, inference_queues, result_queue, data_queue, data_generation_config, best_iter, game_number_counter):
        """Target function for a single self-play worker process."""

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"

        torch.set_num_threads(1)
        process = psutil.Process()

        process.cpu_affinity([core_id])
        process.nice(psutil.HIGH_PRIORITY_CLASS)

        # Create a new logger specific to this worker process
        worker_logger = DataGenerationTask._setup_logger(
            f"SelfPlayWorker_{worker_id}", 
            data_generation_config['worker_logging_level'],
            os.path.join(output_dir, f"data_generation_worker_{worker_id}.log")
        )

        worker_logger.info(f"Setting core to {core_id}")

        # Distribute workers evenly among inference queues
        inference_queue_for_worker = inference_queues[worker_id % len(inference_queues)]
        
        # Instantiate the MCTS player, configured to use the queues
        mcts_player = SelfPlayAgent(
            name=f'best_model_iter_{best_iter} (self-play)',
            logger=worker_logger,
            self_play_config=data_generation_config,
            worker_id=worker_id,
            inference_queue=inference_queue_for_worker,
            result_queue=result_queue
        )
        
        game_manager = DataGenerationGameWorker(
            logger=worker_logger,
            player_1=mcts_player,
            player_2=mcts_player,
            data_generation_config=data_generation_config
        )
        
        while True:  # Run games indefinitely until the main process terminates us
            with game_number_counter.get_lock():
                current_game_number = game_number_counter.value
                game_number_counter.value += 1
            
            game_start = time.time()
            # The run_training_loop method plays a full game and returns the data
            training_data, simulation_count = game_manager.run_training_loop(current_game_number)
            game_end = time.time()

            game_time = game_end - game_start
            num_new_positions = len(training_data)

            if training_data is None or num_new_positions == 0:
                continue

            simulations_per_second = simulation_count / game_time
            
            worker_logger.critical(
                f"Game {current_game_number} completed in {game_time:.2f}s  "
                f"with {simulations_per_second:.2f} simulations per second "
                f"({num_new_positions} positions). Sending data to main process."
            )
            
            data_queue.put((training_data, game_time))
                

    def run_for_n_positions(self, total_positions: int, chunk_size: int):
        """
        Starts the multi-process pipeline, yields data in chunks,
        and then terminates the processes gracefully.
        """
        inference_cores = {12, 13, 14, 15, 28, 29, 30, 31}
        worker_cores = [i for i in range(32) if i not in inference_cores]


        # Create and start all the inference batcher processes
        for i in range(self.num_inference_batchers):
            batcher = InferenceBatcher(
                f'data_generation_{i}',
                self.model_config['best_model_path'],
                self.model_config,
                self.data_generation_config['batch_size_per_worker'] * self.num_workers,
                self.data_generation_config['batch_timeout'],
                self.output_dir,
                self.data_generation_config['inference_logging_level']
            )
            p = mp.Process(
                target=batcher.run,
                args=(self.inference_queues[i], self.result_queues, set(sorted(inference_cores)[i*4:(i+1)*4])),
                daemon=True
            )
            self.inference_processes.append(p)
            p.start()
            self.main_logger.info(f"Inference batcher process {i} started (PID: {p.pid}).")
            time.sleep(2)

        self.main_logger.info(f"Starting pipeline to generate a chunk of up to {total_positions} positions...")



        # Create and start all the worker processes
        for i in range(self.num_workers):
            p = mp.Process(
                target=DataGenerationTask._worker_main,
                args=(i, worker_cores[i], self.output_dir, self.inference_queues, self.result_queues[i], self.data_queue,
                      self.data_generation_config, self.best_iter, self.game_number_counter),
                daemon=True
            )
            self.worker_processes.append(p)
            p.start()
            self.main_logger.info(f"Worker process {i} started (PID: {p.pid}).")
            time.sleep(2)


        positions_collected_total = 0
        positions_in_current_chunk = 0
        collected_data = []
        games_in_chunk = 0

        try:
            while positions_collected_total < total_positions:
                game_data, game_time = self.data_queue.get()              
                games_in_chunk += 1
                
                collected_data.extend(game_data)
                positions_in_current_chunk += len(game_data)
                
                self.main_logger.info(
                    f"Collected a new game with {len(game_data)} positions in {game_time:.4f} seconds. "
                    f"Positions in current chunk: {positions_in_current_chunk}/{chunk_size} positions. "
                )

                if positions_in_current_chunk >= chunk_size:

                    self.main_logger.info(f"Yielding a chunk of {len(collected_data)} positions.")
                    yield collected_data, games_in_chunk
                    
                    positions_collected_total += len(collected_data)
                    positions_in_current_chunk = 0
                    games_in_chunk = 0
                    collected_data = []

                    self.main_logger.info(f"Have processed {positions_collected_total} out of {total_positions} positions.")


            if collected_data:
                self.main_logger.info(f"Yielding final partial chunk of {len(collected_data)} positions.")
                yield collected_data, games_in_chunk
            
            self.main_logger.info(f"Finished collecting {positions_collected_total} positions. Terminating processes...")

        finally:
            for p in self.inference_processes:
                p.terminate()
            for p in self.worker_processes:
                p.terminate()

            for p in self.inference_processes:
                p.join()
            for p in self.worker_processes:
                p.join()
                
            self.main_logger.info("All processes terminated. Data generation task complete.")