import os
import time
import logging
import numpy as np
import multiprocessing as mp
import psutil
import torch
import queue
from datetime import datetime

# Assuming these imports are in your project structure
from self_play_agent import SelfPlayAgent
from evaluation_game_worker import EvaluationGameWorker
from inference_batcher import InferenceBatcher # New import for InferenceBatcher


class EvaluationTask:
    def __init__(self, output_dir, test_model, model_config, evaluation_config, current_iter, best_iter):
        self.output_dir = output_dir
        self.test_model_path = test_model # Renamed for clarity, holds path to test model
        self.model_config = model_config
        self.evaluation_config = evaluation_config
        self.current_iter = current_iter
        self.best_iter = best_iter
        
        # Extract best model path from model_config for clarity
        self.best_model_path = model_config['best_model_path']

        self.num_evaluation_workers = self.evaluation_config['workers']

        # Set up loggers
        self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        self.main_logger = self._setup_logger(
            "EvaluationManager", 
            self.evaluation_config['main_logging_level'],
            os.path.join(self.log_dir, f"evaluation_manager.log")
        )
        
        # Multiprocessing components for two models (test and best)
        # Queues for test model inferences
        self.test_inference_queue = mp.Queue()
        self.test_result_queues = [mp.Queue() for _ in range(self.num_evaluation_workers)]
        
        # Queues for best model inferences
        self.best_inference_queue = mp.Queue()
        self.best_result_queues = [mp.Queue() for _ in range(self.num_evaluation_workers)]
        
        # Queue for sending game assignments to workers
        self.game_job_queue = mp.Queue() 
        # Queue for receiving game results from workers
        self.game_result_queue = mp.Queue() 
        
        # Shared counters for wins/draws (using multiprocessing.Value for atomic updates)
        self.test_model_wins = mp.Value('i', 0)
        self.best_model_wins = mp.Value('i', 0)
        self.draws = mp.Value('i', 0)

        self.worker_processes = []
        self.test_inference_process = None
        self.best_inference_process = None


    @staticmethod
    def _setup_logger(name: str, level: str, log_file: str):
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
    def _eval_worker_main(
        worker_id, output_dir, evaluation_config,
        current_iter, best_iter,
        game_job_queue, game_result_queue,
        test_inference_queue, test_result_queue_for_this_worker,
        best_inference_queue, best_result_queue_for_this_worker,
        test_model_wins_shared, best_model_wins_shared, draws_shared
    ):
        """Target function for a single evaluation worker process."""
        
        # Set CPU affinity and thread limits for performance isolation
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        torch.set_num_threads(1)
        
        process = psutil.Process()
        process.cpu_affinity([worker_id])

        process.nice(psutil.HIGH_PRIORITY_CLASS)


        # Create a new logger specific to this worker process
        log_dir = os.path.join(output_dir, "logs")
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        worker_logger = EvaluationTask._setup_logger(
            f"EvalWorker_{worker_id}", 
            evaluation_config['worker_logging_level'],
            os.path.join(log_dir, f"evaluation_worker_{worker_id}.log")
        )

        # Instantiate players configured to use the inference queues
        mcts_player_best = SelfPlayAgent(
            name=f'best_model_iter_{best_iter}',
            logger=worker_logger,
            self_play_config=evaluation_config,
            worker_id=worker_id,
            inference_queue=best_inference_queue,
            result_queue=best_result_queue_for_this_worker
        )

        mcts_player_test = SelfPlayAgent(
            name=f'test_model_iter_{current_iter}',
            logger=worker_logger,
            self_play_config=evaluation_config,
            worker_id=worker_id,
            inference_queue=test_inference_queue,
            result_queue=test_result_queue_for_this_worker
        )

        
        game_manager = EvaluationGameWorker(
            logger=worker_logger,
            player_1=mcts_player_best,
            player_2=mcts_player_test,
            evaluation_config=evaluation_config
        )

        # Main loop to receive game jobs and play them
        while True:
            try:
                # Get a game job: (game_idx, test_is_white_flag)
                game_idx, test_is_white = game_job_queue.get(timeout=1.0) 
                
                # Signal to terminate workers
                if game_idx is None:
                    worker_logger.info(f"Eval Worker {worker_id}: Received termination signal.")
                    break

                # Determine which player is White/Black for this game
                if test_is_white:
                    game_manager.player_1 = mcts_player_test
                    game_manager.player_2 = mcts_player_best
                    player_roles_msg = "Test (W) vs Best (B)"
                else:
                    game_manager.player_1 = mcts_player_best
                    game_manager.player_2 = mcts_player_test
                    player_roles_msg = "Best (W) vs Test (B)"

                start_time_game = time.time()
                game_result, game_length = game_manager.run_eval_loop(game_idx)
                end_time_game = time.time()
                game_duration = end_time_game - start_time_game
                
                # Update shared counters (atomically) and send result to main process
                with test_model_wins_shared.get_lock():
                    with best_model_wins_shared.get_lock():
                        with draws_shared.get_lock():
                            if game_result == '1-0': # White wins
                                if test_is_white: # Test model was white
                                    test_model_wins_shared.value += 1
                                else: # Best model was white
                                    best_model_wins_shared.value += 1
                            elif game_result == '0-1': # Black wins
                                if test_is_white: # Test model was white (lost)
                                    best_model_wins_shared.value += 1
                                else: # Best model was white (lost)
                                    test_model_wins_shared.value += 1
                            else: # Draw
                                draws_shared.value += 1
                
                worker_logger.info(
                    f"Game {game_idx} ({player_roles_msg}) completed in {game_duration:.2f}s "
                    f"with length {game_length} moves. Result: {game_result}. "
                )
                
                # Send summary data back to the main process for overall logging/progress
                game_result_queue.put({
                    'game_idx': game_idx,
                    'result': game_result,
                    'length': game_length,
                    'duration': game_duration,
                    'player_roles': player_roles_msg
                })

            except queue.Empty:
                # No jobs for a while, continue waiting or break if explicit signal is added
                continue
            except Exception as e:
                worker_logger.error(f"Eval Worker {worker_id}: Error during game play: {e}", exc_info=True)
                # Potentially put an error signal onto game_result_queue for main process to handle
                pass
        
        worker_logger.info(f"Eval Worker {worker_id}: Exiting.")


    def run_for_n_games(self, n_games):
        """
        Evaluates the new model by playing a number of games against the best model
        using multiprocessing. The players are swapped for fairness across games.
        Inference is offloaded to dedicated batcher processes.
        
        Args:
            n_games (int): The total number of games to play for evaluation.
            
        Returns:
            tuple: A tuple containing the final game scores (test_score, best_score).
                    Score is calculated as (wins * 1.0) + (draws * 0.5).
        """
        self.main_logger.info(f"Starting evaluation of new model over {n_games} games with {self.num_evaluation_workers} workers...")
        self.main_logger.info(f"Test Model: Iter {self.current_iter} (path: {self.test_model_path}) vs Best Model: Iter {self.best_iter} (path: {self.best_model_path})")

        # --- Start Inference Batcher Processes ---
        # Test Model Inference Batcher
        test_batcher = InferenceBatcher(
            name='test_model',
            model_path=self.test_model_path,
            model_config=self.model_config,
            batch_size=self.evaluation_config['batch_size_per_worker'] * self.num_evaluation_workers,
            batch_timeout=self.evaluation_config['batch_timeout'],
            log_dir=self.log_dir,
            logging_level=self.evaluation_config['inference_logging_level']
        )
        self.test_inference_process = mp.Process(
            target=test_batcher.run,
            args=(self.test_inference_queue, self.test_result_queues, self.num_evaluation_workers),
            daemon=True
        )
        self.test_inference_process.start()
        self.main_logger.info(f"Test Model Inference batcher process started (PID: {self.test_inference_process.pid}).")

        # Best Model Inference Batcher
        best_batcher = InferenceBatcher(
            name='best_model',
            model_path=self.best_model_path,
            model_config=self.model_config,
            batch_size=self.evaluation_config['batch_size_per_worker'] * self.num_evaluation_workers,
            batch_timeout=self.evaluation_config['batch_timeout'],
            log_dir=self.log_dir,
            logging_level=self.evaluation_config['inference_logging_level']
        )
        self.best_inference_process = mp.Process(
            target=best_batcher.run,
            args=(self.best_inference_queue, self.best_result_queues, self.num_evaluation_workers),
            daemon=True
        )
        self.best_inference_process.start()
        self.main_logger.info(f"Best Model Inference batcher process started (PID: {self.best_inference_process.pid}).")


        # --- Create and Start Worker Processes ---
        for i in range(self.num_evaluation_workers):
            p = mp.Process(
                target=EvaluationTask._eval_worker_main,
                args=(
                    i, # worker_id
                    self.output_dir, # output_dir
                    self.evaluation_config, # evaluation_config
                    self.current_iter, # current_iter
                    self.best_iter, # best_iter
                    self.game_job_queue, # game_job_queue
                    self.game_result_queue, # game_result_queue
                    self.test_inference_queue, # test_inference_queue
                    self.test_result_queues[i], # test_result_queue_for_this_worker
                    self.best_inference_queue, # best_inference_queue
                    self.best_result_queues[i], # best_result_queue_for_this_worker
                    self.test_model_wins, # test_model_wins_shared
                    self.best_model_wins, # best_model_wins_shared
                    self.draws # draws_shared
                ),
                daemon=True
            )
            self.worker_processes.append(p)
            p.start()
            self.main_logger.info(f"Evaluation worker process {i} started (PID: {p.pid}).")
        
        # --- Distribute game jobs ---
        try:
            half_games = n_games // 2
            
            # Phase 1: Best model plays white for the first half of games
            self.main_logger.info(f"Distributing {half_games} games where Best Model plays White.")
            for game_idx in range(1, half_games + 1):
                self.game_job_queue.put((game_idx, False))
            
            # Phase 2: Test model plays white for the remaining games
            remaining_games_start_idx = half_games + 1
            self.main_logger.info(f"Distributing {n_games - half_games} games where Test Model plays White.")
            for game_idx in range(remaining_games_start_idx, n_games + 1):
                self.game_job_queue.put((game_idx, True))
            # --- Collect results from workers ---
            games_processed = 0
            while games_processed < n_games:
                try:
                    # Get results from any worker
                    result_data = self.game_result_queue.get(timeout=5.0) # Add timeout for robustness
                    games_processed += 1

                    self.main_logger.info(
                        f"Main: Game {result_data['game_idx']}/{n_games} ({result_data['player_roles']}) "
                        f"completed. Result: {result_data['result']}. "
                        f"Overall Scores: Test Wins={self.test_model_wins.value}, "
                        f"Best Wins={self.best_model_wins.value}, Draws={self.draws.value}."
                    )
                except queue.Empty:
                    continue 

        finally:
            self.main_logger.info(f"Finished evaluating {games_processed} games. Terminating evaluation processes...")
            
            # Ensure inference batcher processes are terminated and joined
            if self.test_inference_process and self.test_inference_process.is_alive():
                self.test_inference_process.terminate()
            if self.best_inference_process and self.best_inference_process.is_alive():
                self.best_inference_process.terminate()
            
            if self.test_inference_process:
                self.test_inference_process.join()
            if self.best_inference_process:
                self.best_inference_process.join()
            
            # Ensure all worker processes are terminated and joined
            for p in self.worker_processes:
                if p.is_alive():
                    p.terminate()
                p.join() # Wait for the process to clean up

            self.main_logger.info("All evaluation processes terminated. Evaluation task complete.")

        # Final score calculation based on shared values
        final_test_score = self.test_model_wins.value * 1.0 + self.draws.value * 0.5
        final_best_score = self.best_model_wins.value * 1.0 + self.draws.value * 0.5

        self.main_logger.info(f"Final Evaluation Results (out of {n_games} games):")
        self.main_logger.info(f"  Test Model Iter {self.current_iter} Wins: {self.test_model_wins.value}")
        self.main_logger.info(f"  Best Model Iter {self.best_iter} Wins: {self.best_model_wins.value}")
        self.main_logger.info(f"  Draws: {self.draws.value}")
        self.main_logger.info(f"  Total Score: {final_test_score:.1f}-{final_best_score:.1f}")

        return final_test_score, final_best_score