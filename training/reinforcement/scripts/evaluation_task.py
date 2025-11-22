import os, sys
import time
import logging
import multiprocessing as mp
import psutil
import torch
import queue
import chess

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

from self_play_agent import SelfPlayAgent
from evaluation_game_worker import EvaluationGameWorker
from src_shared.inference_batcher import InferenceBatcher
import src_shared.utils


class EvaluationTask:
    def __init__(self, output_dir, best_model_path, test_model_path, model_config, evaluation_config, current_iter, best_iter):
        self.output_dir = output_dir
        self.best_model_path = best_model_path
        self.test_model_path = test_model_path
        self.model_config = model_config
        self.evaluation_config = evaluation_config
        self.current_iter = current_iter
        self.best_iter = best_iter
        
        # Extract best model path from model_config for clarity
        self.num_evaluation_workers = len(self.evaluation_config['game_worker_cores'])

        self.max_batch_size = self.num_evaluation_workers * self.evaluation_config['batch_size_per_worker'] * self.evaluation_config['batch_size_factor']

        # Create Global Shared Buffers (Single Instance)
        # Policy (float16)
        self.shared_input_buffer = torch.zeros(
            self.max_batch_size, src_shared.utils.INPUT_CHANNELS, src_shared.utils.BOARD_DIM, src_shared.utils.BOARD_DIM, dtype=torch.float32
        ).share_memory_()

        self.shared_policy_buffer = torch.zeros(
            self.max_batch_size, src_shared.utils.TOTAL_POLICY_MOVES, dtype=torch.float16
        ).share_memory_()
        # Value (float32)
        self.shared_value_buffer = torch.zeros(
            self.max_batch_size, 1, dtype=torch.float32
        ).share_memory_()
    
        # Create Global Free Index Queue
        self.buffer_free_slots = mp.Queue()
        for i in range(self.max_batch_size):
            self.buffer_free_slots.put(i)


        self.main_logger = self._setup_logger(
            "EvaluationManager", 
            self.evaluation_config['main_logging_level'],
            os.path.join(self.output_dir, f"evaluation_manager.log")
        )

        # Multiprocessing components for two models (test and best)
        self.test_inference_queue = mp.Queue()
        self.test_result_queues = [mp.Queue() for _ in range(self.num_evaluation_workers)]
        
        self.best_inference_queue = mp.Queue()
        self.best_result_queues = [mp.Queue() for _ in range(self.num_evaluation_workers)]
        
        self.game_job_queue = mp.Queue() 
        self.game_result_queue = mp.Queue() 
        
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
    def _generate_openings(book_path, num_openings, logger):
        """
        Generates a specified number of diverse opening lines using a Breadth-First 
        Search (BFS). The search continues until the book runs out or num_openings
        is reached.
        """

        from collections import deque
        import chess.polyglot

        logger.info(f"Generating top {num_openings} diverse opening lines, following book depth...")

        # --- BFS to find unique opening lines ---
        
        queue = deque()
        queue.append((chess.Board(), [])) 

        final_openings = [] # Stores the completed move lists
        seen_positions = set()
        
        logger.info(f"Searching for {num_openings} unique opening lines...")

        while queue and len(final_openings) < num_openings:
            board, move_list = queue.popleft()
            fen = board.fen()
            
            if fen in seen_positions: 
                continue
                
            seen_positions.add(fen)
            
            # Find possible continuations from the book
            with chess.polyglot.open_reader(book_path) as reader:
                entries = list(reader.find_all(board))
            
            has_continuations = len(entries) > 0

            if not has_continuations:
                final_openings.append(move_list)
                continue
                
            for entry in entries:
                new_board = board.copy()
                new_board.push(entry.move)
                new_move_list = move_list + [entry.move]
                queue.append((new_board, new_move_list))

        
        logger.info(f"Generated {len(final_openings)} full opening lines successfully.")
        return final_openings

    @staticmethod
    def _eval_worker_main(
        worker_id, core_id, output_dir, evaluation_config,
        current_iter, best_iter,
        game_job_queue, game_result_queue,
        test_inference_queue, test_result_queue_for_this_worker,
        best_inference_queue, best_result_queue_for_this_worker, 
        shared_input_buffer, shared_policy_buffer, shared_value_buffer,
        buffer_free_slots, test_model_wins_shared, best_model_wins_shared, draws_shared
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
        process.cpu_affinity([core_id])

        process.nice(psutil.HIGH_PRIORITY_CLASS)


        # Create a new logger specific to this worker process
        worker_logger = EvaluationTask._setup_logger(
            f"EvalWorker_{worker_id}", 
            evaluation_config['worker_logging_level'],
            os.path.join(output_dir, f"evaluation_worker_{worker_id}.log")
        )

        # Instantiate players configured to use the inference queues
        mcts_player_best = SelfPlayAgent(
            name=f'best_model_iter_{best_iter}',
            logger=worker_logger,
            self_play_config=evaluation_config,
            worker_id=worker_id,
            inference_queue=best_inference_queue,
            result_queue=best_result_queue_for_this_worker,
            shared_input_buffer=shared_input_buffer, 
            shared_policy_buffer=shared_policy_buffer, 
            shared_value_buffer=shared_value_buffer,
            buffer_free_slots=buffer_free_slots
        )

        mcts_player_test = SelfPlayAgent(
            name=f'test_model_iter_{current_iter}',
            logger=worker_logger,
            self_play_config=evaluation_config,
            worker_id=worker_id,
            inference_queue=test_inference_queue,
            result_queue=test_result_queue_for_this_worker,
            shared_input_buffer=shared_input_buffer, 
            shared_policy_buffer=shared_policy_buffer, 
            shared_value_buffer=shared_value_buffer,
            buffer_free_slots=buffer_free_slots
        )
        
        game_manager = EvaluationGameWorker(
            logger=worker_logger,
            evaluation_config=evaluation_config
        )

        # Main loop to receive game jobs and play them
        while True:
            # Dynamic sleep based on worker ID to stop thundering herd issue
            time.sleep(worker_id*0.4 + 1)
            try:
                # A job is: (game_idx, test_is_white_flag, opening_move_list)
                game_idx, test_is_white, opening_move_list = game_job_queue.get_nowait()
                
                # Signal to terminate workers
                if game_idx is None:
                    worker_logger.info(f"Eval Worker {worker_id}: Received termination signal.")
                    break

                # Determine which player is White/Black for this game
                if test_is_white:
                    game_manager.update_players(
                        player_white=mcts_player_test,
                        player_black=mcts_player_best
                    ),
                    player_roles_msg = "Test (W) vs Best (B)"
                else:
                    game_manager.update_players(
                        player_white=mcts_player_best,
                        player_black=mcts_player_test
                    ),
                    player_roles_msg = "Best (W) vs Test (B)"

                start_time_game = time.time()
                worker_logger.info(
                    f"Starting game with opening moves: {' '.join(m.uci() for m in opening_move_list)} "
                    f"White: {game_manager.players[chess.WHITE].name}, "
                    f"Black: {game_manager.players[chess.BLACK].name}, "          
                )

                game_result, game_length = game_manager.run_eval_loop(game_idx, opening_move_list)
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
                worker_logger.critical(f"Eval Worker {worker_id}: Error during game {game_idx} play: {e}", exc_info=True)
                
                # Update shared counters (atomically) for a draw
                with test_model_wins_shared.get_lock():
                    with best_model_wins_shared.get_lock():
                        with draws_shared.get_lock():
                            draws_shared.value += 1
                
                # Put the draw result on the queue for the main process to handle.
                game_result_queue.put({
                    'game_idx': game_idx,
                    'result': 'Draw (Error)',
                    'length': 0,
                    'duration': 0,
                    'player_roles': 'N/A'
                })
                
                # Shutdown the worker
                break
        
        worker_logger.info(f"Eval Worker {worker_id}: Exiting.")


    def run_for_n_games(self, n_games):
        """
        Evaluates the new model by playing a number of games against the best model
        using multiprocessing. The players are swapped for fairness across games.
        Inference is offloaded to dedicated batcher processes.
        """

        self.main_logger.info(f"Starting evaluation of new model over {n_games} games with {self.num_evaluation_workers} workers...")
        self.main_logger.info(f"Test Model: Iter {self.current_iter} (path: {self.test_model_path}) vs Best Model: Iter {self.best_iter} (path: {self.best_model_path})")

        fixed_evaluation_openings = self._generate_openings(
            os.path.abspath(os.path.join(project_root, self.evaluation_config['opening_book_path'])),
            num_openings=n_games//2,
            logger=self.main_logger
        )

        self.main_logger.info(f"Using {len(fixed_evaluation_openings)} fixed openings for evaluation.")

        # Test Model Inference Batcher
        test_batcher = InferenceBatcher(
            name='eval_test_model',
            model_path=self.test_model_path,
            model_config=self.model_config,
            batch_size=(self.evaluation_config['batch_size_per_worker'] * self.num_evaluation_workers),
            batch_timeout=self.evaluation_config['batch_timeout'],
            log_dir=self.output_dir,
            logging_level=self.evaluation_config['inference_logging_level']
        )
        self.test_inference_process = mp.Process(
            target=test_batcher.run,
            args=(self.test_inference_queue, self.test_result_queues, self.evaluation_config['inference_worker_cores'][0], self.shared_input_buffer, self.shared_policy_buffer, self.shared_value_buffer),
            daemon=True
        )
        self.test_inference_process.start()
        self.main_logger.info(f"Test Model Inference batcher process started (PID: {self.test_inference_process.pid}).")
        time.sleep(2)

        # Best Model Inference Batcher
        best_batcher = InferenceBatcher(
            name='eval_best_model',
            model_path=self.best_model_path,
            model_config=self.model_config,
            batch_size=(self.evaluation_config['batch_size_per_worker'] * self.num_evaluation_workers),
            batch_timeout=self.evaluation_config['batch_timeout'],
            log_dir=self.output_dir,
            logging_level=self.evaluation_config['inference_logging_level']
        )
        self.best_inference_process = mp.Process(
            target=best_batcher.run,
            args=(self.best_inference_queue, self.best_result_queues, self.evaluation_config['inference_worker_cores'][1], self.shared_input_buffer, self.shared_policy_buffer, self.shared_value_buffer),
            daemon=True
        )
        self.best_inference_process.start()
        self.main_logger.info(f"Best Model Inference batcher process started (PID: {self.best_inference_process.pid}).")
        time.sleep(2)


        # --- Create and Start Worker Processes ---
        for i in range(self.num_evaluation_workers):
            p = mp.Process(
                target=EvaluationTask._eval_worker_main,
                args=(
                    i,
                    self.evaluation_config['game_worker_cores'][i],
                    self.output_dir,
                    self.evaluation_config,
                    self.current_iter,
                    self.best_iter,
                    self.game_job_queue,
                    self.game_result_queue,
                    self.test_inference_queue,
                    self.test_result_queues[i],
                    self.best_inference_queue,
                    self.best_result_queues[i],
                    self.shared_input_buffer, 
                    self.shared_policy_buffer, 
                    self.shared_value_buffer,
                    self.buffer_free_slots,
                    self.test_model_wins,
                    self.best_model_wins, 
                    self.draws
                ),
                daemon=True
            )
            self.worker_processes.append(p)
            p.start()
            self.main_logger.info(f"Evaluation worker process {i} started (PID: {p.pid}).")
            time.sleep(2)
        
        # --- MODIFIED: Distribute game jobs with openings ---
        try:
            game_idx_counter = 0
            for opening_move_list in fixed_evaluation_openings:
                # Game 1: Test Model plays White
                game_idx_counter += 1
                self.main_logger.debug(f"Putting game {game_idx_counter} with opening {' '.join(m.uci() for m in opening_move_list)} on queue (Test as White)")
                self.game_job_queue.put((game_idx_counter, True, opening_move_list))
                
                # Game 2: Best Model plays White
                game_idx_counter += 1
                self.main_logger.debug(f"Putting game {game_idx_counter} with opening {' '.join(m.uci() for m in opening_move_list)} on queue (Best as White)")
                self.game_job_queue.put((game_idx_counter, False, opening_move_list))

            # --- Collect results from workers ---
            games_processed = 0
            while games_processed < n_games:
                try:
                    # Get results from any worker
                    result_data = self.game_result_queue.get(timeout=1.0)
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
            
            if self.test_inference_process and self.test_inference_process.is_alive():
                self.test_inference_process.terminate()
            if self.best_inference_process and self.best_inference_process.is_alive():
                self.best_inference_process.terminate()
            
            if self.test_inference_process:
                self.test_inference_process.join()
            if self.best_inference_process:
                self.best_inference_process.join()
            
            for p in self.worker_processes:
                if p.is_alive():
                    p.terminate()
                p.join()

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