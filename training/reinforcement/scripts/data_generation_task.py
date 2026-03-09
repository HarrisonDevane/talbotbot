import os, sys
import time
import logging
import multiprocessing as mp
import psutil
import torch
import chess
import chess.pgn
import queue

current_script_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.abspath(os.path.join(current_script_dir, ".."))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

from src_shared.talbot_agent import TalbotAgent
from src_shared.inference_batcher import InferenceBatcher
import src_shared.utils


class DataGenerationGameWorker:
    """
    Plays a single game of chess between two automated players
    and returns the training data and PGN.
    """
    def __init__(self, logger, player_1, player_2, data_generation_config):
        self.logger = logger
        self.data_generation_config = data_generation_config
        self.player_1 = player_1
        self.player_2 = player_2
        self.players = {
            chess.WHITE: player_1,
            chess.BLACK: player_2,
        }

        self.result = None


    def run_training_loop(self, game_number):
        """
        The main loop for a single game. Runs until a game-ending condition is met.
        Collects raw data, backfills the value targets, and returns the results.
        """
        self.logger.critical(f"\n{'='*60}\n{' '*20}--- GAME {game_number} STARTED ---\n{'='*60}\n")
        
        self.board = chess.Board()
        self.game_over = False
        self.result = None
        self.current_turn = self.board.turn
        
        # Reset the player's internal state for a new game
        for player in self.players.values():
            player.reset_for_new_game()


        ply_count = 1
        raw_training_data = []
        total_simulations = 0
        total_entropy = 0

        phase_budgets = self.data_generation_config['gumbel_search_budget']
        self.logger.info(f"Game {game_number} will use a search depth of {sum(phase_budgets)}")

        while not self.game_over:
            player = self.players[self.current_turn]
            current_board = self.board.copy()

            move, policy_vector, simulation_count, move_entropy = player.get_move(current_board, ply_count, phase_budgets)
            total_simulations += simulation_count
            total_entropy += move_entropy

            if move is None:
                self.result = "1-0" if self.current_turn == chess.BLACK else "0-1"
                self.game_over = True
                self.logger.info(f"Game {game_number} ended by resignation.")

            else:
                board_state_tensor = src_shared.utils.board_to_tensor_69(current_board)                
                raw_training_data.append({
                    "board_state": board_state_tensor,
                    "policy": policy_vector,
                    "turn": current_board.turn,
                    "simulation_count": simulation_count
                })
                
                self.board.push(move)
                self.current_turn = not self.current_turn
                self.logger.info(f"Game {game_number} - Move made: {move.uci()}")
                ply_count += 1
                        
                self._check_game_over(game_number)


        self._generate_pgn(game_number)

        if self.result == '1-0':
            final_game_value = 1.0
        elif self.result == '0-1':
            final_game_value = -1.0
        else:
            final_game_value = 0.0
        
        final_training_data = []
        for i, move_num in enumerate(raw_training_data):            
            final_training_data.append({
                'board_state': move_num['board_state'],
                'policy': move_num['policy'],
                'value_target': final_game_value if move_num['turn'] == chess.WHITE else -final_game_value
            })
            
        return final_training_data, total_simulations, total_entropy


    def _check_game_over(self, game_number):
        """
        Checks if the game has ended and logs the outcome.
        """
        if self.board.is_game_over(claim_draw=True):
            self.game_over = True
            self.result = self.board.result(claim_draw=True)

            if self.board.can_claim_threefold_repetition():
                self.logger.info(f"Game {game_number} ended by threefold repetition claim.")
            elif self.board.can_claim_fifty_moves():
                self.logger.info(f"Game {game_number} ended by 50-move rule claim.")
            else:
                self.logger.info(f"Game {game_number} over. Result: {self.result}")


    def _generate_pgn(self, game_number):
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = f"Self-Play Game {game_number}"
        game.headers["Result"] = self.result
        game.headers["White"] = self.player_1.name if self.players[chess.WHITE] == self.player_1 else self.player_2.name
        game.headers["Black"] = self.player_2.name if self.players[chess.BLACK] == self.player_2 else self.player_1.name

        exporter = chess.pgn.StringExporter(headers=True)
        pgn_string = game.accept(exporter).strip()

        self.logger.critical(f"Game PGN:\n{pgn_string}")


class DataGenerationTask:
    """
    Main class to orchestrate a multi-process AlphaZero-style data generation pipeline.
    It manages the creation of inference and worker processes, and handles inter-process
    communication via queues.
    """
    def __init__(self, output_dir, current_steps, rotation_interval, sync_interval, best_model_path, model_config, data_generation_config, state_config):
        self.output_dir = output_dir
        self.best_model_path = best_model_path
        self.model_config = model_config
        self.data_generation_config = data_generation_config
        self.state_config = state_config
        self.current_steps = current_steps
        self.rotation_interval = rotation_interval
        self.sync_interval = sync_interval
        self.num_workers = len(data_generation_config['game_worker_cores'])
        self.num_inference_batchers = len(data_generation_config['inference_worker_cores'])

        self.max_batch_size = self.num_workers * self.data_generation_config['batch_size_per_worker'] * self.data_generation_config['batch_size_factor']

        use_fp16 = torch.cuda.is_available()
        input_dtype = torch.float16 if use_fp16 else torch.float32

        # Create Global Shared Buffers (Single Instance)
        self.shared_input_buffer = torch.zeros(
            self.max_batch_size, src_shared.utils.INPUT_CHANNELS, src_shared.utils.BOARD_DIM, src_shared.utils.BOARD_DIM, dtype=input_dtype
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

        # Multi-processing components
        self.inference_queues = [mp.Queue() for _ in range(self.num_inference_batchers)]
        self.result_queues = [mp.Queue() for _ in range(self.num_workers)]
        self.data_queue = mp.Queue()
        
        self.inference_processes = []
        self.worker_processes = []
        self.game_number_counter = mp.Value('i', self.state_config['state']['lifetime']['games_played']) 


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
    def _worker_main(worker_id, core_id, output_dir, inference_queues, result_queue, data_queue, data_generation_config, current_steps, rotation_interval, game_number_counter, shared_input_buffer, shared_policy_buffer, shared_value_buffer, buffer_free_slots, stop_event):
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

        # Distribute workers evenly among inference queues
        inference_queue_for_worker = inference_queues[worker_id % len(inference_queues)]
        
        # Track the last used log directory to avoid redundant logger setups
        last_log_dir = None
        worker_logger = None

        while not stop_event.is_set():
            # 1. Dynamic Path Calculation based on current global steps
            with current_steps.get_lock():
                local_step_val = current_steps.value
            
            target_folder_step = (local_step_val // rotation_interval) * rotation_interval
            current_run_dir = os.path.join(output_dir, f"run_step_{target_folder_step:06d}")

            # 2. Update logger if we crossed a step threshold
            if current_run_dir != last_log_dir:
                os.makedirs(current_run_dir, exist_ok=True)
                last_log_dir = current_run_dir
                worker_logger = DataGenerationTask._setup_logger(
                    f"Worker_{worker_id}", 
                    data_generation_config['worker_logging_level'],
                    os.path.join(current_run_dir, f"data_generation_worker_{worker_id}.log")
                )
                worker_logger.info(f"Worker {worker_id} pinned to core {core_id}. Logging to {current_run_dir}")

            with game_number_counter.get_lock():
                current_game_number = game_number_counter.value
                game_number_counter.value += 1
            
            # 3. Instantiate Agent and Manager for the current game
            mcts_player = TalbotAgent(
                name=f'talbot_step_{local_step_val}',
                logger=worker_logger,
                talbot_config=data_generation_config,
                worker_id=worker_id,
                inference_queue=inference_queue_for_worker,
                result_queue=result_queue,
                shared_input_buffer=shared_input_buffer,
                shared_policy_buffer=shared_policy_buffer,
                shared_value_buffer=shared_value_buffer,
                buffer_free_slots=buffer_free_slots
            )
            
            game_manager = DataGenerationGameWorker(
                logger=worker_logger,
                player_1=mcts_player,
                player_2=mcts_player,
                data_generation_config=data_generation_config
            )
            
            game_start = time.time()
            training_data, simulation_count, game_entropy = game_manager.run_training_loop(current_game_number)
            game_end = time.time()

            game_time = game_end - game_start
            num_new_positions = len(training_data)
            simulations_per_second = simulation_count / game_time
            
            worker_logger.critical(
                f"Game {current_game_number} completed in {game_time:.2f}s  "
                f"with {simulations_per_second:.2f} simulations per second "
                f"({num_new_positions} positions). Sending data to main process."
            )
            
            # Use a non-blocking put or check stop_event to ensure we don't hang on shutdown
            try:
                data_queue.put((training_data, game_time, game_entropy), timeout=1.0)
            except:
                worker_logger.warning("Data queue full or shutdown initiated. Dropping game data.")

        if worker_logger:
            worker_logger.info(f"Worker {worker_id} received stop signal. Exiting.")
                

    def run_persistently(self, chunk_size: int):
            """
            Starts the multi-process pipeline and stays alive.
            Yields data in chunks of 'chunk_size' indefinitely.
            """
            # Create a stop event for graceful shutdown
            self.stop_event = mp.Event()

            # 1. Start Inference Batchers
            for i in range(self.num_inference_batchers):
                batcher = InferenceBatcher(
                    f'batcher_{i}',
                    self.best_model_path,
                    self.model_config,
                    self.data_generation_config['batch_size_per_worker'] * self.num_workers,
                    self.data_generation_config['batch_timeout'],
                    self.output_dir,
                    self.data_generation_config['inference_logging_level'],
                    True
                )

                p = mp.Process(
                    target=batcher.run,
                    args=(self.output_dir, self.inference_queues[i], self.result_queues, 
                        self.data_generation_config['inference_worker_cores'][i], 
                        self.shared_input_buffer, self.shared_policy_buffer, 
                        self.shared_value_buffer,
                        self.stop_event, self.current_steps, self.sync_interval, self.rotation_interval),
                    daemon=True
                )
                self.inference_processes.append(p)
                p.start()
                time.sleep(1.0)

            # 2. Start Game Workers
            for i in range(self.num_workers):
                p = mp.Process(
                    target=DataGenerationTask._worker_main,
                    args=(i, self.data_generation_config['game_worker_cores'][i], self.output_dir, 
                        self.inference_queues, self.result_queues[i], self.data_queue,
                        self.data_generation_config, self.current_steps, self.rotation_interval, self.game_number_counter, 
                        self.shared_input_buffer, self.shared_policy_buffer, 
                        self.shared_value_buffer, self.buffer_free_slots,
                        self.stop_event),
                    daemon=True
                )
                self.worker_processes.append(p)
                p.start()
                time.sleep(1.0)

            collected_data = []
            games_in_chunk = 0
            chunk_entropy = 0 

            # 3. Persistent Collection Loop
            while not self.stop_event.is_set():
                try:
                    # Block until a game is finished by any worker
                    game_data, game_time, game_entropy = self.data_queue.get(timeout=1.0)              
                    
                    games_in_chunk += 1
                    chunk_entropy += game_entropy
                    collected_data.extend(game_data)
                    
                    # Check if we have reached the threshold requested by main.py
                    if len(collected_data) >= chunk_size:
                        yield collected_data, games_in_chunk, chunk_entropy
                        
                        # Reset chunk-specific stats but keep processes running
                        collected_data = []
                        games_in_chunk = 0
                        chunk_entropy = 0
                except queue.Empty:
                    continue


    def terminate_all(self):
        """
        Gracefully shuts down all persistent worker and batcher processes.
        """
        # 1. Signal all processes to stop via the shared Event
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        
        # 2. Join and clean up Game Workers
        for i, p in enumerate(self.worker_processes):
            p.join(timeout=2)
            if p.is_alive():
                # Force kill if they don't exit within the timeout
                p.terminate()
                p.join()
        
        # 3. Join and clean up Inference Batchers
        for i, p in enumerate(self.inference_processes):
            p.join(timeout=2)
            if p.is_alive():
                # Force kill to ensure GPU memory is released
                p.terminate()
                p.join()
        
        # 4. Clear the process lists for safety
        self.worker_processes.clear()
        self.inference_processes.clear()