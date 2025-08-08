import os
import time
import logging
import numpy as np
from datetime import datetime

# Assuming these imports are in your project structure
from self_play_agent import TalbotPlayer
from self_play_game_worker import SelfPlayGameWorker

class EvaluationTask:
    def __init__(self, output_dir, test_model, model_config, evaluation_config, current_iter, best_iter):
        self.output_dir = output_dir
        self.test_model = test_model
        self.model_config = model_config
        self.evaluation_config = evaluation_config
        self.current_iter = current_iter
        self.best_iter = best_iter

        # Set up loggers
        self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        self.main_logger = self._setup_logger(
            "SelfPlayManager", 
            self.evaluation_config['main_logging_level'],
            os.path.join(self.log_dir, f"evaluation_{timestamp}.log")
        )
        
        self.worker_logger = self._setup_logger(
            "SelfPlayWorker",
            self.evaluation_config['worker_logging_level'],
            os.path.join(self.log_dir, f"evaluation_games_{timestamp}.log")
        )

        # Instantiate best model and test models
        self.mcts_player_best = TalbotPlayer(
            name=f'best_model_iter_{self.best_iter}',
            logger=self.worker_logger,
            model_path=self.model_config['best_model_path'],
            model_config=self.model_config,
            self_play_config=self.evaluation_config
        )

        self.mcts_player_test = TalbotPlayer(
            name=f'test_model_iter_{self.current_iter}',
            logger=self.worker_logger,
            model_path=self.test_model,
            model_config=self.model_config,
            self_play_config=self.evaluation_config
        )

        self.game_manager = SelfPlayGameWorker(
            logger=self.worker_logger,
            player_1=self.mcts_player_best,
            player_2=self.mcts_player_test,
            model_config=self.model_config,
            self_play_config=self.evaluation_config
        )

        self.game_number = 1

    def _setup_logger(self, name: str, level: str, log_file: str):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if logger.hasHandlers():
            logger.handlers.clear()
        
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger


    def run_for_n_games(self, n_games):
        """
        Evaluates the new model by playing a number of games against the best model.
        The players are swapped halfway through to ensure fairness.
        
        Args:
            n_games (int): The total number of games to play for evaluation.
            
        Returns:
            tuple: A tuple containing the final game results (test_model_wins, best_model_wins, draws).
        """
        self.main_logger.info(f"Starting evaluation of new model over {n_games} games...")
        self.main_logger.info(f"White is {self.mcts_player_best.name}, Black is {self.mcts_player_test.name} Will swap at half way")

        test_model_wins = 0
        best_model_wins = 0
        draws = 0

        test_score = None
        best_score = None
        
        for game_idx in range(1, n_games + 1):
            # Check if it's time to swap players
            if game_idx == (n_games // 2) + 1:
                self.main_logger.info(f"Swapping players for fairness. {self.mcts_player_test.name} will now be White.")
                self.game_manager.player_1 = self.mcts_player_test
                self.game_manager.player_2 = self.mcts_player_best

            start_time_game = time.time()
            
            # The play_one_game method returns the result and length of a single game
            game_result, game_length = self.game_manager.run_eval_loop(game_idx)
            
            end_time_game = time.time()
            
            # Process game result based on who the players were for this game
            current_player_white = self.game_manager.player_1
            current_player_black = self.game_manager.player_2

            if game_result == '1-0':
                if current_player_white == self.mcts_player_test:
                    test_model_wins += 1
                else:
                    best_model_wins += 1
            elif game_result == '0-1':
                if current_player_black == self.mcts_player_test:
                    test_model_wins += 1
                else:
                    best_model_wins += 1
            else:  
                draws += 1

            test_score = test_model_wins * 1.0 + draws * 0.5
            best_score = best_model_wins * 1.0 + draws * 0.5

            player_roles_msg = "Test (W) vs Best (B)" if current_player_white == self.mcts_player_test else "Best (W) vs Test (B)"
            self.main_logger.info(
                f"Game {game_idx}/{n_games} ({player_roles_msg}) completed in {end_time_game - start_time_game:.2f}s "
                f"with length {game_length} moves. Result: {game_result}. "
                f"Current Score: Test Wins={test_model_wins}, Best Wins={best_model_wins}, Draws={draws}. "
                f"Total Score: {test_score:.1f}-{best_score:.1f}"
            )
            
        self.main_logger.info(f"Final Evaluation Results (out of {n_games} games):")
        self.main_logger.info(f"  {self.mcts_player_test.name} Wins: {test_model_wins}")
        self.main_logger.info(f"  {self.mcts_player_best.name} Wins: {best_model_wins}")
        self.main_logger.info(f"  Draws: {draws}")
        self.main_logger.info(f"  Total Score: {test_score:.1f}-{best_score:.1f}")

        return test_score, best_score