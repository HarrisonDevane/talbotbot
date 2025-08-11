import chess
import chess.pgn
import logging
import random
import chess.polyglot
import sys, os
import math
import numpy as np

# Assuming these imports are in your project structure
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

import src_shared.utils as utils

class EvaluationGameWorker:
    """
    Plays a single game of chess between two automated players
    and returns the training data and PGN.
    """
    def __init__(self, logger, player_1, player_2, evaluation_config):
        self.logger = logger
        self.evaluation_config = evaluation_config
        self.player_1 = player_1
        self.player_2 = player_2
        self.players = {
            chess.WHITE: player_1,
            chess.BLACK: player_2,
        }

        self.result = None


    def run_eval_loop(self, game_number):
        """
        The main loop for a single game. Runs until a game-ending condition is met.
        Collects raw data, backfills the value targets, and returns the results.
        """
        self.logger.critical(f"\n{'='*60}\n{' '*20}--- GAME {game_number} STARTED ---\n{'='*60}\n")
        
        self.board = chess.Board()
        self.game_over = False
        self.result = None
        last_move = None
        self.current_turn = self.board.turn
        
        # Reset the player's internal state for a new game
        for player in self.players.values():
            player.reset_for_new_game()


        move_count = 1
        max_opening_moves = random.randint(self.evaluation_config['opening_min_moves'], self.evaluation_config['opening_max_moves'])
        self.logger.info(f"Game {game_number} will use an opening book for the first {max_opening_moves} moves.")

        search_depth = random.choices(self.evaluation_config['search_depth'], weights=self.evaluation_config['search_depth_weights'], k=1)[0]
        self.logger.info(f"Game {game_number} will use a search depth of {search_depth}")

        while not self.game_over:
            player = self.players[self.current_turn]
            best_move = None

            current_board = self.board.copy()
            
            if move_count < max_opening_moves:
                try:
                    with chess.polyglot.open_reader(self.evaluation_config['opening_book_path']) as reader:
                        book_move = reader.weighted_choice(current_board).move
                        best_move = book_move

                except (IndexError, AttributeError):
                    best_move, policy_vector, root_value, simulation_count  = player.get_move(current_board, move_count, search_depth, None)
                    max_opening_moves = 0
            else:
                best_move, policy_vector, root_value, simulation_count = player.get_move(current_board, move_count, search_depth, last_move)
                        
            self.board.push(best_move)
            self.current_turn = not self.current_turn
            self.logger.info(f"Game {game_number} - Move made: {best_move.uci()}")
            last_move = best_move
            move_count += 1
            
            self._check_game_over(game_number)

        return self.result, move_count

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

            # Generate PGN
            game = chess.pgn.Game.from_board(self.board)
            game.headers["Result"] = self.result
            game.headers["White"] = self.player_1.name if self.players[chess.WHITE] == self.player_1 else self.player_2.name
            game.headers["Black"] = self.player_2.name if self.players[chess.BLACK] == self.player_2 else self.player_1.name

            exporter = chess.pgn.StringExporter(headers=True)
            pgn_string = game.accept(exporter).strip()


            self.logger.critical(f"Game PGN:\n{pgn_string}")