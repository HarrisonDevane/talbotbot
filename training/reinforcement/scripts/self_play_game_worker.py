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

import utils

class SelfPlayGameWorker:
    """
    Plays a single game of chess between two automated players
    and returns the training data and PGN.
    """
    def __init__(self, logger: logging.Logger, player, config):
        self.logger = logger
        self.players = {
            chess.WHITE: player,
            chess.BLACK: player,
        }

        self.search_depth = config['self_play']['search_depth']
        self.book_path = config['self_play']['opening_book_path']
        self.opening_min_moves = config['self_play']['opening_min_moves']
        self.opening_max_moves = config['self_play']['opening_max_moves']

        self.transition_factor = config['self_play']['value_transition_factor']
        self.max_transition_move = config['self_play']['value_max_transition_move']
        self.transition_steepness = config['self_play']['value_transition_steepness']
    

    def play_one_game(self, game_number: int):
        """
        Plays a single game from the starting position to the end.
        Returns the PGN string and a list of training examples.
        """
        # Reset game-specific state
        self.board = chess.Board()
        self.game_over = False
        self.current_turn = self.board.turn
        
        # Reset the player's internal state for a new game
        for player in self.players.values():
            player.reset_for_new_game()

        # Call the existing game loop logic (which we'll rename to `_run_game_loop`)
        training_data = self._run_game_loop(game_number)

        return training_data




    def _run_game_loop(self, game_number: int):
        """
        The main loop for a single game. Runs until a game-ending condition is met.
        Collects raw data, backfills the value targets, and returns the results.
        """
        # Use game_number for logging instead of self.current_game
        self.logger.info(f"--- Starting Game {game_number} ---")
        
        move_count = 0
        max_opening_moves = random.randint(self.opening_min_moves, self.opening_max_moves)
        self.logger.debug(f"Game {game_number} will use an opening book for the first {max_opening_moves} moves.")

        raw_training_data = []
        game_length = 0

        while not self.game_over:
            # ... (the rest of your original `game_loop` code, unchanged) ...
            player = self.players[self.current_turn]
            
            move = None
            policy_vector = None
            root_value = None

            current_board = self.board.copy()
            
            if move_count < max_opening_moves:
                try:
                    with chess.polyglot.open_reader(self.book_path) as reader:
                        book_move = reader.weighted_choice(current_board).move
                        move = book_move
                        self.logger.debug(f"Game {game_number} - Book move selected: {move.uci()}")
                        
                        policy_vector = np.zeros(utils.TOTAL_POLICY_MOVES, dtype=np.float32)
                        row, col, channel = utils.move_to_policy_components(move, current_board)
                        flat_index = utils.policy_components_to_flat_index(row, col, channel)
                        policy_vector[flat_index] = 1.0
                        root_value = 0.0

                except (IndexError, AttributeError):
                    move, policy_vector, root_value = player.get_move(current_board, move_count, self.search_depth)
                    max_opening_moves = 0  # Stop using the book
            else:
                move, policy_vector, root_value = player.get_move(current_board, move_count, self.search_depth)
            
            board_state_tensor = utils.board_to_tensor_68(current_board)
            raw_training_data.append({
                "board_state": board_state_tensor,
                "policy": policy_vector,
                "root_value": root_value,
                "turn": current_board.turn,
                "move_count": move_count
            })
            
            self.board.push(move)
            self.current_turn = not self.current_turn
            self.logger.debug(f"Game {game_number} - Move made: {move.uci()}")
            move_count += 1
            
            self._check_game_over(game_number)

        result = self.board.result()
        if result == '1-0':
            final_game_value = 1.0
        elif result == '0-1':
            final_game_value = -1.0
        else:
            final_game_value = 0.0

        game_length = move_count
        
        final_training_data = []
        for i, move_num in enumerate(raw_training_data):
            final_value_for_player = final_game_value if move_num['turn'] == chess.WHITE else -final_game_value
            
            blended_value_target = self._calculate_blended_value(
                mcts_value=move_num['root_value'],
                final_game_value=final_value_for_player,
                move_count=move_num['move_count'],
                game_length=game_length
            )
            
            final_training_data.append({
                'board_state': move_num['board_state'],
                'policy': move_num['policy'],
                'value_target': blended_value_target
            })

        return final_training_data


    def _calculate_blended_value(self, mcts_value: float, final_game_value: float, move_count: int, game_length: int) -> float:
        """
        Calculates a blended value target using a hyperbolic tangent (tanh) function.
        """
        transition_center = min(int(game_length * self.transition_factor), self.max_transition_move)
        final_result_blend_factor = (math.tanh(self.transition_steepness * (move_count - transition_center)) + 1.0) / 2.0
        mcts_blend_factor = 1.0 - final_result_blend_factor
        return (mcts_blend_factor * mcts_value) + (final_result_blend_factor * final_game_value)


    def _check_game_over(self, game_number: int):
        """
        Checks if the game has ended and logs the outcome.
        """
        if self.board.is_game_over():
            self.game_over = True
            result = self.board.result()
            self.logger.info(f"Game {game_number} over. Result: {result}")
            game = chess.pgn.Game.from_board(self.board)
            exporter = chess.pgn.StringExporter(headers=False)
            pgn_string = game.accept(exporter)
            pgn_string = pgn_string.strip() + " " + self.board.result()
            self.logger.info(f"--- Game PGN ---\n{pgn_string}")