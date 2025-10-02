import chess
import chess.pgn
import random
import sys, os

# Assuming these imports are in your project structure
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

import src_shared.utils as utils

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

        search_depth = random.choices(self.data_generation_config['search_depth'], weights=self.data_generation_config['search_depth_weights'], k=1)[0]
        self.logger.info(f"Game {game_number} will use a search depth of {search_depth}")

        while not self.game_over:
            player = self.players[self.current_turn]
            current_board = self.board.copy()

            move, policy_vector, simulation_count = player.get_move(current_board, ply_count, search_depth, None)
            board_state_tensor = utils.board_to_tensor_68(current_board)                

            raw_training_data.append({
                "board_state": board_state_tensor,
                "policy": policy_vector,
                "turn": current_board.turn,
                "simulation_count": simulation_count
            })
            
            self.board.push(move)
            self.current_turn = not self.current_turn
            self.logger.info(f"Game {game_number} - Move made: {move.uci()}")
            total_simulations += simulation_count
            ply_count += 1
                    
            self._check_game_over(game_number)


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
            
        return final_training_data, total_simulations


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