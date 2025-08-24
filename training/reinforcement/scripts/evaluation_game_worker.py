import chess
import chess.pgn
import random


class EvaluationGameWorker:
    """
    Plays a single game of chess between two automated players
    and returns the training data and PGN.
    """
    def __init__(self, logger, evaluation_config):
        self.logger = logger
        self.evaluation_config = evaluation_config
        self.result = None


    def run_eval_loop(self, game_number, opening_move_list):
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
        search_depth = random.choices(self.evaluation_config['search_depth'], weights=self.evaluation_config['search_depth_weights'], k=1)[0]
        self.logger.info(f"Game {game_number} will use a search depth of {search_depth}")

        opening_index = 0
        opening = True

        while not self.game_over:
            player = self.players[self.current_turn]
            current_board = self.board.copy()

            # Try to use the next opening move, if available
            if opening and opening_index < len(opening_move_list):
                best_move = opening_move_list[opening_index]
                opening_index += 1
            else:
                self.logger.info(f"Opening line exhausted at move {move_count}. Switching to search.")
                opening = False
                best_move, policy_vector, root_value, simulation_count = player.get_move(current_board, move_count, search_depth, last_move)

            self.board.push(best_move)
            self.current_turn = not self.current_turn
            self.logger.info(f"Game {game_number} - Move {move_count}: {best_move.uci()}")
            last_move = best_move
            move_count += 1

            self._check_game_over(game_number)

        return self.result, move_count


    def update_players(self, player_white, player_black):
        """
        Updates white and black players
        """
        self.players = {
            chess.WHITE: player_white,
            chess.BLACK: player_black,
        }


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
            game.headers["White"] = self.players[chess.WHITE].name
            game.headers["Black"] = self.players[chess.BLACK].name

            exporter = chess.pgn.StringExporter(headers=True)
            pgn_string = game.accept(exporter).strip()


            self.logger.critical(f"Game PGN:\n{pgn_string}")