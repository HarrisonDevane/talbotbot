import chess
import chess.pgn
import logging
import random
import chess.polyglot

class SelfPlayGameWorker:
    """
    Controls a series of chess games between two automated players.
    """
    def __init__(self, logger: logging.Logger, player, config):
        self.logger = logger
        self.board = chess.Board()

        self.players = {
            chess.WHITE: player,
            chess.BLACK: player,
        }
        
        self.num_games = config['self_play']['num_games_total']
        self.search_depth = config['self_play']['search_depth']
        self.book_path = config['self_play']['opening_book_path']
        self.opening_min_moves = config['self_play']['opening_min_moves']
        self.opening_max_moves = config['self_play']['opening_max_moves']
        
        self.current_game = 1
        self.white_score = 0.0
        self.black_score = 0.0
        
        # State flags for a single game
        self.game_over = False
        self.current_turn = self.board.turn

    def start_game_series(self):
        """
        The main loop for a series of games.
        """
        self.logger.info(f"Starting a series of {self.num_games} games...")
        
        while self.current_game <= self.num_games:
            self.logger.info(f"--- Starting Game {self.current_game}/{self.num_games} ---")
            
            # Reset game-specific state
            self.board.reset()
            self.game_over = False
            self.current_turn = self.board.turn
            
            for player in self.players.values():
                player.reset_for_new_game()

            pgn = self.game_loop()

            if pgn:
                yield pgn

            self.current_game += 1

        self.logger.info("--- All games finished. ---")
        self.logger.info(f"Final Score - White: {self.white_score}, Black: {self.black_score}")


    def game_loop(self):
        """
        The main loop for a single game.
        This function runs until a game-ending condition is met.
        """
        move_count = 0
        # NEW: Randomly choose the number of moves to use the opening book
        max_opening_moves = random.randint(self.opening_min_moves, self.opening_max_moves)
        self.logger.info(f"Game {self.current_game} will use an opening book for the first {max_opening_moves} moves.")

        while not self.game_over:
            player = self.players[self.current_turn]
            move = None
            
            # NEW: Check if we should use the opening book
            if self.book_path and move_count < max_opening_moves:
                try:
                    with chess.polyglot.open_reader(self.book_path) as reader:
                        # Get a weighted random move from the book
                        book_move = reader.weighted_choice(self.board).move
                        move = book_move
                        self.logger.info(f"Game {self.current_game} - Book move selected: {move.uci()}")
                except IndexError:
                    # Position is not in the book; fall back to the player
                    self.logger.warning(f"Game {self.current_game} - Position not in book. Falling back to MCTS.")
                    move = player.get_move(self.board, move_count, self.search_depth)
                    max_opening_moves = 0
            
            # If a move was not selected from the book, get it from the player
            if move is None:
                move = player.get_move(self.board, move_count, self.search_depth)
            
            # Make the move and switch turns
            self.board.push(move)
            self.current_turn = not self.current_turn
            self.logger.info(f"Game {self.current_game} - Move made: {move.uci()}")
            move_count += 1
            
            # Check if the game has ended after the move
            self.check_game_over()
        
        game = chess.pgn.Game.from_board(self.board)
        exporter = chess.pgn.StringExporter(headers=False)
        pgn_string = game.accept(exporter)
        pgn_string = pgn_string.strip() + " " + self.board.result()

        return pgn_string

    def check_game_over(self):
        """
        Checks if the game has ended and handles the outcome.
        """
        if self.board.is_game_over():
            self.game_over = True
            result = self.board.result()
            self.logger.info(f"Game {self.current_game} over. Result: {result}")
            game = chess.pgn.Game.from_board(self.board)

            self.logger.info("--- Game PGN ---")
            self.logger.info(str(game))

            # Update scores based on the result
            if result == "1-0":
                self.white_score += 1
            elif result == "0-1":
                self.black_score += 1
            elif result == "1/2-1/2":
                self.white_score += 0.5
                self.black_score += 0.5

            self.logger.info(f"Current Score - White: {self.white_score}, Black: {self.black_score}")