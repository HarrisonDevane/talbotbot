import chess
import threading
import time

class GameController:
    def __init__(self, white_player, black_player, num_games, gui, initial_fen=None):
        self.initial_fen = initial_fen
        if self.initial_fen:
            self.board = chess.Board(initial_fen)
        else:
            self.board = chess.Board()

        self.players = {
            chess.WHITE: white_player,
            chess.BLACK: black_player,
        }
        self.gui = gui
        
        # Determine current turn based on FEN, or default to WHITE if no FEN
        self.current_turn = self.board.turn 
        
        self.game_over = False

        self.num_games = num_games
        self.current_game = 1
        self.white_score = 0
        self.black_score = 0

        self.selected_square = None
        self.legal_targets = []
        self.last_move = None

        if self.gui:
            self.gui.set_controller(self)

    def start_game(self):
        print(f"Starting game {self.current_game}")
        # self.game_over = False
        # self.current_turn = chess.WHITE
        # self.board.reset()
        # self.selected_square = None
        # self.legal_targets = []
        # self.last_move = None
        self.update_gui()

        if self.gui:
            threading.Thread(target=self.game_loop, daemon=True).start()
        else:
            self.game_loop()



    def game_loop(self):
        while not self.game_over:
            player = self.players[self.current_turn]

            if player.is_human():
                time.sleep(0.1)
            else:
                move = player.get_move(self.board)
                if move and move in self.board.legal_moves:
                    self.make_move(move)
                else:
                    print("Invalid engine move or no move received")
                    self.game_over = True

    def make_move(self, move):
        if move in self.board.legal_moves:
            self.last_move = move
            self.board.push(move)
            self.selected_square = None
            self.legal_targets = []
            self.current_turn = not self.current_turn
            self.update_gui()
            self.check_game_over()
        else:
            print("Illegal move attempted:", move)

    def handle_gui_click(self, square):
        if (self.current_turn == chess.WHITE and self.players[chess.WHITE].is_human()) or \
           (self.current_turn == chess.BLACK and self.players[chess.BLACK].is_human()):

            if self.selected_square is None:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.current_turn:
                    self.selected_square = square
                    self.legal_targets = [
                        move.to_square for move in self.board.legal_moves
                        if move.from_square == square
                    ]
                else:
                    self.selected_square = None
                    self.legal_targets = []
            else:
                move = chess.Move(self.selected_square, square)
                if move in self.board.legal_moves:
                    self.make_move(move)
                else:
                    print("Invalid move")
                    self.selected_square = None
                    self.legal_targets = []

        self.update_gui()

    def update_gui(self):
        if self.gui:
            self.gui.update_board(
                board=self.board,
                last_move=self.last_move,
                legal_moves=self.legal_targets,
                selected_square=self.selected_square
            )

    def check_game_over(self):
        if self.board.is_game_over():
            self.game_over = True
            result = self.board.result()
            print("Game over:", result)

            if result == "1-0":
                self.white_score += 1
            elif result == "0-1":
                self.black_score += 1
            elif result == "1/2-1/2":
                self.white_score += 0.5
                self.black_score += 0.5

            print(f"Score — White: {self.white_score}, Black: {self.black_score}")

            if self.current_game < self.num_games:
                self.current_game += 1
                if self.gui:
                    threading.Timer(1.0, self.start_game).start()
                else:
                    self.start_game()

    def shutdown(self):
        self.game_over = True
        if hasattr(self.players[chess.WHITE], "engine"):
            self.players[chess.WHITE].engine.quit()
        if hasattr(self.players[chess.BLACK], "engine"):
            self.players[chess.BLACK].engine.quit()