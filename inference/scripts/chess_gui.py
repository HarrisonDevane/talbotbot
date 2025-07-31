import tkinter as tk
from PIL import Image, ImageTk
import os
import chess
import logging

class ChessGUI:
    def __init__(self, root, logger: logging.Logger):
        self.root = root
        self.root.title("Chess GUI")

        self.window = root
        self.window.title("Talbot Chess")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.logger = logger

        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.controller = None

        self.images = {}
        self.squares = {}
        self.square_colors = {}

        self.load_images()
        self.draw_board()
        self.update_board()

    def set_controller(self, controller):
        self.controller = controller

    def load_images(self):
        pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
        colors = ['w', 'b']
        assets_path = os.path.join(os.path.dirname(__file__), '../../data', 'assets')

        for color in colors:
            for piece in pieces:
                filename = f"{color}{piece}.png"
                path = os.path.join(assets_path, filename)
                if os.path.exists(path):
                    img = Image.open(path).resize((96, 96))
                    self.images[color + piece] = ImageTk.PhotoImage(img)
                else:
                    self.logger.error(f"Missing image: {path}")

    def draw_board(self):
        for r in range(8):
            for c in range(8):
                color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
                self.square_colors[(r, c)] = color

                frame = tk.Frame(self.root, width=96, height=96)
                frame.grid(row=r, column=c)
                canvas = tk.Canvas(frame, bg=color, width=96, height=96, highlightthickness=0)
                canvas.pack()
                canvas.bind("<Button-1>", lambda e, row=r, col=c: self.on_left_click(row, col))
                self.squares[(r, c)] = canvas

    def on_left_click(self, row, col):
        if self.controller:
            square = chess.square(col, 7 - row)
            self.controller.handle_gui_click(square)

    def update_board(self, board=None, last_move=None, legal_moves=[], selected_square=None):
        if board:
            self.board = board
        self.last_move = last_move
        self.legal_targets = legal_moves
        self.selected_square = selected_square

        for (r, c), canvas in self.squares.items():
            canvas.delete("all")
            square = chess.square(c, 7 - r)

            # Highlight last move
            if self.last_move and (square == self.last_move.from_square or square == self.last_move.to_square):
                canvas.create_rectangle(0, 0, 96, 96, fill="#f6f669", outline="")

            # Highlight selected square
            if self.selected_square == square:
                canvas.create_rectangle(0, 0, 96, 96, fill="#829769", outline="")

            # Draw legal move targets
            if square in self.legal_targets:
                canvas.create_oval(38, 38, 58, 58, fill="#829769", outline="")

            # Draw piece
            piece = self.board.piece_at(square)
            if piece:
                color = 'w' if piece.color else 'b'
                symbol = piece.symbol().upper()
                img = self.images.get(color + symbol)
                if img:
                    canvas.create_image(48, 48, image=img)


    def on_close(self):
        if self.controller:
            self.controller.shutdown()
        self.window.quit()   # Stop the mainloop
        self.window.destroy()  # Close the window