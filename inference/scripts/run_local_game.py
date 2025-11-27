import tkinter as tk
import os
import logging
import sys
import yaml
import multiprocessing as mp 
import atexit 
import torch
import chess
import logging
import time
import threading

from datetime import datetime
from PIL import Image, ImageTk

current_script_dir = os.path.dirname(os.path.abspath(__file__))
inference_root = os.path.abspath(os.path.join(current_script_dir, ".."))
project_root = os.path.abspath(os.path.join(current_script_dir, "../.."))

sys.path.insert(0, project_root)
sys.path.insert(0, inference_root)

from src_shared.self_play_agent import SelfPlayAgent 
from src_shared.inference_batcher import InferenceBatcher 
import src_shared.utils as utils

class HumanPlayer:
    """Minimal class to satisfy the GameController's player interface for human moves."""
    def __init__(self, name="HumanPlayer"): 
        self.name = name
    def reset_for_new_game(self):
        return
    

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
        self.window.quit()
        self.window.destroy() 


class GameController:
    """
    A controller that runs a single game of chess using MCTS/SelfPlay logic,
    designed to run locally with a required GUI interface.
    """
    def __init__(self, logger: logging.Logger, white_player, black_player, evaluation_config: dict, gui, initial_fen=None):

        self.logger = logger
        self.evaluation_config = evaluation_config
        self.gui = gui
        
        self.initial_fen = initial_fen

        if self.initial_fen:
            self.board = chess.Board(initial_fen)
        else:
            self.board = chess.Board()

        self.players = {
            chess.WHITE: white_player,
            chess.BLACK: black_player,
        }

        # Game state variables
        self.game_over = False
        self.result = None
        self.current_turn = self.board.turn 

        # Shutdown control variables (FIXED ATTRIBUTE ERROR HERE)
        self.running = False           
        self._game_thread = None       

        # GUI-related state
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        
        self.gui.set_controller(self)


    def start_game(self):
        """
        Starts the game thread for automated/human play.
        """
        self.logger.critical(f"\n{'='*60}\n{' '*20}--- GAME STARTED ---\n{'='*60}\n")
        
        if self.initial_fen:
            self.board = chess.Board(self.initial_fen)
        else:
            self.board = chess.Board()
            
        self.game_over = False
        self.result = None
        self.last_move = None
        self.current_turn = self.board.turn
            
        self.players[chess.WHITE].reset_for_new_game()
        self.players[chess.BLACK].reset_for_new_game()
        
        fixed_simulations = self.evaluation_config['search_depth'] 
        self.logger.info(f"Game will use a fixed search depth of {fixed_simulations} simulations.")
        
        self.update_gui()
        self.running = True
        self._game_thread = threading.Thread(
            target=self._game_loop_logic, 
            args=(fixed_simulations,),
            daemon=True
        )
        self._game_thread.start()


    def _game_loop_logic(self, fixed_simulations):
        """Internal method containing the main game loop logic for automated play."""
        ply_count = 1
        
        while self.running and not self.game_over: 
            player = self.players[self.current_turn]
            
            if player.name == 'HumanPlayer': 
                time.sleep(0.1)
                continue 
            
            current_board = self.board.copy()
            
            move_info = player.get_move(current_board, ply_count, fixed_simulations, self.last_move)
            best_move, policy_vector, simulation_count = move_info 
            self.logger.info(f"Move {ply_count}: {best_move.uci()} ({simulation_count} sims)")

            self.make_move(best_move)
            ply_count += 1

            if self.game_over:
                break

        if not self.running:
             self.logger.info("Game loop exited due to shutdown request.")


    def make_move(self, move):
        """Processes a move and updates the board/state."""
        if move in self.board.legal_moves:
            self.last_move = move
            self.board.push(move)
            self.current_turn = not self.current_turn
            
            self.selected_square = None
            self.legal_targets = []
            
            self.update_gui() 
            self._check_game_over()
            
        else:
            self.logger.error(f"Attempted illegal move {move.uci()}! Game state error.")


    def handle_gui_click(self, square):
        """Handles a click event from the GUI."""
        
        is_human_turn = self.players[self.current_turn].name == 'HumanPlayer'
        
        if is_human_turn:
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
                    self.selected_square = None
                    self.legal_targets = []

            self.update_gui()


    def update_gui(self):
        """Updates the GUI display with the current board state."""
        self.gui.update_board(
            board=self.board,
            last_move=self.last_move,
            legal_moves=self.legal_targets,
            selected_square=self.selected_square
        )


    def _check_game_over(self):
        """Checks if the game has ended and logs the outcome."""
        if self.board.is_game_over(claim_draw=True):
            self.game_over = True
            self.result = self.board.result(claim_draw=True)

            if self.board.can_claim_threefold_repetition():
                self.logger.info("Game ended by threefold repetition claim.")
            elif self.board.can_claim_fifty_moves():
                self.logger.info("Game ended by 50-move rule claim.")
            else:
                self.logger.info(f"Game over. Result: {self.result}")

            self.shutdown()


    def shutdown(self):
        """
        Gracefully terminates the game thread.
        """
        self.logger.critical("\n--- SHUTDOWN INITIATED: Terminating GameController thread. ---\n")
        self.running = False # Set the flag to stop the while loop in _game_loop_logic
        
        # This check is now safe because self._game_thread is initialized to None in __init__
        if self._game_thread and self._game_thread.is_alive(): 
            # Wait for the thread to finish its current sleep/move and exit
            self._game_thread.join(timeout=1.0)
            if self._game_thread.is_alive():
                self.logger.warning("Game thread did not join gracefully within timeout.")
            else:
                self.logger.info("Game thread joined successfully.")


def main():
    mp.set_start_method('spawn', force=True) 

    full_config_path = os.path.join(inference_root, "config/local_config.yaml")
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    game_config = config['game']
    model_config = config['model']
    evaluation_config = config['evaluation'] 

    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs/local_inference", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("local_main")
    logger.setLevel(evaluation_config['worker_logging_level'])
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(log_dir, "main.log"), mode='w')
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.info("Starting local game environment.")
    
    white_player_type = game_config['white_player']
    black_player_type = game_config['black_player']
        
    inference_process = None
    
    # IPC components initialized to None
    inference_queue = None
    result_queue = None 
    shared_input_buffer = None
    shared_policy_buffer = None
    shared_value_buffer = None
    buffer_free_slots = None
    
    # Max batch size for a single worker
    max_batch_size = evaluation_config['batch_size_per_worker']
    
    # Shared Input Buffer (float32)
    shared_input_buffer = torch.zeros(
        max_batch_size, utils.INPUT_CHANNELS, utils.BOARD_DIM, utils.BOARD_DIM, dtype=torch.float32
    ).share_memory_()

    # Shared Policy Buffer (float16)
    shared_policy_buffer = torch.zeros(
        max_batch_size, utils.TOTAL_POLICY_MOVES, dtype=torch.float16
    ).share_memory_()
    
    # Shared Value Buffer (float32)
    shared_value_buffer = torch.zeros(
        max_batch_size, 1, dtype=torch.float32
    ).share_memory_()
    
    # Global Free Index Queue
    buffer_free_slots = mp.Queue()
    for i in range(max_batch_size):
        buffer_free_slots.put(i)
    
    # Inference Queues
    inference_queue = mp.Queue()
    result_queue = mp.Queue() 

    model_path = os.path.abspath(os.path.join(project_root, model_config['model_path']))
    inference_core_id = evaluation_config['inference_worker_cores'][0]
    
    inference_batcher = InferenceBatcher(
        name='',
        model_path=model_path,
        model_config=model_config,
        batch_size=max_batch_size, 
        batch_timeout=evaluation_config['batch_timeout'],
        log_dir=log_dir, 
        logging_level=evaluation_config['inference_logging_level']
    )
    
    # Batcher expects a list of result queues even if it's just one
    inference_process = mp.Process(target=inference_batcher.run, daemon=True, args=(
        inference_queue, 
        [result_queue], 
        inference_core_id,
        shared_input_buffer, 
        shared_policy_buffer, 
        shared_value_buffer)
    )
    inference_process.start()
    logger.info(f"Inference Batcher started on core {inference_core_id} (PID: {inference_process.pid})")

    worker_id = 0

    def initialize_player(player_type, color_name):
        nonlocal worker_id
        
        if player_type == "talbotbot":

            player = SelfPlayAgent(
                name=f"talbotbot-{color_name.lower()}",
                logger=logger,
                self_play_config=evaluation_config,
                worker_id=worker_id,
                inference_queue=inference_queue,
                result_queue=result_queue,
                shared_input_buffer=shared_input_buffer,
                shared_policy_buffer=shared_policy_buffer,
                shared_value_buffer=shared_value_buffer,
                buffer_free_slots=buffer_free_slots
            )
            worker_id += 1
            return player
        
        elif player_type == "human":
             return HumanPlayer()
        
        else:
            logger.critical(f"Unsupported player type '{player_type}' configured. Only 'talbotbot' and 'human' are supported. Exiting.")
            sys.exit(1)


    white_player = initialize_player(white_player_type, "White")
    black_player = initialize_player(black_player_type, "Black")

    # --- 4. GAME CONTROLLER AND EXECUTION ---
    root = tk.Tk()
    gui = ChessGUI(root, logger) 

    controller = GameController(
        logger=logger,
        white_player=white_player,
        black_player=black_player,
        evaluation_config=evaluation_config,
        gui=gui,
        initial_fen=game_config['initial_fen'],
    )

    gui.set_controller(controller)
    
    # Ensure the Inference Batcher is terminated when the script exits
    def shutdown_processes():
        if inference_process and inference_process.is_alive():
            logger.info("Terminating Inference Batcher process...")
            inference_process.terminate()
            inference_process.join()
            logger.info("Inference Batcher terminated.")
    
    atexit.register(shutdown_processes)

    controller.start_game()

    root.mainloop()

if __name__ == "__main__":
    main()