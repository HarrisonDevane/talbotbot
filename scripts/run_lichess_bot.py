import os
import sys
import logging
import yaml
import berserk
import chess
from datetime import datetime
import requests.exceptions
import threading
import time
import random

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, "..")

src_dir = os.path.join(project_root, "src")
config_dir = os.path.join(project_root, "config")
sys.path.insert(0, src_dir)
sys.path.insert(0, config_dir)

from players import TalbotPlayer

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs/lichess_inference"))
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"lichess_bot_run_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='w'
)

logger = logging.getLogger(__name__)

class LichessBot:
    def __init__(self, lichess_config: dict, time_per_move: float, talbot_player_config: dict):
        self.lichess_config = lichess_config
        self.time_per_move = time_per_move
        self.challenge_interval_seconds = 30
        self.talbot_player_config = talbot_player_config 

        self.session = berserk.TokenSession(lichess_config["api_token"])
        self.client = berserk.Client(session=self.session)

        self.game_threads = {}
        self.stop_event = threading.Event()

        logger.debug(f"Lichess Bot '{lichess_config['bot_id']}' initialized.")

    def _challenge_loop(self):
        """
        Runs in a separate thread to periodically attempt to challenge specific bots.
        """
        logger.debug("Challenge thread started.")
        while not self.stop_event.is_set():
            # Only attempt to challenge if the bot is not currently in any game
            if not self.game_threads:
                logger.info("Challenge thread: No active games. Attempting periodic challenge...")
                self._challenge_specific_bots()
            else:
                logger.debug(f"Challenge thread: Bot is currently in {len(self.game_threads)} game(s). Skipping challenge attempt.")

            self.stop_event.wait(self.challenge_interval_seconds)
        logger.debug("Challenge thread stopped.")


    def run(self):
        logger.debug("Connecting to Lichess API (main event stream)...")

        self.challenge_thread = threading.Thread(target=self._challenge_loop, name="ChallengeThread")
        self.challenge_thread.daemon = True
        self.challenge_thread.start()

        event_stream = None
        while not self.stop_event.is_set():
            if event_stream is None:
                try:
                    event_stream = self.client.bots.stream_incoming_events()
                    logger.info("Successfully connected to Lichess main event stream.")
                except requests.exceptions.RequestException as e:
                    logger.critical(f"Initial Lichess API connection error (main stream): {e}. Retrying in 5 seconds...", exc_info=True)
                    time.sleep(5)
                    continue
                except Exception as e:
                    logger.critical(f"Unexpected error during initial main event stream connection: {e}. Retrying in 5 seconds...", exc_info=True)
                    time.sleep(5)
                    continue

            try:
                event = next(event_stream)
            except StopIteration:
                logger.info("Lichess main event stream ended. Reconnecting...")
                event_stream = None
                continue
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
                logger.warning(f"Lichess main event stream disconnected ({type(e).__name__}). Attempting to reconnect in 5 seconds...", exc_info=True)
                event_stream = None
                time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Unexpected error reading from main event stream: {e}. Reconnecting in 5 seconds...", exc_info=True)
                event_stream = None
                time.sleep(5)
                continue

            event_type = event["type"]
            logger.debug(f"Received main event: {event}")

            if event_type == "challenge":
                self._handle_challenge(event["challenge"])
            elif event_type == "gameStart":
                game_id = event["game"]["id"]
                if game_id not in self.game_threads:
                    logger.debug(f"Starting new game thread for game {game_id}")
                    self.game_threads[game_id] = {
                        "thread": None,
                        "board": None,
                        "moves_on_board": 0,
                        "initial_game_data": event["game"],
                        "wtime_ms": 0, "btime_ms": 0,
                        "winc_ms": 0, "binc_ms": 0,
                        "last_state_update_time": 0
                    }
                    game_thread = threading.Thread(target=self._run_game_loop, args=(game_id, event["game"]))
                    self.game_threads[game_id]["thread"] = game_thread
                    game_thread.start()
                else:
                    logger.warning(f"Game {game_id} already has a running thread. Ignoring duplicate gameStart event.")
            elif event_type == "gameFinish":
                self._handle_game_finish(game_id, event["game"].get("status", "unknown_status"))
            elif event_type == "chatLine":
                logger.debug(f"Received main stream chat: {event['username']}: {event['text']}")
            elif event_type == "gameFull":
                game_state = event.get("state")
                if game_state and game_state.get("type") == "gameState":
                    logger.debug(f"Processing embedded gameState from gameFull event for game {game_id}")
                    if game_id in self.game_threads:
                        self._handle_game_state(
                            game_id,
                            game_state["moves"],
                            game_state["wtime"],
                            game_state["btime"],
                            game_state["winc"],
                            game_state["binc"],
                            game_state["status"],
                            time.time()
                        )
                    else:
                        logger.warning(f"Received gameFull event with state for unknown/inactive game {game_id}. Ignoring.")
                else:
                    logger.debug(f"Received gameFull event for {game_id} without a valid gameState. Ignoring.")
            else:
                logger.debug(f"Unhandled main event type: {event_type}")

        self._shutdown_bot()


    def _run_game_loop(self, game_id, initial_game_data):
        """
        Runs the dedicated stream for a single game, maintaining board state incrementally.
        This is where the TalbotPlayer instance for *this specific game* is created.
        """
        logger.debug(f"Game loop started for game {game_id}.")

        player_instance = TalbotPlayer(
            name=self.lichess_config["bot_id"] + "_" + game_id[:4],
            model_path=self.talbot_player_config.get("model_path"),
            num_residual_blocks=self.talbot_player_config.get("resblocks"),
            cpuct=self.talbot_player_config.get("cpuct"),
            batch_size=self.talbot_player_config.get("batchsize"),
        )
        self.game_threads[game_id]["player_instance"] = player_instance

        try:
            # Apply retry wrapper
            full_game_details = self._make_api_call_with_retries(self.client.games.export, game_id)
            initial_fen = full_game_details.get('initialFen', chess.STARTING_FEN)
            moves_string = full_game_details.get('moves', '')

            current_board = chess.Board(initial_fen)
            moves_on_board = 0

            for move_str in moves_string.split():
                if not move_str:
                    continue
                move = current_board.parse_san(move_str)
                current_board.push(move)
                moves_on_board += 1


            self.game_threads[game_id]["board"] = current_board
            self.game_threads[game_id]["moves_on_board"] = moves_on_board
            logger.debug(f"Game {game_id} board initialized. Current FEN: {current_board.fen()} with {moves_on_board} moves.")

        except (requests.exceptions.RequestException, ValueError, chess.IllegalMoveError, Exception) as e:
            logger.critical(f"FATAL: Initial board setup or game export failed for game {game_id}. Error: {e}. Resigning.", exc_info=True)
            self._try_resign_game(game_id)
            return

        bot_color_str = initial_game_data["color"]
        bot_is_white = (bot_color_str == 'white')

        self.game_threads[game_id]["bot_color_str"] = bot_color_str

        if (current_board.turn == chess.WHITE and bot_is_white) or \
           (current_board.turn == chess.BLACK and not bot_is_white):
            logger.debug(f"It's my turn for the initial move in game {game_id}. Thinking of a move...")
            turn_start_time = time.time()
            self.make_bot_move(game_id, turn_start_time) # Call make_bot_move without passing player_instance
        else:
            logger.debug(f"Game {game_id}: Not my turn for the initial move. Waiting for opponent's move.")

        try:
            # Use the game_id to stream only events for this specific game
            game_stream = self.client.bots.stream_game_state(game_id)
            for event in game_stream:
                if self.stop_event.is_set():
                    logger.debug(f"Stop event detected for game {game_id}. Exiting game loop.")
                    break

                event_type = event["type"]
                logger.debug(f"Received game {game_id} event: {event}")

                if event_type == "gameState":
                    self._handle_game_state(
                        game_id,
                        event.get("moves", ""),
                        event.get("wtime"),
                        event.get("btime"),
                        event.get("winc"),
                        event.get("binc"),
                        event.get("status"),
                        time.time()
                    )
                elif event_type == "chatLine":
                    if hasattr(self, '_handle_chat_line') and callable(self._handle_chat_line):
                        self._handle_chat_line(event)
                elif event_type == "opponentGone":
                    logger.warning(f"Opponent left in game {game_id}! Graceful handling not yet implemented.")
                elif event_type == "gameFull":
                    game_state_data_from_full = event.get("state")
                    if game_state_data_from_full:
                        logger.debug(f"Received gameFull event for {game_id}. Processing full state update.")
                        self._handle_game_state(
                            game_id,
                            game_state_data_from_full.get("moves", ""),
                            game_state_data_from_full.get("wtime"),
                            game_state_data_from_full.get("btime"),
                            game_state_data_from_full.get("winc"),
                            game_state_data_from_full.get("binc"),
                            game_state_data_from_full.get("status"),
                            time.time()
                        )
                    else:
                        logger.warning(f"GameFull event for {game_id} received but missing 'state' data. Skipping board update.")
                else:
                    logger.debug(f"Unhandled game {game_id} event type: {event_type}")

        except (requests.exceptions.RequestException, Exception) as e:
            logger.error(f"Lichess API connection or unexpected error in game {game_id} stream: {e}", exc_info=True)
            logger.warning(f"Game {game_id} stream disconnected. Attempting to rejoin/clean up.")
        finally:
            if game_id in self.game_threads:
                del self.game_threads[game_id]["player_instance"]
                self.game_threads[game_id]["thread"] = None
                self.game_threads[game_id]["board"] = None
                self.game_threads[game_id]["moves_on_board"] = 0
                self.game_threads[game_id]["wtime_ms"] = 0
                self.game_threads[game_id]["btime_ms"] = 0
                self.game_threads[game_id]["winc_ms"] = 0
                self.game_threads[game_id]["binc_ms"] = 0
                self.game_threads[game_id]["last_state_update_time"] = 0
                if "bot_color_str" in self.game_threads[game_id]:
                    del self.game_threads[game_id]["bot_color_str"]
            logger.debug(f"Game loop for {game_id} finished.")


    def _shutdown_bot(self):
        """Gracefully shuts down all game threads and the player, including the challenge thread."""
        logger.debug("Setting stop event for all threads...")
        self.stop_event.set() # Signal all threads to stop

        # First, join the challenge thread to ensure it cleans up
        if self.challenge_thread and self.challenge_thread.is_alive():
            logger.debug("Waiting for challenge thread to finish...")
            self.challenge_thread.join(timeout=5) # Give it 5 seconds to finish
            if self.challenge_thread.is_alive():
                logger.warning("Challenge thread did not terminate gracefully within timeout.")

        # Then, join individual game threads
        for game_id, data in list(self.game_threads.items()):
            thread = data["thread"]
            if thread and thread.is_alive(): # Check if thread object exists and is alive
                logger.debug(f"Waiting for game thread {game_id} to finish...")
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning(f"Game thread {game_id} did not terminate gracefully within timeout.")
            # Clear game data regardless of thread termination success
            del self.game_threads[game_id]

        logger.debug("Lichess Bot completely shut down.")


    def _handle_challenge(self, challenge_data):

        
        challenge_id = challenge_data["id"]
        challenger_id = challenge_data['challenger']['id'].lower()

        if challenger_id == self.lichess_config['bot_id'].lower():
            return

        variant = challenge_data["variant"]["key"].lower()
        rated = challenge_data["rated"]

        time_control_data = challenge_data["timeControl"]

        challenge_time_initial = time_control_data.get("limit")
        challenge_time_min = 0
        if challenge_time_initial is not None:
            try:
                challenge_time_min = int(challenge_time_initial) / 60
            except ValueError:
                logger.warning(f"Could not parse initial time '{challenge_time_initial}' for challenge {challenge_id}. Defaulting to 0.")

        challenge_increment_sec = time_control_data.get("increment", 0)

        logger.debug(f"Received challenge {challenge_id} from {challenger_id}: Variant={variant}, Rated={rated}, Time={challenge_time_min:.1f}min+{challenge_increment_sec}s")


        accept = True
        decline_reason = []

        accepted_variant = self.lichess_config.get("accept_variant", "standard").lower()
        min_time_min_config = self.lichess_config.get("challenge_time_min", 0)
        min_increment_sec_config = self.lichess_config.get("challenge_increment_sec", 0)

        if accepted_variant != "any" and variant != accepted_variant:
            accept = False
            decline_reason.append(f"variant '{variant}' (only '{accepted_variant}' accepted)")

        if challenge_time_min < min_time_min_config:
            accept = False
            decline_reason.append(f"main time {challenge_time_min:.1f}min (min {min_time_min_config}min required)")

        if challenge_increment_sec < min_increment_sec_config:
            accept = False
            decline_reason.append(f"increment {challenge_increment_sec}s (min {min_increment_sec_config}s required)")

        if len(self.game_threads) > 0:
            accept = False
            decline_reason.append(f"bot is already in {len(self.game_threads)} game(s).")
            logger.info(f"Declining challenge {challenge_id} from {challenger_id} because bot is already busy.")


        try:
            if accept:
                self.client.bots.accept_challenge(challenge_id)
                logger.debug(f"Accepted challenge {challenge_id}.")
            else:
                reason_str = ", ".join(decline_reason)
                logger.debug(f"Declining challenge {challenge_id} due to criteria mismatch: {reason_str}.")
                self.client.bots.decline_challenge(challenge_id)
        except berserk.exceptions.ResponseError as e:
            logger.error(f"Error processing challenge {challenge_id} (accept/decline): {e}", exc_info=True)


    def _handle_game_state(self, game_id, moves_str, wtime, btime, winc, binc, status_data, event_receive_time):
        game_data = self.game_threads.get(game_id)
        if not game_data or game_data["board"] is None:
            logger.error(f"Board not found for game {game_id} in _handle_game_state. Cannot process game state. Resigning.")

            if hasattr(self, '_try_resign_game') and callable(self._try_resign_game):
                self._try_resign_game(game_id)
            else:
                logger.warning(f"Cannot resign game {game_id}: _try_resign_game method not found.")
            return

        current_board = game_data["board"]
        moves_on_board = game_data["moves_on_board"]

        # Ensure times are integers (milliseconds), handling potential datetime objects if they occur
        game_data["wtime_ms"] = int(wtime.timestamp() * 1000) if isinstance(wtime, datetime) else wtime
        game_data["btime_ms"] = int(btime.timestamp() * 1000) if isinstance(btime, datetime) else btime
        game_data["winc_ms"] = int(winc.timestamp() * 1000) if isinstance(winc, datetime) else winc
        game_data["binc_ms"] = int(binc.timestamp() * 1000) if isinstance(binc, datetime) else binc
        game_data["last_state_update_time"] = event_receive_time

        game_status_name = None
        if isinstance(status_data, dict):
            game_status_name = status_data.get("name")
        elif isinstance(status_data, str):
            game_status_name = status_data
        else:
            logger.warning(f"Unexpected type for status_data in game {game_id}: {type(status_data)}. Data: {status_data}")
            return

        if game_status_name != "started":
            logger.debug(f"Game {game_id} status: {game_status_name}. No move needed.")
            return

        incoming_moves = moves_str.split()
        num_incoming_moves = len(incoming_moves)

        if num_incoming_moves < moves_on_board:
            logger.critical(f"FATAL: Lichess sent fewer moves ({num_incoming_moves}) than already on board ({moves_on_board}) for game {game_id}. Desync detected! Resigning.")
            if hasattr(self, '_try_resign_game') and callable(self._try_resign_game):
                self._try_resign_game(game_id)
            return
        elif num_incoming_moves == moves_on_board:
            logger.debug(f"Game {game_id}: No new moves to apply. Current FEN: {current_board.fen()}")
        else:
            new_moves_to_apply = incoming_moves[moves_on_board:]
            logger.debug(f"Game {game_id}: Applying {len(new_moves_to_apply)} new moves.")
            try:
                for move_str in new_moves_to_apply:
                    if not move_str: continue

                    move = current_board.parse_san(move_str)
                    current_board.push(move)
                    moves_on_board += 1
                    logger.debug(f"Applied new move: {move_str}. Current FEN: {current_board.fen()}")

                game_data["moves_on_board"] = moves_on_board
            except Exception as e:
                logger.critical(f"FATAL: Unexpected error applying new moves for game {game_id}. Error: {e}. Board will be desynced! Resigning.", exc_info=True)
                if hasattr(self, '_try_resign_game') and callable(self._try_resign_game):
                    self._try_resign_game(game_id)
                return

        logger.debug(f"Game {game_id} state updated. Current FEN: {current_board.fen()}")

        is_my_turn = False
        bot_color_in_game = None

        initial_game_data = game_data.get("initial_game_data")
        if initial_game_data:
            bot_color_str = initial_game_data["color"]
            bot_color_in_game = chess.WHITE if bot_color_str == 'white' else chess.BLACK
        else:
            # This should ideally not happen if initial_game_data is always stored on gameStart
            logger.error(f"Initial game data not found for game {game_id}. Cannot determine bot's color.")

        if bot_color_in_game is not None:
            is_my_turn = (current_board.turn == bot_color_in_game)
        else:
            logger.error(f"Could not determine bot's color in game {game_id} from initial game data. Cannot make a move.")
            return

        if is_my_turn:
            logger.debug(f"It's my turn in game {game_id}. Thinking of a move...")
            # Call make_bot_move without passing player_instance; it will retrieve it from game_data
            self.make_bot_move(game_id, event_receive_time)
        else:
            logger.debug(f"Game {game_id}: Not my turn.")

    def _calculate_thinking_time(self, game_id):
        """
        Calculates the thinking time for the AI based on current game state and time control.
        Prioritizes increment, uses faster thinking for early moves, and a percentage of
        remaining time for later moves, all capped by self.time_per_move.
        Uses locally stored game state data instead of an API export.
        """
        game_data = self.game_threads.get(game_id)
        if not game_data:
            logger.error(f"Game data not found for {game_id}. Cannot calculate thinking time. Falling back to default.")
            return self.time_per_move

        initial_game_data = game_data.get("initial_game_data")
        if not initial_game_data:
            logger.error(f"Initial game data not found for {game_id}. Cannot determine bot color for time calculation. Falling back to default.")
            return self.time_per_move

        bot_color_str = initial_game_data["color"]
        bot_color = chess.WHITE if bot_color_str == 'white' else chess.BLACK

        last_state_update_time = game_data.get("last_state_update_time", 0)
        time_elapsed_since_event = time.time() - last_state_update_time

        if bot_color == chess.WHITE:
            remaining_time_ms = game_data.get('wtime_ms', 0)
            increment_ms = game_data.get('winc_ms', 0)
        else:
            remaining_time_ms = game_data.get('btime_ms', 0)
            increment_ms = game_data.get('binc_ms', 0)

        effective_remaining_time_ms = max(0, remaining_time_ms - int(time_elapsed_since_event * 1000))

        remaining_time_sec = effective_remaining_time_ms / 1000.0
        increment_sec = increment_ms / 1000.0

        move_number = game_data.get("moves_on_board", 0)

        EARLY_GAME_MOVES_THRESHOLD = 10
        EARLY_GAME_THINK_TIME_FACTOR = 0.5
        MID_LATE_GAME_TIME_PERCENTAGE = 0.02

        calculated_time = increment_sec

        if move_number <= EARLY_GAME_MOVES_THRESHOLD:
            early_game_time_budget = self.time_per_move * EARLY_GAME_THINK_TIME_FACTOR
            calculated_time = min(remaining_time_sec, max(increment_sec, early_game_time_budget))
            logger.debug(f"Game {game_id} (Move {move_number}): Early game strategy. Base calculated time: {calculated_time:.2f}s")
        else:
            time_from_percentage = remaining_time_sec * MID_LATE_GAME_TIME_PERCENTAGE
            calculated_time = max(increment_sec, time_from_percentage)
            logger.debug(f"Game {game_id} (Move {move_number}): Mid/Late game strategy. Base calculated time: {calculated_time:.2f}s")

        final_thinking_time = min(calculated_time, self.time_per_move)

        logger.info(f"Game {game_id}: Effective Remaining clock: {remaining_time_sec:.2f}s, Increment: {increment_sec:.2f}s, Move #: {move_number}. Calculated thinking time: {final_thinking_time:.4f}s")
        return final_thinking_time
    

    def _make_api_call_with_retries(self, api_call_func, *args, **kwargs):
        """
        Wrapper to make Lichess API calls with retry logic for connection errors.
        This method will retry on requests.exceptions.ConnectionError and berserk.exceptions.ApiError.
        """
        max_retries = 3
        initial_delay_seconds = 1
        for attempt in range(1, max_retries + 1):
            try:
                result = api_call_func(*args, **kwargs)
                return result # Success!
            except (requests.exceptions.ConnectionError, berserk.exceptions.ApiError) as e:
                logger.warning(f"Attempt {attempt}/{max_retries}: Connection error during API call. Error: {e}")
                if attempt < max_retries:
                    delay = initial_delay_seconds * (2 ** (attempt - 1)) # Exponential backoff
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts failed for API call. Giving up.")
                    raise # Re-raise the exception if all retries fail
            except Exception as e:
                # Catch any other unexpected exceptions and re-raise immediately
                logger.critical(f"An unexpected non-retryable error occurred during API call: {e}", exc_info=True)
                raise # Re-raise immediately for unhandled exceptions

    def make_bot_move(self, game_id, turn_start_time):
        game_data = self.game_threads.get(game_id)
        if not game_data or game_data["board"] is None:
            logger.error(f"Attempted to make a move for game {game_id} but no board found. Resigning.")
            self._try_resign_game(game_id)
            return

        current_board = game_data["board"]

        try:
            time_for_ai = self._calculate_thinking_time(game_id)
            time_spent_since_turn_start = time.time() - turn_start_time
            effective_time_for_ai = max(1, time_for_ai - time_spent_since_turn_start)

            player_instance = self.game_threads[game_id]["player_instance"]
            best_move_obj = player_instance.get_move(current_board, effective_time_for_ai)
            best_move_uci = best_move_obj.uci()

            logger.debug(f"Game {game_id}: Attempting to make move {best_move_uci}")

            self._make_api_call_with_retries(self.client.bots.make_move, game_id, best_move_uci)
            logger.info(f"Successfully sent move {best_move_uci} for game {game_id}.")
            current_board.push_uci(best_move_uci)
            game_data["moves_on_board"] += 1

            if game_data and game_data.get('game_over'):
                logger.warning(f"[{self.name}] Game {game_data.get('game_id', 'N/A')} ended during MCTS calculation. Discarding calculated move: {best_move_uci}.")
                return None

        except berserk.exceptions.ResponseError as e:
            error_message = str(e).lower()
            if "not your turn" in error_message or "too fast" in error_message or "already played" in error_message or "game already over" in error_message:
                logger.warning(f"Lichess reported '{e.message}' for game {game_id}. This indicates a potential desync. Attempting to resync board and re-evaluate turn.")

                try:
                    full_game_details = self._make_api_call_with_retries(self.client.games.export, game_id)

                    if not isinstance(full_game_details, dict):
                        logger.critical(f"FATAL: Lichess game export for {game_id} during error resync returned unexpected type '{type(full_game_details).__name__}'. Content snippet: {str(full_game_details)[:200]}. Resigning.")
                        self._try_resign_game(game_id)
                        return

                    initial_fen = full_game_details.get('initialFen', chess.STARTING_FEN)
                    moves_string = full_game_details.get('moves', '')

                    temp_board = chess.Board(initial_fen)
                    temp_moves_on_board = 0
                    for move_str in moves_string.split():
                        if not move_str: continue
                        move = temp_board.parse_san(move_str) # <<< NOT CHANGED THIS LINE
                        temp_board.push(move)
                        temp_moves_on_board += 1

                    game_data["board"] = temp_board
                    game_data["moves_on_board"] = temp_moves_on_board
                    current_board = temp_board
                    logger.info(f"Game {game_id} board successfully resynced after move error. New FEN: {current_board.fen()}")

                    bot_color_in_game = None
                    white_player_id = full_game_details.get('players', {}).get('white', {}).get('user', {}).get('id', '').lower()
                    black_player_id = full_game_details.get('players', {}).get('black', {}).get('user', {}).get('id', '').lower()

                    if white_player_id == self.lichess_config['bot_id'].lower():
                        bot_color_in_game = chess.WHITE
                    elif black_player_id == self.lichess_config['bot_id'].lower():
                        bot_color_in_game = chess.BLACK

                    if bot_color_in_game is not None and current_board.turn == bot_color_in_game:
                        logger.info(f"After resync, it IS my turn in game {game_id}. Recalculating move...")
                        self.make_bot_move(game_id, turn_start_time)
                    else:
                        logger.info(f"After resync, it is NOT my turn or game is over for {game_id}. Status: {full_game_details.get('status', 'unknown')}. Not making a move.")

                except (requests.exceptions.RequestException, berserk.exceptions.ApiError, ValueError, chess.IllegalMoveError, Exception) as e_resync:
                    logger.critical(f"FATAL: Resync attempt failed for game {game_id} due to error: {e_resync}. Cannot recover. Resigning.", exc_info=True)
                    self._try_resign_game(game_id)
            else:
                logger.error(f"Unhandled Lichess API error on move for game {game_id}: {e}. Resigning.", exc_info=True)
                self._try_resign_game(game_id)
        except Exception as e:
            logger.critical(f"Unexpected error during move generation or sending for game {game_id}: {e}", exc_info=True)
            self._try_resign_game(game_id)


    def _try_resign_game(self, game_id):
        try:
            self.client.bots.resign_game(game_id)
            logger.info(f"Successfully sent resign command for game {game_id}.")
        except Exception as e:
            logger.error(f"Unexpected error during resignation attempt for game {game_id}: {e}", exc_info=True)


    def _handle_game_finish(self, game_id, status_data=None):

        """
        Handles the final cleanup for a game. This should be the single point
        where a game's entry is removed from self.game_threads.
        """

        status_name = status_data.get("name") if isinstance(status_data, dict) else status_data
        if status_name:
            logger.info(f"Game {game_id} finished with status: {status_name}")

        del self.game_threads[game_id]
        logger.info(f"Cleaned up and removed game {game_id} from active games. Active games: {len(self.game_threads)}")


    def _handle_chat_line(self, chat_data):
        game_id = chat_data.get("gameId", "N/A")
        username = chat_data["username"]
        text = chat_data["text"]
        logger.debug(f"Chat in game {game_id} from {username}: {text}")

        

    def _challenge_specific_bots(self):
        """
        Challenges a predefined list of Lichess bots within the 1500-2000 Elo range.
        This function should be called when your bot is not currently in a game.
        """
        # List of bot IDs to challenge. These are generally active and within the target Elo range.
        # You can find more at https://lichess.org/player/bots
        TARGET_BOT_IDS = [
            "maia5",
            "maia9",
            "turkjs",
            "sargon-1ply",
            "jangine",
            "Demolito_L3",
            "melsh_bot",
            "LeelaRogue",
            "AshNostromo",
            "turochamp-1ply",
            "sargon-3ply",
            "mochi_bot",
            "littlePatricia",
            "ScalaQueen",
            "fathzer-jchess",
            "LazyBotJr",
            "ColossusBOT",
            "plumbot"
        ]

        time_limit_seconds = self.lichess_config.get("challenge_time_min", 5) * 60 
        increment_seconds = self.lichess_config.get("challenge_increment_sec", 3)

        opponent_id = random.choice(TARGET_BOT_IDS)

        if len(self.game_threads) > 0:
            logger.info(f"Bot is currently in {len(self.game_threads)} game(s). Skipping challenge to {opponent_id}.")
            return 

        logger.info(f"Attempting to challenge bot: {opponent_id}")

        challenge = self.client.challenges.create(
            opponent_id,
            rated=True,
            variant='standard',
            clock_limit=time_limit_seconds,
            clock_increment=increment_seconds
        )
            
        logger.info(f"Successfully sent challenge to {opponent_id}. Challenge ID: {challenge['id']}")


if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(config_dir, "lichess_config.yaml"))

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config["lichess"]["api_token"] = os.getenv("LICHESS_API_TOKEN")

    if not config["lichess"]["api_token"]:
        logger.critical("LICHESS_API_TOKEN environment variable is not set. Please set it before running the bot.")
        sys.exit(1)

    talbot_config = config.get("talbot", {})
    model_path = talbot_config.get("model_path")
    num_residual_blocks = talbot_config.get("resblocks")
    cpuct = talbot_config.get("cpuct")
    batch_size = talbot_config.get("batchsize")
    time_per_move = talbot_config.get('time_per_move')
    

    bot = LichessBot(config["lichess"], time_per_move, talbot_config)
    bot.run()
