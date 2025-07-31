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
import requests
from bs4 import BeautifulSoup

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, "..")

src_dir = os.path.join(project_root, "src")
config_dir = os.path.join(project_root, "config")
sys.path.insert(0, src_dir)
sys.path.insert(0, config_dir)

from players import TalbotPlayer


class DummyLogger:
    def debug(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def critical(self, *args, **kwargs): pass
    def exception(self, *args, **kwargs): pass


class LichessBot:
    def __init__(self, lichess_config: dict, talbot_player_config: dict, timing_config:dict, challenge_in_config: dict, challenge_out_config: dict, logging_config: dict):
        self.lichess_config = lichess_config
        self.talbot_player_config = talbot_player_config
        self.timing_config = timing_config
        self.challenge_in_config = challenge_in_config
        self.challenge_out_config = challenge_out_config
        self.logging_config = logging_config
        self.bots_to_challenge = None
        self.challenge_thread = None

        self.session = berserk.TokenSession(lichess_config["api_token"])
        self.client = berserk.Client(session=self.session)

        self.games_played = 0
        self.game_threads = {}
        self.stop_event = threading.Event()

        # Set up logging dir + lichess logging
        self.log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs/lichess_inference", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.getLogger("lichess_main")
        self.logger.setLevel(self.logging_config['lichess_logging_level'])

        if not self.logger.handlers:
            handler = logging.FileHandler(os.path.join(self.log_dir, "main.log"), mode='w')
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)



    def _setup_game_logger(self, game_id):
        logger = logging.getLogger(f"{game_id}")
        logger.setLevel(self.logging_config['game_logging_level'])

        if not logger.handlers:
            handler = logging.FileHandler(os.path.join(self.log_dir, f"game_{self.games_played}_{game_id}.log"), mode='w')
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
    

    def _challenge_loop(self):
        """
        Runs in a separate thread to periodically attempt to challenge specific bots.
        """
        self.logger.debug("Challenge thread started.")
        while not self.stop_event.is_set():
            if self.lichess_config['total_games'] is not None and self.games_played >= self.lichess_config['total_games']:
                self.logger.info("Challenge thread: Total game limit reached. Stopping challenge loop.")
                break

            self.logger.info(f"Active games: {len(self.game_threads)}, Total in threads: {self.challenge_in_config['accept_challenge_threads']}, Total out threads: {self.challenge_out_config['challenge_bot_threads']}, Games played: {self.games_played}, Games total: {self.lichess_config['total_games']}")

            # Only attempt to challenge if the number of active bot games < challenge threads
            if len(self.game_threads) < (self.challenge_out_config['challenge_bot_threads']):
                self._challenge_specific_bots()

            self.stop_event.wait(self.challenge_out_config['challenge_bot_interval'])
        self.logger.debug("Challenge thread stopped.")

    def get_bots_page_1(self):
        url = self.challenge_out_config['challenge_bot_url']
        self.logger.info(f"Fetching {url} ...")
        response = requests.get(url)
        if response.status_code != 200:
            self.logger.error(f"Failed to fetch page 1: status code {response.status_code}")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        bot_links = soup.select('a[href^="/@"]')
        
        bots = []
        for link in bot_links:
            bot_name = link.text.strip()
            if bot_name and bot_name not in bots:
                bots.append(bot_name)
        
        cleaned_bots = [bot.replace('BOT\xa0', '') for bot in bots]

        self.logger.info(f"Found {len(bots)} bots on page 1.")
        return cleaned_bots
    

    def _challenge_specific_bots(self):
        """
        Challenges a list of bots on lichess
        """

        opponent_id = random.choice(self.bots_to_challenge)
        self.logger.info(f"Attempting to challenge bot: {opponent_id}")

        try:
            challenge = self.client.challenges.create(
                opponent_id,
                rated=self.challenge_out_config['challenge_bot_rated'],
                variant=self.challenge_out_config['challenge_bot_variant'],
                clock_limit=self.challenge_out_config["challenge_bot_time_min"] * 60,
                clock_increment=self.challenge_out_config["challenge_bot_increment_sec"]
            )
            self.logger.info(f"Successfully sent challenge to {opponent_id}. Challenge ID: {challenge['id']}")
        except Exception as e:
            self.logger.error(f"Unexpected error when challenging: {e}", exc_info=True)



    def run(self):
        self.logger.debug("Connecting to Lichess API (main event stream)...")

        if self.challenge_out_config['challenge_bot']:
            self.bots_to_challenge = self.get_bots_page_1()
            self.challenge_thread = threading.Thread(target=self._challenge_loop, name="ChallengeThread")
            self.challenge_thread.daemon = True
            self.challenge_thread.start()

        event_stream = None
        try:
            while not self.stop_event.is_set():
                if event_stream is None:
                    try:
                        event_stream = self.client.bots.stream_incoming_events()
                        self.logger.info("Successfully connected to Lichess main event stream.")
                    except Exception as e:
                        self.logger.critical(f"Unexpected error during initial main event stream connection: {e}. Retrying in 5 seconds...", exc_info=True)
                        time.sleep(5)
                        continue

                event = next(event_stream)

                event_type = event["type"]
                self.logger.debug(f"Received main event: {event}")
                game_id = None

                if event_type == "challenge":
                    self._handle_challenge(event["challenge"])
                elif event_type == "gameStart":
                    game_id = event["game"]["id"]
                    
                    # Check if more than total games limit
                    if self.lichess_config['total_games'] is not None and self.games_played >= self.lichess_config['total_games']:
                        self.logger.info(f"{game_id}: ignored as total_games limit ({self.lichess_config['total_games'] }) reached.")
                        self.client.bots.resign_game(game_id)
                        continue


                    # check how many threads are running
                    if len(self.game_threads) < (self.challenge_in_config['accept_challenge_threads'] + self.challenge_out_config['challenge_bot_threads']):
                        if game_id not in self.game_threads:
                            self.logger.debug(f"{game_id}: starting new game thread")
                            self.games_played += 1
                            self.game_threads[game_id] = {
                                "thread": None,
                                "board": None,
                                "moves_on_board": 0,
                                "initial_game_data": event["game"],
                                "wtime_ms": 0, "btime_ms": 0,
                                "winc_ms": 0, "binc_ms": 0,
                                "last_state_update_time": 0,
                                "thread_lock": False,
                                "is_over": False
                            }
                            game_thread = threading.Thread(target=self._run_game_loop, args=(game_id, event["game"]))
                            self.game_threads[game_id]["thread"] = game_thread
                            game_thread.start()
                        else:
                            self.logger.warning(f"{game_id}: already has a running thread. Ignoring duplicate gameStart event.")
                    else:
                        self.logger.warning(f"{game_id}: new gamestarted exceeding thread count. Resigning.")
                        self.client.bots.resign_game(game_id)
                elif event_type == "gameFinish":
                    game_id = event["game"]["id"]
                    self._handle_game_finish(game_id, event["game"].get("status", "unknown_status"))
                elif event_type == "gameFull":
                    game_id = event["game"]["id"]
                    game_state = event["state"]
                    if game_state and game_state["type"] == "gameState":
                        self.logger.debug(f"{game_id}: processing embedded gameState from gameFull event")
                        self._handle_game_state(game_id, game_state, time.time())
                    else:
                        self.logger.debug(f"{game_id}: received gameFull event without a valid gameState. Ignoring")
        except KeyboardInterrupt:
            # Resign all games on c
            self.logger.info("Ctrl+C detected! Resigning all active games and shutting down...")
            for game_id in list(self.game_threads.keys()):
                self._try_resign_game(game_id)

        except Exception as e:
            self.logger.critical(f"Unexpected error: {e}", exc_info=True)

        finally:
            self._shutdown_bot()


    def _run_game_loop(self, game_id, initial_game_data):
        """
        Runs the dedicated stream for a single game, maintaining board state incrementally.
        This is where the TalbotPlayer instance for *this specific game* is created.
        """
        self.logger.info(f"{game_id}: game loop started")
        game_logger = self._setup_game_logger(game_id)


        player_instance = TalbotPlayer(
            name=self.lichess_config["bot_id"] + "_" + game_id[:4],
            model_path=self.talbot_player_config["model_path"],
            logger=game_logger,
            num_residual_blocks=self.talbot_player_config["resblocks"],
            num_input_planes=self.talbot_player_config["input_planes"],
            num_filters=self.talbot_player_config["filters"],
            cpuct=self.talbot_player_config["cpuct"],
            batch_size=self.talbot_player_config["batchsize"],
        )

        self.game_threads[game_id]["player_instance"] = player_instance

        full_game_details = self._make_api_call_with_retries(game_id, self.client.games.export, game_id)
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
        game_logger.debug(f"{game_id}: board initialized. Current FEN: {current_board.fen()} with {moves_on_board} moves")

        self.game_threads[game_id]["bot_color_str"] = initial_game_data["color"]

        if (current_board.turn == initial_game_data["color"]):
            game_logger.debug(f"{game_id}: it's my turn for the initial move. Thinking of a move...")
            turn_start_time = time.time()
            self.make_bot_move(game_id, turn_start_time)
        else:
            game_logger.debug(f"{game_id}: not my turn for the initial move. Waiting for opponent's move")


        game_stream = self.client.bots.stream_game_state(game_id)
        for event in game_stream:
            if self.stop_event.is_set():
                game_logger.debug(f"Stop event detected for game {game_id}. Exiting game loop.")
                break

            event_type = event["type"]
            game_logger.debug(f"{game_id}: received game event: {event}")

            if event_type == "gameState":
                self._handle_game_state(game_id, event, time.time())
            elif event_type == "gameFull":
                game_state_data_from_full = event["state"]
                game_logger.debug(f"{game_id}: received gameFull event. Processing full state update")
                self._handle_game_state(game_id, game_state_data_from_full, time.time())
            else:
                game_logger.debug(f"{game_id}: unhandled game event type: {event_type}")


    def _handle_challenge(self, challenge_data):

        
        challenge_id = challenge_data["id"]
        challenger_id = challenge_data['challenger']['id'].lower()
        human_challenge = True

        # Ignore self challenges
        if challenger_id == self.lichess_config['bot_id'].lower():
            return
        
        # Check for human challenges
        if 'title' in challenge_data['challenger'] and \
            challenge_data['challenger']['title'] is not None and \
            challenge_data['challenger']['title'].lower() == 'bot':
            human_challenge = False

        variant = challenge_data["variant"]["key"].lower()
        rated = challenge_data["rated"]

        challenge_time_min = int(challenge_data["timeControl"]["limit"]) / 60
        challenge_increment_sec = challenge_data["timeControl"]["limit"]

        accepted_variant = self.challenge_in_config["accept_variant"]
        min_time_min_config = self.challenge_in_config["accept_min_challenge_time"]
        min_increment_sec_config = self.challenge_in_config["accept_min_challenge_increment"]

        self.logger.debug(f"Received challenge {challenge_id} from {challenger_id}: Variant={variant}, Rated={rated}, Time={challenge_time_min:.1f}min+{challenge_increment_sec}s")


        decline_reason = None

        # List reasons for declining challenge
        if human_challenge and not self.challenge_in_config['accept_challenge_human']:
            decline_reason = "not accepting human challenges"

        elif not human_challenge and not self.challenge_in_config['accept_challenge_bot']:
            decline_reason = "not accepting bot challenges"

        elif rated and not self.challenge_in_config['accept_challenge_rated']:
            decline_reason = "not accepting rated challenges"

        elif accepted_variant != "any" and variant != accepted_variant:
            decline_reason = f"variant '{variant}' not accepted (only '{accepted_variant}' allowed)"

        elif challenge_time_min < min_time_min_config:
            decline_reason = f"main time {challenge_time_min:.1f}min too short (min {min_time_min_config}min required)"

        elif challenge_increment_sec < min_increment_sec_config:
            decline_reason = f"increment {challenge_increment_sec}s too low (min {min_increment_sec_config}s required)"

        elif len(self.game_threads) == (self.challenge_in_config['accept_challenge_threads'] + self.challenge_out_config['challenge_bot_threads']):
            decline_reason = f"bot is already in {len(self.game_threads)} game(s)"
            self.logger.info(f"Declining challenge {challenge_id} from {challenger_id} because bot is already busy.")

        elif self.lichess_config['total_games'] is not None and self.games_played >= self.lichess_config['total_games']:
            decline_reason = "bot has played all games for today"
            self.logger.info(f"Declining challenge {challenge_id} from {challenger_id} because total games have been played.")

        # Decline or accept challenge
        if decline_reason:
            self.logger.debug(f"Declining challenge {challenge_id} due to: {decline_reason}")
            self.client.bots.decline_challenge(challenge_id)
        else:
            self.client.bots.accept_challenge(challenge_id)
            self.logger.debug(f"Accepted challenge {challenge_id}.")


    def _handle_game_state(self, game_id, game_state, event_receive_time):
        game_data = self.game_threads.get(game_id)

        if not game_data or game_data['is_over']:
            return

        if not game_data or game_data["board"] is None:
            self.logger.error(f"{game_id} : board not found for game in _handle_game_state. Cannot process game state. Resigning.")
            self._try_resign_game(game_id)
            return

        current_board = game_data["board"]
        moves_on_board = game_data["moves_on_board"]

        # Ensure times are integers (milliseconds)
        game_data["wtime_ms"] = int(game_state["wtime"].timestamp() * 1000)if isinstance(game_state["wtime"], datetime) else game_state["wtime"]
        game_data["btime_ms"] = int(game_state["btime"].timestamp() * 1000) if isinstance(game_state["btime"], datetime) else game_state["btime"]
        game_data["winc_ms"] = int(game_state["winc"].timestamp() * 1000) if isinstance(game_state["winc"], datetime) else game_state["winc"]
        game_data["binc_ms"] = int(game_state["binc"].timestamp() * 1000) if isinstance(game_state["binc"], datetime) else game_state["binc"]

        game_data["last_state_update_time"] = event_receive_time

        game_status_name = game_state["status"]

        if game_status_name != "started":
            self.logger.debug(f"{game_id}: status: {game_status_name}. No move needed")
            return

        incoming_moves = game_state["moves"].split()
        num_incoming_moves = len(incoming_moves)

        if num_incoming_moves <= moves_on_board:
            self.logger.debug(f"{game_id}: no new moves to apply. Current FEN: {current_board.fen()}")
        else:
            new_moves_to_apply = incoming_moves[moves_on_board:]
            self.logger.debug(f"{game_id}: applying {len(new_moves_to_apply)} new moves")
            for move_str in new_moves_to_apply:
                if not move_str:
                    continue

                move = current_board.parse_san(move_str)
                current_board.push(move)
                moves_on_board += 1
                self.logger.debug(f"{game_id}: applied new move: {move_str}")

            game_data["moves_on_board"] = moves_on_board

        self.logger.debug(f"{game_id}: state updated. Current FEN: {current_board.fen()}")

        # Determine if it's the bot's turn
        bot_color_in_game = chess.WHITE if game_data["initial_game_data"]["color"] == 'white' else chess.BLACK

        if(current_board.turn == bot_color_in_game):
            self.logger.debug(f"{game_id}: my turn, thinking of a move...")
            self.make_bot_move(game_id, event_receive_time)
        else:
            self.logger.debug(f"{game_id}: not my turn")


    def _calculate_thinking_time(self, game_id):
        """
        Calculates the thinking time for the AI based on current game state and time control.
        Prioritizes increment, uses faster thinking for early moves, and a percentage of
        remaining time for later moves.
        """
        game_data = self.game_threads[game_id]
        initial_game_data = game_data["initial_game_data"]

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

        # Set remaining time = to the time from event + overhead
        remaining_time_sec = effective_remaining_time_ms / 1000.0
        increment_sec = increment_ms / 1000.0

        move_number = game_data.get("moves_on_board", 0)

        calculated_time = 0

        # Early game -> fixed move time (quick)
        if move_number <= self.timing_config["mid_game_threshold"]:

            calculated_time = self.timing_config["early_game_movetime"]
            self.logger.debug(f"{game_id}: (move {move_number}): Early game strategy. Base calculated time: {calculated_time:.2f}s")

        # Mid game -> calculated
        elif move_number <= self.timing_config["late_game_threshold"]:
            base_time = remaining_time_sec / self.timing_config["mid_game_movetime_factor"] + increment_sec * self.timing_config["mid_game_increment_factor"]  # midgame
            cap = remaining_time_sec * self.timing_config["mid_game_cap_factor"]
            calculated_time = min(base_time, cap)

            self.logger.debug(f"{game_id}: (move {move_number}): Mid/Late game strategy. Base calculated time: {calculated_time:.2f}s")

        else:
            base_time = remaining_time_sec / self.timing_config["late_game_movetime_factor"] + increment_ms * self.timing_config["late_game_increment_factor"]  # more time late game
            cap = remaining_time_sec * self.timing_config["late_game_cap_factor"]
            calculated_time = min(base_time, cap)

        # Check if time < buffer time to prevent flagging
        if remaining_time_sec < calculated_time + self.timing_config["buffer_time"]:
            calculated_time = max(remaining_time_sec - self.timing_config["buffer_time"], self.timing_config["minimum_time"])  # min 1 second

        self.logger.debug(f"{game_id}: effective Remaining clock: {remaining_time_sec:.2f}s, Increment: {increment_sec:.2f}s, Move #: {move_number}. Calculated thinking time: {calculated_time:.4f}s")
        return calculated_time
    

    def _make_api_call_with_retries(self, game_id, api_call_func, *args, **kwargs):
        """
        Wrapper to make Lichess API calls with retry logic for connection errors.
        This method will retry on requests.exceptions.ConnectionError and berserk.exceptions.ApiError.
        """
        max_retries = 3
        initial_delay_seconds = 1
        for attempt in range(1, max_retries + 1):
            if self.game_threads.get(game_id) is None and not self.game_threads[game_id]["is_over"]:
                return
            
            try:
                result = api_call_func(*args, **kwargs)
                return result # Success!
            except (requests.exceptions.ConnectionError, berserk.exceptions.ApiError) as e:
                self.logger.warning(f"Attempt {attempt}/{max_retries}: Connection error during API call. Error: {e}")
                if attempt < max_retries:
                    delay = initial_delay_seconds * (2 ** (attempt - 1)) # Exponential backoff
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {max_retries} attempts failed for API call. Giving up.")
                    raise # Re-raise the exception if all retries fail
            except Exception as e:
                # Catch any other unexpected exceptions and re-raise immediately
                self.logger.critical(f"An unexpected non-retryable error occurred during API call: {e}", exc_info=True)
                raise # Re-raise immediately for unhandled exceptions


    def make_bot_move(self, game_id, turn_start_time):
        game_data = self.game_threads.get(game_id)
        if not game_data or game_data["board"] is None:
            self.logger.error(f"{game_id}: attempted to make a move but no board found. Resigning")
            self._try_resign_game(game_id)
            return

        current_board = game_data["board"]

        try:
            time_for_ai = self._calculate_thinking_time(game_id)
            time_spent_since_turn_start = time.time() - turn_start_time
            effective_time_for_ai = max(1, time_for_ai - time_spent_since_turn_start)

            player_instance = self.game_threads[game_id]["player_instance"]

            # Lock thread
            self.game_threads[game_id]["thread_lock"] = True
            best_move_obj = player_instance.get_move(current_board, effective_time_for_ai)
            self.game_threads[game_id]["thread_lock"] = False

            # Only make move if game is in play
            if self.game_threads.get(game_id) is not None and not self.game_threads[game_id]["is_over"]:                
                best_move_uci = best_move_obj.uci()
                self.logger.debug(f"{game_id}: attempting to make move {best_move_uci}")

                self._make_api_call_with_retries(game_id, self.client.bots.make_move, game_id, best_move_uci)
                self.logger.debug(f"{game_id}: successfully sent move {best_move_uci}")
                current_board.push_uci(best_move_uci)
                game_data["moves_on_board"] += 1

        except Exception as e:
            self.logger.critical(f"{game_id}: unexpected error during move generation or sending {e}", exc_info=True)
            self._try_resign_game(game_id)


    def _try_resign_game(self, game_id):
        game_data = self.game_threads.get(game_id)
        game_data["is_over"] = True

        try:
            self.client.bots.resign_game(game_id)
            self.logger.info(f"{game_id}: successfully sent resign command")
        except Exception as e:
            self.logger.error(f"{game_id}: unexpected error during resignation attempt for game: {e}", exc_info=True)


    def _handle_game_finish(self, game_id, status_data=None):
        """
        Handles the final cleanup for a game. This should be the single point
        where a game's entry is removed from self.game_threads.
        """

        if status_data is not None:
            status_name = status_data.get("name")
            if status_name:
                self.logger.info(f"{game_id}: finished with status: {status_name}")

        game_data = self.game_threads.get(game_id)

        if not game_data:
            self.logger.debug(f"{game_id}: could not find game, returning")
            return

        # Mark game as over so move logic can check before sending
        game_data["is_over"] = True

        def delayed_cleanup():
            # Wait if move calculation is ongoing
            while game_data["thread_lock"]:
                time.sleep(0.5)
            self.game_threads.pop(game_id, None)
            self.logger.info(f"{game_id}: cleaned up and removed from active games. Active games: {len(self.game_threads)}")

        threading.Thread(target=delayed_cleanup, daemon=True).start()


    def _shutdown_bot(self):
        """Gracefully shuts down all game threads and the player, including the challenge thread."""
        self.logger.debug("Setting stop event for all threads...")
        self.stop_event.set() # Signal all threads to stop

        # First, join the challenge thread to ensure it cleans up
        if self.challenge_thread is not None and self.challenge_thread.is_alive():
            self.logger.debug("Waiting for challenge thread to finish...")
            self.challenge_thread.join(timeout=1) # Give it 5 seconds to finish
            if self.challenge_thread.is_alive():
                self.logger.warning("Challenge thread did not terminate gracefully within timeout.")

        # Then, join individual game threads
        for game_id, data in list(self.game_threads.items()):
            self.logger.debug(f"{game_id}: Handling end of game...")
            self._handle_game_finish(game_id)

            while self.game_threads.get(game_id) is not None:
                time.sleep(0.5)

            thread = data["thread"]

            # Join threads after game has been removed
            if thread and thread.is_alive():
                self.logger.debug(f"{game_id}: Waiting for game thread to finish...")
                thread.join()
                if thread.is_alive():
                    self.logger.warning(f"{game_id}: game thread did not terminate gracefully within timeout.")

        self.logger.debug("Lichess Bot completely shut down.")




if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(config_dir, "lichess_config.yaml"))

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config["lichess"]["api_token"] = os.getenv("LICHESS_API_TOKEN")

    bot = LichessBot(
        lichess_config=config["lichess"], 
        talbot_player_config=config["talbot"], 
        timing_config=config["timing"], 
        challenge_in_config=config["challenge_in"], 
        challenge_out_config=config["challenge_out"],
        logging_config=config["logging"]
    )
    bot.run()