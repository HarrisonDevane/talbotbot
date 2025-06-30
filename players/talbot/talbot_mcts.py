import chess
import cython_chess
import os
import torch
import torch.nn.functional as F
import sys
import math
import random
import time
import logging
from collections import deque

# Get the parent directory path to help with relative imports and paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

import utils
import training.supervised.model as model

# --- Logging Configuration ---
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "training/supervised/v3_hqgames_mcts/logs/"))
print(f"Logging to: {log_dir}")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "mcts_debug.log")
if os.path.exists(log_file_path): os.remove(log_file_path)

# IMPORTANT: Setting level to DEBUG to see the detailed per-selection logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a'
)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---


class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search tree.
    Each node corresponds to a specific board state.
    """
    def __init__(self, board: chess.Board = None, parent=None, move: chess.Move = None):
        self._board = board 
        self.parent = parent
        self.move = move 
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior_probabilities = None 
        self.prior_probability_from_parent = 0.0 
        self.is_expanded = False
        self.is_queued_for_inference = False

    @property
    def board(self) -> chess.Board:
        if self._board is None and self.parent is not None:
            self._board = self.parent.board.copy() 
            self._board.push(self.move)
        return self._board

    def is_leaf(self) -> bool:
        return not self.children and not self.is_expanded

    def is_root(self) -> bool:
        return self.parent is None

    def uct_score(self, cpuct: float, prior_probability_for_this_move: float, sqrt_parent_visits_term: float) -> float:
        if self.visits == 0:
            return float('inf')

        Q = -self.value_sum / self.visits
        U = cpuct * prior_probability_for_this_move * sqrt_parent_visits_term / (1 + self.visits)
        
        return Q + U

class MCTS:
    """
    Monte Carlo Tree Search algorithm for finding the best move.
    """
    def __init__(self, model_player: 'TalbotPlayerMCTS', cpuct: float = 1.0, batch_size: int = 16, target_move_uci: str = None): # Added target_move_uci
        self.model_player = model_player
        self.cpuct = cpuct
        self.root = None
        self.batch_size = batch_size
        self._inference_batch = deque() 
        self._pending_nodes = [] 
        self.force_batch = False
        self.target_move_uci = target_move_uci # Store target move

    def _log_root_children_stats(self, step_type: str = "Interim"):
        """Logs the current stats of the root's direct children."""
        if not self.root or not self.root.children:
            return

        logger.info(f"\n--- MCTS Root Children Analysis ({step_type} State) ---")
        sorted_children = sorted(self.root.children.items(), key=lambda item: item[1].visits, reverse=True)
        sqrt_root_visits_term = math.sqrt(self.root.visits) if self.root.visits > 0 else 0.0

        for move, child_node in sorted_children:
            q_value = -child_node.value_sum / child_node.visits if child_node.visits > 0 else 0.0
            u_value = self.cpuct * child_node.prior_probability_from_parent * sqrt_root_visits_term / (1 + child_node.visits)
            uct_value = q_value + u_value
            
            log_message = (
                f"Move: {move.uci()}, Visits: {child_node.visits}, "
                f"Avg Q-value: {q_value:.4f}, U-value: {u_value:.4f}, "
                f"UCT: {uct_value:.4f}, Prior Probability: {child_node.prior_probability_from_parent:.4f}"
            )
            
            if self.target_move_uci and move.uci() == self.target_move_uci:
                logger.info(f"### TARGET MOVE: {log_message} ###") # Highlight target move
            else:
                logger.info(log_message)
        logger.info("-----------------------------------\n")


    def run_simulations(self, initial_board: chess.Board, time_limit: float):
        self.root = MCTSNode(initial_board.copy()) 
        start_time_total = time.perf_counter()

        simulation_count = 0
        self._log_root_children_stats("Initial (after root expansion)") # Log initial state

        while time.perf_counter() - start_time_total < time_limit:
            simulation_count += 1
            logger.debug(f"\n--- Simulation {simulation_count} Started ---") # New log for start of sim
            node = self.root
            path = [node] 

            # Selection phase
            while not node.is_leaf() and node.is_expanded and \
                    not node.is_queued_for_inference:
                
                best_child = None
                best_uct_score = -float('inf')

                legal_moves = cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL)

                eligible_children = []
                for move in legal_moves:
                    if move in node.children and not node.children[move].is_queued_for_inference:
                        eligible_children.append((move, node.children[move]))

                sqrt_parent_visits_term = math.sqrt(node.visits) if node.visits > 0 else 0.0

                parent_info = f"Move: {node.move.uci()}" if node.move else "Root"
                logger.debug(f"   Selecting child from Node ({parent_info}), Parent Visits: {node.visits}, Sqrt Parent Visits Term: {sqrt_parent_visits_term:.4f}")

                child_uct_details = [] # To store details for all children for logging
                for move, child in eligible_children: 
                    prior_prob_for_child = child.prior_probability_from_parent 
                    uct = child.uct_score(self.cpuct, prior_prob_for_child, sqrt_parent_visits_term)

                    # --- MODIFIED SECTION FOR TIE-BREAKING ---
                    if uct == float('inf'):
                        # If current UCT is infinite, prioritize based on prior probability
                        if best_child is None or prior_prob_for_child > best_prior_for_tie_break:
                            best_uct_score = uct # Still infinite
                            best_prior_for_tie_break = prior_prob_for_child
                            best_child = child
                    elif uct > best_uct_score:
                        # If current UCT is finite and better than current best_uct_score, select it
                        best_uct_score = uct
                        best_prior_for_tie_break = prior_prob_for_child # Keep this updated for potential future ties
                        best_child = child
                    # --- END MODIFIED SECTION ---


                    Q = -child.value_sum / (child.visits if child.visits > 0 else 1e-6) # Avoid division by zero for Q
                    U = self.cpuct * prior_prob_for_child * sqrt_parent_visits_term / (1 + child.visits)

                    child_uct_details.append((move, uct, Q, U, prior_prob_for_child, child.visits))
                    
                    # --- NEW: Log specifically for the target move if the current node is its parent ---
                    if self.target_move_uci and node.move and self.target_move_uci == move.uci():
                        logger.debug(f"      TARGET CHILD CANDIDATE: Move: {move.uci()}, Visits: {child.visits}, Q: {Q:.4f}, U: {U:.4f}, UCT: {uct:.4f}, Prior: {prior_prob_for_child:.4f}, is_queued_for_inference: {child.is_queued_for_inference}")
                    # --- END NEW ---

                    if uct > best_uct_score:
                        best_uct_score = uct
                        best_child = child
                
                # Log details for all eligible children, sorted by UCT descending
                for move, uct_val, q_val, u_val, prior_prob, child_visits in sorted(child_uct_details, key=lambda x: x[1], reverse=True):
                    # Ensure general logging is still there, but target has special log
                    if not (self.target_move_uci and move.uci() == self.target_move_uci):
                        logger.debug(f"      Candidate: Move: {move.uci()}, Visits: {child_visits}, Q: {q_val:.4f}, U: {u_val:.4f}, UCT: {uct_val:.4f}, Prior: {prior_prob:.4f}")

                if best_child is None:
                    logger.debug("    No eligible child found for selection. Breaking from selection loop.")
                    break 
                
                logger.debug(f"   Selected Node: Move: {best_child.move.uci()} with UCT: {best_uct_score:.4f}") # Log the selected move

                node = best_child
                path.append(node)
            
            # Simulation/Expansion phase (queue for inference or handle game over)
            successfully_queued = self.simulate(node) 

            # If not successfully queued (e.g., game over), but there's a batch pending, force processing
            if not successfully_queued and not node.board.is_game_over() and self._inference_batch:
                self.force_batch = True
            else:
                self.force_batch = False

            # Batch inference and backpropagation
            if len(self._inference_batch) >= self.batch_size or \
               (time.perf_counter() - start_time_total >= time_limit and self._inference_batch) or \
               self.force_batch:
                
                self._perform_batched_inference()
                for processed_node, value_from_nn in self._pending_nodes:
                    self.backpropagate(processed_node, value_from_nn)
                self._pending_nodes.clear()
                
                # Log root's children after each batch processing + backpropagation
                # This log is at INFO level, so it will always appear.
                self._log_root_children_stats(f"After Simulation {simulation_count} Batch")


        # Final batch processing if any remain
        if self._inference_batch:
            self._perform_batched_inference()
            for processed_node, value_from_nn in self._pending_nodes:
                self.backpropagate(processed_node, value_from_nn)
            self._pending_nodes.clear()

        total_elapsed_time = time.perf_counter() - start_time_total

        logger.info(f"MCTS completed {simulation_count} simulations within {time_limit:.4f} seconds (actual: {total_elapsed_time:.4f}s).")
        
        # Log final root children analysis
        self._log_root_children_stats("Final")


    def expand(self, node: MCTSNode, policy_probs: torch.Tensor):
        if node.board.is_game_over(): 
            node.is_expanded = True
            node.is_queued_for_inference = False
            return

        legal_moves = cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL)
        
        from_row_ints = []
        from_col_ints = []
        channel_ints = []
        
        child_nodes_in_order = []

        for move in legal_moves:
            from_row_int, from_col_int, channel_int = utils.move_to_policy_components(move, node.board) 
            from_row_ints.append(from_row_int)
            from_col_ints.append(from_col_int)
            channel_ints.append(channel_int)
            
            child_node = MCTSNode(board=None, parent=node, move=move) 
            node.children[move] = child_node
            child_nodes_in_order.append(child_node)
            
        node.prior_probabilities = torch.zeros_like(policy_probs, dtype=torch.float)

        if legal_moves:
            from_row_tensor = torch.tensor(from_row_ints, dtype=torch.long, device=policy_probs.device)
            from_col_tensor = torch.tensor(from_col_ints, dtype=torch.long, device=policy_probs.device)
            channel_tensor = torch.tensor(channel_ints, dtype=torch.long, device=policy_probs.device)

            indices_tensor = utils.policy_components_to_flat_index_torch(
                from_row_tensor, from_col_tensor, channel_tensor
            )
            
            prior_values_for_legal_moves = policy_probs[indices_tensor]
            
            sum_of_legal_priors = prior_values_for_legal_moves.sum()
            
            if sum_of_legal_priors > 0:
                normalized_legal_priors = prior_values_for_legal_moves / sum_of_legal_priors
            else: 
                normalized_legal_priors = prior_values_for_legal_moves
        else:
            indices_tensor = torch.empty(0, dtype=torch.long, device=policy_probs.device)
            normalized_legal_priors = torch.empty(0, device=policy_probs.device) 
        
        if indices_tensor.numel() > 0 and normalized_legal_priors.numel() > 0: 
            node.prior_probabilities.index_put_((indices_tensor,), normalized_legal_priors, accumulate=False)

        if normalized_legal_priors.numel() > 0:
            normalized_priors_list = normalized_legal_priors.cpu().tolist()
            for i, child_node in enumerate(child_nodes_in_order):
                child_node.prior_probability_from_parent = normalized_priors_list[i]

        node.is_expanded = True
        node.is_queued_for_inference = False


    def simulate(self, node: MCTSNode) -> bool:
        """
        Simulation phase: If a node is a leaf, queue it for batched neural network inference.
        If it's a game-over state, determine the value directly.
        Returns: True if node was successfully queued, False otherwise (skipped or game over handled).
        """
        current_board = node.board 

        if current_board.is_game_over():
            result = current_board.result()
            value = 0.0
            if result == "1-0":
                value = 1.0 if current_board.turn == chess.WHITE else -1.0 
            elif result == "0-1":
                value = 1.0 if current_board.turn == chess.BLACK else -1.0 
            else:
                value = 0.0
            self.backpropagate(node, value)
            return False

        if node.is_queued_for_inference:
            return False

        board_tensor_np = utils.board_to_tensor(node.board) 
        board_input = torch.from_numpy(board_tensor_np).float().to(self.model_player.device)
        
        self._inference_batch.append((node, board_input))
        self._pending_nodes.append(node)
        node.is_queued_for_inference = True

        if node.parent:
            all_legal_children_queued = True
            parent_legal_moves = cython_chess.generate_legal_moves(node.parent.board, chess.BB_ALL, chess.BB_ALL)

            for move in parent_legal_moves:
                if move in node.parent.children and not node.parent.children[move].is_queued_for_inference:
                    all_legal_children_queued = False
                    break
            if all_legal_children_queued:
                node.parent.is_queued_for_inference = True

        return True


    def _perform_batched_inference(self):
        if not self._inference_batch:
            return

        nodes_to_process = []
        board_tensors = []
        while self._inference_batch:
            node, board_tensor = self._inference_batch.popleft()
            nodes_to_process.append(node)
            board_tensors.append(board_tensor)

        batch_input = torch.stack(board_tensors)
        
        with torch.no_grad():
            policy_logits_batch, value_output_batch = self.model_player.model(batch_input)
        
        policy_probs_batch = F.softmax(policy_logits_batch, dim=1)

        temp_pending_nodes = [] 
        for i, node in enumerate(nodes_to_process):
            policy_probs = policy_probs_batch[i].squeeze(0)
            value_for_current_node_player = value_output_batch[i].item()

            self.expand(node, policy_probs) 

            temp_pending_nodes.append((node, value_for_current_node_player))
        
        self._pending_nodes = temp_pending_nodes


    def backpropagate(self, node: MCTSNode, value: float):
        current = node
        original_expanded_node_turn = node.board.turn
        
        while current is not None:
            # --- NEW: Reset is_queued_for_inference during backpropagation ---
            if current.is_queued_for_inference:
                logger.debug(f"   Resetting is_queued_for_inference for node {current.move.uci() if current.move else 'Root'}")
                current.is_queued_for_inference = False
            # --- END NEW ---
            
            current.visits += 1
            
            if current.board.turn == original_expanded_node_turn:
                current.value_sum += value
            else:
                current.value_sum -= value 
            
            # --- NEW: Log updates for the specific target move and its parent ---
            if self.target_move_uci and current.move and current.move.uci() == self.target_move_uci:
                logger.debug(f"      TARGET MOVE BACKPROPAGATION: Move: {current.move.uci()}, Visits: {current.visits}, Value Sum: {current.value_sum:.4f}")
            elif self.target_move_uci and current.parent and current.parent.move and current.parent.move.uci() == self.target_move_uci:
                # This logs when the current node is a child of the target move
                logger.debug(f"      Backpropagating through parent of TARGET MOVE ({self.target_move_uci}): Current Node: {current.move.uci()}, Parent's (target) Visits: {current.parent.visits}, Parent's (target) Value Sum: {current.parent.value_sum:.4f}")
            # --- END NEW ---

            current = current.parent


class TalbotPlayerMCTS:
    def __init__(self, model_path: str, name="TalbotMCTS", num_residual_blocks: int = 20, time_per_move: float = 3.0, cpuct: float = 1.0, batch_size: int = 16, target_move_uci: str = None): # Added target_move_uci
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.ChessAIModel(num_input_planes=18, num_residual_blocks=num_residual_blocks, num_filters=128)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict']) 
        logger.info(f"Model loaded successfully from {model_path}")
            
        self.model.to(self.device)
        self.model.eval()

        self.time_per_move = time_per_move
        self.cpuct = cpuct
        self.batch_size = batch_size
        self.target_move_uci = target_move_uci # Pass target move to MCTS

    def get_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over():
            logger.info("Game is already over, no move to make.")
            return None

        mcts = MCTS(self, self.cpuct, self.batch_size, self.target_move_uci) # Pass target_move_uci here
        mcts.run_simulations(board.copy(), self.time_per_move) 

        best_move = None
        max_visits = -1
        
        if not mcts.root.children:
            logger.warning("MCTS root has no children after simulations. Falling back to random legal move.")
            legal_moves = cython_chess.generate_legal_moves(board, chess.BB_ALL, chess.BB_ALL)
            if legal_moves:
                return random.choice(list(legal_moves))
            else:
                logger.error("No legal moves available. Returning None.")
                return None


        for move, child_node in mcts.root.children.items():
            if child_node.visits > max_visits:
                max_visits = child_node.visits
                best_move = move
        
        if best_move is None:
            legal_moves = cython_chess.generate_legal_moves(board, chess.BB_ALL, chess.BB_ALL)
            if legal_moves:
                return random.choice(list(legal_moves))
            else:
                logger.error("No legal moves available. Returning None.")
                return None

        logger.info(f"MCTS selected move: {best_move.uci()} with {max_visits} visits.")
        return best_move
    
    def is_human(self):
        return False

    def close(self):
        pass