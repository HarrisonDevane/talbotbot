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
log_dir = os.path.abspath(os.path.join(parent_dir, "training/supervised/v3_hqgames_mcts/logs/"))
print(f"Logging to: {log_dir}")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "mcts_debug.log")
if os.path.exists(log_file_path): os.remove(log_file_path)

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
    def __init__(self, board: chess.Board, parent=None, move: chess.Move = None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior_probabilities = None
        self.is_expanded = False
        self.is_queued_for_inference = False

    def is_leaf(self) -> bool:
        return not self.children and not self.is_expanded

    def is_root(self) -> bool:
        return self.parent is None

    def uct_score(self, parent_visits: int, cpuct: float, prior_probability_for_this_move: float) -> float:
        """
        Calculates the UCT (Upper Confidence Bound for Trees) score for selecting the best child node.
        prior_probability_for_this_move: The specific prior probability for the move leading to this node.
        """
        if self.visits == 0:
            # If a node has not been visited, give it a very high UCT score to encourage exploration.
            return float('inf')

        # Exploitation term (average reward)
        Q = -self.value_sum / self.visits

        # Exploration term (encourages visiting less explored or high-prior nodes)
        # Use the passed prior_probability_for_this_move, which is a scalar float
        U = cpuct * prior_probability_for_this_move * math.sqrt(parent_visits) / (1 + self.visits)
        
        return Q + U

class MCTS:
    """
    Monte Carlo Tree Search algorithm for finding the best move.
    """
    def __init__(self, model_player: 'TalbotPlayerMCTS', cpuct: float = 1.0, batch_size: int = 16):
        self.model_player = model_player
        self.cpuct = cpuct
        self.root = None
        self.batch_size = batch_size
        self._inference_batch = deque() # Stores (node, board_tensor) tuples waiting for NN inference
        self._pending_nodes = [] # Stores nodes that were sent for batched inference
        self.timing_data = {
            "select": 0.0,
            "expand": 0.0,
            "simulate": 0.0,
            "backpropagate": 0.0,
            "total_mcts_loop": 0.0, # To track total time within the loop excluding setup/teardown
            "nn_prediction": 0.0 # Time specifically for NN inference
        }
        self.force_batch = False

    def run_simulations(self, initial_board: chess.Board, time_limit: float):
        self.root = MCTSNode(initial_board.copy())
        start_time_total = time.perf_counter()

        simulation_count = 0
        while time.perf_counter() - start_time_total < time_limit:
            start_loop_time = time.perf_counter()

            node = self.root
            path = [node] # Keep track of the path for backpropagation

            # --- Selection Phase ---
            start_select = time.perf_counter()
            # Loop while the node is *not* a leaf AND *is* expanded, AND *not* queued for inference,
            # AND (NEW) not all of its children are already queued for inference.
            while not node.is_leaf() and node.is_expanded and \
                  not node.is_queued_for_inference:
                
                best_child = None
                best_uct_score = -float('inf')

                # Filter children to only include legal moves, and ensure they exist
                # and are NOT already queued for inference (eligible children)
                eligible_children = []
                # Use cython_chess to generate legal moves as strings

                for move in cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL):
                    if move in node.children and not node.children[move].is_queued_for_inference:
                        eligible_children.append((move, node.children[move]))

                for move, child in eligible_children: 
                    # Calculate prior probability for the child. If move_to_policy_components fails, treat as 0.
                    prior_prob_for_child = 0.0
                    from_row_norm, from_col_norm, channel = utils.move_to_policy_components(move, node.board)
                    flat_index = utils.policy_components_to_flat_index(from_row_norm, from_col_norm, channel)

                    # Ensure prior_probabilities is not None and flat_index is valid
                    if node.prior_probabilities is not None and flat_index < len(node.prior_probabilities):
                        prior_prob_for_child = node.prior_probabilities[flat_index].item()
                    
                    uct = child.uct_score(node.visits, self.cpuct, prior_prob_for_child)
                    
                    if uct > best_uct_score:
                        best_uct_score = uct
                        best_child = child

                node = best_child
                path.append(node)
            
            end_select = time.perf_counter()
            self.timing_data["select"] += (end_select - start_select)

            # --- Simulation/Expansion Phase (Batched) ---
            start_simulate = time.perf_counter()
            # simulate() now returns a boolean indicating if a node was successfully queued
            successfully_queued = self.simulate(node) 
            end_simulate = time.perf_counter()
            self.timing_data["simulate"] += (end_simulate - start_simulate)

            # NEW LOGIC for Step 2: Update consecutive skipped queues
            if not successfully_queued and not node.board.is_game_over() and self._inference_batch: # Only increment if skipped due to being already queued, not game over
                self.force_batch = True
                logger.debug(f"No new nodes found")
            else:
                self.force_batch = False

            if len(self._inference_batch) >= self.batch_size or \
               (time.perf_counter() - start_time_total >= time_limit and self._inference_batch) or \
               self.force_batch:
                
                logger.info(f"Triggering batch inference: Full ({len(self._inference_batch)}/{self.batch_size}), "
                            f"Time ({time.perf_counter() - start_time_total:.2f}/{time_limit:.2f})")
                self._perform_batched_inference()
                # After inference, nodes in _pending_nodes have their values and policies.
                # Now, backpropagate for all these recently processed nodes.
                for processed_node, value_from_nn in self._pending_nodes:
                    self.backpropagate(processed_node, value_from_nn)
                self._pending_nodes.clear() # Clear for the next batch

            end_loop_time = time.perf_counter()
            self.timing_data["total_mcts_loop"] += (end_loop_time - start_loop_time)
            simulation_count += 1
            
        # Process any remaining nodes in the batch at the end of the time limit
        if self._inference_batch:
            logger.info("Processing final batch due to time limit.")
            self._perform_batched_inference()
            for processed_node, value_from_nn in self._pending_nodes:
                self.backpropagate(processed_node, value_from_nn)
            self._pending_nodes.clear()

        end_time_total = time.perf_counter()
        total_elapsed_time = end_time_total - start_time_total

        logger.info(f"MCTS completed {simulation_count} simulations within {time_limit:.4f} seconds (actual: {total_elapsed_time:.4f}s).")
        logger.info("\n--- MCTS Timing Report ---")
        for step, duration in self.timing_data.items():
            if total_elapsed_time > 0:
                logger.info(f"Time spent in {step}: {duration:.4f} seconds ({100 * duration / total_elapsed_time:.2f}%)")
            else:
                logger.info(f"Time spent in {step}: {duration:.4f} seconds (0.00%) - total_elapsed_time was 0.")
        logger.info("--------------------------")


    def expand(self, node: MCTSNode, policy_probs: torch.Tensor):
        """
        Expansion phase: If a node is visited for the first time, expand it by creating children
        for all legal moves and initializing their prior probabilities from the neural network's policy output.
        """
        if node.board.is_game_over():
            node.is_expanded = True
            node.is_queued_for_inference = False # NEW: Reset flag after processing
            logger.debug("Node represents a game over state, no expansion needed.")
            return

        # Initialize prior_probabilities to zeros, then fill for legal moves
        node.prior_probabilities = torch.zeros_like(policy_probs, dtype=torch.float)

        mapped_legal_moves_count = 0
        # logger.debug(f"Processing {len(legal_moves)} legal moves for expansion:") # Keep this at DEBUG level

        for move in cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL):
            from_row_norm, from_col_norm, channel = utils.move_to_policy_components(move, node.board)
            flat_index = utils.policy_components_to_flat_index(from_row_norm, from_col_norm, channel)
            
            new_board = node.board.copy()
            new_board.push(move)
            child_node = MCTSNode(new_board, parent=node, move=move)
            node.children[move] = child_node
            
            # Assign prior probability from the NN output
            if flat_index < len(policy_probs): # Basic bounds check
                node.prior_probabilities[flat_index] = policy_probs[flat_index].item()
                mapped_legal_moves_count += 1
                logger.debug(f"      - Created child for move {move}, NN Prior Prob: {policy_probs[flat_index].item():.4f}")
        
        # Normalize prior probabilities for only the legal moves that were mapped
        if node.prior_probabilities.sum() > 0:
            original_sum = node.prior_probabilities.sum()
            node.prior_probabilities /= original_sum
            # logger.debug(f"Normalized prior probabilities for legal moves. Original sum: {original_sum:.4f}, New sum: {node.prior_probabilities.sum():.4f}")

        node.is_expanded = True
        node.is_queued_for_inference = False # NEW: Reset flag once expansion is done
        logger.debug("Node expansion complete.")


    def simulate(self, node: MCTSNode) -> bool: # NEW: Return boolean
        """
        Simulation phase: If a node is a leaf, queue it for batched neural network inference.
        If it's a game-over state, determine the value directly.
        Returns: True if node was successfully queued, False otherwise (skipped or game over handled).
        """
        if node.board.is_game_over():
            result = node.board.result()
            value = 0.0
            if result == "1-0":  # White wins
                # Value from perspective of the player whose turn it was at the *simulated* node
                value = 1.0 if node.board.turn == chess.WHITE else -1.0 
            elif result == "0-1":  # Black wins
                value = 1.0 if node.board.turn == chess.BLACK else -1.0 
            else:  # Draw
                value = 0.0
            logger.debug(f"Game over state detected. Result: {result}, Value from current player's perspective: {value:.4f}")
            # Immediately backpropagate for game-over nodes, as they don't need NN inference
            self.backpropagate(node, value)
            return False # Not queued, handled immediately

        if node.is_queued_for_inference: # Check if already queued
            logger.debug(f"Node is already queued for inference. Skipping addition to batch.")
            return False # Skipped

        # Convert board to tensor for the model
        board_tensor_np = utils.board_to_tensor(node.board)
        board_input = torch.from_numpy(board_tensor_np).float().to(self.model_player.device)
        
        # Add node to the batch for later inference
        self._inference_batch.append((node, board_input))
        self._pending_nodes.append(node) # Keep track of nodes in the current batch
        node.is_queued_for_inference = True # Set flag to True

        # NEW LOGIC: Check if all children of the parent are now queued. If true, 'queue' parent too
        if node.parent:
            all_legal_children_queued = True
            
            # Use cython_chess to generate legal moves as strings for the parent
            for move in cython_chess.generate_legal_moves(node.parent.board, chess.BB_ALL, chess.BB_ALL):
                if move in node.parent.children and not node.parent.children[move].is_queued_for_inference:
                    all_legal_children_queued = False
                    break
            if all_legal_children_queued:
                node.parent.is_queued_for_inference = True
                logger.debug(f"Parent node {node.parent.move} now has all its legal children queued for inference.")


        logger.debug(f"Queued node for batched NN inference. Batch size: {len(self._inference_batch)}/{self.batch_size}")
        return True # Successfully queued


    def _perform_batched_inference(self):
        """
        Performs batched neural network inference on the collected states
        and expands the respective nodes.
        """
        if not self._inference_batch:
            return

        # Separate nodes and board tensors from the batch
        nodes_to_process = []
        board_tensors = []
        while self._inference_batch:
            node, board_tensor = self._inference_batch.popleft()
            nodes_to_process.append(node)
            board_tensors.append(board_tensor)

        # Stack board tensors to create a single batch input
        batch_input = torch.stack(board_tensors)
        
        start_nn_prediction = time.perf_counter()
        with torch.no_grad():
            policy_logits_batch, value_output_batch = self.model_player.model(batch_input)
        end_nn_prediction = time.perf_counter()
        self.timing_data["nn_prediction"] += (end_nn_prediction - start_nn_prediction)
        
        policy_probs_batch = F.softmax(policy_logits_batch, dim=1)

        logger.info(f"Performed batched NN inference for {len(nodes_to_process)} nodes.")

        # Distribute results back to the individual nodes
        for i, node in enumerate(nodes_to_process):
            policy_probs = policy_probs_batch[i].squeeze(0)
            value_for_current_node_player = value_output_batch[i].item()

            logger.debug(f"NN Result for node (Move: {node.move}): Value: {value_for_current_node_player:.4f}")
            
            # Now, expand the node using the policy probabilities
            start_expand = time.perf_counter()
            self.expand(node, policy_probs) # expand() will now reset node.is_queued_for_inference and all_children_queued_for_inference for THIS node
            end_expand = time.perf_counter()
            self.timing_data["expand"] += (end_expand - start_expand)

            # Store the value with the node for backpropagation.
            # The value is from the perspective of the player *whose turn it is at the node*.
            # This is crucial for correct backpropagation.
            self._pending_nodes[i] = (node, value_for_current_node_player)


    def backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagation phase: Update the visit count and value sum for all nodes
        from the simulated node up to the root.
        'value' is from the perspective of the player whose turn it was at the 'node'
        that was just simulated/evaluated.
        """
        current = node
        # The 'value' argument is always from the perspective of 'node.board.turn'.
        # We need to flip it for parent nodes if their turn is opposite.
        
        # Determine the initial player for whom the 'value' was obtained.
        # This player's perspective is used as the reference.
        initial_player_turn = node.board.turn 

        while current is not None:
            current.visits += 1
            
            # If the current node's turn is the same as the initial player (whose perspective 'value' is from)
            # then add the value directly.
            if current.board.turn == initial_player_turn:
                current.value_sum += value
                logger.debug(f"    Node (Move: {current.move}, Turn: {'White' if current.board.turn == chess.WHITE else 'Black'}): Added {value:.4f}. New Visits: {current.visits}, New Value Sum: {current.value_sum:.4f}")
            else:
                # If it's the opponent's turn, the value is inverted.
                current.value_sum -= value 
                logger.debug(f"    Node (Move: {current.move}, Turn: {'White' if current.board.turn == chess.WHITE else 'Black'}): Subtracted {value:.4f}. New Visits: {current.visits}, New Value Sum: {current.value_sum:.4f}")
            current = current.parent
        logger.debug("Backpropagation complete.")

class TalbotPlayerMCTS:
    def __init__(self, model_path: str, name="TalbotMCTS", num_residual_blocks: int = 20, time_per_move: float = 3.0, cpuct: float = 1.0, batch_size: int = 16):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.ChessAIModel(num_input_planes=18, num_residual_blocks=num_residual_blocks, num_filters=128)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict']) 
        logger.info(f"Model loaded successfully from {model_path}")
            
        self.model.to(self.device)
        self.model.eval() # Set the model to evaluation mode

        self.time_per_move = time_per_move
        self.cpuct = cpuct
        self.batch_size = batch_size

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Determines the best move using MCTS with a time limit.
        """
        if board.is_game_over():
            logger.info("Game is already over, no move to make.")
            return None

        mcts = MCTS(self, self.cpuct, self.batch_size)
        mcts.run_simulations(board.copy(), self.time_per_move) 

        best_move = None
        max_visits = -1 # AlphaZero-like final selection based on visit count

        logger.info("\n--- Final Move Selection from Root's Children ---")
        child_moves_data = []

        for move, child_node in mcts.root.children.items():
            # Only consider visited nodes for final selection if there are any,
            # otherwise, we might pick an unvisited one if no other option.
            if child_node.visits == 0:
                # logger.debug(f"Move {move} had 0 visits, skipping for final selection display.")
                continue

            # Q-value from the perspective of the player whose turn it is AT THE CHILD NODE
            # (which is the opponent of the root's player).
            q_value_opponent_perspective = child_node.value_sum / child_node.visits 
            
            # Q-value from the perspective of the PLAYER AT THE ROOT NODE (current player)
            # This is the value we would use for intuition (positive=good, negative=bad for current player)
            q_value_root_player_perspective = -q_value_opponent_perspective

            child_moves_data.append({
                'move': move,
                'visits': child_node.visits,
                'value_sum': child_node.value_sum,
                'q_value_opponent_perspective': q_value_opponent_perspective,
                'q_value_root_player_perspective': q_value_root_player_perspective 
            })

            # The actual selection criterion: choose the move with the most visits (AlphaZero)
            if child_node.visits > max_visits:
                max_visits = child_node.visits
                best_move = move
        
        # Sort by visits for clear presentation of the MCTS's preference
        child_moves_data.sort(key=lambda x: x['visits'], reverse=True)

        for data in child_moves_data:
            # Mark the move chosen by max_visits as the best
            is_best = " **(BEST SELECTED MOVE)**" if data['move'] == best_move else ""
            logger.info(f"Move: {data['move']}, Visits: {data['visits']}, "
                                 f"Value Sum: {data['value_sum']:.4f}, "
                                 f"Q-Value (Opponent Persp): {data['q_value_opponent_perspective']:.4f}, "
                                 f"Q-Value (Root Player Persp): {data['q_value_root_player_perspective']:.4f}{is_best}")
        logger.info("--------------------------------------------------")

        return best_move
    
    def is_human(self):
        return False

    def close(self):
        pass