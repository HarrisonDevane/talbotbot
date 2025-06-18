import chess
import os
import torch
import torch.nn.functional as F
import sys
import math
import random
import time
import logging # Import the logging module

# Get the parent directory path to help with relative imports and paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

import utils
import training.supervised.model as model

# --- Logging Configuration ---
# Determine the base directory for logs
# Assuming this script is in 'file_dir', and the model path is relative to it
log_dir = os.path.abspath(os.path.join(parent_dir, "training/supervised/v2_pol_mvplayed_val_sfeval/logs/"))
print(log_dir)
os.makedirs(log_dir, exist_ok=True) # Ensure the directory exists

log_file_path = os.path.join(log_dir, "mcts_debug.log")

# Configure the logger
logging.basicConfig(
    level=logging.INFO, # Set the logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a' # 'a' for append, 'w' for overwrite
)
logger = logging.getLogger(__name__)
# If you also want to see logs in the console while running
# Uncomment the following two lines:
# console_handler = logging.StreamHandler(sys.stdout)
# logger.addHandler(console_handler)
# --- End Logging Configuration ---


class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search tree.
    Each node corresponds to a specific board state.
    """
    def __init__(self, board: chess.Board, parent=None, move: chess.Move = None):
        self.board = board
        self.parent = parent
        self.move = move  # The move that led to this board state from the parent
        self.children = {}  # Map of move -> MCTSNode
        self.visits = 0
        self.value_sum = 0.0  # Sum of values (win/loss/draw) accumulated during simulations
        self.prior_probabilities = None  # Policy probabilities from the neural network for this node's state
        self.is_expanded = False

    def is_leaf(self) -> bool:
        return not self.children and self.is_expanded

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
    def __init__(self, model_player: 'TalbotPlayerMCTS', cpuct: float = 1.0):
        self.model_player = model_player
        self.cpuct = cpuct
        self.root = None

    def run_simulations(self, initial_board: chess.Board, time_limit: float):
        self.root = MCTSNode(initial_board.copy())
        start_time = time.time()
        
        simulation_count = 0
        while time.time() - start_time < time_limit:
            node = self.select(self.root)
            value = self.simulate(node)
            self.backpropagate(node, value)
            simulation_count += 1
        
        logger.info(f"MCTS completed {simulation_count} simulations within {time_limit} seconds.")

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Traverse the tree from the root, selecting children with the highest UCT score.
        """
        if node.is_root():
            logger.info(f"\n--- Starting MCTS Selection from Root (Board Turn: {'White' if node.board.turn == chess.WHITE else 'Black'}) ---")
        else:
            logger.info(f"\n--- MCTS Selection (Current Node Move: {node.move}, Board Turn: {'White' if node.board.turn == chess.WHITE else 'Black'}) ---")

        while not node.is_leaf() and node.is_expanded:
            if not node.children:
                logger.info("No children in current node, breaking selection.")
                break

            best_child = None
            best_uct_score = -float('inf')

            legal_moves = list(node.board.legal_moves)
            legal_children = {move: child for move, child in node.children.items() if move in legal_moves}

            if not legal_children:
                logger.info("No legal children found for selection, breaking.")
                break

            logger.info(f"Parent Visits: {node.visits}. Evaluating children for selection:")
            
            evaluated_children_data = [] 
            for move, child in legal_children.items():
                if child.visits == 0: # Prioritize unvisited nodes
                    best_child = child
                    logger.info(f"    Move: {move}, Visits: {child.visits}, UCT Score: INF (unvisited) - **SELECTED**")
                    break
                
                try:
                    from_row_norm, from_col_norm, channel = utils.move_to_policy_components(move, node.board)
                    flat_index = utils.policy_components_to_flat_index(from_row_norm, from_col_norm, channel)
                    prior_prob_for_child = node.prior_probabilities[flat_index].item()
                except ValueError:
                    prior_prob_for_child = 0.0
                
                uct = child.uct_score(node.visits, self.cpuct, prior_prob_for_child)
                
                evaluated_children_data.append({
                    'move': move,
                    'visits': child.visits,
                    'value_sum': child.value_sum, # ADDED THIS LINE
                    'q_value': child.value_sum / child.visits,
                    'prior_prob': prior_prob_for_child,
                    'uct_score': uct
                })

                if uct > best_uct_score:
                    best_uct_score = uct
                    best_child = child
            
            if evaluated_children_data and best_child:
                # Sort for better readability, by UCT score
                evaluated_children_data.sort(key=lambda x: x['uct_score'], reverse=True)
                for data in evaluated_children_data:
                    is_selected = " **(SELECTED)**" if data['move'] == best_child.move else ""
                    # MODIFIED THE LOGGING STRING HERE
                    logger.info(f"    Move: {data['move']}, Visits: {data['visits']}, Value Sum: {data['value_sum']:.4f}, Q-Value: {data['q_value']:.4f}, "
                                f"Prior: {data['prior_prob']:.4f}, UCT Score: {data['uct_score']:.4f}{is_selected}")

            if best_child is None:
                logger.info("No best child selected in this iteration, breaking selection.")
                break
            node = best_child
        return node

    def expand(self, node: MCTSNode, policy_probs: torch.Tensor):
        """
        Expansion phase: If a node is visited for the first time, expand it by creating children
        for all legal moves and initializing their prior probabilities from the neural network's policy output.
        """
        logger.info(f"\n--- MCTS Expansion Phase (Node Move: {node.move}, Board Turn: {'White' if node.board.turn == chess.WHITE else 'Black'}) ---")

        if node.board.is_game_over():
            node.is_expanded = True
            logger.info("Node represents a game over state, no expansion needed.")
            return

        legal_moves = list(node.board.legal_moves)
        
        node.prior_probabilities = torch.zeros_like(policy_probs, dtype=torch.float)

        if not legal_moves:
            node.is_expanded = True
            logger.info("No legal moves for this board state, no expansion possible.")
            return

        mapped_legal_moves_count = 0
        logger.info(f"Processing {len(legal_moves)} legal moves for expansion:")

        for move in legal_moves:
            try:
                from_row_norm, from_col_norm, channel = utils.move_to_policy_components(move, node.board)
                flat_index = utils.policy_components_to_flat_index(from_row_norm, from_col_norm, channel)
                
                new_board = node.board.copy()
                new_board.push(move)
                child_node = MCTSNode(new_board, parent=node, move=move)
                node.children[move] = child_node
                
                node.prior_probabilities[flat_index] = policy_probs[flat_index].item()
                mapped_legal_moves_count += 1
                logger.info(f"    - Created child for move {move}, NN Prior Prob: {policy_probs[flat_index].item():.4f}")
            except ValueError as e:
                logger.warning(f"    - Warning: Could not convert legal move {move.uci()} to policy components during expansion: {e}")
                pass
        
        if node.prior_probabilities.sum() > 0:
            original_sum = node.prior_probabilities.sum()
            node.prior_probabilities /= original_sum
            logger.info(f"Normalized prior probabilities for legal moves. Original sum: {original_sum:.4f}, New sum: {node.prior_probabilities.sum():.4f}")
        elif mapped_legal_moves_count == 0 and legal_moves: 
            logger.warning("No legal moves could be mapped to policy indices during expansion. Distributing probabilities evenly as fallback.")
            for move in legal_moves:
                try:
                    frn, fcn, ch = utils.move_to_policy_components(move, node.board)
                    flat_idx = utils.policy_components_to_flat_index(frn, fcn, ch)
                    node.prior_probabilities[flat_idx] = 1.0 / len(legal_moves)
                except ValueError:
                    pass 
            if node.prior_probabilities.sum() > 0: 
                node.prior_probabilities /= node.prior_probabilities.sum()
                logger.info("Fallback normalization applied.")

        node.is_expanded = True
        logger.info("Node expansion complete.")

    def simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase: If a node is a leaf, use the neural network to get
        its policy and value. The value is then used to backpropagate.
        """
        logger.info(f"\n--- MCTS Simulation Phase (Node Move: {node.move}, Board Turn: {'White' if node.board.turn == chess.WHITE else 'Black'}) ---")

        if node.board.is_game_over():
            result = node.board.result()
            value = 0.0
            if result == "1-0":  # White wins
                value = 1.0 if node.board.turn == chess.BLACK else -1.0 # Value from perspective of current node's player
            elif result == "0-1":  # Black wins
                value = 1.0 if node.board.turn == chess.WHITE else -1.0 # Value from perspective of current node's player
            else:  # Draw
                value = 0.0
            logger.info(f"Game over state detected. Result: {result}, Value from current player's perspective: {value:.4f}")
            return value

        # Convert board to tensor for the model
        board_tensor_np = utils.board_to_tensor(node.board)
        board_input = torch.from_numpy(board_tensor_np).unsqueeze(0).float().to(self.model_player.device)

        with torch.no_grad():
            policy_logits, value_output = self.model_player.model(board_input)

        # Policy probabilities for expansion
        policy_probs = F.softmax(policy_logits, dim=1).squeeze(0)

        # If your model's value_output is ALREADY from the current player's perspective:
        value_for_current_node_player = value_output.item() 
        
        logger.info(f"Neural Network Prediction: Value (Current Player's perspective): {value_for_current_node_player:.4f}")
            
        if not node.is_expanded:
            logger.info("Node was not expanded, initiating expansion.")
            self.expand(node, policy_probs)

        return value_for_current_node_player
            

    def backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagation phase: Update the visit count and value sum for all nodes
        from the simulated node up to the root.
        'value' is from the perspective of the player whose turn it was at the 'node'
        that was just simulated.
        """
        logger.info(f"\n--- MCTS Backpropagation Phase (Starting from Node Move: {node.move}, Value: {value:.4f}) ---")

        current = node
        while current is not None:
            current.visits += 1
            
            # The value should be added from the perspective of the player
            # whose turn it is at the 'current' node.
            if current.board.turn == node.board.turn: # 'node' is the simulated node
                current.value_sum += value
                logger.info(f"    Node (Move: {current.move}, Turn: {'White' if current.board.turn == chess.WHITE else 'Black'}): Added {value:.4f}. New Visits: {current.visits}, New Value Sum: {current.value_sum:.4f}")
            else:
                current.value_sum -= value # It's the opponent's turn, so their value is opposite
                logger.info(f"    Node (Move: {current.move}, Turn: {'White' if current.board.turn == chess.WHITE else 'Black'}): Subtracted {value:.4f}. New Visits: {current.visits}, New Value Sum: {current.value_sum:.4f}")
            current = current.parent
        logger.info("Backpropagation complete.")

class TalbotPlayerMCTS:
    def __init__(self, model_path: str, name="TalbotMCTS", time_per_move: float = 10.0, cpuct: float = 1.0):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.ChessAIModel(num_input_planes=18, num_residual_blocks=16, num_filters=128)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict']) 
        
        self.model.to(self.device)
        self.model.eval() # Set the model to evaluation mode

        self.time_per_move = time_per_move
        self.cpuct = cpuct

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Determines the best move using MCTS with a time limit.
        """
        if board.is_game_over():
            return None

        mcts = MCTS(self, self.cpuct)
        mcts.run_simulations(board.copy(), self.time_per_move) 

        if not mcts.root.children:
            logger.info("MCTS root has no children. Choosing a random legal move as fallback.")
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
            else:
                return None

        best_move = None
        max_visits = -1 # This is correct for AlphaZero-like final selection

        logger.info("\n--- Final Move Selection from Root's Children ---")
        child_moves_data = []

        for move, child_node in mcts.root.children.items():
            # Only consider visited nodes for final selection, though max_visits handles it.
            if child_node.visits == 0:
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
                'q_value_root_player_perspective': q_value_root_player_perspective # Store both
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

        if best_move is None:
            logger.info("MCTS did not select a best move. Falling back to random legal move.")
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
            else:
                return None

        # Optional: Print winning probability from the root's value (as perceived by the NN)
        board_tensor_np = utils.board_to_tensor(board)
        board_input = torch.from_numpy(board_tensor_np).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            _, value_output = self.model(board_input) # Use self.model instead of self.model_player.model
                                                       # since TalbotPlayerMCTS holds the model directly
                
        prob_current_player_winning = utils.get_win_probability(value_output)
        
        if board.turn == chess.WHITE:
            logger.info(f'Probability White is winning (from initial model prediction): {prob_current_player_winning:.4f}')
        else:
            logger.info(f'Probability Black is winning (from initial model prediction): {prob_current_player_winning:.4f}')

        return best_move
    
    def is_human(self):
        return False

    def close(self):
        pass