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

# Adjust path for internal modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

import utils
import training.supervised.model as model

logger = logging.getLogger(__name__)


class MCTSNode:
    """
    Represents a single state (board position) in our MCTS tree.
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
        # Lazily create board to save memory
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
    Our Monte Carlo Tree Search brain for finding the best move.
    """
    def __init__(self, model_player: 'TalbotMCTSEngine', cpuct: float = 1.0, batch_size: int = 16):
        self.model_player = model_player
        self.cpuct = cpuct
        self.root = None
        self.batch_size = batch_size
        self._inference_batch = deque()
        self._pending_nodes = []
        self.force_batch = False

    def set_new_root(self, board: chess.Board, opponent_last_move: chess.Move, our_last_move: chess.Move):
        """
        Updates the MCTS root based on the sequence of moves.
        If our_last_move and opponent_last_move allow traversal, the tree is kept.
        Otherwise, a new tree is started.
        """
        if self.root is None:
            self.root = MCTSNode(board.copy())
            logger.info("MCTS root initialized (first move).")
            # Clear any stale queues from previous games
            self._inference_batch.clear()
            self._pending_nodes.clear()
            self.force_batch = False
            return

        new_root_found = False
        if our_last_move and our_last_move in self.root.children:
            node_after_our_move = self.root.children[our_last_move]
            if opponent_last_move and opponent_last_move in node_after_our_move.children:
                new_root = node_after_our_move.children[opponent_last_move]
                new_root.parent = None
                
                if new_root._board is None:
                    new_root._board = board.copy()
                
                self.root = new_root
                logger.info(f"MCTS root advanced by sequence: {our_last_move.uci()} -> {opponent_last_move.uci()}")
                new_root_found = True

        if not new_root_found:
            logger.warning(f"MCTS root reset due to unexpected move, no matching branch, or initial state beyond first move.")
            self.root = MCTSNode(board.copy())
        
        # Always clear pending batches if the root changes significantly
        self._inference_batch.clear()
        self._pending_nodes.clear()
        self.force_batch = False


    def run_simulations(self, time_limit: float):
        # The set_new_root method now handles initial board setup and consistency.

        start_time_total = time.perf_counter()
        simulation_count = 0
        
        # Expand the root if it hasn't been yet to get initial policy and value
        if not self.root.is_expanded and not self.root.is_queued_for_inference:
            board_input = torch.from_numpy(utils.board_to_tensor(self.root.board)).float().to(self.model_player.device)
            # CALLING THE NEW get_policy_value METHOD
            policy_logits, value_output = self.model_player.get_policy_value(board_input.unsqueeze(0))
            policy_probs = F.softmax(policy_logits.squeeze(0), dim=0)
            self.expand(self.root, policy_probs)
            self.backpropagate(self.root, value_output.item())

        while time.perf_counter() - start_time_total < time_limit:
            simulation_count += 1
            logger.debug(f"\n--- Simulation {simulation_count} Started ---")
            node = self.root
            path = [node]

            # Selection: Traverse the tree to find a leaf or unvisited node
            while not node.is_leaf() and node.is_expanded and \
                    not node.is_queued_for_inference:

                best_child = None
                best_uct_score = -float('inf')
                best_prior_for_tie_break = -1.0

                legal_moves = cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL)
                eligible_children = []
                for move in legal_moves:
                    if move in node.children and not node.children[move].is_queued_for_inference:
                        eligible_children.append((move, node.children[move]))

                sqrt_parent_visits_term = math.sqrt(node.visits) if node.visits > 0 else 0.0

                parent_info = f"Move: {node.move.uci()}" if node.move else "Root"
                logger.debug(f"     Selecting child from Node ({parent_info}), Parent Visits: {node.visits}, Sqrt Parent Visits Term: {sqrt_parent_visits_term:.4f}")

                for move, child in eligible_children:
                    prior_prob_for_child = child.prior_probability_from_parent
                    uct = child.uct_score(self.cpuct, prior_prob_for_child, sqrt_parent_visits_term)

                    logger.debug(f'Current move: {move.uci()} with UCT: {uct:.4f}')

                    if uct > best_uct_score:
                        best_uct_score = uct
                        best_prior_for_tie_break = prior_prob_for_child
                        best_child = child
                    elif uct == best_uct_score:
                        if prior_prob_for_child > best_prior_for_tie_break:
                            best_uct_score = uct
                            best_prior_for_tie_break = prior_prob_for_child
                            best_child = child
                    
                logger.debug(f'Best move: {best_child.move.uci()} with UCT: {best_uct_score:.4f}')

                if best_child is None:
                    logger.debug("     No eligible child found for selection. Breaking from selection loop.")
                    break

                logger.debug(f"     Selected Node: Move: {best_child.move.uci()}, with UCT: {best_uct_score:.4f}, prior probability: {best_child.prior_probability_from_parent:.4f}")

                node = best_child
                path.append(node)

            # Expansion/Simulation: Queue the selected leaf node for NN inference
            successfully_queued = self.simulate(node)

            # If a game-over state was reached or a node was already queued, and we have a batch, process it
            if not successfully_queued and not node.board.is_game_over() and self._inference_batch:
                self.force_batch = True
            else:
                self.force_batch = False

            # When batch is full or time is running out, run inference and backpropagate
            if len(self._inference_batch) >= self.batch_size or \
               (time.perf_counter() - start_time_total >= time_limit and self._inference_batch) or \
               self.force_batch:

                self._perform_batched_inference()

                for processed_node, value_from_nn in self._pending_nodes:
                    self.backpropagate(processed_node, value_from_nn)
                self._pending_nodes.clear()


        # Process any remaining nodes in the batch before finishing
        if self._inference_batch:
            self._perform_batched_inference()

            for processed_node, value_from_nn in self._pending_nodes:
                self.backpropagate(processed_node, value_from_nn)
            self._pending_nodes.clear()

        total_elapsed_time = time.perf_counter() - start_time_total
        logger.info(f"MCTS completed {simulation_count} simulations in {total_elapsed_time:.4f}s.")

        # Log root children stats at the end of run_simulations
        if self.root and self.root.children:
            logger.info(f"\n--- MCTS Root Children Analysis (Final State) ---")
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
                logger.info(log_message)
            logger.info("-----------------------------------\n")


    def expand(self, node: MCTSNode, policy_probs: torch.Tensor):
        if node.board.is_game_over():
            node.is_expanded = True
            node.is_queued_for_inference = False
            return

        legal_moves = cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL)
        
        from_row_ints, from_col_ints, channel_ints = [], [], []
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

            normalized_legal_priors = prior_values_for_legal_moves / sum_of_legal_priors if sum_of_legal_priors > 0 else prior_values_for_legal_moves
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
        If a node is a leaf, we queue it for neural network inference.
        If it's a game-over state, we calculate the value directly.
        Returns True if queued, False otherwise.
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
            logger.debug(f"     Simulate: Game over detected. Value: {value:.4f}")
            return False

        if node.is_queued_for_inference:
            logger.debug("     Simulate: Node already queued for inference. Skipping.")
            return False

        board_input = torch.from_numpy(utils.board_to_tensor(node.board)).float().to(self.model_player.device)
        self._inference_batch.append((node, board_input))
        self._pending_nodes.append(node)
        node.is_queued_for_inference = True

        # If all legal children of the parent are now queued, mark parent too
        if node.parent:
            all_legal_children_queued = True
            parent_legal_moves = cython_chess.generate_legal_moves(node.parent.board, chess.BB_ALL, chess.BB_ALL)

            for move in parent_legal_moves:
                if move in node.parent.children and not node.parent.children[move].is_queued_for_inference:
                    all_legal_children_queued = False
                    break
            if all_legal_children_queued:
                node.parent.is_queued_for_inference = True
                logger.debug(f"     Simulate: Parent node also marked as queued for inference.")

        return True


    def _perform_batched_inference(self):
        if not self._inference_batch:
            return

        logger.debug(f"   _perform_batched_inference: Processing batch of size {len(self._inference_batch)}")
        nodes_to_process = []
        board_tensors = []
        while self._inference_batch:
            node, board_tensor = self._inference_batch.popleft()
            nodes_to_process.append(node)
            board_tensors.append(board_tensor)

        batch_input = torch.stack(board_tensors)

        # CALLING THE NEW get_policy_value METHOD
        policy_logits_batch, value_output_batch = self.model_player.get_policy_value(batch_input)

        policy_probs_batch = F.softmax(policy_logits_batch, dim=1)

        temp_pending_nodes = []
        for i, node in enumerate(nodes_to_process):
            policy_probs = policy_probs_batch[i].squeeze(0)
            value_for_current_node_player = value_output_batch[i].item()

            self.expand(node, policy_probs)
            temp_pending_nodes.append((node, value_for_current_node_player))

        self._pending_nodes = temp_pending_nodes


    def backpropagate(self, node: MCTSNode, value: float):
        """
        Update visit counts and value sums along the path from the node up to the root.
        Value is from the perspective of the player whose turn it is at 'node.board'.
        """
        current = node
        original_expanded_node_turn = node.board.turn

        while current is not None:
            if current.is_queued_for_inference:
                logger.debug(f"     Resetting is_queued_for_inference for node (in backprop)")
                current.is_queued_for_inference = False

            current.visits += 1

            if current.board.turn == original_expanded_node_turn:
                current.value_sum += value
            else:
                current.value_sum -= value

            current = current.parent


class TalbotMCTSEngine:
    def __init__(self, model_path: str, name="Talbot", num_residual_blocks: int = 20, cpuct: float = 1.0, batch_size: int = 16):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.ChessAIModel(num_input_planes=18, num_residual_blocks=num_residual_blocks, num_filters=128)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.debug(f"Model loaded successfully from {model_path}")

        self.model.to(self.device)
        self.model.eval()

        self.cpuct = cpuct
        self.batch_size = batch_size
        self._mcts = None 
        self._last_own_move = None
        self._move_number = 0

    def get_policy_value(self, board_tensor: torch.Tensor):
        with torch.no_grad():
            policy_logits, value_output = self.model(board_tensor)
        return policy_logits, value_output

    def find_best_move(self, board: chess.Board, time_per_move: float = None, depth_limit: int = None) -> chess.Move:
        self._move_number += 1
        logger.info(f"\n{'='*60}\n{' '*20}--- MOVE {self._move_number} STARTED ---\n{'='*60}\n")

        if board.is_game_over():
            logger.info("Game is already over, no move to make.")
            return None

        opponent_last_move = None
        if board.move_stack:
            opponent_last_move = board.move_stack[-1]

        if self._mcts is None:
            self._mcts = MCTS(self, self.cpuct, self.batch_size)
            self._mcts.set_new_root(board.copy(), None, None) 
        else:
            self._mcts.set_new_root(board.copy(), opponent_last_move, self._last_own_move)

        # You can add logic here to use depth_limit if provided, e.g.,
        # self._mcts.run_simulations_by_depth(depth_limit) instead of time_limit
        self._mcts.run_simulations(time_per_move)

        best_move = None
        max_visits = -1
        
        for move, child_node in self._mcts.root.children.items():
            if child_node.visits > max_visits:
                max_visits = child_node.visits
                best_move = move

        if best_move is None:
            legal_moves = cython_chess.generate_legal_moves(board, chess.BB_ALL, chess.BB_ALL)
            if legal_moves:
                best_move = random.choice(list(legal_moves))
            else:
                logger.error("No legal moves available. This shouldn't happen unless the game is over.")
                return None
        
        self._last_own_move = best_move

        logger.info(f"MCTS for move {self._move_number} picked move: {best_move.uci()} with {max_visits} visits.")
        return best_move