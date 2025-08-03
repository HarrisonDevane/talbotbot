import chess
import cython_chess
import os
import torch
import torch.nn.functional as F
import sys
import math
import time
import logging
from .mcts_node import MCTSNode
from collections import deque

# Adjust path for internal modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, parent_dir)

import utils

class MCTSEngine:
    """
    Our Monte Carlo Tree Search brain for finding the best move.
    """
    def __init__(self, logger: logging.Logger, model_player, cpuct: float = 1.0, batch_size: int = 16):
        self.model_player = model_player
        self.logger = logger
        self.cpuct = cpuct
        self.root = None
        self.batch_size = batch_size
        self.inference_batch = deque()
        self.pending_nodes = []
        self.force_batch = False

    def set_new_root(self, board: chess.Board, opponent_last_move: chess.Move, our_last_move: chess.Move):
        """
        Updates the MCTS root based on the sequence of moves.
        If our_last_move and opponent_last_move allow traversal, the tree is kept.
        Otherwise, a new tree is started.
        """
        if self.root is None:
            self.root = MCTSNode(board.copy())
            self.logger.info("MCTS root initialized (first move).")
            self.inference_batch.clear()
            self.pending_nodes.clear()
            self.force_batch = False
            return

        new_root_found = False
        if our_last_move and our_last_move in self.root.children:
            node_after_our_move = self.root.children[our_last_move]
            if opponent_last_move and opponent_last_move in node_after_our_move.children:
                new_root = node_after_our_move.children[opponent_last_move]
                new_root.parent = None
                
                if new_root.board is None:
                    new_root.board = board.copy()
                
                self.root = new_root
                self.logger.info(f"MCTS root advanced by sequence: {our_last_move.uci()} -> {opponent_last_move.uci()}")
                new_root_found = True

        if not new_root_found:
            self.logger.warning(f"MCTS root reset due to unexpected move, no matching branch, or initial state beyond first move.")
            self.root = MCTSNode(board.copy())
        
        # Always clear pending batches if the root changes significantly
        self.inference_batch.clear()
        self.pending_nodes.clear()
        self.force_batch = False


    def run_simulations(self, time_limit: float):
        # The set_new_root method now handles initial board setup and consistency.

        start_time_total = time.perf_counter()
        simulation_count = 0
        
        # Expand the root if it hasn't been yet to get initial policy and value
        if not self.root.is_expanded and not self.root.is_queued_for_inference:
            board_input = torch.from_numpy(utils.board_to_tensor_68(self.root.board)).float().to(self.model_player.device)
            policy_logits, value_output = self.model_player.get_policy_value(board_input.unsqueeze(0))
            policy_probs = F.softmax(policy_logits.squeeze(0), dim=0)
            self.expand(self.root, policy_probs)
            self.backpropagate(self.root, value_output.item())

        while time.perf_counter() - start_time_total < time_limit:
            simulation_count += 1
            self.logger.debug(f"\n--- Simulation {simulation_count} Started ---")
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
                self.logger.debug(f"     Selecting child from Node ({parent_info}), Parent Visits: {node.visits}, Sqrt Parent Visits Term: {sqrt_parent_visits_term:.4f}")

                for move, child in eligible_children:
                    prior_prob_for_child = child.prior_probability_from_parent
                    uct = child.uct_score(self.cpuct, prior_prob_for_child, sqrt_parent_visits_term)

                    self.logger.debug(f'Current move: {move.uci()} with UCT: {uct:.4f}')

                    if uct > best_uct_score:
                        best_uct_score = uct
                        best_prior_for_tie_break = prior_prob_for_child
                        best_child = child
                    elif uct == best_uct_score:
                        if prior_prob_for_child > best_prior_for_tie_break:
                            best_uct_score = uct
                            best_prior_for_tie_break = prior_prob_for_child
                            best_child = child
                    
                self.logger.debug(f'Best move: {best_child.move.uci()} with UCT: {best_uct_score:.4f}')

                if best_child is None:
                    self.logger.debug("     No eligible child found for selection. Breaking from selection loop.")
                    break

                self.logger.debug(f"     Selected Node: Move: {best_child.move.uci()}, with UCT: {best_uct_score:.4f}, prior probability: {best_child.prior_probability_from_parent:.4f}")

                node = best_child
                path.append(node)

            # Expansion/Simulation: Queue the selected leaf node for NN inference
            successfully_queued = self.simulate(node)

            # If a game-over state was reached or a node was already queued, and we have a batch, process it
            if not successfully_queued and not node.board.is_game_over() and self.inference_batch:
                self.force_batch = True
            else:
                self.force_batch = False

            # When batch is full or time is running out, run inference and backpropagate
            if len(self.inference_batch) >= self.batch_size or \
               (time.perf_counter() - start_time_total >= time_limit and self.inference_batch) or \
               self.force_batch:

                self.perform_batched_inference()

                for processed_node, value_from_nn in self.pending_nodes:
                    self.backpropagate(processed_node, value_from_nn)
                self.pending_nodes.clear()


        # Process any remaining nodes in the batch before finishing
        if self.inference_batch:
            self.perform_batched_inference()

            for processed_node, value_from_nn in self.pending_nodes:
                self.backpropagate(processed_node, value_from_nn)
            self.pending_nodes.clear()

        total_elapsed_time = time.perf_counter() - start_time_total
        self.logger.info(f"MCTS completed {simulation_count} simulations in {total_elapsed_time:.4f}s.")

        # Log root children stats at the end of run_simulations
        if self.root and self.root.children:
            self.logger.info(f"\n--- MCTS Root Children Analysis (Final State) ---")
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
                self.logger.info(log_message)
            self.logger.info("-----------------------------------\n")


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
            self.logger.debug(f"     Simulate: Game over detected. Value: {value:.4f}")
            return False

        if node.is_queued_for_inference:
            self.logger.debug("     Simulate: Node already queued for inference. Skipping.")
            return False

        board_input = torch.from_numpy(utils.board_to_tensor_68(node.board)).float().to(self.model_player.device)
        self.inference_batch.append((node, board_input))
        self.pending_nodes.append(node)
        node.is_queued_for_inference = True

        # If all legal children of the parent are now queued, mark parent too
        current_node = node.parent

        while current_node is not None:
            all_legal_children_queued = True
            
            parent_legal_moves = cython_chess.generate_legal_moves(current_node.board, chess.BB_ALL, chess.BB_ALL)

            for move in parent_legal_moves:
                if move not in current_node.children or not current_node.children[move].is_queued_for_inference:
                    all_legal_children_queued = False
                    break 

            if all_legal_children_queued:
                # If all legal children are queued, mark this parent node too
                current_node.is_queued_for_inference = True
                self.logger.debug(f"    Simulate: Node {current_node.move} (parent of a fully queued subtree) also marked as queued for inference.")
            
            # Move up to the next parent to continue the check
            current_node = current_node.parent

        return True


    def perform_batched_inference(self):
        if not self.inference_batch:
            return

        self.logger.debug(f"   _perform_batched_inference: Processing batch of size {len(self.inference_batch)}")
        nodes_to_process = []
        board_tensors = []
        while self.inference_batch:
            node, board_tensor = self.inference_batch.popleft()
            nodes_to_process.append(node)
            board_tensors.append(board_tensor)

        batch_input = torch.stack(board_tensors)

        policy_logits_batch, value_output_batch = self.model_player.get_policy_value(batch_input)

        policy_probs_batch = F.softmax(policy_logits_batch, dim=1)

        temp_pending_nodes = []
        for i, node in enumerate(nodes_to_process):
            policy_probs = policy_probs_batch[i].squeeze(0)
            value_for_current_node_player = value_output_batch[i].item()

            self.expand(node, policy_probs)
            temp_pending_nodes.append((node, value_for_current_node_player))

        self.pending_nodes = temp_pending_nodes


    def backpropagate(self, node: MCTSNode, value: float):
        """
        Update visit counts and value sums along the path from the node up to the root.
        Value is from the perspective of the player whose turn it is at 'node.board'.
        """
        current = node
        original_expanded_node_turn = node.board.turn

        while current is not None:
            if current.is_queued_for_inference:
                self.logger.debug(f"     Resetting is_queued_for_inference for node (in backprop)")
                current.is_queued_for_inference = False

            current.visits += 1

            if current.board.turn == original_expanded_node_turn:
                current.value_sum += value
            else:
                current.value_sum -= value

            current = current.parent