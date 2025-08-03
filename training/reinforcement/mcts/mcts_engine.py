import chess
import cython_chess
import os
import torch
import torch.nn.functional as F
import sys
import math
import time
import logging
import numpy as np
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
    def __init__(self, logger: logging.Logger, model_player, cpuct: float, batch_size: int, dirichlet_alpha: float, dirichlet_epsilon: float):
        self.model_player = model_player
        self.logger = logger
        self.cpuct = cpuct
        self.root = None
        self.batch_size = batch_size
        self.inference_batch = deque()
        self.pending_nodes = []
        self.force_batch = False
        self.dirichlet_alpha=dirichlet_alpha
        self.dirichlet_epsilon=dirichlet_epsilon


    def _add_dirichlet_noise(self, policy_probs):
        """
        Adds Dirichlet noise to the policy probabilities for the root node.
        """
        num_moves = len(policy_probs)
        dirichlet_noise = torch.distributions.dirichlet.Dirichlet(torch.full((num_moves,), self.dirichlet_alpha, device=policy_probs.device)).sample()
        noisy_policy = (1 - self.dirichlet_epsilon) * policy_probs + self.dirichlet_epsilon * dirichlet_noise
        return noisy_policy
    

    def get_target_vectors(self):
        """
        Computes the policy vector from the visit counts of the root's children.
        """
        policy_vector = np.zeros(utils.TOTAL_POLICY_MOVES, dtype=np.float32)
        total_visits = sum(child.visits for child in self.root.children.values())
        
        for move, child_node in self.root.children.items():
            normalized_prob = child_node.visits / total_visits
            from_row, from_col, channel = utils.move_to_policy_components(move, self.root.board)
            flat_index = utils.policy_components_to_flat_index(from_row, from_col, channel)
            policy_vector[flat_index] = normalized_prob

        root_value = -self.root.value_sum / self.root.visits

        return policy_vector, root_value

    def set_new_root(self, board: chess.Board, last_move: chess.Move):
        """
        Updates the MCTS root based on the sequence of moves.
        Otherwise, a new tree is started.
        """
        if self.root is None:
            self.root = MCTSNode(board.copy())
            self.inference_batch.clear()
            self.pending_nodes.clear()
            self.force_batch = False
            return

        if last_move and last_move in self.root.children:
            new_root = self.root.children[last_move]
            new_root.parent = None
            if new_root.board is None:
                new_root.board = board.copy()
            self.root = new_root
            if self.root.is_expanded and self.root.prior_probabilities is not None:
                self.root.prior_probabilities = self._add_dirichlet_noise(self.root.prior_probabilities)

        self.inference_batch.clear()
        self.pending_nodes.clear()
        self.force_batch = False


    def run_simulations(self, search_depth):
        # Time tracking variables
        total_selection_time = 0.0
        total_simulation_time = 0.0
        total_inference_time = 0.0
        total_backpropagation_time = 0.0
        
        start_time_total = time.perf_counter()
        simulation_count = 0
        
        # Expand the root if it hasn't been yet to get initial policy and value
        if not self.root.is_expanded and not self.root.is_queued_for_inference:
            start_root_expansion_time = time.perf_counter()
            board_input = torch.from_numpy(utils.board_to_tensor_68(self.root.board)).float().to(self.model_player.device)
            policy_logits, value_output = self.model_player.get_policy_value(board_input.unsqueeze(0))
            policy_probs = F.softmax(policy_logits.squeeze(0), dim=0)
            policy_probs = self._add_dirichlet_noise(policy_probs)
            self.expand(self.root, policy_probs)
            self.backpropagate(self.root, value_output.item())
            total_inference_time += (time.perf_counter() - start_root_expansion_time)
            total_backpropagation_time += (time.perf_counter() - start_root_expansion_time)

        while simulation_count < search_depth:
            simulation_count += 1
            node = self.root
            path = [node]

            # Selection: Traverse the tree to find a leaf or unvisited node
            start_selection_time = time.perf_counter()
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

                for move, child in eligible_children:
                    prior_prob_for_child = child.prior_probability_from_parent
                    uct = child.uct_score(self.cpuct, prior_prob_for_child, sqrt_parent_visits_term)

                    if uct > best_uct_score:
                        best_uct_score = uct
                        best_prior_for_tie_break = prior_prob_for_child
                        best_child = child
                    elif uct == best_uct_score:
                        if prior_prob_for_child > best_prior_for_tie_break:
                            best_uct_score = uct
                            best_prior_for_tie_break = prior_prob_for_child
                            best_child = child
                    
                if best_child is None:
                    break

                node = best_child
                path.append(node)
            total_selection_time += (time.perf_counter() - start_selection_time)

            # Expansion/Simulation: Queue the selected leaf node for NN inference
            start_simulation_time = time.perf_counter()
            successfully_queued = self.simulate(node)
            total_simulation_time += (time.perf_counter() - start_simulation_time)
            
            # If a game-over state was reached or a node was already queued, and we have a batch, process it
            if not successfully_queued and not node.board.is_game_over() and self.inference_batch:
                self.force_batch = True
            else:
                self.force_batch = False

            # When batch is full or time is running out, run inference and backpropagate
            if len(self.inference_batch) >= self.batch_size or self.force_batch:
                start_inference_batch_time = time.perf_counter()
                self.perform_batched_inference()
                total_inference_time += (time.perf_counter() - start_inference_batch_time)

                start_backprop_batch_time = time.perf_counter()
                for processed_node, value_from_nn in self.pending_nodes:
                    self.backpropagate(processed_node, value_from_nn)
                self.pending_nodes.clear()
                total_backpropagation_time += (time.perf_counter() - start_backprop_batch_time)

        # Process any remaining nodes in the batch before finishing
        if self.inference_batch:
            start_final_inference_time = time.perf_counter()
            self.perform_batched_inference()
            total_inference_time += (time.perf_counter() - start_final_inference_time)

            start_final_backprop_time = time.perf_counter()
            for processed_node, value_from_nn in self.pending_nodes:
                self.backpropagate(processed_node, value_from_nn)
            self.pending_nodes.clear()
            total_backpropagation_time += (time.perf_counter() - start_final_backprop_time)

        total_elapsed_time = time.perf_counter() - start_time_total
        
        self.logger.debug(f"--- MCTS Move Analysis (Total) ---")
        self.logger.debug(f"Total Simulations: {simulation_count}")
        self.logger.debug(f"Total time spent: {total_elapsed_time:.4f}s")
        self.logger.debug(f"Avg time per simulation: {total_elapsed_time/simulation_count:.4f}s" if simulation_count > 0 else "Avg time per simulation: 0.0s")
        self.logger.debug(f"Selection time: {total_selection_time:.4f}s ({total_selection_time/total_elapsed_time*100:.2f}%)")
        self.logger.debug(f"Simulation/Queuing time: {total_simulation_time:.4f}s ({total_simulation_time/total_elapsed_time*100:.2f}%)")
        self.logger.debug(f"Inference time: {total_inference_time:.4f}s ({total_inference_time/total_elapsed_time*100:.2f}%)")
        self.logger.debug(f"Backpropagation time: {total_backpropagation_time:.4f}s ({total_backpropagation_time/total_elapsed_time*100:.2f}%)")
        self.logger.debug(f"-----------------------------------\n")

        # Log root children stats at the end of run_simulations
        if self.root and self.root.children:
            self.logger.debug(f"\n--- MCTS Root Children Analysis (Final State) ---")
            sorted_children = sorted(self.root.children.items(), key=lambda item: item[1].visits, reverse=True)
            total_visits = sum(child.visits for child in self.root.children.values())
            
            for move, child_node in sorted_children:
                q_value = -child_node.value_sum / child_node.visits if child_node.visits > 0 else 0.0
                normalized_prob = child_node.visits / total_visits if total_visits > 0 else 0.0
                
                log_message = (
                    f"Move: {move.uci()}, Visits: {child_node.visits}, "
                    f"Avg Q-value: {q_value:.4f}, Policy Prob: {normalized_prob:.4f}"
                )
                self.logger.debug(log_message)
            self.logger.debug("-----------------------------------\n")


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

        board_input = torch.from_numpy(utils.board_to_tensor_68(node.board)).float().to(self.model_player.device)
        self.inference_batch.append((node, board_input))
        self.pending_nodes.append(node)
        node.is_queued_for_inference = True

        current_node = node.parent
        while current_node is not None:
            all_legal_children_queued = True
            parent_legal_moves = cython_chess.generate_legal_moves(current_node.board, chess.BB_ALL, chess.BB_ALL)

            for move in parent_legal_moves:
                if move not in current_node.children or not current_node.children[move].is_queued_for_inference:
                    all_legal_children_queued = False
                    break 

            if all_legal_children_queued:
                current_node.is_queued_for_inference = True
            
            current_node = current_node.parent

        return True


    def perform_batched_inference(self):
        if not self.inference_batch:
            return

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
        current = node
        original_expanded_node_turn = node.board.turn

        while current is not None:
            if current.is_queued_for_inference:
                current.is_queued_for_inference = False

            current.visits += 1

            if current.board.turn == original_expanded_node_turn:
                current.value_sum += value
            else:
                current.value_sum -= value

            current = current.parent