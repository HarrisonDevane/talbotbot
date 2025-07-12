import chess
import cython_chess
import sys
import os
import torch
import torch.nn.functional as F
import time
import math
import logging
from .mcts_node import MCTSNode
from collections import deque
from .inference_queue import InferenceQueue  # global batching system

# Adjust path for internal modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

import utils

class MCTSEngineBatched:
    def __init__(self, logger: logging.Logger, model_player, cpuct: float = 1.0):
        """
        Initialize the MCTSEngineBatched instance.

        Args:
            logger (logging.Logger): Logger instance for logging information.
            model_player: The model player used for inference.
            cpuct (float): The exploration constant used in the UCT formula.
        """
        self.model_player = model_player
        self.logger = logger
        self.cpuct = cpuct
        self.root = None

    def set_new_root(self, board: chess.Board, opponent_last_move: chess.Move, our_last_move: chess.Move):
        """
        Set the new root node of the MCTS tree based on the last two moves played.

        Args:
            board (chess.Board): The current board state.
            opponent_last_move (chess.Move): The opponent's last move.
            our_last_move (chess.Move): Our last move.
        """
        if self.root is None:
            self.root = MCTSNode(board.copy())
            self.logger.info("MCTS root initialized (first move).")
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
            self.logger.warning("MCTS root reset due to unexpected move or no matching branch.")
            self.root = MCTSNode(board.copy())

    def run_simulations(self, time_limit: float):
        """
        Run MCTS simulations for a specified amount of time.

        Args:
            time_limit (float): The time limit (in seconds) to run simulations.
        """
        start_time = time.perf_counter()
        simulation_count = 0

        if not self.root.is_expanded and not self.root.is_queued_for_inference:
            self.expand_and_backprop(self.root)

        while time.perf_counter() - start_time < time_limit:
            simulation_count += 1
            self.logger.debug(f"\n--- Simulation {simulation_count} Started ---")

            node = self.root
            path = [node]

            while not node.is_leaf() and node.is_expanded and not node.is_queued_for_inference:
                best_child = self.select_best_child(node)
                if best_child is None:
                    break
                node = best_child
                path.append(node)

            if not self.simulate(node):
                break

        self.logger.info(f"MCTS completed {simulation_count} simulations in {time.perf_counter() - start_time:.4f}s.")

    def select_best_child(self, node: MCTSNode):
        """
        Select the child node with the highest UCT score.

        Args:
            node (MCTSNode): The node from which to select the best child.

        Returns:
            MCTSNode: The best child node.
        """
        best_uct = -float('inf')
        best_child = None
        sqrt_visits = math.sqrt(node.visits) if node.visits > 0 else 0.0

        for move, child in node.children.items():
            if child.is_queued_for_inference:
                continue
            uct = child.uct_score(self.cpuct, child.prior_probability_from_parent, sqrt_visits)
            if uct > best_uct:
                best_uct = uct
                best_child = child
        return best_child

    def simulate(self, node: MCTSNode) -> bool:
        """
        Perform one simulation from the given node.

        Args:
            node (MCTSNode): The node from which to start the simulation.

        Returns:
            bool: True if the simulation proceeded, False otherwise.
        """
        if node.board.is_game_over():
            result = node.board.result()
            value = 0.0
            if result == "1-0":
                value = 1.0 if node.board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                value = 1.0 if node.board.turn == chess.BLACK else -1.0
            self.backpropagate(node, value)
            return False

        if node.is_queued_for_inference:
            return False

        InferenceQueue.queue_inference(node, node.board, self.model_player.device)
        node.is_queued_for_inference = True

        policy_probs, value = InferenceQueue.get_result(node)
        self.expand(node, policy_probs)
        self.backpropagate(node, value)
        return True

    def expand_and_backprop(self, node: MCTSNode):
        """
        Expand a node using inference and then backpropagate its value.

        Args:
            node (MCTSNode): The node to expand and backpropagate from.
        """
        InferenceQueue.queue_inference(node, node.board, self.model_player.device)
        node.is_queued_for_inference = True
        policy_probs, value = InferenceQueue.get_result(node)
        self.expand(node, policy_probs)
        self.backpropagate(node, value)

    def expand(self, node: MCTSNode, policy_probs: torch.Tensor):
        """
        Expand the given node by creating its children using the provided policy probabilities.

        Args:
            node (MCTSNode): The node to expand.
            policy_probs (torch.Tensor): The policy output from the model.
        """
        if node.board.is_game_over():
            node.is_expanded = True
            node.is_queued_for_inference = False
            return

        legal_moves = cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL)

        from_row_ints, from_col_ints, channel_ints = [], [], []
        child_nodes_in_order = []

        for move in legal_moves:
            f_row, f_col, chan = utils.move_to_policy_components(move, node.board)
            from_row_ints.append(f_row)
            from_col_ints.append(f_col)
            channel_ints.append(chan)

            child_node = MCTSNode(board=None, parent=node, move=move)
            node.children[move] = child_node
            child_nodes_in_order.append(child_node)

        node.prior_probabilities = torch.zeros_like(policy_probs, dtype=torch.float)

        if legal_moves:
            indices = utils.policy_components_to_flat_index_torch(
                torch.tensor(from_row_ints), torch.tensor(from_col_ints), torch.tensor(channel_ints)
            ).to(policy_probs.device)

            legal_prior = policy_probs[indices]
            total = legal_prior.sum()
            norm_prior = legal_prior / total if total > 0 else legal_prior

            node.prior_probabilities.index_put_((indices,), norm_prior)

            for i, child_node in enumerate(child_nodes_in_order):
                child_node.prior_probability_from_parent = norm_prior[i].item()

        node.is_expanded = True
        node.is_queued_for_inference = False

    def backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagate the simulation result up the tree.

        Args:
            node (MCTSNode): The leaf node to start backpropagation from.
            value (float): The result of the simulation (from the model or game outcome).
        """
        current = node
        orig_turn = node.board.turn

        while current is not None:
            current.visits += 1
            if current.board.turn == orig_turn:
                current.value_sum += value
            else:
                current.value_sum -= value
            current.is_queued_for_inference = False
            current = current.parent
