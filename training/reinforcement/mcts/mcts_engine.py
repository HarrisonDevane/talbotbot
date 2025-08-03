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
import threading
import queue

# Adjust path for internal modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, parent_dir)

import utils

class MCTSEngine:
    """
    Our Monte Carlo Tree Search brain for finding the best move,
    now with multithreaded workers for parallel selection and queuing.
    """
    def __init__(self, logger: logging.Logger, model_player, cpuct: float, batch_size: int, dirichlet_alpha: float, dirichlet_epsilon: float, num_threads: int):
        self.model_player = model_player
        self.logger = logger
        self.cpuct = cpuct
        self.root = None
        self.num_threads = num_threads
        self.dirichlet_alpha=dirichlet_alpha
        self.dirichlet_epsilon=dirichlet_epsilon
        
        # Thread-safe queues for communication between workers and inference thread
        self.inference_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.batch_size = batch_size
        self.stop_threads = threading.Event()
        self.tree_lock = threading.Lock()
        self.queue_lock = threading.Lock()


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
        with self.tree_lock:
            if self.root is None:
                self.root = MCTSNode(board.copy())
                self.logger.debug("MCTSEngine: New root node created.")
                return

            if last_move and last_move in self.root.children:
                self.logger.debug(f"MCTSEngine: Root changed to child for move {last_move.uci()}.")
                new_root = self.root.children[last_move]
                new_root.parent = None
                if new_root.board is None:
                    new_root.board = board.copy()
                self.root = new_root
                if self.root.is_expanded and self.root.prior_probabilities is not None:
                    self.root.prior_probabilities = self._add_dirichlet_noise(self.root.prior_probabilities)
            else:
                self.logger.warning(f"MCTSEngine: Last move {last_move.uci()} not in current root's children. Rebuilding tree from scratch.")
                self.root = MCTSNode(board.copy())


    def run_simulations(self, search_depth):
        
        self.stop_threads.clear()
        
        self.logger.info(f"Starting {search_depth} simulations with {self.num_threads} worker threads.")

        # Start the inference worker thread
        inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        inference_thread.start()
        self.logger.debug("Inference worker thread started.")

        # Start the MCTS worker threads
        worker_threads = []
        sims_per_thread = search_depth // self.num_threads
        for i in range(self.num_threads):
            worker_thread = threading.Thread(target=self._mcts_worker, args=(sims_per_thread, i,), daemon=True)
            worker_threads.append(worker_thread)
            worker_thread.start()
            self.logger.debug(f"MCTS worker thread {i} started, assigned {sims_per_thread} simulations.")

        # Wait for all worker threads to finish
        for thread in worker_threads:
            thread.join()
        
        self.logger.debug("All MCTS worker threads have finished their simulations.")

        # Signal inference thread to stop and wait for it to finish its last batch
        self.stop_threads.set()
        self.logger.debug("Signaled inference thread to stop. Waiting for final batch.")
        inference_thread.join()
        
        self.logger.info("MCTS search complete.")
    

    def _mcts_worker(self, num_simulations, thread_id):
        """
        Worker thread function for MCTS simulation.
        """
        self.logger.debug(f"[Worker {thread_id}] Started with {num_simulations} simulations.")
        local_sim_count = 0
        while local_sim_count < num_simulations and not self.stop_threads.is_set():
            local_sim_count += 1
            node = self.root
            
            # Selection: Traverse the tree to find a leaf or unvisited node
            with self.tree_lock:
                # The selection logic is atomic with respect to other threads
                while not node.is_leaf() and node.is_expanded:
                    best_child = None
                    best_uct_score = -float('inf')
                    best_prior_for_tie_break = -1.0

                    legal_moves = cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL)
                    eligible_children = [
                        (move, node.children[move]) for move in legal_moves if move in node.children
                    ]

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

            self.logger.debug(f"[Worker {thread_id}] Simulation {local_sim_count}: Selected leaf node.")

            # Expansion/Backpropagation
            if node.board.is_game_over():
                self.logger.debug(f"[Worker {thread_id}] Simulation {local_sim_count}: Found game-over state. Backpropagating.")
                result = node.board.result()
                value = 0.0
                if result == "1-0":
                    value = 1.0 if node.board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    value = 1.0 if node.board.turn == chess.BLACK else -1.0
                with self.tree_lock:
                    self.backpropagate(node, value)
            else:
                self.logger.debug(f"[Worker {thread_id}] Simulation {local_sim_count}: Queuing node for inference.")
                board_input = torch.from_numpy(utils.board_to_tensor_68(node.board)).float().to(self.model_player.device)
                
                request_id = id(node) 
                
                self.inference_queue.put((request_id, board_input, node))
                
                self.logger.debug(f"[Worker {thread_id}] Simulation {local_sim_count}: Waiting for inference result...")
                # Wait for the result from the inference worker
                while True:
                    try:
                        result_id, policy_probs, value_output = self.results_queue.get(timeout=1)
                        if result_id == request_id:
                            with self.tree_lock:
                                self.expand(node, policy_probs)
                                self.backpropagate(node, value_output)
                            self.logger.debug(f"[Worker {thread_id}] Simulation {local_sim_count}: Received result and backpropagated.")
                            break
                        else:
                            # If it's not our result, put it back in the queue for another thread
                            self.results_queue.put((result_id, policy_probs, value_output))
                            self.logger.debug(f"[Worker {thread_id}] Simulation {local_sim_count}: Got wrong result, re-queuing.")
                    except queue.Empty:
                        if self.stop_threads.is_set():
                            self.logger.warning(f"[Worker {thread_id}] Simulation {local_sim_count}: Exiting due to stop signal.")
                            break
        self.logger.debug(f"[Worker {thread_id}] Finished all {local_sim_count} simulations.")

    def _inference_worker(self):
        """
        Dedicated thread for batched neural network inference.
        """
        nodes_to_process = []
        board_tensors = []
        
        self.logger.debug("[Inference Worker] Started.")

        while not self.stop_threads.is_set() or not self.inference_queue.empty():
            try:
                request_id, board_tensor, node = self.inference_queue.get(timeout=0.01)
                nodes_to_process.append(request_id)
                board_tensors.append(board_tensor)
                self.logger.debug(f"[Inference Worker] Added request from node {request_id} to batch.")
            except queue.Empty:
                pass
            
            if len(board_tensors) >= self.batch_size or (self.stop_threads.is_set() and len(board_tensors) > 0):
                if not board_tensors:
                    continue

                self.logger.debug(f"[Inference Worker] Processing batch of size {len(board_tensors)}.")
                batch_input = torch.stack(board_tensors)
                
                with torch.no_grad():
                    policy_logits_batch, value_output_batch = self.model_player.get_policy_value(batch_input)
                
                policy_probs_batch = F.softmax(policy_logits_batch, dim=1)
                
                for i in range(len(nodes_to_process)):
                    request_id = nodes_to_process[i]
                    policy_probs = policy_probs_batch[i].squeeze(0)
                    value_output = value_output_batch[i].item()
                    self.results_queue.put((request_id, policy_probs, value_output))
                    self.logger.debug(f"[Inference Worker] Sent result for node {request_id} to results queue.")

                nodes_to_process = []
                board_tensors = []

        self.logger.debug("[Inference Worker] Exiting loop. Final check on queues.")
        # Process any remaining items in the queue before exiting
        while not self.inference_queue.empty():
            try:
                request_id, board_tensor, node = self.inference_queue.get(timeout=0.01)
                nodes_to_process.append(request_id)
                board_tensors.append(board_tensor)
            except queue.Empty:
                pass
        
        if len(board_tensors) > 0:
            self.logger.debug(f"[Inference Worker] Processing final batch of size {len(board_tensors)}.")
            batch_input = torch.stack(board_tensors)
            with torch.no_grad():
                policy_logits_batch, value_output_batch = self.model_player.get_policy_value(batch_input)
            policy_probs_batch = F.softmax(policy_logits_batch, dim=1)
            for i in range(len(nodes_to_process)):
                request_id = nodes_to_process[i]
                policy_probs = policy_probs_batch[i].squeeze(0)
                value_output = value_output_batch[i].item()
                self.results_queue.put((request_id, policy_probs, value_output))

        self.logger.debug("[Inference Worker] Exited.")


    def expand(self, node: MCTSNode, policy_probs: torch.Tensor):
        if node.board.is_game_over():
            node.is_expanded = True
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


    def backpropagate(self, node: MCTSNode, value: float):
        current = node
        original_expanded_node_turn = node.board.turn

        while current is not None:
            current.visits += 1
            if current.board.turn == original_expanded_node_turn:
                current.value_sum += value
            else:
                current.value_sum -= value
            current = current.parent