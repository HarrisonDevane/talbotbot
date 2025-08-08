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
import threading
import queue

# Adjust path for internal modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, parent_dir)

import src_shared.utils as utils
from.mcts_node import MCTSNode

class MCTSEngine:
    """
    Our Monte Carlo Tree Search brain, now with a truly multithreaded,
    pipelined architecture that separates inference from tree updates.

    Selection workers continuously search and queue nodes for inference.
    A single, dedicated inference worker processes these nodes in batches.
    The results are placed on a new queue, which update workers consume
    to perform expansion and backpropagation, keeping them busy and
    distributing the workload efficiently.
    """
    def __init__(self, logger, model_player, cpuct, batch_size, virtual_loss, dirichlet_alpha, dirichlet_epsilon, selection_workers, update_workers):
        self.model_player = model_player
        self.logger = logger
        self.virtual_loss = virtual_loss
        self.cpuct = cpuct
        self.root = None
        
        # Use dedicated worker counts
        self.selection_workers_count = selection_workers
        self.update_workers_count = update_workers
        self.batch_size = batch_size
        self.worker_batch_size = (self.batch_size + self.selection_workers_count - 1) // self.selection_workers_count

        
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        # Thread-safe queues and locks
        self.inference_queue = queue.Queue(maxsize=self.selection_workers_count) 
        self.update_queue = queue.Queue(maxsize=self.update_workers_count)
        self.stop_threads = threading.Event()
        
        self.simulation_counter = 0 
        self.counter_lock = threading.Lock()
        


    def _add_dirichlet_noise(self, node):
        """
        Adds Dirichlet noise to the policy probabilities for the root node,
        only for legal moves (non-zero probabilities), and adjusts child prior probabilities accordingly.
        """
        self.logger.debug("[Dirichlet Noise] Starting to add noise...")

        with node.lock:
            policy_probs_tensor = node.prior_probabilities

            # Find legal (non-zero) move indices
            legal_indices = (policy_probs_tensor > 0).nonzero(as_tuple=True)[0]

            # Extract legal move probabilities
            legal_probs = policy_probs_tensor[legal_indices]

            # Dirichlet sampling on legal moves only
            alpha = torch.full((len(legal_indices),), self.dirichlet_alpha, device=policy_probs_tensor.device)
            dirichlet_noise = torch.distributions.dirichlet.Dirichlet(alpha).sample()

            # Mix noise with legal probabilities
            noisy_legal_probs = (
                (1 - self.dirichlet_epsilon) * legal_probs +
                self.dirichlet_epsilon * dirichlet_noise
            )

            # Create a copy of the original tensor to update
            noisy_policy_tensor = policy_probs_tensor.clone()

            # Replace legal indices with noisy probabilities
            noisy_policy_tensor[legal_indices] = noisy_legal_probs

            # Assign back to node
            node.prior_probabilities = noisy_policy_tensor

            if node.children:
                from_row_t, from_col_t, channel_t = utils.policy_flat_index_to_components_torch(legal_indices)

                from_row_list = from_row_t.tolist()
                from_col_list = from_col_t.tolist()
                channel_list = channel_t.tolist()
                noisy_legal_probs_list = noisy_legal_probs.tolist()

                for i in range(len(legal_indices)):
                    from_row = from_row_list[i]
                    from_col = from_col_list[i]
                    channel = channel_list[i]

                    move = utils.policy_components_to_move(from_row, from_col, channel, node.board)

                    if move is not None and move in node.children:
                        with node.children[move].lock:
                            node.children[move].prior_probability_from_parent = noisy_legal_probs_list[i]

        self.logger.debug(f"[Dirichlet Noise] Added Dirichlet noise to root")
    
    
    def _expand_root(self):
        """
        Expand root node initially or when root changed
        """
        self.logger.debug("Expanding root node")

        board_input = torch.from_numpy(utils.board_to_tensor_68(self.root.board)).float().to(self.model_player.device)
        policy_logits, value_output = self.model_player.get_policy_value(board_input.unsqueeze(0))
        policy_probs = F.softmax(policy_logits.squeeze(0), dim=0)

        # Expand the root node and backpropagate the value
        self._expand(self.root, policy_probs)
        self._backpropagate(self.root, value_output.item()) 
        self._add_dirichlet_noise(self.root)


    def get_target_vectors(self):
        """
        Computes the policy vector from the visit counts of the root's children.
        """
        policy_vector = np.zeros(utils.TOTAL_POLICY_MOVES, dtype=np.float32)
        total_visits = sum(child.visits for child in self.root.children.values())
        
        if total_visits == 0:
            return policy_vector, 0.0

        for move, child_node in self.root.children.items():
            normalized_prob = child_node.visits / total_visits
            from_row, from_col, channel = utils.move_to_policy_components(move, self.root.board)
            flat_index = utils.policy_components_to_flat_index(from_row, from_col, channel)
            policy_vector[flat_index] = normalized_prob

        root_value = -self.root.value_sum / self.root.visits

        return policy_vector, root_value


    def set_new_root(self, board: chess.Board, our_move: chess.Move, opponent_move: chess.Move):
        """
        Updates the MCTS root based on the sequence of moves.
        Otherwise, a new tree is started.
        """
        if self.root is None:
            self.logger.debug("MCTSEngine: New root node created.")
            self.root = MCTSNode(board.copy())
            self._expand_root()
            return
    

        # If our last move is a child of the root, we update the root to that child.
        if our_move and our_move in self.root.children:
            new_root = self.root.children[our_move]
            new_root.parent = None
            self.root = new_root

            self.logger.debug(f"MCTSEngine: Root changed to child for our move {our_move.uci()}.")

            # If opponent move also exists, we further update the root to our move's child node.
            if opponent_move:
                if opponent_move in self.root.children:
                    new_root = self.root.children[opponent_move]
                    new_root.parent = None
                    self.root = new_root
                    self.logger.debug(f"MCTSEngine: Root changed to child for opponent move {opponent_move.uci()}.")
                else:
                    self.logger.debug("MCTSE Engine: New root node created.")
                    self.root = MCTSNode(board.copy())
                    self._expand_root()

        # After updating the root, we check if it's already expanded.
        if self.root.is_expanded:
            self._add_dirichlet_noise(self.root)
        else:
            self._expand_root()


    def run_simulations(self, search_depth, early_cutoff_simulations, early_cutoff_threshold):
        
        # Start timing for the entire run_simulations method        
        self.stop_threads.clear()
        
        with self.counter_lock:
            self.simulation_counter = 0 # Reset the global counter
        
        self.logger.info(f"Starting {search_depth} simulations with {self.selection_workers_count} selection, 1 inference, and {self.update_workers_count} update workers.")
        
        # Before starting worker threads, perform a single inference for the root at the start of the game
        if not self.root.is_expanded:
            self._expand_root()
        
        # Start the inference worker thread
        inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        inference_thread.start()
        self.logger.debug("Inference worker thread started.")

        # Start the MCTS worker threads
        selection_threads = []
        for i in range(self.selection_workers_count):
            worker_thread = threading.Thread(target=self._selection_worker, args=(search_depth, early_cutoff_simulations, early_cutoff_threshold, i,), daemon=True)
            selection_threads.append(worker_thread)
            worker_thread.start()
            self.logger.debug(f"Selection worker thread {i} started.")
            
        update_threads = []
        for i in range(self.update_workers_count):
            worker_thread = threading.Thread(target=self._update_worker, args=(i,), daemon=True)
            update_threads.append(worker_thread)
            worker_thread.start()
            self.logger.debug(f"Update worker thread {i} started.")

        # Wait for all selection threads to finish.
        for thread in selection_threads:
            thread.join()
        
        self.logger.debug("All selection worker threads have finished their simulations.")

        # Signal other threads to stop and wait for them to finish their last tasks
        self.stop_threads.set()
        self.logger.debug("Signaled inference thread to stop. Waiting for final batch.")
        
        inference_thread.join()

        for thread in update_threads:
            thread.join()

        self.logger.debug("All update worker threads have finished.")

        # Log root children stats at the end of run_simulations
        self.logger.info(f"\n--- MCTS Root Children Analysis (Final State) ---")
        sorted_children = sorted(self.root.children.items(), key=lambda item: item[1].visits, reverse=True)
                    
        for move, child_node in sorted_children:
            prior_prob = child_node.prior_probability_from_parent
            log_message = (
                f"Move: {move.uci()}, Visits: {child_node.visits}, "
                f"Prior Probability: {prior_prob:.4f}, "
                f"Average Value: {-child_node.value_sum / child_node.visits if child_node.visits > 0 else 0.0:.4f}, "
            )
            self.logger.info(log_message)
        self.logger.info("-----------------------------------\n")

    
    def _selection_worker(self, search_depth, early_cutoff_simulations, early_cutoff_threshold, thread_id):
        """
        Dedicated worker thread for MCTS selection.
        Uses per-node locks to traverse and queue leaf nodes for inference.
        This version respects the max_queue_size to apply backpressure.
        """
        batch_buffer = []
        self.logger.debug(f"[Worker {thread_id}] Initializing with worker_batch_size: {self.worker_batch_size}")

        while not self.stop_threads.is_set():

            # Check for global simulation cutoff
            check_for_cutoff = False
            with self.counter_lock:
                if self.simulation_counter >= search_depth:
                    self.stop_threads.set()
                    self.logger.debug(f"[Worker {thread_id}] Global simulation count reached. Shutting down.")
                    break
                elif self.simulation_counter >= early_cutoff_simulations:
                    check_for_cutoff = True

            # Early cutoff if one child dominates
            if check_for_cutoff:
                children = list(self.root.children.values())
                total_visits = sum(child.visits for child in children)
                if total_visits > 0:
                    max_visits = max(child.visits for child in children)
                    if (max_visits / total_visits) > early_cutoff_threshold:
                        self.stop_threads.set()
                        self.logger.info(f"[Worker {thread_id}] Early cutoff due to a single child having > {early_cutoff_threshold * 100:.1f}% of visits. Visits: {max_visits}/{total_visits}")
                        break
            
            # Put batch on queue
            if len(batch_buffer) >= self.worker_batch_size:
                try:
                    self.inference_queue.put_nowait(batch_buffer)
                    self.logger.debug(f"[Selection worker {thread_id}] Pushed a full batch of size {len(batch_buffer)} to inference queue.")
                    batch_buffer = []
                except queue.Full:
                    self.logger.debug(f"[Selection worker {thread_id}] Inference queue full - waiting.")
                    time.sleep(0.001)
                    continue

            # Use root queue as proxy for forced batch
            with self.root.lock:
                if self.root.is_queued_for_inference:
                    self.inference_queue.put(batch_buffer)
                    self.logger.debug(f"[Worker {thread_id}] Root queued for inference. Pushed batch of size {len(batch_buffer)} to inference queue.")
                    batch_buffer = []
                    time.sleep(0.001)
                    continue

            # Begin selection
            node = self.root
            path = [node]

            # Lock-free traversal (stale reads are acceptable)
            while not node.is_leaf() and node.is_expanded and not node.is_queued_for_inference:
                best_child = None
                best_uct_score = -float('inf')
                best_prior_for_tie_break = -1.0

                with node.lock:
                    # Get children to consider for the next step.
                    # We only consider children that are expanded and not yet queued.
                    eligible_children = []
                    for move, child in node.children.items():
                        if not child.is_queued_for_inference:
                            eligible_children.append(child)
                    
                    if not eligible_children:
                        break
                    
                    sqrt_parent_visits_term = math.sqrt(node.visits) if node.visits > 0 else 0.0

                # UCT calculation is done without holding a lock on the parent node.
                # It's a "stale read" but is an acceptable race condition for performance.
                for child in eligible_children:
                    prior_prob_for_child = child.prior_probability_from_parent
                    uct = child.uct_score(self.cpuct, prior_prob_for_child, sqrt_parent_visits_term)
                    if uct > best_uct_score or (uct == best_uct_score and prior_prob_for_child > best_prior_for_tie_break):
                        best_uct_score = uct
                        best_prior_for_tie_break = prior_prob_for_child
                        best_child = child

                node = best_child
                path.append(node)

            # Check if this node was already queued by another worker
            # This is the final check before queuing. We need to lock the node to do this safely.
            with node.lock:
                if node.is_queued_for_inference:
                    continue

                with self.counter_lock:
                    self.logger.debug(f"[Selection worker {thread_id}] Selected node ID: {id(node)}, incremented simulation count: {self.simulation_counter}, Queue size: {self.inference_queue.qsize()}")
                
                # Leaf is terminal -> backprop directly
                if node.board.is_game_over() or node.board.can_claim_threefold_repetition() or node.board.can_claim_fifty_moves():
                    result = node.board.result(claim_draw=True)
                    value = 0.0
                    if result == "1-0":
                        value = 1.0 if node.board.turn == chess.WHITE else -1.0
                    elif result == "0-1":
                        value = 1.0 if node.board.turn == chess.BLACK else -1.0
                    
                    self._backpropagate(node, value)
                    with self.counter_lock:
                        self.simulation_counter += 1
                    continue

                # Otherwise -> queue for inference
                node.is_queued_for_inference = True
                board_input = torch.from_numpy(utils.board_to_tensor_68(node.board)).float().to(self.model_player.device)
                batch_buffer.append((board_input, node))
                self._apply_virtual_loss(node)

                with self.counter_lock:
                    self.simulation_counter += 1

        # Batch remaining nodes before shutdown
        while batch_buffer:
            try:
                self.inference_queue.put_nowait(batch_buffer)
                self.logger.debug(f"[Selection worker {thread_id}] Flushed final partial batch of size {len(batch_buffer)} to inference queue.")
                batch_buffer = []
            except queue.Full:
                self.logger.debug(f"[Selection worker {thread_id}] Inference queue still full during final flush. Retrying.")
                time.sleep(0.001)


    def _update_worker(self, thread_id):
        """
        Dedicated worker thread for MCTS tree updates.
        This worker consumes a batch of inference results and performs expansion and backpropagation
        for each result in the batch.
        """
        self.logger.debug(f"[Update worker {thread_id}] Started.")
        
        while True:
            try:
                results_batch = self.update_queue.get(timeout=0.001)

                # --- Shutdown logic ---
                if results_batch is None:
                    self.logger.debug(f"[Update worker {thread_id}] Received poison pill. Exiting.")
                    self.update_queue.task_done()
                    break
                
                # --- Normal processing of a batch ---
                for node, policy_probs, value_output in results_batch:
                    if not node.is_expanded:
                        self._expand(node, policy_probs)
                    self._backpropagate(node, value_output - self.virtual_loss, 0)
                    
                    self.logger.debug(f"[Update worker {thread_id}] Expanding and backpropagating on node ID: {id(node)}")
                
                self.update_queue.task_done()
                
            except queue.Empty:
                pass
            

    def _inference_worker(self):
        """
        Dedicated thread for batched neural network inference.
        This worker consumes requests from the inference queue, processes them,
        and then places the results on the update queue. It does NOT touch the tree.
        """
        self.logger.debug("[Inference Worker] Started.")

        while True:
            nodes_to_process = []
            board_tensors = []
            chunks_retrieved_count = 0  # Counter for how many chunks we've pulled from the queue
            
            # This will be set to True if we receive a partial batch from a selection worker
            force_process_batch = False

            # --- REFACTORED BATCH FILLING LOOP ---
            try:
                # Block until the first chunk arrives or a shutdown signal is given.
                # Using a timeout prevents a deadlock if stop_threads is set while queue is empty.
                if self.stop_threads.is_set():
                    chunk = self.inference_queue.get_nowait()
                else:
                    chunk = self.inference_queue.get(timeout=0.001)

                chunks_retrieved_count += 1

                # Unpack the chunk and add to our local batch
                for board_tensor, node in chunk:
                    nodes_to_process.append(node)
                    board_tensors.append(board_tensor)
                
                # If the received chunk is smaller than a full worker batch,
                if len(chunk) < self.worker_batch_size:
                    force_process_batch = True
                    
                # Quickly drain the queue of any other chunks to fill the rest of the batch
                while len(board_tensors) < self.batch_size and not force_process_batch:
                    try:
                        next_chunk = self.inference_queue.get_nowait()
                        chunks_retrieved_count += 1
                        for board_tensor, node in next_chunk:
                            nodes_to_process.append(node)
                            board_tensors.append(board_tensor)
                        # If this chunk is also a partial one, force the batch
                        if len(next_chunk) < self.worker_batch_size:
                            force_process_batch = True
                    except queue.Empty:
                        break
                
            except queue.Empty:
                if self.stop_threads.is_set():
                    self.logger.debug("[Inference Worker] Exiting loop due to stop signal and empty queue.")
                    break # Exit the loop cleanly
                continue # Continue waiting for more items

            if not nodes_to_process:
                continue

            self.logger.debug(f"[Inference Worker] Processing batch of size {len(board_tensors)}. Stopping: {self.stop_threads.is_set()}")

            # Run inference
            batch_input = torch.stack(board_tensors)
            policy_logits_batch, value_output_batch = self.model_player.get_policy_value(batch_input)

            if self.model_player.device.type == 'cuda':
                torch.cuda.synchronize()

            # Process results
            policy_probs_batch = F.softmax(policy_logits_batch, dim=1)
            results_list = []
            for i in range(len(nodes_to_process)):
                results_list.append((
                    nodes_to_process[i],
                    policy_probs_batch[i].squeeze(0),
                    value_output_batch[i].item()
                ))
            
            num_results = len(results_list)
            num_chunks = self.update_workers_count
            chunk_size = (num_results + num_chunks - 1) // num_chunks

            # Split the results_list into chunks and put each chunk on the update queue
            for i in range(0, num_results, chunk_size):
                chunk = results_list[i:i + chunk_size]
                while True:
                    try:
                        self.update_queue.put_nowait(chunk)
                        self.logger.debug(f"[Inference Worker] Put chunk of size {len(chunk)} on update queue.")
                        break
                    except queue.Full:
                        self.logger.debug("[Inference Worker] Update queue full — waiting.")
                        time.sleep(0.001)

            for _ in range(chunks_retrieved_count):
                self.inference_queue.task_done()

        self.logger.debug("[Inference Worker] Exiting loop. Sending None signal to update workers.")

        for _ in range(self.update_workers_count):
            self.update_queue.put(None)


    def _expand(self, node: MCTSNode, policy_probs: torch.Tensor):
        legal_moves = cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL)

        from_row_ints, from_col_ints, channel_ints = [], [], []
        child_nodes_in_order = []

        with node.lock:
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

            normalized_legal_priors = (
                prior_values_for_legal_moves / sum_of_legal_priors
                if sum_of_legal_priors > 0 else prior_values_for_legal_moves
            )
        else:
            indices_tensor = torch.empty(0, dtype=torch.long, device=policy_probs.device)
            normalized_legal_priors = torch.empty(0, device=policy_probs.device)

        if indices_tensor.numel() > 0 and normalized_legal_priors.numel() > 0:
            with node.lock:
                node.prior_probabilities.index_put_((indices_tensor,), normalized_legal_priors, accumulate=False)

        if normalized_legal_priors.numel() > 0:
            normalized_priors_list = normalized_legal_priors.cpu().tolist()
            for i, child_node in enumerate(child_nodes_in_order):
                with child_node.lock:
                    child_node.prior_probability_from_parent = normalized_priors_list[i]

        with node.lock:
            node.is_expanded = True


    def _backpropagate(self, node, value, visit_increment = 1):
        """
        Propagate value and visits up the tree using counter-level locking.
        Includes virtual loss
        """
        current = node
        original_turn = node.board.turn

        while current is not None:
            with current.inference_lock:
                current.is_queued_for_inference = False

            with current.visits_value_lock:
                current.visits += visit_increment
                if current.board.turn == original_turn:
                    current.value_sum += value
                else:
                    current.value_sum -= value

            current = current.parent


    def _apply_virtual_loss(self, node):
        """
        Applies a virtual loss to a node and its ancestors.
        This is a temporary penalty to discourage other threads from exploring
        the same path while the node is queued for inference.
        """
        self.logger.debug(f"[Virtual Loss] Applying loss of {self.virtual_loss} to node ID: {id(node)}")
        current = node
        original_turn = node.board.turn
        while current is not None:
            with current.visits_value_lock:
                current.visits += 1
                if current.board.turn == original_turn:
                    current.value_sum += self.virtual_loss
                else:
                    current.value_sum -= self.virtual_loss
            current = current.parent