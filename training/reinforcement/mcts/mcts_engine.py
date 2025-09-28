import chess
import cython_chess
import os
import torch
import torch.nn.functional as F
import sys
import math
import time
import logging
import queue
import numpy as np
import torch.multiprocessing as mp
from .mcts_node import MCTSNode

# Adjust path for internal modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, parent_dir)

import src_shared.utils as utils

class MCTSEngine:
    """
    Our Monte Carlo Tree Search brain for finding the best move,
    designed to work as a single worker within a multi-process
    pipeline. It submits nodes for batched inference and waits
    for results.
    """
    def __init__(self, logger: logging.Logger, worker_batch_size: int, inference_queue, result_queue, worker_id: int, cpuct: float, virtual_loss: float, dirichlet_alpha: float, dirichlet_epsilon: float, draw_cutoff: float):
        self.logger = logger
        self.worker_batch_size = worker_batch_size
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.cpuct = cpuct
        self.virtual_loss = virtual_loss
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.draw_cutoff = draw_cutoff

        self.root = None
        self.next_uid = 0
        self.in_flight_nodes = {}
        
        # Set the number of threads for internal PyTorch CPU operations.
        torch.set_num_threads(1)
        
        # Determine the device here to inform data type handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device.type == 'cuda'


    def set_new_root(self, board: chess.Board, our_move: chess.Move, opponent_move: chess.Move):
        """
        Updates the MCTS root based on the sequence of moves.
        If our_move and opponent_move allow traversal, the tree is kept.
        Otherwise, a new tree is started.
        """
        if self.root is None:
            self.logger.info("MCTSEngine: New root node created.")
            self.root = MCTSNode(board.copy())
            self._expand_root()
            return
        
        # If our last move is a child of the root, we update the root to that child.
        if our_move and our_move in self.root.children:
            new_root = self.root.children[our_move]
            new_root.move = None
            new_root.parent = None
            self.root = new_root

            self.logger.info(f"MCTSEngine: Root changed to child for our move {our_move.uci()}.")

            # If opponent move also exists, we further update the root to our move's child node.
            if opponent_move:
                if opponent_move in self.root.children:
                    new_root = self.root.children[opponent_move]
                    new_root.move = None
                    new_root.parent = None
                    self.root = new_root
                    self.logger.info(f"MCTSEngine: Root changed to child for opponent move {opponent_move.uci()}.")
                else:
                    self.logger.info("MCTSE Engine: New root node created due to opponent move not in tree.")
                    self.root = MCTSNode(board.copy())
                    self._expand_root()

        # After updating the root, we check if it's already expanded.
        if self.root.expanded:
            self._add_dirichlet_noise(self.root)
        else:
            self.logger.warning("MCTSE Engine: New root node created due to no matching branch or initial state.")
            self.root = MCTSNode(board.copy())
            self._expand_root()


    def run_simulations(self, search_depth: int):
        """
        Runs a specified number of MCTS simulations. Each simulation involves
        selection, queuing for inference, waiting for a result, and finally
        expansion and backpropagation.
        """
        
        # Ensure the root is expanded at the start of a game, or if a new tree was created.
        if not self.root.expanded:
            self._expand_root()

        simulation_count = 0
        inference_sent = 0
        inference_received = 0
        batch_buffer = []
        check_for_cutoff = False

        time_selection = 0
        time_expansion = 0
        time_backpropagation = 0
        time_inference = 0
        time_misc = 0
        time_shutdown = 0
            
        while simulation_count < search_depth:
            # Process any available results first (non-blocking)
            while not self.result_queue.empty():
                try:
                    # Retrieve raw outputs from the queue
                    time_inference_start = time.perf_counter()
                    node_uid, raw_policy_probs, raw_value_output = self.result_queue.get_nowait()

                    node = self.in_flight_nodes.pop(node_uid)
                    inference_received += 1

                    # Convert dtypes if necessary
                    policy_probs_dtype = torch.float16 if self.use_fp16 else torch.float32
                    policy_probs = raw_policy_probs.to(policy_probs_dtype)
                    value_output = raw_value_output.float().item()
                    
                    time_inference_end = time.perf_counter()
                    time_inference += (time_inference_end - time_inference_start)

                    if not node.expanded:
                        time_expansion_start = time.perf_counter()
                        self._expand(node, policy_probs)
                        time_expansion_end = time.perf_counter()
                        time_expansion += (time_expansion_end - time_expansion_start)
                    
                    time_backpropagation_start = time.perf_counter()
                    self._backpropagate(node, value_output, is_terminal=False)
                    time_backpropagation_end = time.perf_counter()
                    time_backpropagation += (time_backpropagation_end - time_backpropagation_start)
                                                
                except queue.Empty:
                    break

            # Put batch on queue if worker_batch_size is reached
            if len(batch_buffer) >= self.worker_batch_size:
                time_inference_start = time.perf_counter()
                self.inference_queue.put_nowait(batch_buffer)
                inference_sent += len(batch_buffer)
                self.logger.debug(f"[Misc] Pushed a full batch of size {len(batch_buffer)} to inference queue. Inferences sent: {inference_sent}")
                batch_buffer = []
                time_inference_end = time.perf_counter()
                time_inference += (time_inference_end - time_inference_start)
            
            # Check if root is queued for inference (this handles when all nodes in tree are queued)
            if self.root.selected:
                if len(batch_buffer) > 0:
                    self.inference_queue.put_nowait(batch_buffer)
                    inference_sent += len(batch_buffer)
                    self.logger.debug(f"[Misc] Pushed a batch of size {len(batch_buffer)} to inference queue. Inferences sent: {inference_sent}")
                    batch_buffer = []

                # Root is queued + not waiting for inference results -> break
                # Happens when all nodes are terminal
                if inference_received >= inference_sent:
                    self.logger.info(f"Only terminal nodes remaning - breaking MCTS loop")
                    break

                time.sleep(0.001)
                continue

            node = self.root
            path = [node]

            time_selection_start = time.perf_counter()
            uct = None

            while node.children and node.expanded and not node.selected:
                best_child = None
                best_uct_score = -float('inf')
                best_prior_for_tie_break = -1.0

                eligible_children = [(move, child) for move, child in node.children.items() if not child.selected]

                # UCT selection
                sqrt_parent_visits_term = math.sqrt(node.visits) if node.visits > 0 else 0.0

                if not eligible_children:
                    break

                for move, child in eligible_children:
                    prior_prob_for_child = child.prior_probability_from_parent
                    uct = child.uct_score(self.cpuct, prior_prob_for_child, sqrt_parent_visits_term)

                    if uct > best_uct_score or (uct == best_uct_score and prior_prob_for_child > best_prior_for_tie_break):
                        best_uct_score = uct
                        best_prior_for_tie_break = prior_prob_for_child
                        best_child = child

                node = best_child
                path.append(node)

            time_selection_end = time.perf_counter()
            time_selection += (time_selection_end - time_selection_start)

            if node == self.root:
                self.logger.info(f"Root chosen - restaring loop")
                time.sleep(0.001)
                continue

            # Expansion/Simulation: Check for game-over or queue for inference
            if node.board.is_game_over(claim_draw=True):
                time_expansion_start = time.perf_counter()
                result = node.board.result(claim_draw=True)
                node.selected = True
                value = 0.0
                if result == "1-0":
                    value = 1.0 if node.board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    value = 1.0 if node.board.turn == chess.BLACK else -1.0

                time_expansion_end = time.perf_counter()
                time_expansion += (time_expansion_end - time_expansion_start)

                time_backpropagation_start = time.perf_counter()
                self._backpropagate(node, value, is_terminal=True)
                time_backpropagation_end = time.perf_counter()
                time_backpropagation += (time_backpropagation_end - time_backpropagation_start)

                simulation_count += 1
                self.logger.debug(f"[Misc] Game-over handling completed. Simulation count: {simulation_count}")

            # Otherwise -> queue for inference
            else:
                time_inference_start = time.perf_counter()
                node.selected = True
                node.uid = self.next_uid
                self.next_uid += 1
                self.in_flight_nodes[node.uid] = node
                
                # Board to tensor conversion
                numpy_board = utils.board_to_tensor_68(node.board)
                board_input = torch.from_numpy(numpy_board).float()
                
                if self.use_fp16:
                    board_input = board_input.half()

                board_input = board_input.pin_memory()

                batch_buffer.append((self.worker_id, node.uid, board_input))

                time_inference_end = time.perf_counter()
                time_inference += (time_inference_end - time_inference_start)
                
                time_backpropagation_start = time.perf_counter()
                self._virtual_loss(node, is_applying=True)
                time_backpropagation_end = time.perf_counter()
                time_backpropagation += (time_backpropagation_end - time_backpropagation_start)

                simulation_count += 1
                self.logger.debug(f"[Misc] Node queued for inference. Simulation count: {simulation_count}, batch size: {len(batch_buffer)}")

            # If all legal children of the parent are now queued, mark parent too
            time_misc_start = time.perf_counter()
            current_node = node.parent

            while current_node is not None:
                # If all children have been selected, mark parent as selected too
                if all(child.selected for child in current_node.children.values()):
                    current_node.selected = True
                    self.logger.debug(f"[Misc] Node {current_node.move} (parent of a fully queued subtree) also marked as selected.")
                else:
                    break

                # Move up to the next parent to continue the check
                current_node = current_node.parent

            time_misc_end = time.perf_counter()
            time_misc += (time_misc_end - time_misc_start)
        
        # Cleanup
        time_shutdown_start = time.perf_counter()
        
        # Batch remaining nodes before shutdown
        self.logger.debug(f"[Misc] Final flush: batch_buffer size: {len(batch_buffer)}")
        if batch_buffer: # Only put if there's something to flush
            self.inference_queue.put(batch_buffer) # Use blocking put here for final flush
            inference_sent += len(batch_buffer)
            self.logger.debug(f"[Misc] Flushed final partial batch of size {len(batch_buffer)} to inference queue. Inferences sent: {inference_sent}")
            batch_buffer = []

        # Wait for remaining nodes - this is explicitly acknowledged as a blocking shutdown step
        while inference_received < inference_sent:
            try:
                node_uid, raw_policy_probs, raw_value_output = self.result_queue.get(timeout=0.01)

                node = self.in_flight_nodes.pop(node_uid)
                inference_received += 1
                
                policy_probs_dtype = torch.float16 if self.use_fp16 else torch.float32
                policy_probs = raw_policy_probs.to(policy_probs_dtype)         
                
                value_output = raw_value_output.float().item()
                
                if not node.expanded:
                    self._expand(node, policy_probs)
                
                self._backpropagate(node, value_output, is_terminal=False)                
                self.logger.debug(f"[Backpropagation] Expanding and backpropagating on node during final wait.")
                        
            except queue.Empty:
                self.logger.debug(f"[Misc] Result queue empty during final wait (inference_received={inference_received}, inference_sent={inference_sent}). Waiting for more results...")
                time.sleep(0.01)
        
        time_shutdown_end = time.perf_counter()
        time_shutdown += (time_shutdown_end - time_shutdown_start)

        # Log root children stats at the end of run_simulations
        self.logger.info(f"\n--- MCTS Root Children Analysis (Final State) ---")
        self.logger.info(
                f"Root node: Visits: {self.root.visits}, "
                f"Average Value: {self.root.value_sum / self.root.visits if self.root.visits > 0 else 0.0:.4f}, "
            )

        sorted_children = sorted(self.root.children.items(), key=lambda item: item[1].visits, reverse=True)
        
        # Only calculate this if set to info
        for move, child_node in sorted_children:
            sqrt_parent_visits_term = math.sqrt(child_node.visits) if child_node.visits > 0 else 0.0
            prior_prob = child_node.prior_probability_from_parent
            uct = child_node.uct_score(self.cpuct, prior_prob, sqrt_parent_visits_term)

            log_message = (
                f"Move: {move.uci()}, "
                f"Prior Probability: {prior_prob:.4f}, "
                f"Visits: {child_node.visits}, "
                f"Average Value: {-child_node.value_sum / child_node.visits if child_node.visits > 0 else 0.0:.4f}, "
                f"UCT Score: {uct:.4f}, "
                f"Forced outcome: {child_node.forced_outcome}, "
                f"Distance to mate: {child_node.distance_to_mate}"
            )
            self.logger.info(log_message)

        # --- Aggregate Profiling Output ---
        self.logger.info(f"\n--- Aggregate Selection Phase Timings ({simulation_count} simulations) ---")
        self.logger.info(f"Selection time: {time_selection:.4f}")
        self.logger.info(f"Inference time: {time_inference:.4f}")
        self.logger.info(f"Expansion time: {time_expansion:.4f}")
        self.logger.info(f"Backpropagation time: {time_backpropagation:.4f}")
        self.logger.info(f"Shutdown time: {time_shutdown:.4f}")

        return simulation_count

    

    def _add_dirichlet_noise(self, node):
        """
        Adds Dirichlet noise to the policy probabilities for the root node,
        only for legal moves (non-zero probabilities), and adjusts child prior probabilities accordingly.
        This is done only once at the start of a new search.
        """

        self.logger.debug("[Dirichlet Noise] Starting to add noise...")
        policy_probs_tensor = node.prior_probabilities.clone() 
        legal_indices = (policy_probs_tensor > 0).nonzero(as_tuple=True)[0]

        # Always convert legal_probs to float32 before computations involving Dirichlet distribution
        legal_probs_float32 = policy_probs_tensor[legal_indices].float() 

        # Dirichlet distribution concentration parameter 'alpha' must be float32
        alpha = torch.full((len(legal_indices),), self.dirichlet_alpha, device=policy_probs_tensor.device, dtype=torch.float32)
        dirichlet_noise = torch.distributions.dirichlet.Dirichlet(alpha).sample()


        # Perform the weighted sum in float32
        noisy_legal_probs_float32 = (
            (1 - self.dirichlet_epsilon) * legal_probs_float32 +
            self.dirichlet_epsilon * dirichlet_noise
        )

        noisy_policy_tensor = policy_probs_tensor.clone() # This will retain the original dtype (e.g., float16)
        
        # Cast the float32 result back to the original dtype (e.g., float16) before assigning
        noisy_policy_tensor[legal_indices] = noisy_legal_probs_float32.to(noisy_policy_tensor.dtype)

        if node.children:
            from_row_t, from_col_t, channel_t = utils.policy_flat_index_to_components_torch(legal_indices)

            from_row_list = from_row_t.tolist()
            from_col_list = from_col_t.tolist()
            channel_list = channel_t.tolist()
            # Convert to list and ensure it matches the original float type of policy_probs_tensor
            noisy_legal_probs_list = noisy_legal_probs_float32.to(policy_probs_tensor.dtype).tolist()

            for i in range(len(legal_indices)):
                from_row = from_row_list[i]
                from_col = from_col_list[i]
                channel = channel_list[i]

                move = utils.policy_components_to_move(from_row, from_col, channel, node.board)

                if move is not None and move in node.children:
                    node.children[move].prior_probability_from_parent = noisy_legal_probs_list[i]

        self.logger.debug(f"[Dirichlet Noise] Added Dirichlet noise to root")


    def _expand_root(self):
        """
        A helper method to perform a single initial expansion of the root node
        when the tree is first created or reset.
        """
        # Cast board input to FP16 if `use_fp16` is true for inference batcher
        self.root.uid = self.next_uid
        self.next_uid += 1

        board_input = torch.from_numpy(utils.board_to_tensor_68(self.root.board)).float()
        if self.use_fp16:
            board_input = board_input.half()
        board_input = board_input.pin_memory()
        self._virtual_loss(self.root, is_applying=True)
        self.inference_queue.put([(self.worker_id, self.root.uid, board_input)])

        while True:
            try:
                # Retrieve raw outputs from the queue
                _, raw_policy_probs, raw_value_output = self.result_queue.get_nowait()
                
                policy_probs_dtype = torch.float16 if self.use_fp16 else torch.float32
                policy_probs = raw_policy_probs.to(policy_probs_dtype)
                
                value_output = raw_value_output.float().item()

                self._expand(self.root, policy_probs)
                self._backpropagate(self.root, value_output, is_terminal=False)
                self._add_dirichlet_noise(self.root)
                break
            except queue.Empty:
                time.sleep(0.001)
                pass


    def _expand(self, node: MCTSNode, policy_probs: torch.Tensor):

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

        # Initialize node.prior_probabilities with the same dtype as policy_probs
        node.prior_probabilities = torch.zeros_like(policy_probs, dtype=policy_probs.dtype)

        if legal_moves:
            from_row_tensor = torch.tensor(from_row_ints, dtype=torch.long)
            from_col_tensor = torch.tensor(from_col_ints, dtype=torch.long)
            channel_tensor = torch.tensor(channel_ints, dtype=torch.long)

            indices_tensor = utils.policy_components_to_flat_index_torch(
                from_row_tensor, from_col_tensor, channel_tensor
            )
            # Ensure prior_values_for_legal_moves also has the correct dtype by slicing policy_probs
            prior_values_for_legal_moves = policy_probs[indices_tensor]
            sum_of_legal_priors = prior_values_for_legal_moves.sum()
            normalized_legal_priors = prior_values_for_legal_moves / sum_of_legal_priors if sum_of_legal_priors > 0 else prior_values_for_legal_moves
        else:
            indices_tensor = torch.empty(0, dtype=torch.long)
            # Ensure empty tensor has correct dtype
            normalized_legal_priors = torch.empty(0, dtype=policy_probs.dtype) 

        if indices_tensor.numel() > 0 and normalized_legal_priors.numel() > 0:
            node.prior_probabilities.index_put_((indices_tensor,), normalized_legal_priors, accumulate=False)

        if normalized_legal_priors.numel() > 0:
            normalized_priors_list = normalized_legal_priors.tolist()
            for i, child_node in enumerate(child_nodes_in_order):
                child_node.prior_probability_from_parent = normalized_priors_list[i]

        node.expanded = True


    def _backpropagate_minimax(self, node: MCTSNode):
        """
        Checks for forced wins, forced losses and draws by decision
        """
        if node.children:
            # Rule 1: Check for a winning move (any child is a loss for opponent)
            winning_children = [c for c in node.children.values() if c.forced_outcome == -1]
            if winning_children:
                node.forced_outcome = 1
                best_win = min(winning_children, key=lambda c: c.distance_to_mate)
                node.distance_to_mate = best_win.distance_to_mate + 1

            # Rule 2: Check for draw (only if no win above), and the current position is losing
            # This is draw by decision. If the bot thinks this position is losing, and a forced draw is available, take the draw
            elif any(child.forced_outcome == 0 for child in node.children.values()) and (node.value_sum / node.visits <= self.draw_cutoff):
                node.forced_outcome = 0
                node.distance_to_mate = 0

            # Rule 3: Check for forced loss (only if no win or draw)
            elif all(c.forced_outcome == 1 for c in node.children.values()):
                losing_children = [c for c in node.children.values() if c.forced_outcome == 1]
                if losing_children:
                    node.forced_outcome = -1
                    worst_loss = max(losing_children, key=lambda c: c.distance_to_mate)
                    node.distance_to_mate = worst_loss.distance_to_mate + 1
            else:
                node.forced_outcome = None
                node.distance_to_mate = None 


    def _backpropagate(self, node: MCTSNode, value: float, is_terminal: bool):
        """
        Updates visit counts, value sums, and RAVE values along the path from a node up to the root.
        Handles both terminal and inference-based backpropagation.
        """
        current_node = node
        value_for_backprop = value
        path_moves = set()

        # If this is the start of a terminal backpropagation, set the initial forced outcome.
        if is_terminal:
            current_node.forced_outcome = int(value)
            current_node.distance_to_mate = 0
        else:
            # For inference backpropagation, remove the virtual loss.
            self._virtual_loss(current_node, is_applying=False)

        while current_node is not None:
            # Standard MCTS updates
            if not is_terminal:
                 current_node.selected = False 
            
            current_node.visits += 1
            current_node.value_sum += value_for_backprop

            path_moves.add(current_node.move)
            
            # Call the minimax helper to update forced outcomes and DTM
            self._backpropagate_minimax(current_node)

            # Alternate perspective for next node up
            value_for_backprop = -value_for_backprop
            current_node = current_node.parent

            
    def _virtual_loss(self, node: MCTSNode, is_applying: bool):
        """
        Applies or removes a virtual loss to a node and its ancestors.
        """
        multiplier = 1 if is_applying else -1

        current_node = node

        while current_node is not None:
            current_node.visits += 1 * multiplier
            current_node.value_sum += self.virtual_loss * multiplier
                 
            current_node = current_node.parent