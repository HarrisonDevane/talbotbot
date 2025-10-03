import chess
import cython_chess
import os
import torch
import sys
import math
import time
import logging
import queue
import numpy as np
cimport numpy as cnp
import src_shared.utils
cimport src_shared.mcts_node as MCTSNode_c

cnp.import_array() 

cdef extern from "math.h":
    double sqrt(double x)

cdef extern from "Python.h":
    double Py_BLOCK_THREADS
    double PyThreadState_Swap(void *tstate)
    void *PyGILState_Ensure()
    void PyGILState_Release(void *state)

def _visits_key_func(item):
    """
    Key function for sorting MCTSNode children by visits.
    Casts item[1] to the C-type MCTSNode for direct attribute access.
    """
    cdef MCTSNode_c.MCTSNode child_node
    child_node = <MCTSNode_c.MCTSNode>item[1]
    return child_node.visits


cdef class MCTSEngine:
    """
    Our Monte Carlo Tree Search brain for finding the best move,
    designed to work as a single worker within a multi-process
    pipeline. It submits nodes for batched inference and waits
    for results.
    """
    cdef public int worker_batch_size
    cdef public int worker_id
    cdef public double cpuct
    cdef public double virtual_loss
    cdef public double draw_cutoff
    cdef public int simulation_count
    cdef public int inference_sent
    cdef public int inference_received
    cdef public bint use_fp16
    cdef public double dirichlet_alpha
    cdef public double dirichlet_epsilon

    cdef public double time_selection
    cdef public double time_expansion
    cdef public double time_backpropagation
    cdef public double time_retrieval
    cdef public double time_queueing
    cdef public double time_misc
    cdef public double time_shutdown    

    cdef public MCTSNode_c.MCTSNode root 
    
    cdef public object logger
    cdef public object inference_queue
    cdef public object result_queue
    cdef public object in_flight_nodes
    cdef public object batch_buffer
    cdef public object device
    cdef public object policy_probs_dtype
    cdef public object shared_input_buffer
    cdef public object shared_policy_buffer
    cdef public object shared_value_buffer
    cdef public object buffer_free_slots

    def __init__(self, logger: logging.Logger, worker_batch_size: int, inference_queue, result_queue, worker_id: int, cpuct: float, virtual_loss: float, dirichlet_alpha: float, 
                dirichlet_epsilon: float, draw_cutoff: float, shared_input_buffer, shared_policy_buffer, shared_value_buffer, buffer_free_slots):

        self.logger = logger
        self.worker_batch_size = worker_batch_size
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.shared_input_buffer = shared_input_buffer
        self.shared_policy_buffer = shared_policy_buffer
        self.shared_value_buffer = shared_value_buffer
        self.buffer_free_slots = buffer_free_slots
        self.worker_id = worker_id
        self.cpuct = cpuct
        self.virtual_loss = virtual_loss
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.draw_cutoff = draw_cutoff

        self.root = None
        self.in_flight_nodes = {}

        # Initializing for run simulations method
        self.simulation_count = 0
        self.inference_sent = 0
        self.inference_received = 0
        self.batch_buffer = []

        # Timing info
        self.time_selection = 0.0
        self.time_expansion = 0.0
        self.time_backpropagation = 0.0
        self.time_retrieval = 0.0
        self.time_queueing = 0.0
        self.time_misc = 0.0
        self.time_shutdown = 0.0
        
        # Set the number of threads for internal PyTorch CPU operations.
        torch.set_num_threads(1)
        
        # Determine the device here to inform data type handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device.type == 'cuda'
        self.policy_probs_dtype = torch.float16 if self.use_fp16 else torch.float32


    cpdef set_new_root(self, board: chess.Board, our_move: chess.Move, opponent_move: chess.Move):
        """
        Updates the MCTS root based on the sequence of moves.
        If our_move and opponent_move allow traversal, the tree is kept.
        Otherwise, a new tree is started.
        """
        cdef MCTSNode_c.MCTSNode new_root         
        cdef object current_our_move = our_move
        cdef object current_opponent_move = opponent_move
        self.logger.debug("A")

        if self.root is None:
            self.logger.info("MCTSEngine: New root node created.")
            self.root = MCTSNode_c.MCTSNode(board.copy())
            self._expand_root()
            return

        self.logger.debug("B")
        
        # If our last move is a child of the root, we update the root to that child.
        if current_our_move and current_our_move in self.root.children:
            self.logger.debug("C")
            new_root = self.root.children[current_our_move] 
            new_root.move = None
            new_root.parent = None
            self.root = new_root
            self.logger.debug("D")

            self.logger.info(f"MCTSEngine: Root changed to child for our move {current_our_move.uci()}.")

            # If opponent move also exists, we further update the root to our move's child node.
            if current_opponent_move:
                if current_opponent_move in self.root.children:
                    self.logger.debug("E")
                    new_root = self.root.children[current_opponent_move]
                    new_root.move = None
                    new_root.parent = None
                    self.root = new_root
                    self.logger.info(f"MCTSEngine: Root changed to child for opponent move {current_opponent_move.uci()}.")
                else:
                    self.logger.info("MCTSE Engine: New root node created due to opponent move not in tree.")
                    self.root = MCTSNode_c.MCTSNode(board.copy())
                    self._expand_root()

        # After updating the root, we check if it's already expanded.
        if self.root.expanded:
            self._add_dirichlet_noise(self.root)
        else:
            self.logger.warning("MCTSE Engine: New root node created due to no matching branch or initial state.")
            self.root = MCTSNode_c.MCTSNode(board.copy())
            self._expand_root()


    cdef _shutdown(self):
        """
        Handles the final flush of the inference queue and waits for all
        remaining results to complete backpropagation.
        """             
        cdef double time_shutdown_start = time.perf_counter()
        cdef int buffer_index
        cdef object raw_policy_probs, raw_value_output
        cdef MCTSNode_c.MCTSNode node
        cdef double value_output
        cdef object policy_probs

        cdef int batch_buffer_size = len(self.batch_buffer)

        self.logger.debug(f"[Misc] Final flush: self.batch_buffer size: {batch_buffer_size}")
        if self.batch_buffer:
            self._submit_batch()
            self.logger.debug(f"[Misc] Flushed final partial batch of size {batch_buffer_size} to inference queue. Inferences sent: {self.inference_sent}")

        # Wait for remaining nodes - this is explicitly acknowledged as a blocking shutdown step
        while self.inference_received < self.inference_sent:
            try:
                buffer_index = self.result_queue.get(timeout=0.01)

                node = self.in_flight_nodes.pop(buffer_index)
                self.inference_received += 1

                raw_policy_probs = self.shared_policy_buffer[buffer_index] 
                raw_value_output = self.shared_value_buffer[buffer_index]

                policy_probs = raw_policy_probs.to(self.policy_probs_dtype)
                value_output = raw_value_output.item()

                self.buffer_free_slots.put(buffer_index) 
                
                if not node.expanded:
                    self._expand(node, policy_probs)
                
                self._backpropagate(node, value_output, is_terminal=False)                
                self.logger.debug(f"[Backpropagation] Expanding and backpropagating on node during final wait.")

            except queue.Empty:
                self.logger.debug(f"[Misc] Result queue empty during final wait (self.inference_received={self.inference_received}, self.inference_sent={self.inference_sent}). Waiting for more results...")
                time.sleep(0.01)

        self.time_shutdown += (time.perf_counter() - time_shutdown_start)

    
    cpdef _select(self):
        """
        Traverses the MCTS tree from the root to a leaf node using the 
        Upper Confidence Bound for Trees (UCT) selection rule.

        Returns:
            MCTSNode: The node chosen for the next step (expansion or evaluation).
        """
        # Timing remains a Python object operation, as requested.
        cdef double time_selection_start = time.perf_counter()

        cdef MCTSNode_c.MCTSNode node = self.root
        cdef MCTSNode_c.MCTSNode best_child = None
        cdef double best_uct_score = -float('inf')
        cdef double best_prior_for_tie_break = -1.0
        cdef double sqrt_parent_visits_term
        cdef double prior_prob_for_child
        cdef double uct
        cdef MCTSNode_c.MCTSNode child
        
        path = [node]

        while node.children and node.expanded and not node.selected:
            
            best_child = None
            best_uct_score = -float('inf')
            best_prior_for_tie_break = -1.0

            eligible_children = [child_py for move, child_py in node.children.items() if not child_py.selected]
            sqrt_parent_visits_term = sqrt(node.visits) if node.visits > 0 else 0.0

            if not eligible_children:
                break
            
            for child in eligible_children:
                
                prior_prob_for_child = child.prior_probability_from_parent 
                uct = child.uct_score(self.cpuct, prior_prob_for_child, sqrt_parent_visits_term)

                if uct > best_uct_score or (uct == best_uct_score and prior_prob_for_child > best_prior_for_tie_break):
                    best_uct_score = uct
                    best_prior_for_tie_break = prior_prob_for_child
                    best_child = child

            node = best_child
            path.append(node)
        
        # Timing remains a Python object operation
        self.time_selection += (time.perf_counter() - time_selection_start)
        return node
    

    def _mark_selected(self, MCTSNode_c.MCTSNode node):

        cdef double time_misc_start = time.perf_counter()
        cdef MCTSNode_c.MCTSNode current_node = node.parent

        while current_node is not None:
            if all(child.selected for child in current_node.children.values()):
                current_node.selected = True
                self.logger.debug(f"[Misc] Node {current_node.move} (parent of a fully queued subtree) also marked as selected.")
            else:
                break

            current_node = current_node.parent

        self.time_misc += (time.perf_counter() - time_misc_start)


    cdef _retrieve_infernce(self):
        cdef double time_retrieval_start
        cdef int buffer_index
        cdef object raw_policy_probs, raw_value_output
        cdef MCTSNode_c.MCTSNode node
        cdef double value_output
        cdef object policy_probs

        while True:
            try:
                time_retrieval_start = time.perf_counter()
                buffer_index = self.result_queue.get_nowait()

                node = self.in_flight_nodes.pop(buffer_index)
                self.inference_received += 1

                raw_policy_probs = self.shared_policy_buffer[buffer_index] 
                raw_value_output = self.shared_value_buffer[buffer_index]

                policy_probs = raw_policy_probs.to(self.policy_probs_dtype)
                value_output = raw_value_output.item()

                self.buffer_free_slots.put(buffer_index) 

                if not node.expanded:
                    self._expand(node, policy_probs)
                self._backpropagate(node, value_output, is_terminal=False)
                                        
            except queue.Empty:
                break


    cdef _submit_batch(self):
        cdef double time_queueing_start = time.perf_counter()
        cdef int batch_size = len(self.batch_buffer)
        self.inference_queue.put(self.batch_buffer)
    
        self.inference_sent += batch_size
        self.logger.debug(f"[Misc] Pushed a full batch of size {batch_size} to inference queue. Inferences sent: {self.inference_sent}")
        self.batch_buffer = []
        self.time_queueing += (time.perf_counter() - time_queueing_start)



    cpdef run_simulations(self, int search_depth): # search_depth is C-typed
        """
        Runs a specified number of MCTS simulations. Each simulation involves
        selection, queuing for inference, waiting for a result, and finally
        expansion and backpropagation.
        """
        
        cdef MCTSNode_c.MCTSNode node
        cdef double time_expansion_start, time_misc_start
        cdef double value
        cdef object result
        cdef object numpy_board, board_input
        cdef int current_batch_size
        
        if not self.root.expanded:
            self._expand_root()

        self.simulation_count = 0
        self.inference_sent = 0
        self.inference_received = 0
        self.batch_buffer = []

        self.time_selection = 0.0
        self.time_expansion = 0.0
        self.time_backpropagation = 0.0
        self.time_retrieval = 0.0
        self.time_queueing = 0.0
        self.time_misc = 0.0
        self.time_shutdown = 0.0
            
        while self.simulation_count < search_depth:
            # Process any available results first (non-blocking)
            self._retrieve_infernce()

            # Put batch on queue if worker_batch_size is reached
            current_batch_size = len(self.batch_buffer)
            if current_batch_size >= self.worker_batch_size:
                self._submit_batch()
            
            # Check if root is queued for inference (this handles when all nodes in tree are queued)
            if self.root.selected:
                if current_batch_size > 0:
                    self._submit_batch()

                # Root is queued + not waiting for inference results -> break
                if self.inference_received >= self.inference_sent:
                    self.logger.info(f"Only terminal nodes remaning - breaking MCTS loop")
                    break

                time.sleep(0.001)
                continue

            node = self._select()

            if node == self.root:
                self.logger.debug(f"Root chosen - restaring loop")
                time.sleep(0.001)
                continue

            if self.buffer_free_slots.qsize() == 0:
                self.logger.debug(f"No free buffer indicies")
                time.sleep(0.001)
                continue

            # Expansion/Simulation: Check for game-over or queue for inference
            if node.board.is_game_over(claim_draw=True):
                time_expansion_start = time.perf_counter()
                result = node.board.result(claim_draw=True)
                node.selected = True
                value = 0.0
                
                # Fast comparisons and assignments
                if result == "1-0":
                    value = 1.0 if node.board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    value = 1.0 if node.board.turn == chess.BLACK else -1.0
                
                self.time_expansion += (time.perf_counter() - time_expansion_start)
                self._backpropagate(node, value, is_terminal=True)
                self.simulation_count += 1
                self.logger.debug(f"[Misc] Game-over handling completed. Simulation count: {self.simulation_count}")

            # Otherwise -> queue for inference
            else:
                time_misc_start = time.perf_counter()
                buffer_index = self.buffer_free_slots.get() 
                node.selected = True
                self.in_flight_nodes[buffer_index] = node
                self.logger.debug(f"Free Nodes: {self.buffer_free_slots.qsize()}")
                
                numpy_board = src_shared.utils.board_to_tensor_68(node.board)
                board_input = torch.from_numpy(numpy_board).float()
                
                if self.use_fp16:
                    board_input = board_input.half()

                self.shared_input_buffer[buffer_index].copy_(board_input)
    
                self.batch_buffer.append((self.worker_id, buffer_index))
                self.time_misc = (time.perf_counter() - time_misc_start) 
                self._virtual_loss(node, is_applying=True)

                self.simulation_count += 1
                self.logger.debug(f"[Misc] Node queued for inference. Simulation count: {self.simulation_count}, batch size: {len(self.batch_buffer)}")

            # If all legal children of the parent are now queued, mark parent too
            self._mark_selected(node)
        
        # Cleanup
        self._shutdown()
        
        self.logger.info(f"\n--- MCTS Root Children Analysis (Final State) ---")
        self.logger.info(
                f"Root node: Visits: {self.root.visits}, "
                f"Average Value: {self.root.value_sum / self.root.visits if self.root.visits > 0 else 0.0:.4f}, "
            )

        cdef double sqrt_parent_visits_term
        cdef double prior_prob
        cdef double uct
        cdef MCTSNode_c.MCTSNode child_node
        cdef object move_obj
        cdef object log_message

        self.logger.debug(f"Free Nodes: {self.buffer_free_slots.qsize()}")
        
        sorted_children = sorted(self.root.children.items(), key=_visits_key_func, reverse=True)

        for move_obj, child_node in sorted_children:
            sqrt_parent_visits_term = sqrt(child_node.visits) if child_node.visits > 0 else 0.0
            prior_prob = child_node.prior_probability_from_parent
            uct = child_node.uct_score(self.cpuct, prior_prob, sqrt_parent_visits_term)

            log_message = (
                f"Move: {move_obj.uci()}, "
                f"Prior Probability: {prior_prob:.4f}, "
                f"Visits: {child_node.visits}, "
                f"Average Value: {-child_node.value_sum / child_node.visits if child_node.visits > 0 else 0.0:.4f}, "
                f"UCT Score: {uct:.4f}, "
                f"Forced outcome: {child_node.forced_outcome}, "
                f"Distance to mate: {child_node.distance_to_mate}"
            )
            self.logger.info(log_message)

        self.logger.info(f"\n--- Aggregate Selection Phase Timings ({self.simulation_count} simulations) ---")
        self.logger.info(f"{'Selection time:':<25}{self.time_selection:.4f}")
        self.logger.info(f"{'Queueing time:':<25}{self.time_queueing:.4f}")
        self.logger.info(f"{'Retrieving time:':<25}{self.time_retrieval:.4f}")
        self.logger.info(f"{'Expansion time:':<25}{self.time_expansion:.4f}")
        self.logger.info(f"{'Backpropagation time:':<25}{self.time_backpropagation:.4f}")
        self.logger.info(f"{'Misc time:':<25}{self.time_misc:.4f}")
        self.logger.info(f"{'Shutdown time:':<25}{self.time_shutdown:.4f}")

        return self.simulation_count


    cdef _add_dirichlet_noise(self, MCTSNode_c.MCTSNode node):
        """
        Adds Dirichlet noise to the policy probabilities for the root node,
        only for legal moves (non-zero probabilities), and adjusts child prior probabilities accordingly.
        This is done only once at the start of a new search.
        """
        cdef double time_misc_start = time.perf_counter()
        cdef object policy_probs_tensor, legal_indices, legal_probs_float32, alpha, dirichlet_noise
        cdef object noisy_legal_probs_float32, noisy_policy_tensor
        cdef object from_row_t, from_col_t, channel_t
        cdef list from_row_list, from_col_list, channel_list, noisy_legal_probs_list
        cdef int i, from_row, from_col, channel
        cdef object move
        
        self.logger.debug("[Dirichlet Noise] Starting to add noise...")

        policy_probs_tensor = node.prior_probabilities.clone() 
        legal_indices = (policy_probs_tensor > 0).nonzero(as_tuple=True)[0]

        # Always convert legal_probs to float32 before computations involving Dirichlet distribution
        legal_probs_float32 = policy_probs_tensor[legal_indices].float() 
        alpha = torch.full((len(legal_indices),), self.dirichlet_alpha, device=policy_probs_tensor.device, dtype=torch.float32)
        dirichlet_noise = torch.distributions.dirichlet.Dirichlet(alpha).sample()

        noisy_legal_probs_float32 = (
            (1.0 - self.dirichlet_epsilon) * legal_probs_float32 +
            self.dirichlet_epsilon * dirichlet_noise
        )

        noisy_policy_tensor = policy_probs_tensor.clone()
        noisy_policy_tensor[legal_indices] = noisy_legal_probs_float32.to(noisy_policy_tensor.dtype)

        if node.children:
            from_row_t, from_col_t, channel_t = src_shared.utils.policy_flat_index_to_components_torch(legal_indices)

            from_row_list = from_row_t.tolist()
            from_col_list = from_col_t.tolist()
            channel_list = channel_t.tolist()
            noisy_legal_probs_list = noisy_legal_probs_float32.to(policy_probs_tensor.dtype).tolist()

            for i in range(len(legal_indices)):
                from_row = from_row_list[i]
                from_col = from_col_list[i]
                channel = channel_list[i]

                move = src_shared.utils.policy_components_to_move(from_row, from_col, channel, node.board)

                if move is not None and move in node.children:
                    node.children[move].prior_probability_from_parent = noisy_legal_probs_list[i]

        self.logger.debug(f"[Dirichlet Noise] Added Dirichlet noise to root")
        self.time_misc += (time.perf_counter() - time_misc_start)


    cdef _expand_root(self):
        """
        A helper method to perform a single initial expansion of the root node
        when the tree is first created or reset.
        """
        cdef object board_input, raw_policy_probs
        cdef object policy_probs 
        cdef object raw_value_output
        cdef int buffer_index

        buffer_index = self.buffer_free_slots.get() 

        board_input = torch.from_numpy(src_shared.utils.board_to_tensor_68(self.root.board)).float()
        if self.use_fp16:
            board_input = board_input.half()
        self.shared_input_buffer[buffer_index].copy_(board_input)
        
        self._virtual_loss(self.root, is_applying=True)
        self.inference_queue.put([(self.worker_id, buffer_index)])

        buffer_index = self.result_queue.get()
        self.inference_received += 1

        raw_policy_probs = self.shared_policy_buffer[buffer_index] 
        raw_value_output = self.shared_value_buffer[buffer_index]

        policy_probs = raw_policy_probs.to(self.policy_probs_dtype)
        value_output = raw_value_output.item()

        self.buffer_free_slots.put(buffer_index) 
        
        self._expand(self.root, policy_probs)
        self._backpropagate(self.root, value_output, is_terminal=False)
        self._add_dirichlet_noise(self.root)


    cpdef _expand(self, MCTSNode_c.MCTSNode node, policy_probs: torch.Tensor):
        cdef double time_expansion_start = time.perf_counter()
        cdef object legal_moves
        cdef list from_row_list = []
        cdef list from_col_list = []
        cdef list channel_list = []
        cdef list child_nodes_in_order = []

        cdef int from_row_int, from_col_int, channel_int
        cdef object move
        cdef MCTSNode_c.MCTSNode child_node

        legal_moves = cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL)

        for move in legal_moves:
            from_row_int, from_col_int, channel_int = src_shared.utils.move_to_policy_components(move, node.board)

            from_row_list.append(from_row_int)
            from_col_list.append(from_col_int)
            channel_list.append(channel_int)

            child_node = MCTSNode_c.MCTSNode(board=None, parent=node, move=move)
            node.children[move] = child_node
            child_nodes_in_order.append(child_node)

        node.prior_probabilities = torch.zeros_like(policy_probs, dtype=policy_probs.dtype)

        cdef object normalized_legal_priors_pyobj = None
        cdef cnp.ndarray[cnp.float32_t, ndim=1] prior_array = None
        cdef float [:] priors_view
        cdef int i

        if legal_moves:
            from_row_tensor = torch.tensor(from_row_list, dtype=torch.long)
            from_col_tensor = torch.tensor(from_col_list, dtype=torch.long)
            channel_tensor = torch.tensor(channel_list, dtype=torch.long)

            indices_tensor = src_shared.utils.policy_components_to_flat_index_torch(
                from_row_tensor, from_col_tensor, channel_tensor
            )

            prior_values_for_legal_moves = policy_probs.flatten()[indices_tensor]
            sum_of_legal_priors = prior_values_for_legal_moves.sum()

            normalized_legal_priors_pyobj = torch.where(
                sum_of_legal_priors > 0,
                prior_values_for_legal_moves / sum_of_legal_priors,
                prior_values_for_legal_moves
            )

            if normalized_legal_priors_pyobj.numel() > 0:
                prior_array = normalized_legal_priors_pyobj.cpu().float().numpy()
                priors_view = prior_array

                for i, child_node in enumerate(child_nodes_in_order):
                    child_node.prior_probability_from_parent = priors_view[i]

        node.expanded = True
        self.time_expansion += (time.perf_counter() - time_expansion_start)


    cdef _backpropagate_minimax(self, MCTSNode_c.MCTSNode node):
        """
        Checks for forced wins, forced losses and draws by decision
        """
        cdef double time_backpropagation_start = time.perf_counter()
        cdef double avg_value
        cdef object winning_children
        cdef object best_win
        cdef object losing_children
        cdef object worst_loss
        
        if node.children:
            # Check for a winning move (any child is a loss for opponent)
            winning_children = [c for c in node.children.values() if c.forced_outcome == -1]
            
            if node.visits > 0:
                avg_value = node.value_sum / node.visits
            else:
                avg_value = 0.0

            if winning_children:
                node.forced_outcome = 1
                best_win = min(winning_children, key=lambda c: c.distance_to_mate)
                node.distance_to_mate = best_win.distance_to_mate + 1

            # Rule 2: Check for draw (only if no win above), and the current position is losing
            elif any(child.forced_outcome == 0 for child in node.children.values()) and (avg_value <= self.draw_cutoff):
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

        self.time_backpropagation += (time.perf_counter() - time_backpropagation_start)


    cdef _backpropagate(self, MCTSNode_c.MCTSNode node, double value, bint is_terminal):
        """
        Updates visit counts, value sums, and RAVE values along the path from a node up to the root.
        Handles both terminal and inference-based backpropagation.
        """
        cdef double time_backpropagation_start = time.perf_counter()
        cdef MCTSNode_c.MCTSNode current_node = node
        cdef double value_for_backprop = value
        cdef object path_moves = set() 
        
        if is_terminal:
            current_node.forced_outcome = int(value) 
            current_node.distance_to_mate = 0
        else:
            self._virtual_loss(current_node, is_applying=False)

        while current_node is not None:
            if not is_terminal:
                current_node.selected = False 
            
            current_node.visits += 1
            current_node.value_sum += value_for_backprop
            path_moves.add(current_node.move) 
            
            self._backpropagate_minimax(current_node)

            value_for_backprop = -value_for_backprop
            current_node = current_node.parent

        self.time_backpropagation += (time.perf_counter() - time_backpropagation_start)

            
    cdef _virtual_loss(self, MCTSNode_c.MCTSNode node, bint is_applying):
        """
        Applies or removes a virtual loss to a node and its ancestors.
        """
        cdef double time_backpropagation_start = time.perf_counter()
        cdef int multiplier
        cdef MCTSNode_c.MCTSNode current_node = node
        
        if is_applying:
            multiplier = 1
        else:
            multiplier = -1

        while current_node is not None:
            current_node.visits += 1 * multiplier
            current_node.value_sum += self.virtual_loss * multiplier
            current_node = current_node.parent

        self.time_backpropagation += (time.perf_counter() - time_backpropagation_start)