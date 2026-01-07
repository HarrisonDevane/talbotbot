import chess
import cython_chess
import os
import torch
import sys
import math
import time
import logging
import queue
import operator
import numpy as np
cimport numpy as cnp
import src_shared.utils
cimport src_shared.mcts_node as MCTSNode_c

cnp.import_array() 

cdef extern from "math.h":
    double sqrt(double x)


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
    cdef public int k_candidates
    cdef public double sigma_scale
    cdef public double noise
    cdef public bint use_fp16
    cdef public bint training

    cdef public double time_selection
    cdef public double time_expansion
    cdef public double time_backpropagation
    cdef public double time_retrieval
    cdef public double time_queueing
    cdef public double time_misc
    cdef public double time_wait_for_inference    

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

    def __init__(self, logger: logging.Logger, training: bool, worker_batch_size: int, inference_queue, result_queue, worker_id: int, cpuct: float, virtual_loss: float,
             draw_cutoff: float, k_candidates: int, sigma_scale: float, noise: float, board: chess.Board, shared_input_buffer, shared_policy_buffer, shared_value_buffer, buffer_free_slots):

        self.logger = logger
        self.training = training
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
        self.draw_cutoff = draw_cutoff

        self.noise = noise
        self.k_candidates = k_candidates
        self.sigma_scale = sigma_scale

        self.root = MCTSNode_c.MCTSNode(board.copy())
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
        self.time_wait_for_inference = 0.0
        
        # Set the number of threads for internal PyTorch CPU operations.
        torch.set_num_threads(1)
        
        # Determine the device here to inform data type handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device.type == 'cuda'
        self.policy_probs_dtype = torch.float16 if self.use_fp16 else torch.float32


    cdef _wait_for_inference(self):
        """
        Handles the final flush of the inference queue and waits for all
        remaining results to complete backpropagation.
        """             
        cdef double time_wait_for_inference_start = time.perf_counter()
        cdef int buffer_index
        cdef object raw_policy_probs, raw_value_output
        cdef MCTSNode_c.MCTSNode node
        cdef double value_output
        cdef object policy_probs

        cdef int batch_buffer_size = len(self.batch_buffer)

        self.logger.debug(f"[Misc] Flush: self.batch_buffer size: {batch_buffer_size}")
        if self.batch_buffer:
            self._submit_batch()
            self.logger.debug(f"[Misc] Flushed final partial batch of size {batch_buffer_size} to inference queue. Inferences sent: {self.inference_sent}")

        # Wait for remaining nodes - this is explicitly acknowledged as a blocking wait_for_inference step
        while self.inference_received < self.inference_sent:
            try:
                buffer_index = self.result_queue.get()

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
                break

        self.time_wait_for_inference += (time.perf_counter() - time_wait_for_inference_start)

    
    cpdef _select(self, MCTSNode_c.MCTSNode start_node):
        """
        Traverses the MCTS tree from the root to a leaf node using the 
        Upper Confidence Bound for Trees (UCT) selection rule.

        Returns:
            MCTSNode: The node chosen for the next step (expansion or evaluation).
        """
        # Timing remains a Python object operation, as requested.
        cdef double time_selection_start = time.perf_counter()

        cdef MCTSNode_c.MCTSNode node = start_node
        cdef MCTSNode_c.MCTSNode best_child = None
        cdef double best_uct_score = -float('inf')
        cdef double best_prior_for_tie_break = -1.0
        cdef double sqrt_parent_visits_term
        cdef double prior_prob_for_child
        cdef double uct
        cdef MCTSNode_c.MCTSNode child
        
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
        
        # Timing remains a Python object operation
        self.time_selection += (time.perf_counter() - time_selection_start)
        return node
    

    def _mark_selected(self, MCTSNode_c.MCTSNode node):

        cdef double time_misc_start = time.perf_counter()
        cdef MCTSNode_c.MCTSNode current_node = node

        while current_node is not None:
            if all(child.selected for child in current_node.children.values()):
                current_node.selected = True
                self.logger.debug(f"[Misc] Node {current_node.move} (parent of a fully queued subtree) also marked as selected.")
            else:
                break

            current_node = current_node.parent

        self.time_misc += (time.perf_counter() - time_misc_start)


    cdef _retrieve_inference(self):
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
                self.time_retrieval += (time.perf_counter() - time_retrieval_start)

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


    cdef _handle_terminal_node(self, MCTSNode_c.MCTSNode leaf):
        cdef double time_expansion_start = time.perf_counter()
        cdef object result
        cdef double value = 0.0
        
        result = leaf.board.result(claim_draw=True)
        self._mark_selected(leaf)
        
        if result == "1-0":
            value = 1.0 if leaf.board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            value = 1.0 if leaf.board.turn == chess.BLACK else -1.0
            
        self.time_expansion += (time.perf_counter() - time_expansion_start)
        self._backpropagate(leaf, value, is_terminal=True)
        self.simulation_count += 1


    cdef _queue_leaf_for_inference(self, MCTSNode_c.MCTSNode leaf):
        cdef double time_misc_start = time.perf_counter()
        cdef int buffer_index
        cdef object numpy_board, board_input
        
        while self.buffer_free_slots.qsize() == 0:
            self._retrieve_inference()
            
            if self.batch_buffer:
                self._submit_batch()
            
            time.sleep(0.001)

        buffer_index = self.buffer_free_slots.get() 
        self._mark_selected(leaf)
        self.in_flight_nodes[buffer_index] = leaf
        
        numpy_board = src_shared.utils.board_to_tensor_69(leaf.board)
        board_input = torch.from_numpy(numpy_board).float()
        
        if self.use_fp16:
            board_input = board_input.half()

        self.shared_input_buffer[buffer_index].copy_(board_input)

        self.batch_buffer.append((self.worker_id, buffer_index))
        self._virtual_loss(leaf, is_applying=True)

        self.time_misc += (time.perf_counter() - time_misc_start) 
        self.simulation_count += 1


    cdef _run_single_async_simulation(self, MCTSNode_c.MCTSNode start_node):
        """ 
        Runs one descent starting from start_node, handles queueing. 
        """
        cdef MCTSNode_c.MCTSNode leaf
        cdef int current_batch_size
        cdef double time_misc_start

        while True:
            self._retrieve_inference()
            current_batch_size = len(self.batch_buffer)

            if current_batch_size >= self.worker_batch_size:
                self._submit_batch()
            
            # Check if start node is queued for inference (this handles when all nodes in tree are queued)
            if start_node.selected:
                if current_batch_size > 0:
                    self._submit_batch()

                # Root is queued + not waiting for inference results -> break
                if self.inference_received >= self.inference_sent:
                    self.logger.info(f"Only terminal nodes remaining - breaking MCTS loop")
                    return

                time.sleep(0.001)
                continue

            leaf = self._select(start_node)

            if leaf == start_node:
                self.logger.debug(f"Start node chosen - restarting loop")
                time.sleep(0.001)
                continue

            if self.buffer_free_slots.qsize() == 0:
                self.logger.debug(f"No free buffer indices")
                time.sleep(0.001)
                continue
    
            if leaf.board.is_game_over(claim_draw=True):
                self._handle_terminal_node(leaf)
            else:
                self._queue_leaf_for_inference(leaf)

            return


    cpdef run_simulations(self, int total_simulations):
        """
        Executes the Gumbel MuZero 'Sequential Halving' search.
        """
        cdef int num_phases
        cdef int sim_budget_per_phase
        cdef int sims_per_candidate
        cdef list all_moves
        cdef list active_candidates
        cdef MCTSNode_c.MCTSNode child
        cdef int actual_k
        cdef int i
        
        self.simulation_count = 0
        self.inference_sent = 0
        self.inference_received = 0
        self.batch_buffer = []

        # Reset timings
        self.time_selection = 0.0
        self.time_expansion = 0.0
        self.time_backpropagation = 0.0
        self.time_retrieval = 0.0
        self.time_queueing = 0.0
        self.time_misc = 0.0
        self.time_wait_for_inference = 0.0

    
        self._queue_leaf_for_inference(self.root)
        self._submit_batch()
        self._wait_for_inference()

        all_moves = list(self.root.children.keys())
        actual_k = min(self.k_candidates, len(all_moves))
        
        # Calculate phases: e.g. log2(16) = 4 phases
        num_phases = int(math.ceil(math.log2(actual_k)))
        if num_phases < 1: num_phases = 1
        
        sim_budget_per_phase = total_simulations // num_phases
        if sim_budget_per_phase < 1: sim_budget_per_phase = 1

        # 2. Gumbel Injection (Calculate Noise & Logits Locally)
        candidate_data = []

        for move in all_moves:
            child = self.root.children[move]
            
            # Recover logit: ln(P)
            logit = math.log(max(child.prior_probability_from_parent, 1e-8))
            
            if self.training:
                noise = np.random.gumbel(0, self.noise)
            else:
                noise = 0.0

            child.gumbel_noise = noise                
            
            candidate_data.append({
                'move': move,
                'node': child,
                'logit': logit,
                'noise': noise,
                'score': logit + noise
            })

        # 3. Initial Candidates (Score = Logit + Noise)
        candidate_data.sort(key=operator.itemgetter('score'), reverse=True)
        active_candidates = candidate_data[:actual_k]

        for cand in active_candidates:
            child = cand['node']

            if child.board.is_game_over(claim_draw=True):
                self._handle_terminal_node(child)
            else:
                self._queue_leaf_for_inference(child)

        self._submit_batch()

        # Barrier
        self._wait_for_inference()

        # 4. Sequential Halving Phase Loop
        for phase in range(num_phases):
            if len(active_candidates) == 0:
                break

            sims_per_candidate = sim_budget_per_phase // len(active_candidates)
            if sims_per_candidate < 1: sims_per_candidate = 1
            
            # A. Run Batch Simulations
            for i in range(sims_per_candidate):
                for candidate in active_candidates:
                    child = candidate['node']
                    self._run_single_async_simulation(child)

            # B. Sync Barrier (Wait for all GPUs to finish)
            self._wait_for_inference() 

            # C. Score Update & Pruning
            if len(active_candidates) > 1 and phase < (num_phases - 1):

                min_q = float('inf')
                max_q = -float('inf')
                
                # 1. Find Min/Max
                for cand in active_candidates:
                    node_ptr = cand['node']
                    if node_ptr.visits > 0:
                        q = -node_ptr.value_sum / node_ptr.visits
                        if q < min_q: min_q = q
                        if q > max_q: max_q = q
                
                # Safety for first pass or equal values
                if min_q > max_q: # Should not happen if visits > 0
                    min_q = -1.0
                    max_q = 1.0
                
                if max_q - min_q < 1e-5:
                    max_q += 1e-5 # Prevent divide by zero
                    
                q_range = max_q - min_q

                # 2. Dynamic Sigma
                max_visits = 0
                for cand in active_candidates:
                    if cand['node'].visits > max_visits:
                        max_visits = cand['node'].visits
                
                dynamic_sigma = (self.sigma_scale + max_visits) * 1.0

                # 3. Apply Score
                for cand in active_candidates:
                    node_ptr = cand['node']
                    if node_ptr.visits > 0:
                        q = -node_ptr.value_sum / node_ptr.visits
                        # Rel Norm: Maps min->0, max->1
                        q_norm = (q - min_q) / q_range
                    else:
                        # Unvisited nodes are usually treated as neutral or pessimistic
                        # In Gumbel Top-K, if we are pruning, we generally want to keep unvisiteds 
                        # alive or score them 0. Let's score 0.0 to be safe (worst case).
                        q_norm = 0.0 

                    cand['score'] = cand['logit'] + cand['noise'] + (dynamic_sigma * q_norm)

                active_candidates.sort(key=operator.itemgetter('score'), reverse=True)
                
                cutoff = len(active_candidates) // 2
                active_candidates = active_candidates[:cutoff]


        self.simulation_count = total_simulations

        # --- CHILD STATS (Sorted by Q) ---
        self.logger.info(f"\n{'Move':<8} {'Visits':>8} {'Prior':>8} {'Q-Val':>8} {'Outcome':>8}")
        self.logger.info("-" * 50)

        cdef list stats = []
        cdef double q_val
        cdef object outcome

        for move, child in self.root.children.items():
            if child.visits > 0:
                # Calculate Q from Parent Perspective (-child value)
                q_val = -child.value_sum / child.visits
                stats.append((child, q_val))

        # Sort by Q-Value Descending
        stats.sort(key=operator.itemgetter(1), reverse=True)

        for child, q_val in stats:
            outcome = str(child.forced_outcome) if child.forced_outcome is not None else ""
            self.logger.info(f"{child.move.uci():<8} {child.visits:>8} {child.prior_probability_from_parent:>8.4f} {q_val:>8.4f} {outcome:>8}")

        # Logging
        self.logger.info(f"\n--- Gumbel Search ({total_simulations} sims) Timings ---")
        self.logger.info(f"{'Selection time:':<25}{self.time_selection:.4f}")
        self.logger.info(f"{'Queueing time:':<25}{self.time_queueing:.4f}")
        self.logger.info(f"{'Retrieving time:':<25}{self.time_retrieval:.4f}")
        self.logger.info(f"{'Expansion time:':<25}{self.time_expansion:.4f}")
        self.logger.info(f"{'Backpropagation time:':<25}{self.time_backpropagation:.4f}")
        self.logger.info(f"{'Forced waiting for inference time:':<25}{self.time_wait_for_inference:.4f}")

        return self.simulation_count


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

            # Rule 2: Check for draw (only if no win above), and the current position is evaluated as losing
            elif any(child.forced_outcome == 0 for child in node.children.values()) and (avg_value <= 0.0):
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


    cdef _backpropagate(self, MCTSNode_c.MCTSNode node, double value, bint is_terminal):
        """
        Updates visit counts, value sums, and RAVE values along the path from a node up to the root.
        Handles both terminal and inference-based backpropagation.
        """
        cdef double time_backpropagation_start = time.perf_counter()
        cdef MCTSNode_c.MCTSNode current_node = node
        cdef double value_for_backprop = value
        
        if is_terminal:
            current_node.forced_outcome = int(value) 
            current_node.distance_to_mate = 0
        else:
            self._virtual_loss(current_node, is_applying=False)

        current_node.raw_value = value

        while current_node is not None:
            if not is_terminal:
                current_node.selected = False 
            
            current_node.visits += 1
            current_node.value_sum += value_for_backprop
            
            self._backpropagate_minimax(current_node)

            value_for_backprop = -value_for_backprop
            current_node = current_node.parent

        self.time_backpropagation += (time.perf_counter() - time_backpropagation_start)

            
    cdef _virtual_loss(self, MCTSNode_c.MCTSNode node, bint is_applying):
        """
        Applies or removes a virtual loss to a node and its ancestors.
        """
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