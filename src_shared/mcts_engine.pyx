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
    cdef public double virtual_loss
    cdef public double draw_cutoff
    cdef public int simulation_count
    cdef public int inference_sent
    cdef public int inference_received
    cdef public int gumbel_k
    cdef public double gumbel_c_base
    cdef public double gumbel_c_scale
    cdef public double gumbel_noise
    cdef public double gumbel_min_scale
    cdef public double search_min_q
    cdef public double search_max_q
    cdef public bint use_fp16
    cdef public bint gumbel_first_round
    cdef public bint gumbel_final_round

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

    def __init__(self, logger: logging.Logger, worker_batch_size: int, inference_queue, result_queue, worker_id: int, virtual_loss: float,
             draw_cutoff: float, gumbel_k: int, gumbel_c_base: float, gumbel_c_scale: float, gumbel_noise: float, gumbel_first_round: bool, gumbel_final_round: bool, gumbel_min_scale: double, board: chess.Board, shared_input_buffer, shared_policy_buffer, shared_value_buffer, buffer_free_slots):

        self.logger = logger
        self.worker_batch_size = worker_batch_size
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.shared_input_buffer = shared_input_buffer
        self.shared_policy_buffer = shared_policy_buffer
        self.shared_value_buffer = shared_value_buffer
        self.buffer_free_slots = buffer_free_slots
        self.worker_id = worker_id
        self.virtual_loss = virtual_loss
        self.draw_cutoff = draw_cutoff

        self.gumbel_noise = gumbel_noise
        self.gumbel_k = gumbel_k
        self.gumbel_c_base = gumbel_c_base
        self.gumbel_c_scale = gumbel_c_scale
        self.gumbel_min_scale = gumbel_min_scale
        self.gumbel_first_round = gumbel_first_round
        self.gumbel_final_round = gumbel_final_round
        self.search_min_q = 1.0
        self.search_max_q = -1.0

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
        Traverses using Deterministic Policy Matching.
        Delegates score calculation to child.calculate_gumbel_score().
        """
        cdef double time_selection_start = time.perf_counter()
        
        cdef MCTSNode_c.MCTSNode node = start_node
        cdef MCTSNode_c.MCTSNode best_child
        cdef MCTSNode_c.MCTSNode child
        
        # Variables for selection
        cdef double max_visits, sum_visits
        cdef double max_score_logit, sum_score_exp
        cdef double pi_prime, child_n_norm, deficit, best_deficit
        cdef double score
        
        cdef list children_list
        cdef int i, n_children
        
        while node.children and node.expanded and not node.selected:
            best_child = None
            best_deficit = -999999999.0
            
            children_list = list(node.children.values())
            n_children = len(children_list)
            if n_children == 0: break

            max_visits = 0.0
            sum_visits = 0.0
            for child in children_list:
                if child.visits > max_visits: max_visits = child.visits
                sum_visits += child.visits
            
            max_score_logit = -999999999.0
            v_mix = node.calculate_v_mix()

            for i in range(n_children):
                child = children_list[i]
                score = child.calculate_gumbel_score(self.gumbel_c_base, self.gumbel_c_scale, max_visits, self.search_min_q, self.search_max_q, self.gumbel_min_scale, v_mix)
                
                if score > max_score_logit:
                    max_score_logit = score
            
            sum_score_exp = 0.0
            for i in range(n_children):
                child = children_list[i]
                sum_score_exp += math.exp(child.gumbel_score - max_score_logit)
            

            for i in range(n_children):
                child = children_list[i]
                
                # Target Probability (Pi_prime)
                pi_prime = math.exp(child.gumbel_score - max_score_logit) / sum_score_exp
                
                # Actual Visit Portion
                child_n_norm = child.visits / (1.0 + sum_visits)
                
                deficit = pi_prime - child_n_norm
                
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_child = child
            
            node = best_child
        
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

    cpdef _log_tournament_results(self, candidates, phase_name):
        cdef double min_q = self.search_min_q
        cdef double max_q = self.search_max_q
        cdef double root_v_mix = self.root.calculate_v_mix()
        
        # Calculate scale for display context only (logic matches node calc)
        cdef double scale = max_q - min_q
        if scale < self.gumbel_min_scale:
            scale = self.gumbel_min_scale
        
        # Header with Search Context
        self.logger.info(f"\n--- {phase_name} ---")
        self.logger.info(f"Tree Stats: MinQ={min_q:.4f}, MaxQ={max_q:.4f}, Scale={scale:.4f}, Root v_mix={root_v_mix:.4f}")
        self.logger.info(f"{'Move':<8} {'Visits':>8} {'Prior':>8} {'Noise':>8} {'Raw Q':>8} {'Norm Q':>8} {'Score':>8} {'Outcome':>8} {'DTM':>8}")
        self.logger.info("-" * 95)
        
        # Sort by visits (desc), then score (desc)
        sorted_cands = sorted(candidates, key=operator.attrgetter('visits', 'gumbel_score'), reverse=True)        
        
        for node in sorted_cands:
            # DIRECT ACCESS: No recalculation needed
            self.logger.info(f"{node.move.uci():<8} {node.visits:>8} {node.prior_probability_from_parent:>8.4f} {node.gumbel_noise:>8.4f} {node.q_val:>8.4f} {node.q_norm:>8.4f} {node.gumbel_score:>8.4f} {str(node.forced_outcome):>8} {str(node.distance_to_mate):>8}")
        
        self.logger.info("-" * 95)
        
    cpdef run_simulations(self, int total_simulations):
        """
        Executes the Gumbel MuZero 'Sequential Halving' search.
        """
        cdef int num_phases
        cdef int sim_budget_per_phase
        cdef int sims_per_candidate
        cdef list all_moves
        cdef list active_candidates
        cdef list child_candidates
        cdef MCTSNode_c.MCTSNode child
        cdef int actual_k
        cdef int i
        cdef int cutoff
        cdef double min_q, max_q, scale, max_visits_phase
        cdef double final_min_q, final_scale, final_max_visits
        
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
    
        # 1. Expand Root
        self._queue_leaf_for_inference(self.root)
        self._submit_batch()
        self._wait_for_inference()

        all_moves = list(self.root.children.keys())
        actual_k = min(self.gumbel_k, len(all_moves))
        
        # Calculate phases: e.g. log2(16) = 4 phases
        num_phases = int(math.ceil(math.log2(actual_k))) - 1
        final_candidate_threshold = 1

        # Add phases from config
        if self.gumbel_final_round:
            num_phases += 1
            final_candidate_threshold = 0

        if self.gumbel_first_round:
            num_phases += 1

        if num_phases < 1: num_phases = 1
        
        sim_budget_per_phase = (total_simulations - actual_k) // num_phases
        self.logger.info(f"Phases: {num_phases}, Sims per move: {sim_budget_per_phase}")

        active_candidates = []
        
        # 2. Initialize Gumbel Noise
        for move in all_moves:
            child = self.root.children[move]
            child.gumbel_noise = np.random.gumbel(0, self.gumbel_noise)
            child.gumbel_score = child.gumbel_noise + math.log(max(child.prior_probability_from_parent, 1e-8))
            
            active_candidates.append(child)

        # 3. Initial Pruning (Top-k by prior + noise)
        active_candidates.sort(key=operator.attrgetter('gumbel_score'), reverse=True)
        child_candidates = list(active_candidates)
        active_candidates = active_candidates[:actual_k]
        
        # Initial Queueing for the chosen candidates
        for child in active_candidates:
            if child.board.is_game_over(claim_draw=True):
                self._handle_terminal_node(child)
            else:
                self._queue_leaf_for_inference(child)

        self._submit_batch()
        self._wait_for_inference()

        self._log_tournament_results(active_candidates, f"Initial candidates:")

        # Include these again for first round of sequential halving if specified
        if not self.gumbel_first_round:
            active_candidates.sort(key=operator.attrgetter('gumbel_score'), reverse=True)
            cutoff = (len(active_candidates) + 1) // 2
            active_candidates = active_candidates[:cutoff]

        # 4. Sequential Halving Loop
        for phase in range(num_phases):
            if len(active_candidates) <= final_candidate_threshold: break

            sims_per_candidate = sim_budget_per_phase // len(active_candidates)
            if sims_per_candidate < 1: sims_per_candidate = 1
            
            self.logger.info(f"Phase {phase}: Running {sims_per_candidate} sims for {len(active_candidates)} candidates.")

            # A. Run Batch Simulations
            for i in range(sims_per_candidate):
                for child in active_candidates:
                    self._run_single_async_simulation(child)

            # B. Sync Barrier
            self._wait_for_inference() 
            
            # C. Update Global Stats (Using Helper)
            max_visits_phase = max([c.visits for c in active_candidates if c.visits > 0], default=1.0)

            # D. Update Scores on Nodes
            root_v_mix = self.root.calculate_v_mix()

            for child in active_candidates:
                child.calculate_gumbel_score(self.gumbel_c_base, self.gumbel_c_scale, max_visits_phase, self.search_min_q, self.search_max_q, self.gumbel_min_scale, root_v_mix)

            # E. Log Results
            self._log_tournament_results(active_candidates, f"Phase {phase} End")

            # F. Prune
            if len(active_candidates) > 1 and phase < (num_phases - 1):
                active_candidates.sort(key=operator.attrgetter('gumbel_score'), reverse=True)
                cutoff = (len(active_candidates) + 1) // 2
                active_candidates = active_candidates[:cutoff]

        
        # 5. Final Score Update
        max_visits_final = max([c.visits for c in child_candidates if c.visits > 0], default=1.0)
        root_v_mix = self.root.calculate_v_mix()

        for child in child_candidates:
            child.calculate_gumbel_score(self.gumbel_c_base, self.gumbel_c_scale, max_visits_final, self.search_min_q, self.search_max_q, self.gumbel_min_scale, root_v_mix)

        
        # Log Final Standings
        self._log_tournament_results(child_candidates, 'Final scores')

        # Logging Timings
        self.logger.info(f"--- Gumbel Search ({self.simulation_count} sims) Timings ---")
        self.logger.info(f"{'Selection time:':<25}{self.time_selection:.4f}")
        self.logger.info(f"{'Queueing time:':<25}{self.time_queueing:.4f}")
        self.logger.info(f"{'Retrieving time:':<25}{self.time_retrieval:.4f}")
        self.logger.info(f"{'Expansion time:':<25}{self.time_expansion:.4f}")
        self.logger.info(f"{'Backpropagation time:':<25}{self.time_backpropagation:.4f}")
        self.logger.info(f"{'Forced waiting for inference time:':<25}{self.time_wait_for_inference:.4f}")

        return self.simulation_count


    cpdef _expand(self, MCTSNode_c.MCTSNode node, policy_probs: torch.Tensor):
        cdef double time_expansion_start = time.perf_counter()
        
        # 1. Generate moves
        cdef list legal_moves = list(cython_chess.generate_legal_moves(node.board, chess.BB_ALL, chess.BB_ALL))
        cdef int num_moves = len(legal_moves)
        
        # 2. Early exit for terminal nodes (prevents 0-size array errors)
        if num_moves == 0:
            node.expanded = True
            self.time_expansion += (time.perf_counter() - time_expansion_start)
            return

        # 3. Pre-allocate NumPy arrays (int64 maps to torch.long)
        cdef cnp.ndarray[cnp.int64_t, ndim=1] from_row_arr = np.empty(num_moves, dtype=np.int64)
        cdef cnp.ndarray[cnp.int64_t, ndim=1] from_col_arr = np.empty(num_moves, dtype=np.int64)
        cdef cnp.ndarray[cnp.int64_t, ndim=1] channel_arr = np.empty(num_moves, dtype=np.int64)
        
        cdef cnp.int64_t[:] from_row_view = from_row_arr
        cdef cnp.int64_t[:] from_col_view = from_col_arr
        cdef cnp.int64_t[:] channel_view = channel_arr

        cdef list child_nodes_in_order = [None] * num_moves
        cdef int i
        cdef int from_row_int, from_col_int, channel_int
        cdef object move
        cdef MCTSNode_c.MCTSNode child_node

        # 5. Fast Loop
        for i in range(num_moves):
            move = legal_moves[i]
            from_row_int, from_col_int, channel_int = src_shared.utils.move_to_policy_components(move, node.board)
            
            # Direct C-array assignment
            from_row_view[i] = from_row_int
            from_col_view[i] = from_col_int
            channel_view[i] = channel_int

            child_node = MCTSNode_c.MCTSNode(board=None, parent=node, move=move)
            node.children[move] = child_node
            child_nodes_in_order[i] = child_node

    

        # 6. Tensor Creation (Zero-copy where possible)
        from_row_tensor = torch.from_numpy(from_row_arr)
        from_col_tensor = torch.from_numpy(from_col_arr)
        channel_tensor = torch.from_numpy(channel_arr)

        indices_tensor = src_shared.utils.policy_components_to_flat_index_torch(
            from_row_tensor, from_col_tensor, channel_tensor
        )

        prior_values_for_legal_moves = policy_probs.flatten()[indices_tensor]
        sum_of_legal_priors = prior_values_for_legal_moves.sum()

        cdef object normalized_legal_priors_pyobj = torch.where(
            sum_of_legal_priors > 0,
            prior_values_for_legal_moves / sum_of_legal_priors,
            prior_values_for_legal_moves
        )

        cdef cnp.ndarray[cnp.float32_t, ndim=1] prior_array
        cdef float [:] priors_view

        if normalized_legal_priors_pyobj.numel() > 0:
    
            prior_array = normalized_legal_priors_pyobj.cpu().float().numpy()
            priors_view = prior_array

            for i in range(num_moves):
                child_nodes_in_order[i].prior_probability_from_parent = priors_view[i]

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
        cdef object worst_los
        
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

            if value_for_backprop < self.search_min_q: 
                self.search_min_q = value_for_backprop
            if value_for_backprop > self.search_max_q: 
                self.search_max_q = value_for_backprop
            
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