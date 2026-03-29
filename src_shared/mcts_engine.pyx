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
from libc.math cimport exp
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
    cdef public double gumbel_c_visit
    cdef public double gumbel_c_scale
    cdef public double gumbel_noise
    cdef public bint use_fp16

    cdef public double time_selection
    cdef public double time_expansion
    cdef public double time_backpropagation
    cdef public double time_retrieval
    cdef public double time_queueing
    cdef public double time_misc
    cdef public double time_wait_for_inference    

    cdef public MCTSNode_c.MCTSNode root 
    
    cdef public object logger
    cdef public object root_board
    cdef public object inference_queue
    cdef public object result_queue
    cdef public object in_flight_nodes
    cdef public object batch_buffer
    cdef public object device
    cdef public object policy_logits_dtype
    cdef public object shared_input_buffer
    cdef public object shared_policy_buffer
    cdef public object shared_value_buffer
    cdef public object buffer_free_slots

    def __init__(self, logger: logging.Logger, worker_batch_size: int, inference_queue, result_queue, worker_id: int, virtual_loss: float,
             draw_cutoff: float, gumbel_c_visit: float, gumbel_c_scale: float, gumbel_noise: float, board: chess.Board, shared_input_buffer, shared_policy_buffer, shared_value_buffer, buffer_free_slots):

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
        self.gumbel_c_visit = gumbel_c_visit
        self.gumbel_c_scale = gumbel_c_scale

        self.root_board = board
        self.root = MCTSNode_c.MCTSNode()
        self.in_flight_nodes = [None] * len(shared_input_buffer)

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
        self.policy_logits_dtype = torch.float16 if self.use_fp16 else torch.float32


    cdef _sync_board_to_node(self, MCTSNode_c.MCTSNode node):
        """
        Resets root_board to the current position, then pushes moves 
        down the path to 'node'.
        """
        while self.root_board.move_stack:
            self.root_board.pop()
        
        cdef list path = []
        cdef MCTSNode_c.MCTSNode current = node
        while current.parent is not None:
            path.append(current.move)
            current = current.parent
            
        for i in range(len(path) - 1, -1, -1):
            self.root_board.push(path[i])


    cdef _wait_for_inference(self):
        """
        Handles the final flush of the inference queue and waits for all
        remaining results to complete backpropagation.
        """             
        cdef double time_wait_for_inference_start = time.perf_counter()
        cdef int buffer_index
        cdef object raw_policy_logits, raw_value_output
        cdef MCTSNode_c.MCTSNode node
        cdef double value_output
        cdef object policy_logits
        cdef list completed_indices

        cdef int batch_buffer_size = len(self.batch_buffer)

        self.logger.debug(f"[Misc] Flush: self.batch_buffer size: {batch_buffer_size}")
        if self.batch_buffer:
            self._submit_batch()
            self.logger.debug(f"[Misc] Flushed final partial batch of size {batch_buffer_size} to inference queue. Inferences sent: {self.inference_sent}")

        # Wait for remaining nodes - this is explicitly acknowledged as a blocking wait_for_inference step
        while self.inference_received < self.inference_sent:
            try:
                completed_indices = self.result_queue.get()

                for buffer_index in completed_indices:
                    node = self.in_flight_nodes[buffer_index]
                    self.in_flight_nodes[buffer_index] = None
                    self.inference_received += 1

                    raw_policy_logits = self.shared_policy_buffer[buffer_index] 
                    raw_value_output = self.shared_value_buffer[buffer_index]

                    policy_logits = raw_policy_logits.to(self.policy_logits_dtype, copy=True)
                    value_output = raw_value_output.item()

                    self.buffer_free_slots.put(buffer_index) 
                    
                    # DEFERRED EXPANSION
                    if not node.expanded:
                        node.pending_logits = policy_logits
                     
                    self._backpropagate(node, value_output, is_terminal=False)                
                    self.logger.debug(f"[Backpropagation] Backpropagating on node during final wait.")

            except queue.Empty:
                self.logger.debug(f"[Misc] Result queue empty during final wait (self.inference_received={self.inference_received}, self.inference_sent={self.inference_sent}). Waiting for more results...")
                break

        self.time_wait_for_inference += (time.perf_counter() - time_wait_for_inference_start)
        


    cpdef _select(self, MCTSNode_c.MCTSNode start_node, list simulation_path):
        """
        Optimized Traversal: Single-pass selection with zero list allocations.
        """
        cdef double time_selection_start = time.perf_counter()
        
        cdef MCTSNode_c.MCTSNode node = start_node
        cdef MCTSNode_c.MCTSNode child, best_child
        
        cdef double max_visits, sum_visits, v_mix
        cdef double max_score_logit, sum_score_exp, score
        cdef double pi_prime, child_n_norm, deficit, best_deficit
        cdef int i
        cdef int num_children
        
        while True:
            # JUST-IN-TIME EXPANSION
            if node.pending_logits is not None:
                self._expand(node, node.pending_logits)
                node.pending_logits = None
                
            if not node.children or not node.expanded or node.selected:
                break

            best_child = None
            best_deficit = -1e20
            
            # 1. First Pass: Get Max Visits and v_mix
            max_visits = 0.0
            sum_visits = 0.0


            num_children = len(node.child_list)

            for i in range(num_children):
                child = <MCTSNode_c.MCTSNode>node.child_list[i]
                if child.forced_outcome == 1:
                    continue
                    
                if child.visits > max_visits:
                    max_visits = child.visits
                sum_visits += child.visits
            
            v_mix = node.calculate_v_mix()
            
            # 2. Second Pass: Calculate Gumbel scores and find max logit for stable Softmax
            max_score_logit = -1e20
            for i in range(num_children):
                child = <MCTSNode_c.MCTSNode>node.child_list[i]
                if child.forced_outcome == 1:
                    continue 
                score = child.calculate_gumbel_score(self.gumbel_c_visit, self.gumbel_c_scale, max_visits, v_mix)
                if score > max_score_logit:
                    max_score_logit = score
            
            # 3. Third Pass: Sum exps and find Best Deficit (Deterministic Policy Matching)
            sum_score_exp = 0.0
            for i in range(num_children):
                child = <MCTSNode_c.MCTSNode>node.child_list[i]
                if child.forced_outcome == 1:
                    continue
                sum_score_exp += exp(child.gumbel_score - max_score_logit)

            for i in range(num_children):
                child = <MCTSNode_c.MCTSNode>node.child_list[i]
                if child.forced_outcome == 1:
                    continue
                pi_prime = exp(child.gumbel_score - max_score_logit) / sum_score_exp
                child_n_norm = child.visits / (1.0 + sum_visits)
                
                deficit = pi_prime - child_n_norm
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_child = child
            
            if best_child is None:
                break

            self.root_board.push(best_child.move) 
            simulation_path.append(best_child)
            node = best_child
        
        self.time_selection += (time.perf_counter() - time_selection_start)
        return node


    cdef _mark_selected(self, MCTSNode_c.MCTSNode node):
        """
        Optimized: Only crawls up the tree when a node's last child is queued.
        """
        cdef MCTSNode_c.MCTSNode current_node = node
        cdef MCTSNode_c.MCTSNode parent
        
        current_node.selected = True
        
        parent = current_node.parent
        while parent is not None:
            parent.num_unselected_children -= 1
            if parent.num_unselected_children > 0:
                break
                
            parent.selected = True
            current_node = parent
            parent = current_node.parent


    cdef _unmark_selected(self, MCTSNode_c.MCTSNode node):
        cdef MCTSNode_c.MCTSNode current_node = node
        cdef MCTSNode_c.MCTSNode parent
        
        current_node.selected = False
        parent = current_node.parent
        
        while parent is not None:
            parent.num_unselected_children += 1
            
            # If it is exactly 1, it means it was previously 0 (fully selected).
            # We must unselect it and tell ITS parent that a slot opened up.
            if parent.num_unselected_children == 1:
                parent.selected = False
                current_node = parent
                parent = current_node.parent
            else:
                # If it's > 1, the parent wasn't fully selected anyway. We can stop.
                break


    cdef _retrieve_inference(self, bint block):
        cdef double time_retrieval_start
        cdef int buffer_index
        cdef object completed_indices
        cdef object raw_policy_logits, raw_value_output
        cdef MCTSNode_c.MCTSNode node
        cdef double value_output
        cdef object policy_logits

        while True:
            try:
                time_retrieval_start = time.perf_counter()
                
                # We now receive a list of indices from the queue
                completed_indices = self.result_queue.get(block=block)
                block = False
                
                for buffer_index in completed_indices:
                    node = self.in_flight_nodes[buffer_index]
                    self.in_flight_nodes[buffer_index] = None

                    self.inference_received += 1

                    raw_policy_logits = self.shared_policy_buffer[buffer_index] 
                    raw_value_output = self.shared_value_buffer[buffer_index]

                    policy_logits = raw_policy_logits.to(self.policy_logits_dtype, copy=True)
                    value_output = raw_value_output.item()

                    self.buffer_free_slots.put(buffer_index) 
                    
                    if not node.expanded:
                        node.pending_logits = policy_logits
                        
                    self._backpropagate(node, value_output, is_terminal=False)

                self.time_retrieval += (time.perf_counter() - time_retrieval_start)

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
        
        result = self.root_board.result(claim_draw=True)
        self._mark_selected(leaf)
        
        if result == "1-0":
            value = 1.0 if self.root_board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            value = 1.0 if self.root_board.turn == chess.BLACK else -1.0
            
        self.time_expansion += (time.perf_counter() - time_expansion_start)
        self._backpropagate(leaf, value, is_terminal=True)
        self.simulation_count += 1


    cdef _queue_leaf_for_inference(self, MCTSNode_c.MCTSNode leaf):
        cdef double time_misc_start = time.perf_counter()
        cdef int buffer_index
        cdef object numpy_board, board_input
        
        # Fast spin-wait from the old engine
        while self.buffer_free_slots.empty():
            self._retrieve_inference(block=False)
            if self.batch_buffer:
                self._submit_batch()
            time.sleep(0.001)

        buffer_index = self.buffer_free_slots.get()
        self.in_flight_nodes[buffer_index] = leaf
        self._mark_selected(leaf)
        
        numpy_board = src_shared.utils.board_to_tensor_69(self.root_board)
        board_input = torch.from_numpy(numpy_board).float()
        
        if self.use_fp16:
            board_input = board_input.half()

        self.shared_input_buffer[buffer_index].copy_(board_input)

        self.batch_buffer.append((self.worker_id, buffer_index))
        self._virtual_loss(leaf, is_applying=True)

        if len(self.batch_buffer) >= self.worker_batch_size:
            self._submit_batch()

        self.time_misc += (time.perf_counter() - time_misc_start) 
        self.simulation_count += 1


    cdef _run_single_async_simulation(self, MCTSNode_c.MCTSNode start_node):
        """ 
        Runs one descent starting from start_node, handles queueing.
        """
        cdef MCTSNode_c.MCTSNode leaf
        cdef int current_batch_size
        cdef double time_misc_start
        cdef list simulation_path = []
        cdef int start_path_len

        self.root_board.push(start_node.move)
        simulation_path.append(start_node)
        
        while True:
            self._retrieve_inference(block=False)
            current_batch_size = len(self.batch_buffer)

            if current_batch_size >= self.worker_batch_size:
                self._submit_batch()
            
            # Check if start node is fully queued
            if start_node.selected:
                if current_batch_size > 0:
                    self._submit_batch()

                # Root is queued + all inferences received -> break (Terminal state reached)
                if self.inference_received >= self.inference_sent:
                    self.logger.debug(f"Only terminal nodes remaining - breaking MCTS loop")
                    break

                time.sleep(0.001)
                continue

            start_path_len = len(simulation_path)
            leaf = self._select(start_node, simulation_path)

            if self.root_board.is_game_over(claim_draw=True):
                self._handle_terminal_node(leaf)
                break
                
            if leaf.selected or self.buffer_free_slots.empty():
                if self.batch_buffer:
                    self._submit_batch()

                if leaf.selected and self.inference_received >= self.inference_sent:
                    self._backpropagate(leaf, leaf.calculate_v_mix(), is_terminal=False)
                    self.simulation_count += 1
                    break
                    
                while len(simulation_path) > start_path_len:
                    self.root_board.pop()
                    simulation_path.pop()
                
                time.sleep(0.001)
                continue

            # 3. Normal leaf expansion
            self._queue_leaf_for_inference(leaf)
            break

        # Final cleanup: backtrack to the root
        while simulation_path:  
            self.root_board.pop()
            simulation_path.pop()

        return


    cpdef _log_tournament_results(self, candidates, phase_name):
        cdef double root_v_mix = self.root.calculate_v_mix()
        
        # Header with Search Context
        self.logger.info(f"\n--- {phase_name} ---")
        self.logger.info(f"Tree Stats: Root v_mix={root_v_mix:.4f}")
        self.logger.info(f"{'Move':<8} {'Visits':>8} {'Logit':>8} {'Noise':>8} {'Raw Q':>8} {'Norm Q':>8} {'Score':>8} {'Outcome':>8} {'DTM':>8}")
        self.logger.info("-" * 95)
        
        # Sort by visits (desc), then score (desc)
        sorted_cands = sorted(candidates, key=operator.attrgetter('visits', 'gumbel_score'), reverse=True)        
        
        for node in sorted_cands:
            # DIRECT ACCESS: No recalculation needed
            self.logger.info(f"{node.move.uci():<8} {node.visits:>8} {node.raw_logit:>8.4f} {node.gumbel_noise:>8.4f} {node.q_val:>8.4f} {node.q_norm:>8.4f} {node.gumbel_score:>8.4f} {str(node.forced_outcome):>8} {str(node.distance_to_mate):>8}")
        
        self.logger.info("-" * 95)
            

    cpdef run_simulations(self, int search_depth, int max_m):
        """
        Executes the Gumbel MuZero 'Sequential Halving' search dynamically.
        Instead of a hardcoded list, it allocates the simulation budget
        across log2(m) phases.
        """
        cdef int sims_per_candidate
        cdef list all_nodes
        cdef list active_candidates
        cdef MCTSNode_c.MCTSNode child
        cdef int m
        cdef int num_phases
        cdef int phase_budget
        cdef int remaining_search_depth
        cdef int num_cands
        cdef int i
        cdef int cutoff
        cdef int phase_idx
        cdef double max_visits_phase
        cdef double max_visits_final
        cdef double root_v_mix
        
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

        # ENSURE ROOT EXPANSION BEFORE PHASE 0
        if self.root.pending_logits is not None:
            self._expand(self.root, self.root.pending_logits)
            self.root.pending_logits = None

        all_nodes = list(self.root.children.values())
        active_candidates = []
        
        # 2. Initialize Gumbel Noise
        for child in all_nodes:
            child.gumbel_noise = np.random.gumbel(0, self.gumbel_noise)
            child.gumbel_score = child.gumbel_noise + child.raw_logit

            # Check for terminal nodes in all children
            self.root_board.push(child.move)
            if self.root_board.is_game_over(claim_draw=True):
                self._handle_terminal_node(child)
            else:
                active_candidates.append(child)
            self.root_board.pop()

        # 3. Determine actual m (bounded by available legal moves)
        m = min(max_m, len(active_candidates))

        if m == 0:
            return self.simulation_count

        self._log_tournament_results(all_nodes, "Initial candidates:")

        # 4. Initial Pruning (Top-m by prior + noise)
        active_candidates.sort(key=operator.attrgetter('gumbel_score'), reverse=True)
        active_candidates = active_candidates[:m]

        # Calculate phases
        if m <= 1:
            num_phases = 1
        else:
            num_phases = int(math.ceil(math.log2(m)))
            
        phase_budget = search_depth // num_phases
        remaining_search_depth = search_depth

        # 5. Dynamic Sequential Halving Loop
        for phase_idx in range(num_phases):
            num_cands = len(active_candidates)
            if num_cands <= 1:
                break

            self.logger.info(f"Phase {phase_idx}: Budget {phase_budget} -> {phase_budget // num_cands} sims for {num_cands} candidates.")
            
            # Root expansion budget tracking and initialization for Phase 0
            if phase_idx == 0:
                for child in active_candidates:
                    remaining_search_depth -= 1
                    self.root_board.push(child.move)
                    if not self.root_board.is_game_over(claim_draw=True):
                        self._queue_leaf_for_inference(child)
                    self.root_board.pop()

                self._submit_batch()
                self._wait_for_inference()

            # Calculate exact visits for this phase
            if phase_idx == num_phases - 1:
                # The final phase consumes all remaining extra visits to exactly hit n
                sims_per_candidate = remaining_search_depth // num_cands
            else:
                # Standard phase follows max(1, floor(budget / candidates))
                sims_per_candidate = max(1, phase_budget // num_cands)
                
                # Deduct the 1 visit we already spent on expansion during Phase 0
                if phase_idx == 0:
                    sims_per_candidate = max(0, sims_per_candidate - 1)

            # Safety bound to never exceed remaining budget
            sims_per_candidate = max(0, min(sims_per_candidate, remaining_search_depth // num_cands))

            # A. Run Batch Simulations
            for i in range(sims_per_candidate):
                for child in active_candidates:
                    self._run_single_async_simulation(child)
                    remaining_search_depth -= 1

            # B. Sync Barrier
            self._wait_for_inference() 
            
            # C. Update Global Stats 
            max_visits_phase = max([c.visits for c in active_candidates if c.visits > 0], default=1.0)

            # D. Update Scores on Nodes
            root_v_mix = self.root.calculate_v_mix()

            for child in active_candidates:
                child.calculate_gumbel_score(self.gumbel_c_visit, self.gumbel_c_scale, max_visits_phase, root_v_mix)

            # E. Log Results
            self._log_tournament_results(active_candidates, f"Phase {phase_idx} End")

            # F. Prune (Halve the candidates, ensuring we don't accidentally drop below 2 before the end)
            if num_cands > 1 and phase_idx < (num_phases - 1):
                active_candidates = [c for c in active_candidates if c.forced_outcome != 1]
                if len(active_candidates) == 0:
                    break
                
                if len(active_candidates) > 1:
                    active_candidates.sort(key=operator.attrgetter('gumbel_score'), reverse=True)
                    # Use integer division to exactly halve, or round up if odd
                    cutoff = (len(active_candidates) + 1) // 2
                    active_candidates = active_candidates[:cutoff]

        # 6. Final Score Update
        max_visits_final = max([c.visits for c in all_nodes if c.visits > 0], default=1.0)
        root_v_mix = self.root.calculate_v_mix()

        for child in all_nodes:
            child.calculate_gumbel_score(self.gumbel_c_visit, self.gumbel_c_scale, max_visits_final, root_v_mix)

        # Log Final Standings
        self._log_tournament_results(all_nodes, 'Final scores')

        # Logging Timings
        self.logger.info(f"--- Gumbel Search ({self.simulation_count} sims) Timings ---")
        self.logger.info(f"{'Selection time:':<25}{self.time_selection:.4f}")
        self.logger.info(f"{'Queueing time:':<25}{self.time_queueing:.4f}")
        self.logger.info(f"{'Retrieving time:':<25}{self.time_retrieval:.4f}")
        self.logger.info(f"{'Expansion time:':<25}{self.time_expansion:.4f}")
        self.logger.info(f"{'Backpropagation time:':<25}{self.time_backpropagation:.4f}")
        self.logger.info(f"{'Forced waiting for inference time:':<25}{self.time_wait_for_inference:.4f}")

        return self.simulation_count


    cpdef _expand(self, MCTSNode_c.MCTSNode node, policy_logits: torch.Tensor):
        cdef double time_expansion_start = time.perf_counter()
        
        cdef list legal_moves = list(cython_chess.generate_legal_moves(self.root_board, chess.BB_ALL, chess.BB_ALL))
        cdef int num_moves = len(legal_moves)
        
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

        for i in range(num_moves):
            move = legal_moves[i]
            from_row_int, from_col_int, channel_int = src_shared.utils.move_to_policy_components(move, self.root_board)
            
            from_row_view[i] = from_row_int
            from_col_view[i] = from_col_int
            channel_view[i] = channel_int

            child_node = MCTSNode_c.MCTSNode(parent=node, move=move)
            node.children[move] = child_node
            child_nodes_in_order[i] = child_node

        if num_moves > 0:
            from_row_tensor = torch.from_numpy(from_row_arr)
            from_col_tensor = torch.from_numpy(from_col_arr)
            channel_tensor = torch.from_numpy(channel_arr)

            indices_tensor = src_shared.utils.policy_components_to_flat_index_torch(
                from_row_tensor, from_col_tensor, channel_tensor
            )

            raw_logits_for_legal_moves = policy_logits.flatten()[indices_tensor]

            for i in range(num_moves):
                current_child = <MCTSNode_c.MCTSNode>child_nodes_in_order[i]
                current_child.raw_logit = float(raw_logits_for_legal_moves[i])

        node.num_unselected_children = num_moves
        node.expanded = True
        node.child_list = child_nodes_in_order
        self.time_expansion += (time.perf_counter() - time_expansion_start)


    cdef _backpropagate_minimax(self, MCTSNode_c.MCTSNode node):
        """
        Pure Exact-Value Solver for Mates Only
        """
        if not node.children:
            return

        cdef MCTSNode_c.MCTSNode child
        cdef int best_win_dtm = 999999
        cdef int worst_loss_dtm = -1
        cdef bint all_children_are_wins = True
        cdef bint has_winning_child = False
        cdef int i
        cdef int num_children = len(node.child_list)

        # Single pass over children dictionary
        for i in range(num_children):
            child = <MCTSNode_c.MCTSNode>node.child_list[i]            
            if child.forced_outcome == -1:
                has_winning_child = True
                if child.distance_to_mate < best_win_dtm:
                    best_win_dtm = child.distance_to_mate
            
            if child.forced_outcome != 1:
                all_children_are_wins = False
            else:
                if child.distance_to_mate > worst_loss_dtm:
                    worst_loss_dtm = child.distance_to_mate

        # Logic Implementation
        if has_winning_child:
            node.forced_outcome = 1
            node.distance_to_mate = best_win_dtm + 1
        elif all_children_are_wins:
            node.forced_outcome = -1
            node.distance_to_mate = worst_loss_dtm + 1
        else:
            node.forced_outcome = None
            node.distance_to_mate = None
            

    cdef _backpropagate(self, MCTSNode_c.MCTSNode node, double value, bint is_terminal):
        cdef double time_backpropagation_start = time.perf_counter()
        
        # 1. Handle Tree Locking State
        if is_terminal:
            node.forced_outcome = int(value) 
            node.distance_to_mate = 0
            # CRITICAL: We do nothing to the selected flags. 
            # The leaf stays selected forever, and the parent permanently loses 1 from its unselected_children counter.
        else:
            self._virtual_loss(node, is_applying=False)
            # CRITICAL: Unmark the leaf, which safely cascades the counter restorations up the tree.
            self._unmark_selected(node)

        # 2. Standard Value Accumulation
        cdef MCTSNode_c.MCTSNode current_node = node
        cdef double value_for_backprop = value
        current_node.raw_value = value

        while current_node is not None:
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