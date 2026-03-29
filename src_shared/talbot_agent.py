import os, sys
import numpy as np
import time
import chess
import random

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

import src_shared.utils
from src_shared.mcts_engine import MCTSEngine

class TalbotAgent:
    """
    A chess player wrapper for an MCTS engine designed for a multiprocessing
    environment with a central batcher. This class manages the game state
    for a single game worker and communicates with the MCTS instance.
    """
    def __init__(self, name, logger, talbot_config, worker_id, inference_queue, result_queue, shared_input_buffer, shared_policy_buffer, shared_value_buffer, buffer_free_slots):
        self.name = name
        self.logger = logger
        self.talbot_config = talbot_config
        
        # The agent now has direct access to the queues for its MCTS engine
        self.worker_id = worker_id
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.worker_batch_size = self.talbot_config['batch_size_per_worker']
        self.shared_input_buffer = shared_input_buffer
        self.shared_policy_buffer = shared_policy_buffer
        self.shared_value_buffer = shared_value_buffer
        self.buffer_free_slots = buffer_free_slots

        # These are reset each game
        self.mcts = None
        self.use_resignation = None
    
    def get_move(self, board, ply_count, phase_budgets):
        """
        Runs MCTS simulations and selects a move based on a temperature schedule.
        """
        move_number = ((ply_count - 1) // 2) + 1

        self.logger.info(f"\n{'='*60}\n{' '*20}--- MOVE {move_number}: {'White' if board.turn == chess.WHITE else 'Black'}, PLY {ply_count} STARTED ---\n{'='*60}\n")
        move_start_time = time.time()
        
        self.logger.info(f"Current player: {self.name}")

        self.mcts = MCTSEngine(
            logger=self.logger, 
            worker_id=self.worker_id,
            worker_batch_size=self.worker_batch_size,
            inference_queue=self.inference_queue,
            result_queue=self.result_queue,
            virtual_loss=self.talbot_config['virtual_loss'],
            draw_cutoff=self.talbot_config['draw_cutoff'],
            gumbel_c_visit=self.talbot_config['gumbel_c_visit'],
            gumbel_c_scale=self.talbot_config['gumbel_c_scale'],
            gumbel_noise=self.talbot_config['gumbel_noise'],
            board=board,
            shared_input_buffer=self.shared_input_buffer,
            shared_policy_buffer=self.shared_policy_buffer,
            shared_value_buffer=self.shared_value_buffer,
            buffer_free_slots=self.buffer_free_slots
        )

        simulation_count = self.mcts.run_simulations(phase_budgets)

        # --- 1. CALCULATE BASE MCTS POLICY (All Legal Moves) ---
        all_children = list(self.mcts.root.children.values())
        all_moves = [c.move for c in all_children]

        # Extract base probabilities using the noiseless Gumbel scores for ALL legal moves
        base_logits = np.array([c.gumbel_score - c.gumbel_noise for c in all_children], dtype=np.float32)
        base_logits = base_logits - np.max(base_logits)
        base_probs = np.exp(base_logits) / np.sum(np.exp(base_logits))

        # --- 2. ISOLATE NODE CLASSIFICATIONS (All Legal Moves) ---
        visited_nodes = [c for c in all_children if c.visits > 0]

        winning_nodes = [c for c in all_children if c.forced_outcome == -1]
        losing_nodes = [c for c in all_children if c.forced_outcome == 1]
        draw_nodes = [c for c in all_children if c.forced_outcome == 0]
        non_forced_nodes = [c for c in all_children if c.forced_outcome is None]
        non_forced_visited = [c for c in non_forced_nodes if c.visits > 0]

        # --- 3. TARGET GENERATION (final_probs over ALL legal moves) ---
        smoothing_factor = self.talbot_config['minimax_smoothing_factor']

        if winning_nodes:
            # Find the fastest mate(s)
            min_dtm = min(child.distance_to_mate for child in winning_nodes)
            fastest_wins = [child for child in winning_nodes if child.distance_to_mate == min_dtm]
            
            # Rule: Treat all FASTEST forced wins with equal probability for the prior
            minimax_probs = np.zeros(len(all_children), dtype=np.float32)
            prob_per_best = 1.0 / len(fastest_wins)
            for i, child in enumerate(all_children):
                if child in fastest_wins:
                    minimax_probs[i] = prob_per_best
                    
            final_probs = (1.0 - smoothing_factor) * base_probs + (smoothing_factor * minimax_probs)
            self.logger.info(f"{len(fastest_wins)} fastest win(s) found (DTM {min_dtm}).")

        elif draw_nodes and (self.mcts.root.calculate_v_mix() <= self.talbot_config['draw_cutoff']):
            # Rule: Treat draw like a winning node if condition is satisfied
            minimax_probs = np.zeros(len(all_children), dtype=np.float32)
            prob_per_best = 1.0 / len(draw_nodes)
            for i, child in enumerate(all_children):
                if child in draw_nodes:
                    minimax_probs[i] = prob_per_best
            final_probs = (1.0 - smoothing_factor) * base_probs + (smoothing_factor * minimax_probs)
            self.logger.info(f"Forced draw condition met for {len(draw_nodes)}.")

        elif losing_nodes and non_forced_nodes:
            # Rule: Inverse blend. Boost safe moves evenly. Losing moves retain (smoothing_factor * base_probs).
            minimax_probs = np.zeros(len(all_children), dtype=np.float32)
            prob_per_best = 1.0 / len(non_forced_nodes)
            for i, child in enumerate(all_children):
                if child in non_forced_nodes:
                    minimax_probs[i] = prob_per_best
            final_probs = (1.0 - smoothing_factor) * base_probs + (smoothing_factor * minimax_probs)
            self.logger.info(f"{len(losing_nodes)} forced loss(es) found.")

        else:
            # Rule: No wins, no losses, and draw condition not met. Ignore and use base.
            final_probs = base_probs

        # --- 4. MOVE SELECTION ---
        # Rule A: If a move is winning, pick it (lowest DTM)
        if winning_nodes:
            min_dtm = min(c.distance_to_mate for c in winning_nodes)
            best_move = random.choice([c.move for c in winning_nodes if c.distance_to_mate == min_dtm])

        # Rule B: If a move is drawing and the draw condition is satisfied, pick it
        elif draw_nodes and (self.mcts.root.calculate_v_mix() <= self.talbot_config['draw_cutoff']):
            best_move = random.choice([c.move for c in draw_nodes])
            
        # Rule C: Filter losing moves out of the pool. Select from safe visited nodes.
        elif non_forced_visited:
            if ply_count <= self.talbot_config['temperature_ply_cutoff']:
                visits = np.array([c.visits for c in non_forced_visited], dtype=np.float32)
                indices = list(range(len(non_forced_visited)))
                indices.sort(key=lambda i: (non_forced_visited[i].visits, non_forced_visited[i].gumbel_score), reverse=True)
                top_indices = indices[:2]
                
                best_idx = top_indices[np.argmax([non_forced_visited[i].gumbel_score for i in top_indices])]
                best_q_val = -non_forced_visited[best_idx].calculate_v_mix()
                
                valid_indices = [i for i, node in enumerate(non_forced_visited) 
                                 if (best_q_val - (-node.calculate_v_mix())) <= self.talbot_config['temperature_blunder_threshold']]
                
                act_probs = np.zeros(len(non_forced_visited), dtype=np.float32)
                top_prob = self.talbot_config['temperature_top_move']
                act_probs[best_idx] = top_prob
                
                remaining_prob = 1.0 - top_prob
                other_indices = [i for i in valid_indices if i != best_idx]
                
                if remaining_prob > 0 and len(other_indices) > 0:
                    other_visits = visits[other_indices]
                    sum_other_visits = np.sum(other_visits)
                    if sum_other_visits > 0:
                        act_probs[other_indices] = (other_visits / sum_other_visits) * remaining_prob
                    else:
                        act_probs[best_idx] = 1.0
                else:
                    act_probs[best_idx] = 1.0
                
                act_probs /= np.sum(act_probs)
                chosen_idx = np.random.choice(len(non_forced_visited), p=act_probs)
                best_move = non_forced_visited[chosen_idx].move
            else:
                # Late game: Greedy choice strictly from safe moves
                sorted_by_visits = sorted(non_forced_visited, key=lambda c: (c.visits, c.gumbel_score), reverse=True)
                best_move = max(sorted_by_visits[:2], key=lambda c: c.gumbel_score).move
                
        # Rule D: If all visited moves are forced losses, delay the mate as long as possible
        else:
            if draw_nodes:
                # Take the draw over getting checkmated
                best_move = random.choice(draw_nodes).move
            elif losing_nodes:
                # Delay the mate as long as possible
                best_move = max(losing_nodes, key=lambda c: c.distance_to_mate).move
            else:
                # Absolute fallback
                best_move = random.choice(all_moves)

        # Check Resignation (After target generation, before return)
        if (self.use_resignation and self.mcts.root.calculate_v_mix() < self.talbot_config['resignation_cutoff']):
            return None, np.zeros(src_shared.utils.TOTAL_POLICY_MOVES, dtype=np.float32), simulation_count, 0.0

        # --- 5. MAP TO GLOBAL POLICY TENSOR ---
        policy_vector = np.zeros(src_shared.utils.TOTAL_POLICY_MOVES, dtype=np.float32)
        for i, move in enumerate(all_moves):
            from_row, from_col, channel = src_shared.utils.move_to_policy_components(move, board)
            flat_index = src_shared.utils.policy_components_to_flat_index(from_row, from_col, channel)
            policy_vector[flat_index] = final_probs[i]

        entropy = -np.sum(final_probs * np.log(final_probs + 1e-10))

        move_end_time = time.time()
        total_move_time = move_end_time - move_start_time
        simulation_speed = (simulation_count / total_move_time) if total_move_time > 0 else 0

        self.logger.info(f"Total move time: {total_move_time:.4f}, with {simulation_speed:.4f} simulations per second")
        self.logger.info(f"Total entropy: {entropy:.4f}")
        self.logger.info(f"Average root node value: {self.mcts.root.calculate_v_mix():.4f}")

        return best_move, policy_vector, simulation_count, entropy


    def reset_for_new_game(self):
        """
        Resets the player's state for a new game.
        """
        self.logger.debug(f"Resetting state for a new game.")
        self.mcts = None
        self.use_resignation = random.random() < self.talbot_config['resignation_probability']