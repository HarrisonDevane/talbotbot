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
    
    def get_move(self, board, ply_count, search_depth):
            """
            Runs MCTS simulations and selects a move based on a temperature schedule.
            """
            move_number = ((ply_count - 1) // 2) + 1

            self.logger.info(f"\n{'='*60}\n{' '*20}--- MOVE {move_number}: {'White' if board.turn == chess.WHITE else 'Black'}, PLY {ply_count} STARTED ---\n{'='*60}\n")
            move_start_time = time.time()
            
            self.logger.info(f"Current player: {self.name}")

            if not self.talbot_config['training']:                
                if ply_count < self.talbot_config['gumbel_move_cutoff']*2:
                    current_noise = self.talbot_config['gumbel_noise']
                else:
                    current_noise = 1e-6
            else:
                # Training Mode: Always full noise
                current_noise = self.talbot_config['gumbel_noise']

            self.mcts = MCTSEngine(
                logger=self.logger, 
                worker_id=self.worker_id,
                training=self.talbot_config['training'],
                worker_batch_size=self.worker_batch_size,
                inference_queue=self.inference_queue,
                result_queue=self.result_queue,
                cpuct=self.talbot_config['cpuct'],
                virtual_loss=self.talbot_config['virtual_loss'],
                draw_cutoff=self.talbot_config['draw_cutoff'],
                gumbel_k=self.talbot_config['gumbel_k'],
                gumbel_sigma=self.talbot_config['gumbel_sigma'],
                gumbel_noise=current_noise,
                board=board,
                shared_input_buffer=self.shared_input_buffer,
                shared_policy_buffer=self.shared_policy_buffer,
                shared_value_buffer=self.shared_value_buffer,
                buffer_free_slots=self.buffer_free_slots
            )

            simulation_count = self.mcts.run_simulations(search_depth)

            best_move = None
            policy_vector = np.zeros(src_shared.utils.TOTAL_POLICY_MOVES, dtype=np.float32)

            # If a forced win is detected, choose the fastest winning move.
            if (self.mcts.root.forced_outcome == 1) and any(c.forced_outcome == -1 for c in self.mcts.root.children.values()):
                winning_moves = [
                    move for move, child in self.mcts.root.children.items() 
                    if child.forced_outcome == -1
                ]
                min_dtm = min(self.mcts.root.children[move].distance_to_mate for move in winning_moves)
                best_winning_moves = [
                    move for move in winning_moves 
                    if self.mcts.root.children[move].distance_to_mate == min_dtm
                ]
                
                best_move = np.random.choice(best_winning_moves)
                prob_per_move = 1.0 / len(best_winning_moves)
                for move in best_winning_moves:
                    from_row, from_col, channel = src_shared.utils.move_to_policy_components(move, self.mcts.root.board)
                    flat_index = src_shared.utils.policy_components_to_flat_index(from_row, from_col, channel)
                    policy_vector[flat_index] = prob_per_move
                
                self.logger.info(f"{len(best_winning_moves)} forced win/s in {min_dtm} moves were found.")

            # If a position is losing (defined by the average root node value) but has a forced draw available, take the draw based on the cutoff
            elif (self.mcts.root.forced_outcome == 0) and any(c.forced_outcome == 0 for c in self.mcts.root.children.values()) and (self.mcts.root.value_sum / self.mcts.root.visits <= self.talbot_config['draw_cutoff']):
                draw_moves = [
                    move for move, child in self.mcts.root.children.items()
                    if child.forced_outcome == 0
                ]

                # Calculate the average value for each draw move
                draw_move_values = {}
                for move in draw_moves:
                    child = self.mcts.root.children[move]
                    average_value = child.value_sum / child.visits
                    draw_move_values[move] = average_value

                # Find the maximum average value
                max_avg_value = max(draw_move_values.values())
                best_draw_moves = [
                    move for move, avg_value in draw_move_values.items()
                    if abs(avg_value - max_avg_value) < 1e-6
                ]

                best_move = np.random.choice(best_draw_moves)

                prob_per_move = 1.0 / len(best_draw_moves)
                for move in best_draw_moves:
                    from_row, from_col, channel = src_shared.utils.move_to_policy_components(move, self.mcts.root.board)
                    flat_index = src_shared.utils.policy_components_to_flat_index(from_row, from_col, channel)
                    policy_vector[flat_index] = prob_per_move

                self.logger.info(f"A forced draw is the best possible outcome. Choosing one of {len(best_draw_moves)} moves with the highest average value ({max_avg_value:.4f}).")

            # If a forced loss is detected, choose the move that leads to the longest mate for the opponent.
            elif (self.mcts.root.forced_outcome == -1) and any(c.forced_outcome == 1 for c in self.mcts.root.children.values()):
                losing_child_for_parent = max(
                    (child for child in self.mcts.root.children.values()),
                    key=lambda c: c.distance_to_mate
                )
                
                best_move = losing_child_for_parent.move
                
                # Distribute probabilities equally among all children that lead to the same longest mate.
                longest_mate_moves = [
                    move for move, child in self.mcts.root.children.items() 
                    if child.distance_to_mate == losing_child_for_parent.distance_to_mate
                ]
                
                prob_per_move = 1.0 / len(longest_mate_moves)
                for move in longest_mate_moves:
                    from_row, from_col, channel = src_shared.utils.move_to_policy_components(move, self.mcts.root.board)
                    flat_index = src_shared.utils.policy_components_to_flat_index(from_row, from_col, channel)
                    policy_vector[flat_index] = prob_per_move

                self.logger.info(f"Forced loss detected at the root. Selecting move that delays the loss the longest ({losing_child_for_parent.distance_to_mate} moves).")
            
            # Resign if below threshold
            elif (self.use_resignation and self.mcts.root.value_sum / self.mcts.root.visits < self.talbot_config['resignation_cutoff']):
                return None, policy_vector, simulation_count

           
            else:
                # 1. Select Eligible Moves
                eligible_moves = {
                    move: child for move, child in self.mcts.root.children.items()
                    if child.forced_outcome not in [0, 1] and child.visits > 0
                }
                
                if not eligible_moves:
                    eligible_moves = {m: c for m, c in self.mcts.root.children.items() if c.visits > 0}

                if eligible_moves:
                   
                    # 1. Dynamic Sigma
                    max_visits = max(child.visits for child in eligible_moves.values())
                    sigma = self.talbot_config['gumbel_sigma'] + max_visits

                    scores = []
                    moves = list(eligible_moves.keys())
                    
                    for move in moves:
                        child = eligible_moves[move]
                        
                        prior = max(child.prior_probability_from_parent, 1e-8)
                        logit = np.log(prior)
                        
                        if child.visits > 0:
                            raw_q = -child.value_sum / child.visits
                            q_norm = (raw_q + 1.0) / 2.0
                        else:
                            q_norm = 0.0

                        noise = getattr(child, 'gumbel_noise', 0.0)
                        
                        # Calculate Final Score
                        scores.append(logit + noise + (sigma * q_norm))

                    scores = np.array(scores)
                    scores -= np.max(scores)
                    exps = np.exp(scores)
                    softmax_probs = exps / np.sum(exps)

                    # Assign Policy Vector
                    for move, prob in zip(moves, softmax_probs):
                        from_row, from_col, channel = src_shared.utils.move_to_policy_components(move, self.mcts.root.board)
                        flat_index = src_shared.utils.policy_components_to_flat_index(from_row, from_col, channel)
                        policy_vector[flat_index] = prob
                    
                    # Select Move
                    best_move = moves[np.argmax(scores)]                    
                    self.logger.info(f"Generated Q-based policy for {len(moves)} eligible moves.")
                    
                               
            move_end_time = time.time()
            total_move_time = move_end_time - move_start_time
            simulation_speed = (simulation_count / total_move_time) if total_move_time > 0 else 0

            self.logger.info(f"Total move time: {total_move_time:.4f}, with {simulation_speed:.4f} simulations per second")

            return best_move, policy_vector, simulation_count
    
    def reset_for_new_game(self):
        """
        Resets the player's state for a new game.
        """
        self.logger.debug(f"Resetting state for a new game.")
        self.mcts = None
        self.use_resignation = self.talbot_config['training'] and random.random() < self.talbot_config['resignation_probability']