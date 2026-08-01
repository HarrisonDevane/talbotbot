#include "action_selector.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>

ActionSelector::ActionSelector(
    std::string name, int worker_id, ActionSelectorConfig config, Logger& logger
) : name(name), worker_id(worker_id), config(config), logger(logger) {
    std::random_device rd;
    auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(rd() ^ worker_id ^ time_seed);
    reset_for_new_game();
}

void ActionSelector::reset_for_new_game() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    use_resignation = dist(rng) < config.resignation_probability;
    logger.log("DEBUG", "Agent state reset. Resignation allowed: " + std::string(use_resignation ? "True" : "False"));
}

SelectionResult ActionSelector::select_move(MCTSNode* root, int ply_count) {
    SelectionResult result;
    
    int num_children = root->num_children;
    if (num_children == 0) return result;

    std::vector<MCTSNode*> all_children;
    for(int i = 0; i < num_children; ++i) {
        all_children.push_back(root->first_child + i);
    }

    std::vector<MCTSNode*> winning_nodes, losing_nodes, draw_nodes, non_forced_visited;
    for (MCTSNode* child : all_children) {
        if (child->forced_outcome.has_value()) {
            if (child->forced_outcome.value() == -1) winning_nodes.push_back(child);
            else if (child->forced_outcome.value() == 1) losing_nodes.push_back(child);
            else draw_nodes.push_back(child);
        } else {
            // Check if the opponent's best response to this move is a forced draw.
            // If so, this move's true value is a draw regardless of its Q.
            bool is_practical_draw = false;
            if (child->expanded && child->num_children > 0) {
                MCTSNode* best_grandchild = nullptr;
                double best_gq = -2.0;
                for (int i = 0; i < child->num_children; ++i) {
                    MCTSNode* gc = child->first_child + i;
                    // Skip grandchildren that are proven wins for us (opponent avoids them)
                    if (gc->forced_outcome.has_value() && gc->forced_outcome.value() == -1) continue;
                    if (gc->visits == 0) continue;
                    double gq = -gc->expected_value(config.contempt);  // opponent's perspective
                    if (gq > best_gq) { best_gq = gq; best_grandchild = gc; }
                }
                if (best_grandchild != nullptr &&
                    best_grandchild->forced_outcome.has_value() &&
                    best_grandchild->forced_outcome.value() == 0) {
                    is_practical_draw = true;
                }
            }

            if (is_practical_draw) {
                draw_nodes.push_back(child);
                logger.log("INFO", "Practical draw detected: " + chess::uci::moveToUci(child->move)
                    + " (opponent's best response is a forced draw)");
            } else if (child->visits > 0) {
                non_forced_visited.push_back(child);
            }
        }
    }

    MCTSNode* top_node;
    double best_q = -2.0;

    // Find best Q of child nodes
    if (!non_forced_visited.empty()) {
        std::sort(non_forced_visited.begin(), non_forced_visited.end(), [](MCTSNode* a, MCTSNode* b) {
            if (a->visits != b->visits) return a->visits > b->visits;
            return a->gumbel_score > b->gumbel_score;
        });
        MCTSNode* m1 = non_forced_visited[0];
        MCTSNode* m2 = (non_forced_visited.size() > 1) ? non_forced_visited[1] : m1;
        top_node = (m1->gumbel_score > m2->gumbel_score) ? m1 : m2;
        best_q = -top_node->calculate_v_mix(config.contempt);
    }

    // Rule A: Win
    if (!winning_nodes.empty()) {
        int min_dtm = 999999;
        for (MCTSNode* c : winning_nodes) if (c->distance_to_mate.value() < min_dtm) min_dtm = c->distance_to_mate.value();
        std::vector<chess::Move> best_moves;
        for (MCTSNode* c : winning_nodes) if (c->distance_to_mate.value() == min_dtm) best_moves.push_back(c->move);
        
        std::uniform_int_distribution<> dist(0, best_moves.size() - 1);
        result.best_move = best_moves[dist(rng)];

    // Rule B: Draw
    } else if (!draw_nodes.empty() && best_q <= config.draw_cutoff) {
        std::uniform_int_distribution<> dist(0, draw_nodes.size() - 1);
        result.best_move = draw_nodes[dist(rng)]->move;

    // Rule C: Resign if below threshold
    } else if (use_resignation && !non_forced_visited.empty() && best_q < config.resignation_cutoff) {
        // Check for !non_forced_visited.empty() means if a forced loss is found, it will be played out to mate
        logger.log("INFO", "Best Value (" + std::to_string(best_q) + ") is below cutoff. Triggering Resignation.");
        result.resigned = true;
        result.best_move = chess::Move::NO_MOVE;

    // Rule D: Temperature / Deterministic
    } else if (!non_forced_visited.empty()) {
        if (ply_count <= config.temperature_ply_cutoff) {
            // q̃(a) = Q(a) − σ(a)/√visits(a),  σ² from search-averaged WDL
            // weight(a) = visits(a) * exp(-q_drop(a) / temperature)
            double temp = config.temperature_q_decay;
            size_t n = non_forced_visited.size();

            std::vector<double> q_tilde(n);
            double best_q_tilde = -2.0;
            for (size_t i = 0; i < n; ++i) {
                MCTSNode* c = non_forced_visited[i];
                double q = -c->calculate_v_mix(config.contempt);

                // Search-averaged WDL (c->visits > 0 guaranteed by upstream filter)
                double pw = c->w_sum / c->visits;
                double pl = c->l_sum / c->visits;

                // Var[v], v ∈ {-1, 0, +1}: E[v] = pw − pl, E[v²] = pw + pl
                // Symmetric in w/l, so the child-perspective flip needs no sign handling
                double ev     = pw - pl;
                double sigma2 = (pw + pl) - ev * ev;
                double sigma  = std::sqrt(std::max(sigma2, 0.0));

                q_tilde[i] = q - sigma / std::sqrt(static_cast<double>(c->visits));
                if (q_tilde[i] > best_q_tilde) best_q_tilde = q_tilde[i];
            }

            std::vector<double> weights(n);
            double total_weight = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double q_drop = best_q_tilde - q_tilde[i];  // >= 0 by construction
                weights[i] = non_forced_visited[i]->visits * std::exp(-q_drop / temp);
                total_weight += weights[i];
            }

            // Log sampling distribution
            if (logger.get_level() <= 20) {
                logger.log("INFO", "");
                logger.log("INFO", "--- Temperature Sampling (Ply " + std::to_string(ply_count) + ") ---");

                std::stringstream rss;
                rss << "Temp=" << std::fixed << std::setprecision(3) << temp
                    << " | Moves=" << n << " | BestQt=" << std::setprecision(4) << best_q_tilde;
                logger.log("INFO", rss.str());

                char table_header[256];
                snprintf(table_header, sizeof(table_header),
                    "%-8s %8s %8s %8s %8s %8s %8s %8s",
                    "Move", "Visits", "Q", "Sigma", "Qt", "Qdrop", "Weight", "P%");
                logger.log("INFO", table_header);
                logger.log("INFO", std::string(72, '-'));

                // Sort display order by visits without disturbing the sampling arrays
                std::vector<size_t> order(n);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    return non_forced_visited[a]->visits > non_forced_visited[b]->visits;
                });

                for (size_t idx : order) {
                    MCTSNode* c = non_forced_visited[idx];
                    double q  = -c->calculate_v_mix(config.contempt);
                    double pw = c->w_sum / c->visits;
                    double pl = c->l_sum / c->visits;
                    double ev = pw - pl;
                    double sigma = std::sqrt(std::max((pw + pl) - ev * ev, 0.0));
                    double q_drop = best_q_tilde - q_tilde[idx];
                    double pct = (weights[idx] / total_weight) * 100.0;

                    char line[256];
                    snprintf(line, sizeof(line),
                        "%-8s %8d %8.4f %8.4f %8.4f %8.4f %8.4f %8.1f",
                        chess::uci::moveToUci(c->move).c_str(), c->visits,
                        q, sigma, q_tilde[idx], q_drop, weights[idx], pct);
                    logger.log("INFO", line);
                }

                logger.log("INFO", std::string(72, '-'));
                logger.log("INFO", "");
            }

            std::discrete_distribution<> d(weights.begin(), weights.end());
            result.best_move = non_forced_visited[d(rng)]->move;
        } else {
            // Deterministic: pick the gumbel winner
            std::sort(non_forced_visited.begin(), non_forced_visited.end(), [](MCTSNode* a, MCTSNode* b) {
                if (a->visits != b->visits) return a->visits > b->visits;
                return a->gumbel_score > b->gumbel_score;
            });
            MCTSNode* m1 = non_forced_visited[0];
            MCTSNode* m2 = (non_forced_visited.size() > 1) ? non_forced_visited[1] : m1;
            MCTSNode* gumbel_winner = (m1->gumbel_score > m2->gumbel_score) ? m1 : m2;

            // MLH tiebreak (static cutoff)
            MCTSNode* shortest = nullptr;
            double best_mlh = 0.0;
            for (MCTSNode* c : non_forced_visited) {
                double cq = -c->calculate_v_mix(config.contempt);
                if (cq < config.mlh_tiebreak_cutoff) continue;
                double c_mlh = c->mlh_sum / c->visits;
                if (shortest == nullptr || c_mlh < best_mlh) { best_mlh = c_mlh; shortest = c; }
            }
            if (shortest != nullptr) {
                if (shortest != gumbel_winner) {
                    logger.log("INFO", "MLH tiebreak: " + chess::uci::moveToUci(shortest->move)
                        + " over " + chess::uci::moveToUci(gumbel_winner->move)
                        + " (mean ML " + std::to_string(best_mlh) + " norm)");
                }
                result.best_move = shortest->move;
            } else {
                result.best_move = gumbel_winner->move;
            }
        }
        
    // Rule E: Delay Mate
    } else {
        // Delay mate
        if (!losing_nodes.empty()) {
            MCTSNode* best_delay = losing_nodes[0];
            for (MCTSNode* c : losing_nodes) {
                if (c->distance_to_mate.value() > best_delay->distance_to_mate.value()) best_delay = c;
            }
            result.best_move = best_delay->move;
        // Should never be reached
        } else {
            std::uniform_int_distribution<> dist(0, num_children - 1);
            result.best_move = all_children[dist(rng)]->move;
        }
    }

    return result;
}