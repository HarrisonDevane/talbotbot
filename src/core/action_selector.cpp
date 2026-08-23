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

SelectionResult ActionSelector::select_move(MCTSNode* root, int ply_count, MCTSEngine* engine) {
    SelectionResult result;
    
    int num_children = root->num_children;
    if (num_children == 0) return result;

    // Under the edge/node split, the movable+priored thing is MCTSEdge;
    // MCTSNode (accessed via edge->child) is nullable for unvisited edges.
    std::vector<MCTSEdge*> all_edges;
    all_edges.reserve(num_children);
    for(int i = 0; i < num_children; ++i) {
        all_edges.push_back(root->first_edge + i);
    }

    std::vector<MCTSEdge*> winning_edges, losing_edges, draw_edges, non_forced_visited;
    for (MCTSEdge* edge : all_edges) {
        MCTSNode* child = edge->child;

        // Unmaterialised child -> was never selected during search. Treated
        // the same as a materialised-but-visits==0 child under the old scheme:
        // it can't be a forced outcome, it can't be visited. Falls off the
        // decision set entirely.
        if (child == nullptr) continue;

        if (child->has_forced_outcome()) {
            if      (child->forced_outcome == -1) winning_edges.push_back(edge);
            else if (child->forced_outcome ==  1) losing_edges.push_back(edge);
            else                                  draw_edges.push_back(edge);
        } else {
            // Practical-draw detection: if the opponent's best reply is a
            // forced draw, this move's true value is a draw regardless of
            // its Q. Walk the grandchild edges; skip unmaterialised ones
            // (equivalent to old "visits == 0" skip since a nullptr grandchild
            // has never been searched).
            bool is_practical_draw = false;
            if (child->is_expanded() && child->num_children > 0) {
                MCTSNode* best_grandchild = nullptr;
                double best_gq = -2.0;
                for (int i = 0; i < child->num_children; ++i) {
                    MCTSEdge* ge = child->first_edge + i;
                    MCTSNode* gc = ge->child;
                    if (gc == nullptr) continue;
                    // Skip grandchildren that are proven wins for us (opponent avoids them)
                    if (gc->has_forced_outcome() && gc->forced_outcome == -1) continue;
                    if (gc->visits == 0) continue;
                    double gq = -gc->expected_value(config.contempt);  // opponent's perspective
                    if (gq > best_gq) { best_gq = gq; best_grandchild = gc; }
                }
                if (best_grandchild != nullptr &&
                    best_grandchild->has_forced_outcome() &&
                    best_grandchild->forced_outcome == 0) {
                    is_practical_draw = true;
                }
            }

            if (is_practical_draw) {
                draw_edges.push_back(edge);
                logger.log("INFO", "Practical draw detected: " + chess::uci::moveToUci(edge->move)
                    + " (opponent's best response is a forced draw)");
            } else if (child->visits > 0) {
                non_forced_visited.push_back(edge);
            }
        }
    }

    MCTSEdge* top_edge = nullptr;
    double best_q = -2.0;

    // gumbel_score cache was dropped from MCTSNode -- recompute inline via
    // MCTSEdge::calculate_gumbel_score with noise from engine->root_gumbel_noise.
    // max_visits/v_mix computed once over the candidate set.
    auto gscore = [&](MCTSEdge* e, double mv, double vm) -> double {
        int idx = static_cast<int>(e - root->first_edge);
        double noise = (idx >= 0 && idx < (int)engine->root_gumbel_noise.size())
                     ? engine->root_gumbel_noise[idx] : 0.0;
        return e->calculate_gumbel_score(
            config.contempt, engine->gumbel_c_visit, engine->gumbel_c_scale,
            mv, vm, noise);
    };

    if (!non_forced_visited.empty()) {
        double mv = 0.0;
        for (MCTSEdge* e : non_forced_visited) if (e->child->visits > mv) mv = e->child->visits;
        double vm = root->calculate_v_mix(config.contempt);
        std::sort(non_forced_visited.begin(), non_forced_visited.end(),
                  [&](MCTSEdge* a, MCTSEdge* b) {
            if (a->child->visits != b->child->visits) return a->child->visits > b->child->visits;
            return gscore(a, mv, vm) > gscore(b, mv, vm);
        });
        MCTSEdge* e1 = non_forced_visited[0];
        MCTSEdge* e2 = (non_forced_visited.size() > 1) ? non_forced_visited[1] : e1;
        top_edge = (gscore(e1, mv, vm) > gscore(e2, mv, vm)) ? e1 : e2;
        best_q = -top_edge->child->expected_value(config.contempt);
    }

    // Rule A: Win
    if (!winning_edges.empty()) {
        int min_dtm = 999999;
        for (MCTSEdge* e : winning_edges) if (e->child->distance_to_mate < min_dtm) min_dtm = e->child->distance_to_mate;
        std::vector<chess::Move> best_moves;
        for (MCTSEdge* e : winning_edges) if (e->child->distance_to_mate == min_dtm) best_moves.push_back(e->move);
        
        std::uniform_int_distribution<> dist(0, (int)best_moves.size() - 1);
        result.best_move = best_moves[dist(rng)];

    // Rule B: Draw
    } else if (!draw_edges.empty() && best_q <= config.draw_cutoff) {
        std::uniform_int_distribution<> dist(0, (int)draw_edges.size() - 1);
        result.best_move = draw_edges[dist(rng)]->move;

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
                MCTSNode* c = non_forced_visited[i]->child;
                double q = -c->expected_value(config.contempt);

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
                weights[i] = non_forced_visited[i]->child->visits * std::exp(-q_drop / temp);
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
                    return non_forced_visited[a]->child->visits > non_forced_visited[b]->child->visits;
                });

                for (size_t idx : order) {
                    MCTSEdge* e = non_forced_visited[idx];
                    MCTSNode* c = e->child;
                    double q  = -c->expected_value(config.contempt);
                    double pw = c->w_sum / c->visits;
                    double pl = c->l_sum / c->visits;
                    double ev = pw - pl;
                    double sigma = std::sqrt(std::max((pw + pl) - ev * ev, 0.0));
                    double q_drop = best_q_tilde - q_tilde[idx];
                    double pct = (weights[idx] / total_weight) * 100.0;

                    char line[256];
                    snprintf(line, sizeof(line),
                        "%-8s %8d %8.4f %8.4f %8.4f %8.4f %8.4f %8.1f",
                        chess::uci::moveToUci(e->move).c_str(), c->visits,
                        q, sigma, q_tilde[idx], q_drop, weights[idx], pct);
                    logger.log("INFO", line);
                }

                logger.log("INFO", std::string(72, '-'));
                logger.log("INFO", "");
            }

            std::discrete_distribution<> d(weights.begin(), weights.end());
            result.best_move = non_forced_visited[d(rng)]->move;
        } else {
            // Deterministic: pick the gumbel winner. Same formula as the
            // dropped gumbel_score field, computed inline via gscore().
            double mv = 0.0;
            for (MCTSEdge* e : non_forced_visited) if (e->child->visits > mv) mv = e->child->visits;
            double vm = root->calculate_v_mix(config.contempt);
            std::sort(non_forced_visited.begin(), non_forced_visited.end(),
                      [&](MCTSEdge* a, MCTSEdge* b) {
                if (a->child->visits != b->child->visits) return a->child->visits > b->child->visits;
                return gscore(a, mv, vm) > gscore(b, mv, vm);
            });
            MCTSEdge* e1 = non_forced_visited[0];
            MCTSEdge* e2 = (non_forced_visited.size() > 1) ? non_forced_visited[1] : e1;
            result.best_move = (gscore(e1, mv, vm) > gscore(e2, mv, vm)) ? e1->move : e2->move;
        }
        
    // Rule E: Delay Mate
    } else {
        // Delay mate
        if (!losing_edges.empty()) {
            MCTSEdge* best_delay = losing_edges[0];
            for (MCTSEdge* e : losing_edges) {
                if (e->child->distance_to_mate > best_delay->child->distance_to_mate) best_delay = e;
            }
            result.best_move = best_delay->move;
        // Should never be reached
        } else {
            std::uniform_int_distribution<> dist(0, num_children - 1);
            result.best_move = all_edges[dist(rng)]->move;
        }
    }

    logger.log("INFO", "Move selected: " + chess::uci::moveToUci(result.best_move));
    return result;
}