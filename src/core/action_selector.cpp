#include "action_selector.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>
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

    std::vector<MCTSEdge*> all_edges;
    all_edges.reserve(num_children);
    for(int i = 0; i < num_children; ++i) {
        all_edges.push_back(root->first_edge + i);
    }

    std::vector<MCTSEdge*> winning_edges, losing_edges, draw_edges, mhd_edges, non_forced_visited;
    for (MCTSEdge* edge : all_edges) {
        MCTSNode* child = edge->child;

        if (child == nullptr) continue;

        if (child->has_forced_outcome()) {
            if      (child->forced_outcome == -1) winning_edges.push_back(edge);
            else if (child->forced_outcome ==  1) losing_edges.push_back(edge);
            else                                  draw_edges.push_back(edge);
        } else if (child->mover_has_draw()) {
            // Not proven, but PV / no-escape says this child's mover heads to
            // a draw. Treat as draw-like for selection (Rule C), independent
            // of visit count. Excluded from non_forced_visited so temperature
            // sampling can't pick it.
            mhd_edges.push_back(edge);
        } else if (child->visits > 0) {
            non_forced_visited.push_back(edge);
        }
    }

    // Log any child carrying mover_has_draw for post-mortem verification.
    // Distinguishes proven-draw (fo=0) sources from PV/no-escape sources.
    if (logger.get_level() <= 20) {
        for (MCTSEdge* edge : all_edges) {
            MCTSNode* c = edge->child;
            if (c != nullptr && c->mover_has_draw()) {
                const char* origin =
                    c->has_forced_outcome() && c->forced_outcome == 0 ? "proven-draw"
                  : c->has_forced_outcome()                            ? "forced-non-draw(!)"
                  :                                                      "PV/no-escape";
                char line[256];
                snprintf(line, sizeof(line),
                    "mover_has_draw set on %s (%s, visits=%d, cached_q=%.4f)",
                    chess::uci::moveToUci(edge->move).c_str(),
                    origin, c->visits, static_cast<double>(c->cached_q));
                logger.log("INFO", line);
            }
        }
    }

    MCTSEdge* top_edge = nullptr;
    double best_q = -2.0;

    // gscore reads gumbel_c_visit/gumbel_c_scale from the engine directly --
    // ActionSelectorConfig no longer carries those fields after the split.
    // (They lived on the old catch-all config but were never read via
    // config.*; the engine has always been the source of truth here.)
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

    // Rule B: Proven Draw
    } else if (!draw_edges.empty() && best_q <= config.draw_cutoff) {
        std::uniform_int_distribution<> dist(0, (int)draw_edges.size() - 1);
        result.best_move = draw_edges[dist(rng)]->move;

    // Rule C: Mover-Has-Draw (PV / no-escape draw preference, not proven)
    // Same trigger as Rule B: best non-forced option is at-or-below the draw
    // cutoff, so the flagged draw-heading line is preferable to letting the
    // NN's noisy positive estimate on non-forced children win. Checked after
    // Rule B so a proven draw always beats a preference-only draw.
    } else if (!mhd_edges.empty() && best_q <= config.draw_cutoff) {
        std::uniform_int_distribution<> dist(0, (int)mhd_edges.size() - 1);
        result.best_move = mhd_edges[dist(rng)]->move;

    // Rule D: Resign if below threshold
    } else if (use_resignation && !non_forced_visited.empty() && best_q < config.resignation_cutoff) {
        // Check for !non_forced_visited.empty() means if a forced loss is found, it will be played out to mate
        logger.log("INFO", "Best Value (" + std::to_string(best_q) + ") is below cutoff. Triggering Resignation.");
        result.resigned = true;
        result.best_move = chess::Move::NO_MOVE;

    // Rule E: Temperature / Deterministic
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
        
    // Rule F: Delay Mate
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

// -----------------------------------------------------------------------------
// Shared YAML loader for MctsConfig + ActionSelectorConfig.
//
// Single source of truth; called from both DataGenerator (training) and
// load_config in main_uci.cpp (inference). Adding a new MCTS or selection
// knob means editing this function only -- both binaries pick it up.
//
// contempt and draw_cutoff live in both structs by design; each YAML key is
// read exactly once here and mirrored into both, so drift between the two
// fields is impossible by construction.
//
// require_gumbel_m: see header. Training passes true (missing key -> throw);
// UCI passes false (missing key -> default 0, field is not consumed at
// inference anyway).
// -----------------------------------------------------------------------------
LoadedConfigs load_configs(const YAML::Node& mcts_n, const YAML::Node& sel_n,
                           bool require_gumbel_m) {
    LoadedConfigs out;

    // Shared fields: read once, mirror into both structs.
    const double contempt    = mcts_n["contempt"].as<double>();
    const double draw_cutoff = sel_n["draw_cutoff"].as<double>();

    MctsConfig& m = out.mcts;
    m.contempt              = contempt;
    m.draw_cutoff           = draw_cutoff;
    m.deficit_eps           = mcts_n["deficit_eps"].as<double>();
    m.policy_softmax_temp   = mcts_n["policy_softmax_temp"].as<double>();
    m.virtual_loss          = mcts_n["virtual_loss"].as<double>();
    m.gumbel_c_visit        = mcts_n["gumbel_c_visit"].as<double>();
    m.gumbel_c_scale        = mcts_n["gumbel_c_scale"].as<double>();
    m.gumbel_noise          = mcts_n["gumbel_noise"].as<double>();
    m.gumbel_search_depth   = mcts_n["gumbel_search_depth"].as<int>();
    m.batch_size_per_worker = mcts_n["worker_minibatch_size"].as<int>();

    // gumbel_m: required for training, optional for UCI. Explicit throw with
    // a named field beats yaml-cpp's generic "key not found" when training
    // yamls lose the key by accident.
    if (require_gumbel_m && !mcts_n["gumbel_m"]) {
        throw std::runtime_error(
            "load_configs: mcts.gumbel_m is required (training) but missing");
    }
    m.gumbel_m = mcts_n["gumbel_m"] ? mcts_n["gumbel_m"].as<double>() : 0.0;

    ActionSelectorConfig& s = out.selector;
    s.contempt                = contempt;
    s.draw_cutoff             = draw_cutoff;
    s.temperature_ply_cutoff  = sel_n["temperature_ply_cutoff"].as<int>();
    s.temperature_q_decay     = sel_n["temperature_q_decay"].as<double>();
    s.resignation_probability = sel_n["resignation_probability"].as<double>();
    s.resignation_cutoff      = sel_n["resignation_cutoff"].as<double>();

    return out;
}