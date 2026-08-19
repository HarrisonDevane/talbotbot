// =============================================================================
// gumbel_action_selector.cpp
//
// Rule D (temperature + deterministic) and _best_q lifted verbatim from
// the original action_selector.cpp. No logic changes.
// =============================================================================

#include "gumbel_action_selector.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>

GumbelActionSelector::GumbelActionSelector(
    std::string name, int worker_id, Config cfg, Logger& logger
) : ActionSelectorBase(
        std::move(name), worker_id,
        static_cast<SharedConfig>(cfg),   // slice off gumbel-only fields
        logger),
    temperature_q_decay_(cfg.temperature_q_decay)
{}

// -----------------------------------------------------------------------------
// _best_q: preserve the historical top-node-by-(visits, gumbel_score) formula.
// Sorts in place -- the original code did the same, and Rule D deterministic
// re-sorts (idempotent) so nothing downstream is affected.
// -----------------------------------------------------------------------------
double GumbelActionSelector::_best_q(std::vector<MCTSNode*>& non_forced_visited) {
    if (non_forced_visited.empty()) return -2.0;

    std::sort(non_forced_visited.begin(), non_forced_visited.end(),
        [](MCTSNode* a, MCTSNode* b) {
            if (a->visits != b->visits) return a->visits > b->visits;
            return a->gumbel_score > b->gumbel_score;
        });

    MCTSNode* m1 = non_forced_visited[0];
    MCTSNode* m2 = (non_forced_visited.size() > 1) ? non_forced_visited[1] : m1;
    MCTSNode* top_node = (m1->gumbel_score > m2->gumbel_score) ? m1 : m2;
    return -top_node->expected_value(shared_cfg.contempt);
}

// -----------------------------------------------------------------------------
// Rule D. Lifted verbatim from the original ActionSelector::select_move.
// -----------------------------------------------------------------------------
chess::Move GumbelActionSelector::_select_from_visited(
    const std::vector<MCTSNode*>& non_forced_visited_in,
    int ply_count)
{
    // Local mutable copy so the mid-branch re-sort in the deterministic path
    // works exactly as before, without touching the caller's vector.
    std::vector<MCTSNode*> non_forced_visited = non_forced_visited_in;

    if (ply_count <= shared_cfg.temperature_ply_cutoff) {
        // q̃(a) = Q(a) − σ(a)/√visits(a),  σ² from search-averaged WDL
        // weight(a) = visits(a) * exp(-q_drop(a) / temperature)
        double temp = temperature_q_decay_;
        size_t n = non_forced_visited.size();

        std::vector<double> q_tilde(n);
        double best_q_tilde = -2.0;
        for (size_t i = 0; i < n; ++i) {
            MCTSNode* c = non_forced_visited[i];
            double q = -c->expected_value(shared_cfg.contempt);

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
                double q  = -c->expected_value(shared_cfg.contempt);
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
        return non_forced_visited[d(rng)]->move;
    }

    // Deterministic: pick the gumbel winner
    std::sort(non_forced_visited.begin(), non_forced_visited.end(),
        [](MCTSNode* a, MCTSNode* b) {
            if (a->visits != b->visits) return a->visits > b->visits;
            return a->gumbel_score > b->gumbel_score;
        });
    MCTSNode* m1 = non_forced_visited[0];
    MCTSNode* m2 = (non_forced_visited.size() > 1) ? non_forced_visited[1] : m1;
    return (m1->gumbel_score > m2->gumbel_score) ? m1->move : m2->move;
}