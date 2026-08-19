// =============================================================================
// puct_action_selector.cpp
//
// Rule D for PUCT: visits^(1/tau) sampling below the ply cutoff,
// argmax-by-visits (tie-break by Q) above it.
// =============================================================================

#include "puct_action_selector.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

PuctActionSelector::PuctActionSelector(
    std::string name, int worker_id, Config cfg, Logger& logger
) : ActionSelectorBase(
        std::move(name), worker_id,
        static_cast<SharedConfig>(cfg),   // slice off puct-only fields
        logger),
    temperature_visits_(cfg.temperature_visits)
{}

chess::Move PuctActionSelector::_select_from_visited(
    const std::vector<MCTSNode*>& non_forced_visited,
    int ply_count)
{
    size_t n = non_forced_visited.size();

    // ---- Sampling branch: N(a)^(1/tau) ----
    if (ply_count <= shared_cfg.temperature_ply_cutoff) {
        // temperature_visits <= 0 is treated as "greedy" -- fall through to
        // the deterministic branch. Also guard against division-by-zero.
        if (temperature_visits_ > 1e-9) {
            const double inv_tau = 1.0 / temperature_visits_;

            std::vector<double> weights(n);
            double total = 0.0;
            for (size_t i = 0; i < n; ++i) {
                weights[i] = std::pow(static_cast<double>(non_forced_visited[i]->visits), inv_tau);
                total += weights[i];
            }

            if (logger.get_level() <= 20) {
                logger.log("INFO", "");
                logger.log("INFO", "--- PUCT Temperature Sampling (Ply " + std::to_string(ply_count) + ") ---");

                std::stringstream rss;
                rss << "Tau=" << std::fixed << std::setprecision(3) << temperature_visits_
                    << " | Moves=" << n;
                logger.log("INFO", rss.str());

                char table_header[256];
                snprintf(table_header, sizeof(table_header),
                    "%-8s %8s %8s %10s %6s",
                    "Move", "Visits", "Q", "Weight", "P%");
                logger.log("INFO", table_header);
                logger.log("INFO", std::string(46, '-'));

                std::vector<size_t> order(n);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    return non_forced_visited[a]->visits > non_forced_visited[b]->visits;
                });

                for (size_t idx : order) {
                    MCTSNode* c = non_forced_visited[idx];
                    double q   = -c->expected_value(shared_cfg.contempt);
                    double pct = (weights[idx] / total) * 100.0;
                    char line[256];
                    snprintf(line, sizeof(line),
                        "%-8s %8d %8.4f %10.4f %6.1f",
                        chess::uci::moveToUci(c->move).c_str(),
                        c->visits, q, weights[idx], pct);
                    logger.log("INFO", line);
                }
                logger.log("INFO", std::string(46, '-'));
                logger.log("INFO", "");
            }

            std::discrete_distribution<> d(weights.begin(), weights.end());
            return non_forced_visited[d(rng)]->move;
        }
        // else: fall through to deterministic
    }

    // ---- Deterministic: argmax N, tie-break by Q ----
    MCTSNode* best = non_forced_visited[0];
    double    best_q = -best->expected_value(shared_cfg.contempt);
    for (size_t i = 1; i < n; ++i) {
        MCTSNode* c = non_forced_visited[i];
        if (c->visits > best->visits) {
            best = c;
            best_q = -c->expected_value(shared_cfg.contempt);
        } else if (c->visits == best->visits) {
            double q = -c->expected_value(shared_cfg.contempt);
            if (q > best_q) { best = c; best_q = q; }
        }
    }
    return best->move;
}