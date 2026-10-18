// =============================================================================
// game_worker.cpp
// =============================================================================

#include "game_worker.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

// -----------------------------------------------------------------------------
// SearchAgent::think -- one move of search + selection.
//
//   use_halving == true  -> run_simulations_fixed(search_budget, gumbel_m)
//                           (sequential halving at the root; training search)
//   use_halving == false -> run_simulations_fixed_inference(search_budget)
//                           (deficit or PUCT selection, per MCTSEngine::use_puct)
//                           (deficit selection at every level; UCI search)
//
// select_move switches its final-move rule on engine.root_used_halving, which
// each search sets itself, so no extra plumbing is needed here.
//
// Tree reuse (reuse_active()): instead of reset(), walk the previous search's
// tree through the moves played since, one reset_reuse() per ply -- the same
// scheme main_uci uses. Any failure falls back to a full reset().
//
// The pool_sizing_cfg must have been set on the engine at construction time
// (host is responsible).
// -----------------------------------------------------------------------------
bool SearchAgent::try_reuse(const chess::Board& board,
                            const std::vector<chess::Board>& history,
                            const std::vector<chess::Move>& moves,
                            const PoolTargets& pt) {
    if (!has_tree || moves.size() <= searched_at ||
        moves.size() - searched_at > kMaxReuseWalk) {
        return false;
    }

    chess::Board walk_board = last_board;
    std::vector<chess::Board> walk_history = last_history;
    for (size_t i = searched_at; i < moves.size(); ++i) {
        const chess::Move mv = moves[i];
        const bool last = (i + 1 == moves.size());

        chess::Board next_board = walk_board;
        next_board.makeMove(mv);
        std::vector<chess::Board> next_history = walk_history;
        next_history.insert(next_history.begin(), walk_board);
        if (next_history.size() > kHistoryPlies) next_history.pop_back();

        // The final step must land exactly on the session's position.
        if (last && next_board.hash() != board.hash()) return false;

        // Final step uses the session's own board/history (authoritative).
        if (!engine.reset_reuse(last ? board : next_board,
                                last ? history : next_history,
                                mv, pt.node_target, pt.edge_target)) {
            return false;
        }
        walk_board   = next_board;
        walk_history = next_history;
    }
    return true;
}

SelectionResult SearchAgent::think(const chess::Board& board,
                                   const std::vector<chess::Board>& history,
                                   int ply,
                                   const std::vector<chess::Move>& moves) {
    const auto t_start = std::chrono::steady_clock::now();
    PoolTargets pt = engine.predict_pool_needs(search_budget);

    const bool reused = reuse_active() && try_reuse(board, history, moves, pt);
    if (!reused) {
        engine.reset(board, history, pt.node_target, pt.edge_target);
    }

    if (use_halving) {
        engine.run_simulations_fixed(search_budget, gumbel_m);
    } else {
        engine.run_simulations_fixed_inference(search_budget);
    }

    if (reuse_active()) {
        has_tree     = true;
        searched_at  = moves.size();
        last_board   = board;
        last_history = history;
    }
    SelectionResult result = selector.select_move(engine.root, ply, &engine);
    if (stats_out) {
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_start).count();
        write_stats(board, ply, result, ms);
    }
    return result;
}

// -----------------------------------------------------------------------------
// write_stats -- one JSON line per search:
//   {"player","search","game","ply","budget","sims","root_visits","ms","root_q",
//    "fen","played","pv_len","max_depth","avg_depth","moves":[{"m","p","n","q"}, ...]}
//   sims        : simulations run by THIS search (excludes reused visits)
//   root_visits : root visit count at the end (includes reused visits)
//   ms          : wall time of the whole think() (reuse/reset + search + select)
//   root_q      : root value from the side to move's perspective on [-1, 1]
//   max_depth   : deepest selection path this search (edges from the root)
//   avg_depth   : mean selection path length over this search's selections
//   p           : prior = softmax of raw_logit over all root moves
//                 (raw_logit already divided by policy_softmax_temp)
//   q           : child value from the mover's perspective on [-1, 1];
//                 null when the move has no visits
//   pv_len      : length of the most-visited line from the root
// -----------------------------------------------------------------------------
void SearchAgent::write_stats(const chess::Board& board, int ply,
                              const SelectionResult& r, double search_ms) const {
    MCTSNode* root = engine.root;
    if (!root || root->num_children == 0 || !root->first_edge) return;
    const int n = root->num_children;

    double max_logit = -1e300;
    for (int i = 0; i < n; ++i)
        max_logit = std::max(max_logit, static_cast<double>(root->first_edge[i].raw_logit));
    std::vector<double> prior(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        prior[i] = std::exp(static_cast<double>(root->first_edge[i].raw_logit) - max_logit);
        sum += prior[i];
    }
    for (double& p : prior) p = (sum > 0.0) ? p / sum : 0.0;

    int pv_len = 0;
    for (MCTSNode* node = root; node && node->num_children > 0 && node->first_edge;) {
        MCTSNode* best = nullptr;
        int best_v = 0;
        for (int i = 0; i < node->num_children; ++i) {
            MCTSNode* c = node->first_edge[i].child;
            if (c && c->visits > best_v) { best_v = c->visits; best = c; }
        }
        if (!best) break;
        ++pv_len;
        node = best;
    }

    const std::string& search = search_label;
    std::ostringstream o;
    o << std::setprecision(6);
    o << "{\"player\":\"" << player << "\",\"search\":\"" << search
      << "\",\"game\":" << game << ",\"ply\":" << ply
      << ",\"budget\":" << search_budget << ",\"sims\":" << engine.simulation_count
      << ",\"root_visits\":" << root->visits
      << ",\"ms\":" << search_ms
      << ",\"root_q\":" << static_cast<double>(root->cached_q)
      << ",\"fen\":\"" << board.getFen() << "\",\"played\":\""
      << ((r.resigned || r.best_move == chess::Move::NO_MOVE) ? std::string("0000")
                                                              : chess::uci::moveToUci(r.best_move))
      << "\",\"pv_len\":" << pv_len
      << ",\"max_depth\":" << engine.max_selection_depth
      << ",\"avg_depth\":" << (engine.selection_count > 0
                                   ? static_cast<double>(engine.sum_selection_depth) / engine.selection_count
                                   : 0.0)
      << ",\"moves\":[";
    for (int i = 0; i < n; ++i) {
        const MCTSEdge& e = root->first_edge[i];
        const MCTSNode* c = e.child;
        const int visits = c ? c->visits : 0;
        if (i) o << ',';
        o << "{\"m\":\"" << chess::uci::moveToUci(e.move) << "\",\"p\":" << prior[i]
          << ",\"n\":" << visits << ",\"q\":";
        if (visits > 0) o << -static_cast<double>(c->cached_q); else o << "null";
        o << '}';
    }
    o << "]}\n";
    (*stats_out) << o.str();
    stats_out->flush();
}

// -----------------------------------------------------------------------------
GameWorker::GameWorker(int worker_id, SearchAgent primary, Logger& logger)
    : worker_id(worker_id), primary(primary), logger(logger) {}

// -----------------------------------------------------------------------------
// run_one_game
//
// Termination ordering matters. We check is_over() at the TOP of the loop so
// that a game which ended because of the OPPONENT's last move (mate/draw the
// session detected after await_opponent_move advanced its board) is caught
// before we try to search a finished position.
//
// Ends we can reach:
//   1. is_over() true            -> board-detected end (mate / draw rules /
//                                   ply limit). Session fills `result`.
//   2. our resignation           -> ActionSelector returns resigned/NO_MOVE.
//                                   We score it and stop.
//   3. opponent non-move outcome -> await_opponent_move returns
//                                   game_continues=false (opponent resigned,
//                                   aborted, disconnected). Session fills it.
// -----------------------------------------------------------------------------
SessionResult GameWorker::run_one_game(GameSession& session) {
    primary.new_game();
    session.on_game_start();

    SessionResult result;
    result.reason = SessionEndReason::NOT_OVER;

    while (true) {
        // (1) Has the game already ended on the board?
        if (session.is_over(result)) break;

        if (session.our_turn()) {
            const chess::Board& pos = session.current_position();
            int ply = session.ply_count();

            SelectionResult choice = primary.think(pos, session.history(), ply,
                                                   session.moves_played());

            // (2) ActionSelector resigned on our behalf.
            if (choice.resigned || choice.best_move == chess::Move::NO_MOVE) {
                chess::Color us = pos.sideToMove();
                result.reason      = SessionEndReason::RESIGNATION;
                // We resign => the other side wins.
                result.white_value = (us == chess::Color::WHITE) ? -1.0 : 1.0;
                logger.log("INFO",
                    "[worker " + std::to_string(worker_id) +
                    "] resigned at ply " + std::to_string(ply));
                break;
            }

            logger.log("DEBUG",
                "[worker " + std::to_string(worker_id) + "] our move: " +
                chess::uci::moveToUci(choice.best_move));

            // submit_our_move advances the session's internal position.
            session.submit_our_move(choice.best_move);

            // Post-submit end-check: if our move ended the game (mate, draw,
            // ply limit), pick that up now rather than trying to continue.
            if (session.is_over(result)) break;
        }
        else {
            // Block until the opponent acts. For SelfPlaySession this runs the
            // opponent model's MCTS internally and advances the board.
            MoveOutcome outcome = session.await_opponent_move();

            if (!outcome.game_continues) {
                // (3) Opponent resigned / aborted / disconnected.
                result = outcome.result;
                logger.log("INFO",
                    "[worker " + std::to_string(worker_id) +
                    "] game ended via opponent path; reason=" +
                    std::to_string(static_cast<int>(result.reason)));
                break;
            }
            // Normal opponent move -- session already advanced its board.
        }
    }

    session.on_game_end(result);
    return result;
}