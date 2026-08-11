#include "mcts_engine.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include <sstream>
#include "board_utils.hpp"
#include "tbprobe.h"

#define NOW() std::chrono::high_resolution_clock::now()
#define ELAPSED(start, end) std::chrono::duration<double>(end - start).count()

MCTSEngine::MCTSEngine(
    int node_pool_capacity, int worker_batch_size, moodycamel::ConcurrentQueue<std::pair<int, int>>& inference_queue, 
    ThreadSafeQueue<std::vector<int>>& result_queue, int worker_id, double deficit_eps, double virtual_loss, double contempt,
    double draw_cutoff, double gumbel_c_visit, double gumbel_c_scale, double gumbel_noise,
    const chess::Board& board, const std::vector<chess::Board>& base_history, Logger& logger, 
    std::vector<torch::Tensor>& shared_input_buffer, std::vector<torch::Tensor>& shared_policy_buffer, std::vector<torch::Tensor>& shared_value_buffer,
    ThreadSafeQueue<int>& buffer_free_slots, std::atomic<int>* core_wait_count, int workers_per_core,
    bool use_tablebase
) : worker_batch_size(worker_batch_size), worker_id(worker_id), virtual_loss(virtual_loss), deficit_eps(deficit_eps), contempt(contempt),
    draw_cutoff(draw_cutoff), gumbel_c_visit(gumbel_c_visit), gumbel_c_scale(gumbel_c_scale), 
    gumbel_noise(gumbel_noise), root_board(board), base_history(base_history), 
    node_pool(node_pool_capacity), logger(logger), inference_queue(inference_queue), result_queue(result_queue), 
    buffer_free_slots(buffer_free_slots), shared_input_buffer(shared_input_buffer), 
    shared_policy_buffer(shared_policy_buffer), shared_value_buffer(shared_value_buffer),
    core_wait_count(core_wait_count), workers_per_core(workers_per_core), use_tablebase(use_tablebase)
{
    torch::set_num_threads(1);

    device = torch::kCUDA;
    policy_logits_dtype = torch::kFloat16;

    in_flight_nodes.resize(shared_input_buffer.size(), nullptr);
    root = node_pool.allocate();
    simulation_count = 0;
    inference_sent = 0;
    inference_received = 0;
    std::random_device rd;
    rng.seed(rd() ^ worker_id ^ std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

void MCTSEngine::reset(const chess::Board& board, const std::vector<chess::Board>& history) {
    if (!batch_buffer.empty()) {
        _submit_batch();
    }
    
    while (inference_received < inference_sent) {
        std::vector<int> completed_indices = result_queue.pop_wait();
        for (int buffer_index : completed_indices) {
            buffer_free_slots.push(buffer_index);
            inference_received++;
            in_flight_nodes[buffer_index] = nullptr;
        }
    }

    std::vector<int> stray;
    while (result_queue.try_pop(stray)) {
        for (int idx : stray) buffer_free_slots.push(idx);
    }

    std::fill(in_flight_nodes.begin(), in_flight_nodes.end(), nullptr);

    node_pool.reset();
    root_board = board;
    base_history = history;
    root = node_pool.allocate();
    
    simulation_count = 0;
    inference_sent = 0;
    inference_received = 0;
    batch_buffer.clear();

    time_selection = 0.0;
    time_expansion = 0.0;
    time_backpropagation = 0.0;
    time_retrieval = 0.0;
    time_queueing = 0.0;
    time_misc = 0.0;
}

MCTSNode* MCTSEngine::_select(MCTSNode* start_node, std::vector<MCTSNode*>& simulation_path) {
    auto start_time = NOW();
    MCTSNode* node = start_node;
    double exp_cache[256];
    double prior_cache[256];

    while (true) {
        if (node->num_children == 0 || !node->expanded || node->unavailable_for_selection || node->forced_outcome.has_value()) break;

        MCTSNode* best_child = nullptr;
        double best_deficit = -1e20;
        double max_visits = 0.0;
        double sum_visits = 0.0;
        int num_children = node->num_children;

        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->forced_outcome.has_value() || child->unavailable_for_selection) continue;
            if (child->visits > max_visits) max_visits = child->visits;
            sum_visits += child->visits;
        }

        double v_mix = node->calculate_v_mix(contempt);
        double max_score_logit = -1e20;
        double max_raw_logit   = -1e20;

        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->forced_outcome.has_value() || child->unavailable_for_selection) continue;
            double score = child->calculate_gumbel_score(contempt, gumbel_c_visit, gumbel_c_scale, max_visits, v_mix);
            if (score > max_score_logit) max_score_logit = score;
            if (child->raw_logit > max_raw_logit) max_raw_logit = child->raw_logit;
        }

        double sum_score_exp = 0.0;
        double sum_prior_exp = 0.0;
        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->forced_outcome.has_value() || child->unavailable_for_selection) {
                exp_cache[i] = 0.0;
                prior_cache[i] = 0.0;
                continue;
            }
            exp_cache[i] = std::exp(child->gumbel_score - max_score_logit);
            sum_score_exp += exp_cache[i];

            prior_cache[i] = std::exp(child->raw_logit - max_raw_logit);
            sum_prior_exp += prior_cache[i];
        }

        double inv_sum_visits = 1.0 / (1.0 + sum_visits);
        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (exp_cache[i] == 0.0) continue;

            // Deficit target: gumbel-score policy, floored with an eps share of
            // the raw prior so a low-prior move can never be permanently locked
            // out when sigma drives the gumbel softmax to one-hot.
            double pi_prime = (1.0 - deficit_eps) * (exp_cache[i] / sum_score_exp)
                            +        deficit_eps  * (prior_cache[i] / sum_prior_exp);

            double child_n_norm = child->visits * inv_sum_visits;
            double deficit = pi_prime - child_n_norm;

            if (deficit > best_deficit) {
                best_deficit = deficit;
                best_child = child;
            }
        }

        if (best_child == nullptr) break;

        root_board.makeMove(best_child->move);
        simulation_path.push_back(best_child);
        node = best_child;
    }
    time_selection += ELAPSED(start_time, NOW());
    return node;
}

void MCTSEngine::_mark_selected(MCTSNode* node) {
    MCTSNode* current_node = node;
    current_node->unavailable_for_selection = true;
    MCTSNode* parent = current_node->parent;

    while (parent != nullptr) {
        parent->num_available_children -= 1;
        if (parent->num_available_children > 0) break;
        parent->unavailable_for_selection = true;
        current_node = parent;
        parent = current_node->parent;
    }
}

void MCTSEngine::_unmark_selected(MCTSNode* node) {
    MCTSNode* current_node = node;
    current_node->unavailable_for_selection = false;
    MCTSNode* parent = current_node->parent;

    while (parent != nullptr) {
        parent->num_available_children += 1;
        if (parent->num_available_children == 1) {
            parent->unavailable_for_selection = false;
            current_node = parent;
            parent = current_node->parent;
        } else break;
    }
}

template <typename Predicate, typename WorkFn>
void MCTSEngine::_spin_wait(Predicate should_keep_waiting, WorkFn work_fn) {
    if (workers_per_core <= 1) {
        while (should_keep_waiting()) {
            work_fn();
        }
        return;
    }

    core_wait_count->fetch_add(1, std::memory_order_acquire);
    while (should_keep_waiting()) {
        work_fn();
        if (core_wait_count->load(std::memory_order_relaxed) == workers_per_core) {
            _mm_pause();
        } else {
            std::this_thread::yield();
        }
    }
    core_wait_count->fetch_sub(1, std::memory_order_release);
}

void MCTSEngine::_retrieve_inference(bool block) {
    auto start_time = NOW();
    std::vector<int> completed_indices;

    while (true) {
        if (block) {
            completed_indices = result_queue.pop_wait();
            block = false;
        } else {
            if (!result_queue.try_pop(completed_indices)) break;
        }

        // if (logger.get_level() <= 10) {
        //     logger.log("DEBUG", "Received " + std::to_string(completed_indices.size()) + " inferences from batcher.");
        // }

        for (int buffer_index : completed_indices) {
            MCTSNode* node = in_flight_nodes[buffer_index];
            in_flight_nodes[buffer_index] = nullptr;
            inference_received++;

            c10::Half* policy_ptr = shared_policy_buffer[buffer_index].data_ptr<c10::Half>();
            c10::Half* wdl_ptr = shared_value_buffer[buffer_index].data_ptr<c10::Half>();
            float p_win = (float)wdl_ptr[0];
            float p_draw = (float)wdl_ptr[1];
            float p_loss = (float)wdl_ptr[2];

            buffer_free_slots.push(buffer_index);

            if (node != nullptr) {
                if (!node->expanded) {
                    auto exp_start = NOW();
                    for (int i = 0; i < node->num_children; ++i) {
                        MCTSNode* child = node->first_child + i;
                        child->raw_logit = policy_ptr[child->policy_flat_index];
                    }
                    node->expanded = true;
                    time_expansion += ELAPSED(exp_start, NOW());
                }
                _backpropagate(node, p_win, p_draw, p_loss, false);
            }
        }
    }
    time_retrieval += ELAPSED(start_time, NOW());
}

void MCTSEngine::_submit_batch() {
    auto start_time = NOW();
    int b_size = batch_buffer.size();
    if (b_size == 0) return;

    // if (logger.get_level() <= 10) {
    //     logger.log("DEBUG", "Submitting batch of " + std::to_string(b_size) + " states to inference queue.");
    // }
    
    inference_queue.enqueue_bulk(batch_buffer.data(), b_size);
    
    inference_sent += b_size;
    batch_buffer.clear();

    time_queueing += ELAPSED(start_time, NOW());
}

void MCTSEngine::_handle_terminal_node(MCTSNode* leaf) {
    auto start_time = NOW();
    auto result = root_board.isGameOver(); 
    
    double w = 0.0, d = 0.0, l = 0.0;
    std::string term_type = "Draw";

    if (result.second == chess::GameResult::LOSE) {
        l = 1.0; 
        term_type = "Loss (Mate)";
    } else if (result.second == chess::GameResult::DRAW || root_board.isRepetition(2)) {
        d = 1.0;
    }

    // if (logger.get_level() <= 10) {
    //     logger.log("DEBUG", "Terminal node reached during search. Result: " + term_type);
    // }

    _mark_selected(leaf);
    time_expansion += ELAPSED(start_time, NOW());
    
    _backpropagate(leaf, w, d, l, true);
    simulation_count++;
}

// Probe Syzygy WDL for the position currently on root_board (which selection
// has advanced to the leaf). On a hit, resolve the leaf as a proven terminal
// using the SAME path as _handle_terminal_node -- _mark_selected then
// _backpropagate(is_terminal=true) -- so Q backprop, forced_outcome, and the
// minimax pass all behave exactly as they do for a mate/draw. Returns true iff
// the leaf was resolved here; false falls through to normal NN inference.
//
// Fathom WDL is reported from the side-to-move's perspective, which is the
// leaf's perspective -- the same convention _backpropagate expects -- so the
// values go in with NO manual flip. Cursed win / blessed loss collapse to draw
// (we respect the 50-move rule; there is no root-DTZ path to convert them).
//
// NOTE: the chess.hpp accessors below are the only library-version-dependent
// lines here. Verify these names against your copy of the chess-library:
//   pieces(PieceType, Color), Bitboard::getBits(), Bitboard::count(),
//   castlingRights()/CastlingRights::has(Color, Side), enpassantSq()/Square,
//   halfMoveClock(), sideToMove().
bool MCTSEngine::_try_tablebase(MCTSNode* leaf) {
    using chess::PieceType;
    using chess::Color;

    const chess::Bitboard wp = root_board.pieces(PieceType::PAWN,   Color::WHITE);
    const chess::Bitboard wn = root_board.pieces(PieceType::KNIGHT, Color::WHITE);
    const chess::Bitboard wb = root_board.pieces(PieceType::BISHOP, Color::WHITE);
    const chess::Bitboard wr = root_board.pieces(PieceType::ROOK,   Color::WHITE);
    const chess::Bitboard wq = root_board.pieces(PieceType::QUEEN,  Color::WHITE);
    const chess::Bitboard wk = root_board.pieces(PieceType::KING,   Color::WHITE);

    const chess::Bitboard bp = root_board.pieces(PieceType::PAWN,   Color::BLACK);
    const chess::Bitboard bn = root_board.pieces(PieceType::KNIGHT, Color::BLACK);
    const chess::Bitboard bb = root_board.pieces(PieceType::BISHOP, Color::BLACK);
    const chess::Bitboard br = root_board.pieces(PieceType::ROOK,   Color::BLACK);
    const chess::Bitboard bq = root_board.pieces(PieceType::QUEEN,  Color::BLACK);
    const chess::Bitboard bk = root_board.pieces(PieceType::KING,   Color::BLACK);

    const chess::Bitboard white_bb = wp | wn | wb | wr | wq | wk;
    const chess::Bitboard black_bb = bp | bn | bb | br | bq | bk;

    // Only probe positions the loaded tables actually cover.
    if ((white_bb | black_bb).count() > (int)TB_LARGEST) return false;

    // Fathom's WDL probe assumes no castling rights. Positions with <=N pieces
    // and castling are rare but legal (via FEN); guard and fall through.
    const auto& cr = root_board.castlingRights();
    const bool any_castle =
        cr.has(Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE)  ||
        cr.has(Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE) ||
        cr.has(Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE)  ||
        cr.has(Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE);
    if (any_castle) return false;

    const chess::Square ep_sq = root_board.enpassantSq();
    const unsigned ep = (ep_sq == chess::Square::NO_SQ) ? 0u
                                                        : (unsigned)ep_sq.index();
    const unsigned rule50        = (unsigned)root_board.halfMoveClock();
    const bool     white_to_move = (root_board.sideToMove() == Color::WHITE);

    const unsigned wdl = tb_probe_wdl(
        white_bb.getBits(), black_bb.getBits(),
        (wk | bk).getBits(), (wq | bq).getBits(), (wr | br).getBits(),
        (wb | bb).getBits(), (wn | bn).getBits(), (wp | bp).getBits(),
        rule50, /*castling=*/0u, ep, white_to_move);

    if (wdl == TB_RESULT_FAILED) return false;

    // Side-to-move perspective -> leaf perspective (what _backpropagate wants).
    double w = 0.0, d = 0.0, l = 0.0;
    switch (wdl) {
        case TB_WIN:          w = 1.0; break;
        case TB_LOSS:         l = 1.0; break;
        case TB_CURSED_WIN:                 // 50-move rule: treat as draw
        case TB_BLESSED_LOSS:               // 50-move rule: treat as draw
        case TB_DRAW:         d = 1.0; break;
        default:              return false; // unexpected value -> use the NN
    }

    _mark_selected(leaf);
    _backpropagate(leaf, w, d, l, true);
    simulation_count++;
    return true;
}

void MCTSEngine::_queue_leaf_for_inference(MCTSNode* leaf, const std::vector<MCTSNode*>& simulation_path) {
    auto start_time = NOW();
    int buffer_index;

    _spin_wait(
        [&]() { return !buffer_free_slots.try_pop(buffer_index); },
        [&]() { _retrieve_inference(false); if (!batch_buffer.empty()) _submit_batch(); }
    );

    in_flight_nodes[buffer_index] = leaf;
    _mark_selected(leaf);
    
    auto exp_start = NOW();
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, root_board);
    leaf->num_children = moves.size();
    leaf->num_available_children = leaf->num_children;

    if (leaf->num_children > 0) {
        leaf->first_child = node_pool.allocate(leaf, moves[0]);
        PolicyComponent pc = move_to_policy_components(moves[0], root_board);
        leaf->first_child->policy_flat_index = policy_components_to_flat_index(pc.row, pc.col, pc.channel);

        for (int i = 1; i < leaf->num_children; ++i) {
            MCTSNode* child = node_pool.allocate(leaf, moves[i]);
            pc = move_to_policy_components(moves[i], root_board);
            child->policy_flat_index = policy_components_to_flat_index(pc.row, pc.col, pc.channel);
        }
    }
    time_expansion += ELAPSED(exp_start, NOW());

    std::vector<chess::Board> combined_history;
    std::vector<chess::Move> unmade_moves;

    for (int i = (int)simulation_path.size() - 1; i >= 0 && combined_history.size() < 7; --i) {
        root_board.unmakeMove(simulation_path[i]->move);
        unmade_moves.push_back(simulation_path[i]->move);
        combined_history.push_back(root_board);
    }

    for (size_t i = 0; i < base_history.size() && combined_history.size() < 7; ++i) {
        combined_history.push_back(base_history[i]);
    }

    for (int i = (int)unmade_moves.size() - 1; i >= 0; --i) {
        root_board.makeMove(unmade_moves[i]);
    }

    c10::Half* destination_ptr = shared_input_buffer[buffer_index].data_ptr<c10::Half>();
    board_to_tensor(root_board, combined_history, destination_ptr);

    batch_buffer.push_back({worker_id, buffer_index});
    _virtual_loss(leaf, true);

    if (batch_buffer.size() >= (size_t)worker_batch_size) { 
        _submit_batch();
        _spin_wait(
            [&]() { return inference_sent > inference_received; },
            [&]() { _retrieve_inference(true); }
        );
    }

    time_misc += ELAPSED(start_time, NOW());
    simulation_count++;
}


bool MCTSEngine::_run_single_async_simulation(MCTSNode* start_node) {
    std::vector<MCTSNode*> simulation_path;
    root_board.makeMove(start_node->move);
    simulation_path.push_back(start_node);

    bool completed = false;
    
    int start_path_len = simulation_path.size();
    int loop_iterations = 0;
    int unavailable_continues = 0;
    int select_unavailable_continues = 0;

    while (true) {
        loop_iterations++;
        _retrieve_inference(false);
        if (batch_buffer.size() >= (size_t)worker_batch_size) { 
            _spin_wait(
                [&]() { return inference_sent > inference_received; },
                [&]() { _retrieve_inference(true); }
            );
            _submit_batch();
        }

        if (start_node->unavailable_for_selection || buffer_free_slots.empty()) {
            if (start_node->unavailable_for_selection) unavailable_continues++;
            if (!batch_buffer.empty()) _submit_batch();
            if (inference_received >= inference_sent) {
                // No-op exit: nothing was simulated and nothing is in flight for
                // this worker. completed stays false; the caller must NOT charge
                // budget for this call.
                logger.log("WARNING", "No-op sim exit: unavailable=" +
                           std::to_string(start_node->unavailable_for_selection) +
                           " slots_empty=" + std::to_string(buffer_free_slots.empty()) +
                           " unavailable_continues=" + std::to_string(unavailable_continues) +
                           " select_unavailable_continues=" + std::to_string(select_unavailable_continues));
                break;
            }
            _retrieve_inference(true);
            continue;
        }

        MCTSNode* leaf = _select(start_node, simulation_path);

        if (logger.get_level() <= 10)         {
            std::string path_str = "";
            std::string root_move = "";
            MCTSNode* curr = leaf;
            while (curr != nullptr && curr->move != chess::Move::NO_MOVE) {
                std::string uci = chess::uci::moveToUci(curr->move);
                path_str = uci + (path_str.empty() ? "" : " ") + path_str;
                root_move = uci;
                curr = curr->parent;
            }
            if (root_move == "e3h6") {
                logger.log("DEBUG", "Selected path: " + path_str);
            }
        }

        if (root_board.isGameOver().second != chess::GameResult::NONE || root_board.isRepetition(2)) {
            _handle_terminal_node(leaf);
            completed = true;
            break;
        }

        // Exact endgame via Syzygy WDL (UCI only; use_tablebase is false in
        // self-play/tournament). Runs only on non-terminal positions, so the
        // probe never sees mate/stalemate/50-move/repetition. On a hit the leaf
        // is resolved as a proven terminal -- no NN inference is queued.
        if (use_tablebase && _try_tablebase(leaf)) {
            completed = true;
            break;
        }

        if (leaf->expanded) {
            logger.log("WARNING", "_select returned an already-expanded interior node (" +
                       chess::uci::moveToUci(leaf->move) + "); skipping re-queue.");
            while (simulation_path.size() > 1) {
                root_board.unmakeMove(simulation_path.back()->move);
                simulation_path.pop_back();
            }
            if (!batch_buffer.empty()) _submit_batch();
            if (inference_received >= inference_sent) break;
            _retrieve_inference(true);
            continue;
        }

        if (start_node->unavailable_for_selection) {
            select_unavailable_continues++;
            while (simulation_path.size() > 1) {
                root_board.unmakeMove(simulation_path.back()->move);
                simulation_path.pop_back();
            }
            continue;
        }

        _queue_leaf_for_inference(leaf, simulation_path);
        completed = true;
        break;
    }

    while (!simulation_path.empty()) {
        root_board.unmakeMove(simulation_path.back()->move);
        simulation_path.pop_back();
    }
    return completed;
}

void MCTSEngine::_log_tournament_results(const std::vector<MCTSNode*>& candidates,
                                         const std::string& phase_name,
                                         int remaining_search_depth,
                                         int phase_budget,
                                         int sims_completed) {
    if (logger.get_level() > 20) return;

    double root_v_mix = root->calculate_v_mix(contempt);

    logger.log("INFO", "");
    logger.log("INFO", "--- " + phase_name + " ---");

    std::stringstream rss;
    rss << "Tree Stats: Root v_mix=" << std::fixed << std::setprecision(4) << root_v_mix;
    logger.log("INFO", rss.str());

    // Budget accounting: only print when the caller supplied it (defaults = -1).
    {
        int active = 0, forced = 0, total_visits = 0;
        for (MCTSNode* n : candidates) {
            if (n->forced_outcome.has_value()) forced++; else active++;
            total_visits += n->visits;
        }
        char bud[256];
        snprintf(bud, sizeof(bud),
            "Budget: remaining=%d phase_budget=%d sims_completed=%d | "
            "cands=%d (active=%d forced=%d) sum_visits=%d",
            remaining_search_depth, phase_budget, sims_completed,
            (int)candidates.size(), active, forced, total_visits);
        logger.log("INFO", bud);
    }

    char table_header[256];
    snprintf(table_header, sizeof(table_header),
        "%-8s %8s %8s %8s %8s %8s %8s %8s %8s %8s",
        "Move", "Logit", "Visits", "Win%", "Draw%", "Loss%", "Norm Q", "Score", "Outcome", "DTM");
    logger.log("INFO", table_header);
    logger.log("INFO", std::string(95, '-'));

    std::vector<MCTSNode*> sorted_cands = candidates;
    std::sort(sorted_cands.begin(), sorted_cands.end(), [](MCTSNode* a, MCTSNode* b) {
        if (a->visits != b->visits) return a->visits > b->visits;
        return a->gumbel_score > b->gumbel_score;
    });

    for (MCTSNode* node : sorted_cands) {
        char line[512];
        std::string outcome_str = node->forced_outcome.has_value() ? std::to_string(node->forced_outcome.value()) : "None";
        std::string dtm_str = node->distance_to_mate.has_value() ? std::to_string(node->distance_to_mate.value()) : "None";

        double w_pct = (node->visits > 0) ? (node->l_sum / node->visits) * 100.0 : node->raw_l * 100.0;
        double d_pct = (node->visits > 0) ? (node->d_sum / node->visits) * 100.0 : node->raw_d * 100.0;
        double l_pct = (node->visits > 0) ? (node->w_sum / node->visits) * 100.0 : node->raw_w * 100.0;

        double q_val = (node->visits > 0) ? -node->expected_value(contempt) : root_v_mix;
        double q_norm = (q_val + 1.0) / 2.0;

        snprintf(line, sizeof(line),
            "%-8s %8.4f %8d %8.1f %8.1f %8.1f %8.4f %8.4f %8s %8s",
            chess::uci::moveToUci(node->move).c_str(), node->raw_logit, node->visits,
            w_pct, d_pct, l_pct, q_norm, node->gumbel_score, outcome_str.c_str(), dtm_str.c_str());
        logger.log("INFO", line);
    }

    logger.log("INFO", std::string(95, '-'));
    logger.log("INFO", "");
}

// _log_node_by_path -- walk from root following a UCI move sequence, then dump
// the target node's RAW network value and its children.
//
// The KEY line is the target's raw WDL: that is what the value head actually
// returned for the target position (in the target's side-to-move perspective).
// If a catastrophically-losing position shows a high raw win there, the value
// head is blind to the loss -- the search arithmetic is downstream and innocent.
//
// PERSPECTIVE: the target's expected_value / raw value are in the TARGET's
// own mover perspective. Its children are the opponent-to-move, so we print
// each child's value negated back to the TARGET mover ("Q(target)") -- a high
// number means "good for whoever is to move at the target".
void MCTSEngine::_log_node_by_path(const std::vector<std::string>& uci_path, int top_n) {
    if (logger.get_level() > 20) return;

    MCTSNode* node = root;
    std::string walked;
    for (const std::string& uci : uci_path) {
        if (node == nullptr) break;
        MCTSNode* next = nullptr;
        for (int i = 0; i < node->num_children; ++i) {
            MCTSNode* c = node->first_child + i;
            if (chess::uci::moveToUci(c->move) == uci) { next = c; break; }
        }
        if (next == nullptr) {
            logger.log("INFO", "[path dump] move '" + uci + "' not found below '" + walked + "' (node not in tree)");
            return;
        }
        walked += (walked.empty() ? "" : " ") + uci;
        node = next;
    }
    if (node == nullptr) return;

    logger.log("INFO", "");
    logger.log("INFO", "=== Node-by-path dump: [" + walked + "] ===");

    // THE diagnostic line: what the value head returned for THIS position.
    char head[512];
    double tgt_q_own = (node->visits > 0) ? node->expected_value(contempt) : ((node->raw_w - node->raw_l) + contempt * node->raw_d);
    double tgt_vmix  = node->expanded ? node->calculate_v_mix(contempt) : 0.0;
    snprintf(head, sizeof(head),
        "TARGET raw network WDL (own mover persp): W=%.4f D=%.4f L=%.4f  -> raw_value(own)=%+.4f",
        node->raw_w, node->raw_d, node->raw_l, (node->raw_w - node->raw_l));
    logger.log("INFO", head);
    snprintf(head, sizeof(head),
        "TARGET visits=%d  expected_value(own)=%+.4f  v_mix(own)=%+.4f  logit=%.3f  expanded=%d  outcome=%s",
        node->visits, tgt_q_own, tgt_vmix, node->raw_logit, node->expanded ? 1 : 0,
        node->forced_outcome.has_value() ? std::to_string(node->forced_outcome.value()).c_str() : "None");
    logger.log("INFO", head);

    if (!node->expanded || node->num_children == 0) {
        logger.log("INFO", "  (target is a leaf / unexpanded -- no children)");
        logger.log("INFO", "=== end node-by-path dump ===");
        logger.log("INFO", "");
        return;
    }

    std::vector<MCTSNode*> kids;
    for (int i = 0; i < node->num_children; ++i) kids.push_back(node->first_child + i);
    std::sort(kids.begin(), kids.end(), [](MCTSNode* a, MCTSNode* b) {
        if (a->visits != b->visits) return a->visits > b->visits;
        return a->gumbel_score > b->gumbel_score;
    });
    int shown = (top_n > 0 && (int)kids.size() > top_n) ? top_n : (int)kids.size();

    char gh[256];
    snprintf(gh, sizeof(gh), "  %-8s %8s %8s %10s %10s %8s",
             "reply", "logit", "visits", "Q(target)", "rawV(child)", "outcome");
    logger.log("INFO", gh);

    for (int j = 0; j < shown; ++j) {
        MCTSNode* c = kids[j];
        // child is opponent-to-move; negate to express from target's mover view.
        double q_target = (c->visits > 0) ? -c->expected_value(contempt)
                                          : -((c->raw_w - c->raw_l) + contempt * c->raw_d);
        // child's own raw network value, in ITS mover perspective (opponent).
        double child_rawV = (c->raw_w - c->raw_l);
        std::string oc = c->forced_outcome.has_value() ? std::to_string(c->forced_outcome.value()) : "None";
        char line[512];
        snprintf(line, sizeof(line),
            "  %-8s %8.3f %8d %+10.4f %+10.4f %8s",
            chess::uci::moveToUci(c->move).c_str(), c->raw_logit, c->visits,
            q_target, child_rawV, oc.c_str());
        logger.log("INFO", line);
    }
    if (shown < (int)kids.size()) {
        char more[64];
        snprintf(more, sizeof(more), "  ... (%d more replies)", (int)kids.size() - shown);
        logger.log("INFO", more);
    }
    logger.log("INFO", "=== end node-by-path dump ===");
    logger.log("INFO", "");
}

// ---------------------------------------------------------------------------
// Shared sequential-halving building blocks. Extracted verbatim from the
// original run_simulations so run_simulations_fixed and run_simulations_timed
// reuse identical logic; only the orchestration loop differs between the two.
// ---------------------------------------------------------------------------

// Evaluate the root and drain the result so children/policy are populated.
void MCTSEngine::_expand_root() {
    _queue_leaf_for_inference(root, {});
    _submit_batch();
    while (inference_received < inference_sent) {
        _retrieve_inference(true);
    }
}

// Assign gumbel noise/scores to every root child, route terminals out, and
// build the active candidate set sorted+truncated to m. Returns m (0 => none).
int MCTSEngine::_build_candidates(int max_m, std::vector<MCTSNode*>& all_nodes,
                                  std::vector<MCTSNode*>& active_candidates) {
    all_nodes.clear();
    for (int i = 0; i < root->num_children; ++i) {
        all_nodes.push_back(root->first_child + i);
    }
    active_candidates.clear();

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (MCTSNode* child : all_nodes) {
        double u = dist(rng);
        child->gumbel_noise = -gumbel_noise * std::log(-std::log(u));
        child->gumbel_score = child->gumbel_noise + child->raw_logit;

        root_board.makeMove(child->move);
        if (root_board.isGameOver().second != chess::GameResult::NONE || root_board.isRepetition(2)) {
            _handle_terminal_node(child);
        } else {
            active_candidates.push_back(child);
        }
        root_board.unmakeMove(child->move);
    }

    int m = std::min(max_m, (int)active_candidates.size());
    if (m == 0) return 0;

    std::sort(active_candidates.begin(), active_candidates.end(), [](MCTSNode* a, MCTSNode* b) {
        return a->gumbel_score > b->gumbel_score;
    });
    active_candidates.resize(m);
    return m;
}

// Round 0: one simulation against each candidate, drain, log. Decrements the
// caller's remaining budget by the number of candidates touched.
void MCTSEngine::_run_round0(std::vector<MCTSNode*>& active_candidates, int& remaining_search_depth) {
    int ply_count = ((root_board.fullMoveNumber() - 1) * 2) + (root_board.sideToMove() == chess::Color::BLACK ? 2 : 1);

    // Calculate move and color 
    int current_move = (ply_count + 1) / 2;
    std::string color = (root_board.sideToMove() == chess::Color::WHITE) ? "WHITE" : "BLACK";

    // Print the header
    logger.log("INFO", "===============================================================================================");
    logger.log("INFO", " MOVE " + std::to_string(current_move) + " | PLY " + std::to_string(ply_count) + " | " + color);
    logger.log("INFO", "===============================================================================================");    


    for (MCTSNode* child : active_candidates) {
        remaining_search_depth -= 1;
        root_board.makeMove(child->move);
        if (root_board.isGameOver().second == chess::GameResult::NONE && !root_board.isRepetition(2)) {
            // Syzygy probe for root children. Round 0 queues candidates directly,
            // bypassing _run_single_async_simulation -- without this check, depth-1
            // leaves are never probed and TB-provable root moves get NN draw evals.
            // On a hit, _try_tablebase marks/backprops/counts the sim itself, so
            // it fully substitutes for the queued inference.
            if (!(use_tablebase && _try_tablebase(child))) {
                _queue_leaf_for_inference(child, {child});
            }
        }
        root_board.unmakeMove(child->move);
    }
    _submit_batch();
    while (inference_received < inference_sent) {
        _retrieve_inference(true);
    }
}

// Recompute gumbel scores for a node set against its current max-visit count.
// Used both per-phase (active set) and for final scoring (all root children).
void MCTSEngine::_rescore(std::vector<MCTSNode*>& nodes) {
    double max_visits = 1.0;
    for (MCTSNode* child : nodes) {
        if (child->visits > max_visits) max_visits = child->visits;
    }
    double root_v_mix = root->calculate_v_mix(contempt);
    for (MCTSNode* child : nodes) {
        child->calculate_gumbel_score(contempt, gumbel_c_visit, gumbel_c_scale, max_visits, root_v_mix);
    }
}

// Sequential-halving cut: drop proven-loss-for-us candidates, then keep the
// top half by gumbel score.
void MCTSEngine::_halve(std::vector<MCTSNode*>& active_candidates) {
    active_candidates.erase(
        std::remove_if(active_candidates.begin(), active_candidates.end(),
        [](MCTSNode* c) { return c->forced_outcome.has_value(); }),
        active_candidates.end()
    );
    if (active_candidates.size() > 1) {
        std::sort(active_candidates.begin(), active_candidates.end(), [](MCTSNode* a, MCTSNode* b) {
            return a->gumbel_score > b->gumbel_score;
        });
        int cutoff = (active_candidates.size() + 1) / 2;
        active_candidates.resize(cutoff);
    }
}

// Final flush: submit any pending batch and drain all in-flight inferences.
void MCTSEngine::_flush_inflight() {
    if (!batch_buffer.empty()) _submit_batch();
    while (inference_received < inference_sent) {
        _retrieve_inference(true);
    }
}

// EWMA on the ratio (overall speed over the window); nps_alpha_ weights recent
// moves. Searches that completed cleanly and ones cut by the deadline both feed
// in; if you later want cleaner samples, skip recording when the search aborted.
void MCTSEngine::_record_nps(int sims, double seconds) {
    if (sims <= 0 || seconds <= 0.0) return;
    const double inst = static_cast<double>(sims) / seconds;
    nps_ewma_ = (nps_ewma_ <= 0.0) ? inst : (nps_alpha_ * inst + (1.0 - nps_alpha_) * nps_ewma_);
}

// Self-play / fixed-budget path. Full sequential-halving schedule to completion.
int MCTSEngine::run_simulations_fixed(int search_depth, int max_m) {
    if (logger.get_level() <= 20) {
        logger.log("INFO", "Starting Sequential Halving MCTS. Budget: " + std::to_string(search_depth));
    }

    const auto wall_start = std::chrono::steady_clock::now();
    _expand_root();

    std::vector<MCTSNode*> all_nodes;
    std::vector<MCTSNode*> active_candidates;
    int m = _build_candidates(max_m, all_nodes, active_candidates);
    if (m == 0) return simulation_count;

    int remaining_search_depth = search_depth;
    bool did_round0 = false;
    int r0_spent = 0;
    int phase_idx = 0;

    while (active_candidates.size() > 1 && remaining_search_depth > 0) {
        int num_cands = active_candidates.size();

        if (!did_round0) {
            int before = remaining_search_depth;
            _run_round0(active_candidates, remaining_search_depth);
            r0_spent = before - remaining_search_depth;
            did_round0 = true;
            active_candidates.erase(
                std::remove_if(active_candidates.begin(), active_candidates.end(),
                    [](MCTSNode* c){ return c->forced_outcome.has_value(); }),
                active_candidates.end());
            num_cands = active_candidates.size();
            if (num_cands <= 1) break;
        }

        int phases_left = std::max(1, (int)std::ceil(std::log2((double)num_cands)));
        int current_phase_budget;
        if (phases_left <= 1) {
            current_phase_budget = remaining_search_depth;
        } else {
            int pool = remaining_search_depth + (phase_idx == 0 ? r0_spent : 0);
            current_phase_budget = pool / phases_left;
            if (phase_idx == 0) current_phase_budget -= r0_spent;
        }
        current_phase_budget = std::max(0, std::min(current_phase_budget, remaining_search_depth));

        int active_idx = 0;
        int no_progress_streak = 0;
        while (current_phase_budget > 0 && num_cands > 0) {
            MCTSNode* child = active_candidates[active_idx];

            if (child->forced_outcome.has_value()) {
                active_candidates.erase(active_candidates.begin() + active_idx);
                num_cands = active_candidates.size();
                if (num_cands == 0) break;
                if (active_idx >= num_cands) active_idx = 0;
                continue;
            }

            if (_run_single_async_simulation(child)) {
                remaining_search_depth -= 1;
                current_phase_budget -= 1;
                no_progress_streak = 0;
            } else {
                no_progress_streak += 1;
                if (no_progress_streak >= num_cands) {
                    logger.log("WARNING", "Phase stalled: all " + std::to_string(num_cands) +
                               " candidates returned no-op with nothing in flight. Ending phase early (budget left: " +
                               std::to_string(current_phase_budget) + ").");
                    break;
                }
            }

            active_idx++;
            if (active_idx >= num_cands) active_idx = 0;
        }

        while (inference_received < inference_sent) {
            _retrieve_inference(true);
        }

        _rescore(active_candidates);
        _log_tournament_results(active_candidates,
                        "Phase " + std::to_string(phase_idx) + " End",
                        remaining_search_depth, current_phase_budget, simulation_count);

        if (active_candidates.size() > 2) {
            _halve(active_candidates);
        } else {
            break;
        }
        phase_idx++;
    }

    _rescore(all_nodes);
    _log_tournament_results(all_nodes, "Final scores");

    if (logger.get_level() <= 20) {
        logger.log("INFO", "Simulations complete. Total: " + std::to_string(simulation_count));
        logger.log("INFO", "--- Gumbel Search (" + std::to_string(simulation_count) + " sims) Timings ---");

        char buffer[128];
        auto log_timer = [&](const char* label, double value) {
            snprintf(buffer, sizeof(buffer), "%-35s %.4f", label, value);
            logger.log("INFO", buffer);
        };

        log_timer("Selection time:", time_selection);
        log_timer("Queueing time:", time_queueing);
        log_timer("Retrieving time:", time_retrieval);
        log_timer("Expansion time:", time_expansion);
        log_timer("Backpropagation time:", time_backpropagation);
        log_timer("Forced waiting for inference time:", time_wait_for_inference);
    }

    _flush_inflight();
    _record_nps(simulation_count, std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
    return simulation_count;
}

// Clocked sibling of run_simulations_fixed. Same schedule via shared helpers,
// plus: (1) plan the budget from the NPS estimate; (2) stop at a phase boundary
// past the soft deadline; (3) abandon mid-phase at the hard deadline, ALWAYS
// draining in-flight inferences before reading the tree. Self-play never calls this.
int MCTSEngine::run_simulations_timed(int max_m,
                                      std::chrono::steady_clock::time_point soft_deadline,
                                      std::chrono::steady_clock::time_point hard_deadline) {
    const auto wall_start = std::chrono::steady_clock::now();

    double target_s = std::chrono::duration<double>(soft_deadline - wall_start).count();
    if (target_s < 0.0) target_s = 0.0;
    int search_depth = nps_ewma_ * target_s;

    if (logger.get_level() <= 20) {
        logger.log("INFO", "Starting Timed Sequential Halving MCTS. Planned budget: " +
                   std::to_string(search_depth) + " (nps~" + std::to_string((long long)nps_ewma_) + ")");
    }

    _expand_root();

    std::vector<MCTSNode*> all_nodes;
    std::vector<MCTSNode*> active_candidates;
    int m = _build_candidates(max_m, all_nodes, active_candidates);
    if (m == 0) {
        _record_nps(simulation_count, std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
        return simulation_count;
    }

    int remaining_search_depth = search_depth;
    bool did_round0 = false;
    bool aborted = false;
    int r0_spent = 0;
    int phase_idx = 0;

    while (active_candidates.size() > 1 && remaining_search_depth > 0) {
        // SOFT deadline: clean stop at a phase boundary (never before Round 0).
        if (did_round0 && std::chrono::steady_clock::now() >= soft_deadline) break;

        int num_cands = active_candidates.size();

        if (!did_round0) {
            int before = remaining_search_depth;
            _run_round0(active_candidates, remaining_search_depth);
            r0_spent = before - remaining_search_depth;
            did_round0 = true;
            active_candidates.erase(
                std::remove_if(active_candidates.begin(), active_candidates.end(),
                    [](MCTSNode* c){ return c->forced_outcome.has_value(); }),
                active_candidates.end());
            num_cands = active_candidates.size();
            if (num_cands <= 1) break;
        }

        int phases_left = std::max(1, (int)std::ceil(std::log2((double)num_cands)));
        int current_phase_budget;
        if (phases_left <= 1) {
            current_phase_budget = remaining_search_depth;
        } else {
            int pool = remaining_search_depth + (phase_idx == 0 ? r0_spent : 0);
            current_phase_budget = pool / phases_left;
            if (phase_idx == 0) current_phase_budget -= r0_spent;
        }
        current_phase_budget = std::max(0, std::min(current_phase_budget, remaining_search_depth));

        int active_idx = 0;
        int since_check = 0;
        int no_progress_streak = 0;
        while (current_phase_budget > 0 && num_cands > 0) {
            MCTSNode* child = active_candidates[active_idx];

            if (child->forced_outcome.has_value()) {
                active_candidates.erase(active_candidates.begin() + active_idx);
                num_cands = active_candidates.size();
                if (num_cands == 0) break;
                if (active_idx >= num_cands) active_idx = 0;
                continue;
            }

            if (_run_single_async_simulation(child)) {
                remaining_search_depth -= 1;
                current_phase_budget -= 1;
                no_progress_streak = 0;
            } else {
                no_progress_streak += 1;
                if (no_progress_streak >= num_cands) {
                    logger.log("WARNING", "Phase stalled: all " + std::to_string(num_cands) +
                               " candidates returned no-op with nothing in flight. Ending phase early (budget left: " +
                               std::to_string(current_phase_budget) + ").");
                    break;
                }
            }

            // HARD deadline: abandon mid-phase. Drain below still runs.
            if (++since_check >= 128) {
                since_check = 0;
                if (std::chrono::steady_clock::now() >= hard_deadline) { aborted = true; break; }
            }

            active_idx++;
            if (active_idx >= num_cands) active_idx = 0;
        }

        while (inference_received < inference_sent) {
            _retrieve_inference(true);
        }

        _rescore(active_candidates);
        _log_tournament_results(active_candidates, "Phase " + std::to_string(phase_idx) + " End");

        if (aborted) break;

        if (active_candidates.size() > 2) {
            _halve(active_candidates);
        } else {
            break;
        }
        phase_idx++;
    }

    _rescore(all_nodes);
    _log_tournament_results(all_nodes, "Final scores");
    _log_node_by_path({"e6g5"}, 20);

    _flush_inflight();
    _record_nps(simulation_count, std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
    return simulation_count;
}
void MCTSEngine::_backpropagate_minimax(MCTSNode* node) {
    if (node->num_children == 0) return;

    int best_win_dtm = 999999;
    int worst_loss_dtm = -1;
    
    bool has_winning_child = false;
    bool has_drawing_child = false;
    bool all_children_proven = true;
    bool all_children_are_losses = true;
    
    bool had_outcome = node->forced_outcome.has_value();

    for (int i = 0; i < node->num_children; ++i) {
        MCTSNode* child = node->first_child + i;
        
        if (child->forced_outcome.has_value()) {
            int outcome = child->forced_outcome.value();
            
            // child outcome -1 means the child loses, so the current node wins
            if (outcome == -1) { 
                has_winning_child = true;
                if (child->distance_to_mate.value() < best_win_dtm) best_win_dtm = child->distance_to_mate.value();
            } 
            // child outcome 0 is a draw
            else if (outcome == 0) {
                has_drawing_child = true;
                all_children_are_losses = false;
            } 
            // child outcome 1 means the child wins, so the current node loses
            else if (outcome == 1) { 
                if (child->distance_to_mate.value() > worst_loss_dtm) worst_loss_dtm = child->distance_to_mate.value();
            }
        } else {
            all_children_proven = false;
            all_children_are_losses = false;
        }
    }

    // Rule 1: We can win (a child loses) — take shortest mate
    if (has_winning_child) {
        node->forced_outcome = 1;
        node->distance_to_mate = best_win_dtm + 1;
    } 
    // Rule 2: All children fully proven
    else if (all_children_proven) {
        if (has_drawing_child) {
            node->forced_outcome = 0;
            node->distance_to_mate = 0; 
        } else if (all_children_are_losses) {
            node->forced_outcome = -1;
            node->distance_to_mate = worst_loss_dtm + 1;
        }
    }
    // Rule 3: Nothing conclusive
    else {
        node->forced_outcome = std::nullopt;
        node->distance_to_mate = std::nullopt;
    }

    // If this node just became proven, remove from parent's available children
    if ((!had_outcome && node->forced_outcome.has_value()) && node->parent != nullptr) {
        if (!node->unavailable_for_selection) {
            node->parent->num_available_children -= 1;
            if (node->parent->num_available_children <= 0) {
                node->parent->unavailable_for_selection = true;
            }
        }
    }
}

void MCTSEngine::_backpropagate(MCTSNode* node, double w, double d, double l, bool is_terminal) {
    auto start_time = NOW();
    
    if (is_terminal) {
        node->forced_outcome = (w > 0.0) ? 1 : ((l > 0.0) ? -1 : 0);
        node->distance_to_mate = 0;
    } else {
        _virtual_loss(node, false);
        _unmark_selected(node);
    }

    MCTSNode* current_node = node;
    current_node->raw_w = w;
    current_node->raw_d = d;
    current_node->raw_l = l;

    double current_w = w;
    double current_d = d;
    double current_l = l;
    
    // if (logger.get_level() <= 10) {
    //     logger.log("DEBUG", chess::uci::moveToUci(current_node->move) + " raw WDL: " + std::to_string(w) + "/" + std::to_string(d) + "/" + std::to_string(l));
    // }

    while (current_node != nullptr) {
        current_node->visits += 1;
        current_node->w_sum += current_w;
        current_node->d_sum += current_d;
        current_node->l_sum += current_l;

        // if (logger.get_level() <= 10) {
        //     logger.log("DEBUG", chess::uci::moveToUci(current_node->move) + " updated WDL sums: " + std::to_string(current_node->w_sum) + "/" + std::to_string(current_node->d_sum) + "/" + std::to_string(current_node->l_sum));
        // }
        
        _backpropagate_minimax(current_node);

        // Flip Perspective for the Parent
        double temp_w = current_w;
        current_w = current_l;
        current_l = temp_w;

        current_node = current_node->parent;
    }
    time_backpropagation += ELAPSED(start_time, NOW());
}

void MCTSEngine::_virtual_loss(MCTSNode* node, bool is_applying) {
    int multiplier = is_applying ? 1 : -1;
    MCTSNode* current_node = node;

    while (current_node != nullptr) {
        current_node->visits += (1 * multiplier);
        current_node->l_sum += (virtual_loss * multiplier);
        current_node = current_node->parent;
    }
}