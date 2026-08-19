// =============================================================================
// mcts_base.cpp
//
// Implementations of the shared MCTS machinery. Every method here was lifted
// from the original mcts_engine.cpp with no logic changes -- only the ctor
// arg list was trimmed (cpuct, draw_cutoff, gumbel_* dropped; those now live
// in the derived engines' ctors).
// =============================================================================

#include "mcts_base.hpp"
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

MctsBase::MctsBase(
    int node_pool_capacity,
    int worker_batch_size,
    moodycamel::ConcurrentQueue<std::pair<int, int>>& inference_queue,
    ThreadSafeQueue<std::vector<int>>& result_queue,
    int worker_id,
    double virtual_loss,
    double contempt,
    double policy_softmax_temp,
    const chess::Board& board,
    const std::vector<chess::Board>& base_history,
    Logger& logger,
    std::vector<torch::Tensor>& shared_input_buffer,
    std::vector<torch::Tensor>& shared_policy_buffer,
    std::vector<torch::Tensor>& shared_value_buffer,
    ThreadSafeQueue<int>& buffer_free_slots,
    std::atomic<int>* core_wait_count,
    int workers_per_core,
    bool two_fold_repetition,
    bool use_tablebase
) : worker_batch_size(worker_batch_size),
    worker_id(worker_id),
    two_fold_repetition(two_fold_repetition),
    virtual_loss(virtual_loss),
    contempt(contempt),
    policy_softmax_temp(policy_softmax_temp),
    root_board(board),
    base_history(base_history),
    node_pool(node_pool_capacity),
    logger(logger),
    inference_queue(inference_queue),
    result_queue(result_queue),
    buffer_free_slots(buffer_free_slots),
    shared_input_buffer(shared_input_buffer),
    shared_policy_buffer(shared_policy_buffer),
    shared_value_buffer(shared_value_buffer),
    core_wait_count(core_wait_count),
    workers_per_core(workers_per_core),
    use_tablebase(use_tablebase)
{
    torch::set_num_threads(1);

    device = torch::kCUDA;
    policy_logits_dtype = torch::kFloat16;

    in_flight_nodes.resize(shared_input_buffer.size(), nullptr);
    root = node_pool.allocate();
    simulation_count = 0;
    inference_sent = 0;
    inference_received = 0;
}

void MctsBase::reset(const chess::Board& board, const std::vector<chess::Board>& history) {
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

void MctsBase::_mark_selected(MCTSNode* node) {
    MCTSNode* current_node = node;
    current_node->set_unavailable(true);
    MCTSNode* parent = current_node->parent;

    while (parent != nullptr) {
        parent->num_available_children -= 1;
        if (parent->num_available_children > 0) break;
        parent->set_unavailable(true);
        current_node = parent;
        parent = current_node->parent;
    }
}

void MctsBase::_unmark_selected(MCTSNode* node) {
    MCTSNode* current_node = node;
    current_node->set_unavailable(false);
    MCTSNode* parent = current_node->parent;

    while (parent != nullptr) {
        parent->num_available_children += 1;
        if (parent->num_available_children == 1) {
            parent->set_unavailable(false);
            current_node = parent;
            parent = current_node->parent;
        } else break;
    }
}

// _spin_wait's definition lives in mcts_base.hpp -- it's a template, so every
// TU that instantiates it (gumbel_mcts.cpp, puct_mcts.cpp, and mcts_base.cpp
// itself) needs the full definition visible at the point of use.

void MctsBase::_retrieve_inference(bool block) {
    auto start_time = NOW();
    std::vector<int> completed_indices;

    while (true) {
        if (block) {
            completed_indices = result_queue.pop_wait();
            block = false;
        } else {
            if (!result_queue.try_pop(completed_indices)) break;
        }

        if (logger.get_level() <= 10) {
            logger.log("DEBUG", "Received " + std::to_string(completed_indices.size()) + " inferences from batcher.");
        }

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
                if (!node->is_expanded()) {
                    auto exp_start = NOW();
                    for (int i = 0; i < node->num_children; ++i) {
                        MCTSNode* child = node->first_child + i;
                        child->raw_logit = policy_ptr[child->policy_flat_index] / policy_softmax_temp;
                    }
                    node->set_expanded(true);
                    time_expansion += ELAPSED(exp_start, NOW());
                }
                _backpropagate(node, p_win, p_draw, p_loss, false);
            }
        }
    }
    time_retrieval += ELAPSED(start_time, NOW());
}

void MctsBase::_submit_batch() {
    auto start_time = NOW();
    int b_size = batch_buffer.size();
    if (b_size == 0) return;

    if (logger.get_level() <= 10) {
        logger.log("DEBUG", "Submitting batch of " + std::to_string(b_size) + " states to inference queue.");
    }

    inference_queue.enqueue_bulk(batch_buffer.data(), b_size);

    inference_sent += b_size;
    batch_buffer.clear();

    time_queueing += ELAPSED(start_time, NOW());
}

void MctsBase::_handle_terminal_node(MCTSNode* leaf) {
    auto start_time = NOW();
    auto result = root_board.isGameOver();

    double w = 0.0, d = 0.0, l = 0.0;
    std::string term_type = "Draw";

    if (result.second == chess::GameResult::LOSE) {
        l = 1.0;
        term_type = "Loss (Mate)";
    } else if (result.second == chess::GameResult::DRAW || root_board.isRepetition(two_fold_repetition ? 1 : 2)) {
        d = 1.0;
    }

    if (logger.get_level() <= 10) {
        logger.log("DEBUG", "Terminal node reached during search. Result: " + term_type);
    }

    _mark_selected(leaf);
    time_expansion += ELAPSED(start_time, NOW());

    _backpropagate(leaf, w, d, l, true);
    simulation_count++;
}

// Syzygy WDL probe for the position currently on root_board.
bool MctsBase::_try_tablebase(MCTSNode* leaf) {
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

    if ((white_bb | black_bb).count() > (int)TB_LARGEST) return false;

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

    double w = 0.0, d = 0.0, l = 0.0;
    switch (wdl) {
        case TB_WIN:          w = 1.0; break;
        case TB_LOSS:         l = 1.0; break;
        case TB_CURSED_WIN:
        case TB_BLESSED_LOSS:
        case TB_DRAW:         d = 1.0; break;
        default:              return false;
    }

    _mark_selected(leaf);
    _backpropagate(leaf, w, d, l, true);
    simulation_count++;
    return true;
}

void MctsBase::_queue_leaf_for_inference(MCTSNode* leaf, const std::vector<MCTSNode*>& simulation_path) {
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

void MctsBase::_expand_root() {
    _queue_leaf_for_inference(root, {});
    _submit_batch();
    while (inference_received < inference_sent) {
        _retrieve_inference(true);
    }
}

void MctsBase::_flush_inflight() {
    if (!batch_buffer.empty()) _submit_batch();
    while (inference_received < inference_sent) {
        _retrieve_inference(true);
    }
}

// EWMA on the ratio (overall speed over the window); nps_alpha_ weights recent
// moves. Searches cut by the deadline still feed in; skip recording if you want
// cleaner samples.
void MctsBase::_record_nps(int sims, double seconds) {
    if (sims <= 0 || seconds <= 0.0) return;
    const double inst = static_cast<double>(sims) / seconds;
    nps_ewma_ = (nps_ewma_ <= 0.0) ? inst : (nps_alpha_ * inst + (1.0 - nps_alpha_) * nps_ewma_);
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
void MctsBase::_log_node_by_path(const std::vector<std::string>& uci_path, int top_n) {
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

    char head[512];
    double tgt_q_own = (node->visits > 0) ? node->expected_value(contempt) : ((node->raw_w - node->raw_l) + contempt * node->raw_d);
    double tgt_vmix  = node->is_expanded() ? node->calculate_v_mix(contempt) : 0.0;
    snprintf(head, sizeof(head),
        "TARGET raw network WDL (own mover persp): W=%.4f D=%.4f L=%.4f  -> raw_value(own)=%+.4f",
        node->raw_w, node->raw_d, node->raw_l, (node->raw_w - node->raw_l));
    logger.log("INFO", head);
    snprintf(head, sizeof(head),
        "TARGET visits=%d  expected_value(own)=%+.4f  v_mix(own)=%+.4f  logit=%.3f  expanded=%d  outcome=%s",
        node->visits, tgt_q_own, tgt_vmix, node->raw_logit, node->is_expanded() ? 1 : 0,
        node->has_forced_outcome() ? std::to_string(node->forced_outcome).c_str() : "None");
    logger.log("INFO", head);

    if (!node->is_expanded() || node->num_children == 0) {
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
        double q_target = (c->visits > 0) ? -c->expected_value(contempt)
                                          : -((c->raw_w - c->raw_l) + contempt * c->raw_d);
        double child_rawV = (c->raw_w - c->raw_l);
        std::string oc = c->has_forced_outcome() ? std::to_string(c->forced_outcome) : "None";
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

void MctsBase::_backpropagate_minimax(MCTSNode* node) {
    if (node->num_children == 0) return;

    int best_win_dtm = 999999;
    int worst_loss_dtm = -1;

    bool has_winning_child = false;
    bool has_drawing_child = false;
    bool all_children_proven = true;
    bool all_children_are_losses = true;

    bool had_outcome = node->has_forced_outcome();

    for (int i = 0; i < node->num_children; ++i) {
        MCTSNode* child = node->first_child + i;

        if (child->has_forced_outcome()) {
            int outcome = child->forced_outcome;

            if (outcome == -1) {
                has_winning_child = true;
                if (child->distance_to_mate < best_win_dtm) best_win_dtm = child->distance_to_mate;
            }
            else if (outcome == 0) {
                has_drawing_child = true;
                all_children_are_losses = false;
            }
            else if (outcome == 1) {
                if (child->distance_to_mate > worst_loss_dtm) worst_loss_dtm = child->distance_to_mate;
            }
        } else {
            all_children_proven = false;
            all_children_are_losses = false;
        }
    }

    if (has_winning_child) {
        node->forced_outcome = 1;
        node->distance_to_mate = static_cast<int16_t>(best_win_dtm + 1);
    }
    else if (all_children_proven) {
        if (has_drawing_child) {
            node->forced_outcome = 0;
            node->distance_to_mate = 0;
        } else if (all_children_are_losses) {
            node->forced_outcome = -1;
            node->distance_to_mate = static_cast<int16_t>(worst_loss_dtm + 1);
        }
    }

    if ((!had_outcome && node->has_forced_outcome()) && node->parent != nullptr) {
        if (!node->is_unavailable()) {
            node->parent->num_available_children -= 1;
            if (node->parent->num_available_children <= 0) {
                node->parent->set_unavailable(true);
            }
        }
    }
}

void MctsBase::_backpropagate(MCTSNode* node, double w, double d, double l, bool is_terminal) {
    auto start_time = NOW();

    if (is_terminal) {
        node->forced_outcome = static_cast<int8_t>((w > 0.0) ? 1 : ((l > 0.0) ? -1 : 0));
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

    if (logger.get_level() <= 10) {
        logger.log("DEBUG", chess::uci::moveToUci(current_node->move) + " raw WDL: " + std::to_string(w) + "/" + std::to_string(d) + "/" + std::to_string(l));
    }

    while (current_node != nullptr) {
        current_node->visits += 1;
        current_node->w_sum += static_cast<float>(current_w);
        current_node->d_sum += static_cast<float>(current_d);
        current_node->l_sum += static_cast<float>(current_l);

        if (logger.get_level() <= 10) {
            logger.log("DEBUG", chess::uci::moveToUci(current_node->move) + " updated WDL sums: " + std::to_string(current_node->w_sum) + "/" + std::to_string(current_node->d_sum) + "/" + std::to_string(current_node->l_sum));
        }

        _backpropagate_minimax(current_node);

        double temp_w = current_w;
        current_w = current_l;
        current_l = temp_w;

        current_node = current_node->parent;
    }
    time_backpropagation += ELAPSED(start_time, NOW());
}

void MctsBase::_virtual_loss(MCTSNode* node, bool is_applying) {
    int multiplier = is_applying ? 1 : -1;
    MCTSNode* current_node = node;

    while (current_node != nullptr) {
        current_node->visits += (1 * multiplier);
        current_node->l_sum += static_cast<float>(virtual_loss * multiplier);
        current_node = current_node->parent;
    }
}

// Explicit template instantiations for the two _spin_wait usages in this TU
// are not required -- callers are the derived engines' TUs, which will get
// their own instantiations from the header definition.