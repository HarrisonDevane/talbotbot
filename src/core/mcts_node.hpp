#pragma once

#include <cstdint>
#include <climits>
#include "chess.hpp"

// Compact 72-byte MCTSNode.
//
// Size history: originally ~136 B on x86-64 (8 doubles + 2 std::optional<int>
// + 2 bools with padding). This layout drops to 72 B (~1.9x more nodes per
// RAM budget) by:
//   - narrowing accumulators, scores, and NN outputs to float,
//   - encoding proven outcomes as a sentinel-tagged int8 rather than
//     std::optional<int>,
//   - packing the two bool status fields into a single bitfield,
//   - reordering to minimise alignment padding.
//
// The 8-byte parent / first_child pointers are unchanged at this level -- a
// pointer->index refactor saves another 8 B but is deferred.
//
// Float safety: w_sum / d_sum / l_sum accumulate per-visit values in [0,1]
// plus virtual-loss offsets. Float has 24 bits of mantissa (~16M exact
// integers), so precision only starts to matter above ~10M visits on a
// single node -- well above any per-search visit count in self-play or
// analysis. Roots in very long analysis could conceivably approach this
// bound; the accumulators are only used for Q derivation, and Q there is
// dominated by the sum ratio, not the last decimal.
//
// Proven-outcome encoding:
//   forced_outcome == INT8_MIN  -> unresolved (equivalent to old std::nullopt)
//   forced_outcome in {-1, 0, 1} -> {loss, draw, win} from node's own perspective
// distance_to_mate is meaningful iff has_forced_outcome() is true. int16
// (±32k) covers DTM in any practical search or 7-piece Syzygy DTM (max ~550
// half-moves). int8 would be tight; not worth the risk.
struct MCTSNode {
    MCTSNode* parent = nullptr;               //  8
    MCTSNode* first_child = nullptr;          //  8

    // Accumulated WDL from visits, node's own perspective.
    float w_sum = 0.0f;                       //  4
    float d_sum = 0.0f;                       //  4
    float l_sum = 0.0f;                       //  4

    // NN outputs cached on this node. raw_logit is the policy logit written
    // by the PARENT's inference callback (indexed via policy_flat_index);
    // raw_w / raw_d / raw_l are written by THIS node's own inference callback.
    float raw_logit = 0.0f;                   //  4
    float raw_w = 0.0f;                       //  4
    float raw_d = 0.0f;                       //  4
    float raw_l = 0.0f;                       //  4

    float gumbel_noise = 0.0f;                //  4  -- per-child Gumbel-top-k noise
    float gumbel_score = 0.0f;                //  4  -- cached; refreshed by
                                              //         calculate_gumbel_score() /
                                              //         _rescore().

    int32_t visits = 0;                       //  4  -- root can exceed uint16

    chess::Move move = chess::Move::NO_MOVE;  //  2  -- library Move is a uint16 wrapper
    int16_t policy_flat_index = -1;           //  2  -- max policy index 4671 fits
    uint16_t num_children = 0;                //  2  -- max legal chess moves is 218
    uint16_t num_available_children = 0;      //  2

    int16_t distance_to_mate = 0;             //  2  -- meaningful iff has_forced_outcome()

    // Sentinel INT8_MIN == "unresolved". Any of {-1, 0, +1} == proven outcome.
    int8_t forced_outcome = INT8_MIN;         //  1

    // bit 0 = expanded, bit 1 = unavailable_for_selection.
    uint8_t flags = 0;                        //  1

    // -> 72 bytes on x86-64 (was ~136 B).

    static constexpr uint8_t FLAG_EXPANDED    = 0x1;
    static constexpr uint8_t FLAG_UNAVAILABLE = 0x2;

    MCTSNode(MCTSNode* p = nullptr, chess::Move m = chess::Move::NO_MOVE);

    // Status helpers -- replace the old bool fields.
    bool is_expanded()    const { return (flags & FLAG_EXPANDED)    != 0; }
    bool is_unavailable() const { return (flags & FLAG_UNAVAILABLE) != 0; }
    void set_expanded(bool v) {
        flags = static_cast<uint8_t>(v ? (flags |  FLAG_EXPANDED)
                                       : (flags & ~FLAG_EXPANDED));
    }
    void set_unavailable(bool v) {
        flags = static_cast<uint8_t>(v ? (flags |  FLAG_UNAVAILABLE)
                                       : (flags & ~FLAG_UNAVAILABLE));
    }

    // Forced-outcome helpers -- replace std::optional accessors.
    bool has_forced_outcome() const { return forced_outcome != INT8_MIN; }
    void clear_forced_outcome()     { forced_outcome = INT8_MIN; distance_to_mate = 0; }

    MCTSNode* get_child(chess::Move m) const;

    double expected_value(double contempt) const;
    double calculate_gumbel_score(double contempt, double gumbel_c_visit,
                                  double gumbel_c_scale, double max_visits,
                                  double v_mix);
    double calculate_v_mix(double contempt) const;
};

// Compile-time size assertion. Trips if a future edit balloons the struct.
// Adjust deliberately if you WANT more fields.
static_assert(sizeof(MCTSNode) <= 72,
              "MCTSNode grew past 72 bytes; deep-search RAM budget will suffer.");