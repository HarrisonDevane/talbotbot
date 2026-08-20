#pragma once

#include <cstdint>
#include <climits>
#include "chess.hpp"

// Compact 56-byte MCTSNode.
//
// Size history: original ~136 B (doubles + std::optional<int> + bools). First
// trim brought it to 72 B (float sums, sentinel-tagged forced_outcome, packed
// flags). This second pass drops three more fields to reach 56 B:
//
//   - raw_l dropped: derivable as (1 - raw_w - raw_d) since NN outputs a
//     valid probability distribution. See raw_l() helper.
//   - gumbel_noise dropped: only meaningful for root's direct children.
//     Stored off-node in MCTSEngine::root_gumbel_noise (indexed by child
//     position). calculate_gumbel_score() takes noise as an explicit param.
//   - gumbel_score dropped: was a cache for the score formula. Callers now
//     compute inline (cheap: a few flops per call) or hold a local scratch
//     vector when they need many scores at once. _rescore() went away with
//     the field -- there's no cache left to refresh.
//
// Together these save 12 B of fields; alignment collapses the resulting
// padding, so total drops 16 B (72 -> 56). ~22% memory saving on the pool.
//
// Float safety: w_sum / d_sum / l_sum accumulate per-visit values in [0,1]
// plus virtual-loss offsets. Float mantissa (~16M exact integers) tolerates
// >10M visits per node.
//
// Proven-outcome encoding:
//   forced_outcome == INT8_MIN  -> unresolved (was std::nullopt).
//   forced_outcome in {-1,0,1}  -> {loss,draw,win} from node's own perspective.
// distance_to_mate is meaningful iff has_forced_outcome().
struct MCTSNode {
    MCTSNode* parent = nullptr;               //  8
    MCTSNode* first_child = nullptr;          //  8

    // Accumulated WDL from visits, node's own perspective.
    float w_sum = 0.0f;                       //  4
    float d_sum = 0.0f;                       //  4
    float l_sum = 0.0f;                       //  4

    // NN outputs cached on this node. raw_logit is the policy logit written
    // by the PARENT's inference callback (indexed via policy_flat_index);
    // raw_w / raw_d are written by THIS node's own inference callback.
    // raw_l is derived on read: raw_l() below.
    float raw_logit = 0.0f;                   //  4
    float raw_w = 0.0f;                       //  4
    float raw_d = 0.0f;                       //  4

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

    // -> 56 bytes on x86-64 (was 72, originally ~136).

    static constexpr uint8_t FLAG_EXPANDED    = 0x1;
    static constexpr uint8_t FLAG_UNAVAILABLE = 0x2;

    MCTSNode(MCTSNode* p = nullptr, chess::Move m = chess::Move::NO_MOVE);

    // Derived value for the WDL loss probability. NN outputs sum to 1 by
    // construction (softmax at the value head), so this recomputation is
    // exact within float precision.
    float raw_l() const { return 1.0f - raw_w - raw_d; }

    // Status helpers.
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

    // Forced-outcome helpers.
    bool has_forced_outcome() const { return forced_outcome != INT8_MIN; }
    void clear_forced_outcome()     { forced_outcome = INT8_MIN; distance_to_mate = 0; }

    MCTSNode* get_child(chess::Move m) const;

    double expected_value(double contempt) const;

    // Gumbel score = raw_logit + noise + sigma * v_mix_completion.
    // NO CACHING -- returns the computed value. Caller stores if needed.
    // noise: 0 for non-root descendants during selection; root children pull
    // their noise from MCTSEngine::root_gumbel_noise[child - root->first_child].
    double calculate_gumbel_score(double contempt, double gumbel_c_visit,
                                  double gumbel_c_scale, double max_visits,
                                  double v_mix, double noise) const;

    double calculate_v_mix(double contempt) const;
};