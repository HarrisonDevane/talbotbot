#pragma once

#include <cstdint>
#include <climits>
#include "chess.hpp"

// Forward declaration -- MCTSEdge and MCTSNode reference each other.
struct MCTSNode;

// -----------------------------------------------------------------------------
// MCTSEdge -- 16 bytes. One per legal move at every expanded node.
// -----------------------------------------------------------------------------
//
// Allocated in a contiguous block from EdgePool at expansion time. Holds only
// what the parent's selection needs to score and dispatch this move, plus a
// pointer to the child node once it has been materialised.
//
// child == nullptr means "this edge has never been descended into". Everything
// that lives on a node -- visits, W/D/L sums, forced_outcome, unavailability,
// grandchildren -- does not exist yet. Selection treats null child as
// visits=0 and uses v_mix for Q, exactly the same as an unvisited node under
// the old scheme.
//
// The move that led here and the parent's policy prior for that move live on
// the edge (rather than on the eventual child node) because both are needed
// even when no node exists -- e.g. target_generator computing the policy
// target reads raw_logit off every edge, visited or not.
// -----------------------------------------------------------------------------
struct MCTSEdge {
    MCTSNode* child = nullptr;                 //  8  nullptr until first visit
    float raw_logit = 0.0f;                    //  4  policy prior from parent's NN eval
    int16_t policy_flat_index = -1;            //  2
    chess::Move move = chess::Move::NO_MOVE;   //  2  (uint16 wrapper)

    // -> 16 bytes on x86-64. Naturally aligned; no padding.

    // Gumbel score for this edge as a candidate for descent from its parent.
    // Same formula as the previous MCTSNode::calculate_gumbel_score, but
    // visits/Q are read through the child pointer -- null child means
    // visits=0, so q_val falls back to v_mix (identical behaviour to a
    // 0-visit node under the old scheme).
    //
    // noise: 0 for non-root descendants during selection; root's edges pull
    // their noise from MCTSEngine::root_gumbel_noise indexed by edge slot
    // (edge - root->first_edge).
    double calculate_gumbel_score(double contempt, double gumbel_c_visit,
                                  double gumbel_c_scale, double max_visits,
                                  double v_mix, double noise) const;
};

// -----------------------------------------------------------------------------
// MCTSNode -- 48 bytes. Allocated lazily by selection when it first descends
// into an edge.
// -----------------------------------------------------------------------------
//
// Size history:
//   ~136 B  original (doubles + std::optional + bools)
//     72 B  first trim (float sums, packed flags, sentinel-tagged outcome)
//     56 B  second trim (raw_l derived, gumbel noise/score moved off-node)
//     48 B  this pass: move, policy_flat_index, and raw_logit moved to
//           MCTSEdge; first_child renamed to first_edge (still 8 B ptr).
//
// The three fields removed here are precisely what an unvisited child needed.
// Under the new scheme unvisited children have no node at all, so those
// fields find their permanent home on the edge instead.
//
// Invariants worth stating explicitly:
//   * A node is only allocated when selection descends into an edge. Virtual
//     loss is applied *after* materialisation, so a fresh node's visits go
//     from 0 to 1 in the same descent step.
//   * is_unavailable() and forced_outcome are only ever written from _select
//     or _backpropagate_minimax, both of which operate on materialised
//     nodes. There is no code path that marks an unvisited edge dead --
//     an unvisited edge is trivially available.
//   * num_children == number of edges in the block pointed to by first_edge.
//
// Float safety: w_sum / d_sum / l_sum accumulate per-visit values in [0,1]
// plus virtual-loss offsets. Float mantissa (~16M exact integers) tolerates
// >10M visits per node.
//
// Proven-outcome encoding:
//   forced_outcome == INT8_MIN  -> unresolved.
//   forced_outcome in {-1,0,1}  -> {loss,draw,win} from node's own perspective.
// distance_to_mate is meaningful iff has_forced_outcome().
// -----------------------------------------------------------------------------
struct MCTSNode {
    MCTSNode* parent = nullptr;                //  8
    MCTSEdge* first_edge = nullptr;            //  8

    // Accumulated WDL from visits, node's own perspective.
    float w_sum = 0.0f;                        //  4
    float d_sum = 0.0f;                        //  4
    float l_sum = 0.0f;                        //  4

    // NN outputs written by THIS node's own inference callback. raw_l is
    // derived on read (raw_l() below). The parent's policy prior for the
    // move that led here now lives on the incoming edge, not on the node.
    float raw_w = 0.0f;                        //  4
    float raw_d = 0.0f;                        //  4

    int32_t visits = 0;                        //  4  root can exceed uint16

    // Cache of expected_value(contempt) computed with the ENGINE's fixed
    // contempt. Updated inline in _virtual_loss and _backpropagate whenever
    // visits/w_sum/d_sum/l_sum mutate. _select reads this directly (bypassing
    // the expected_value() method) to skip the divide in its hot loop.
    // Value is 0.0 whenever visits == 0 -- no reader inspects it in that case
    // (they all fall back to v_mix), so init default (0.0f) is safe.
    float cached_q = 0.0f;                     //  4

    uint16_t num_children = 0;                 //  2  = number of edges in block
    uint16_t num_available_children = 0;       //  2

    int16_t distance_to_mate = 0;              //  2  meaningful iff has_forced_outcome()

    int8_t forced_outcome = INT8_MIN;          //  1  INT8_MIN = unresolved
    uint8_t flags = 0;                         //  1  FLAG_EXPANDED | FLAG_UNAVAILABLE

    // -> 52 bytes of fields, padded to 56 for 8-byte alignment (pointer members).

    static constexpr uint8_t FLAG_EXPANDED    = 0x1;
    static constexpr uint8_t FLAG_UNAVAILABLE = 0x2;

    explicit MCTSNode(MCTSNode* p = nullptr);
    static constexpr uint8_t FLAG_MOVER_HAS_DRAW = 0x4;

    // Derived WDL loss probability. NN outputs sum to 1 by construction, so
    // this is exact within float precision.
    float raw_l() const { return 1.0f - raw_w - raw_d; }

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

    bool has_forced_outcome() const { return forced_outcome != INT8_MIN; }
    void clear_forced_outcome()     { forced_outcome = INT8_MIN; distance_to_mate = 0; }

    // Look up the edge for a given move. Returns nullptr if no such edge
    // (move not legal here, or this node not yet expanded). Replaces the
    // old get_child(move): callers that want the child node do
    //   MCTSEdge* e = node->get_edge(m);
    //   MCTSNode* c = (e != nullptr) ? e->child : nullptr;
    // and handle the null-child (unvisited) case explicitly.
    MCTSEdge* get_edge(chess::Move m) const;

    // Value from this node's own perspective. Behaviour unchanged.
    double expected_value(double contempt) const;
    bool mover_has_draw() const { return (flags & FLAG_MOVER_HAS_DRAW) != 0; }
    void set_mover_has_draw(bool v) {
        flags = static_cast<uint8_t>(v ? (flags |  FLAG_MOVER_HAS_DRAW)
                                    : (flags & ~FLAG_MOVER_HAS_DRAW));
    }

    // Recompute cached_q. MUST be called after any mutation to visits/w_sum/
    // d_sum/l_sum -- inlined into _virtual_loss and _backpropagate for that
    // purpose. Falls back to 0.0f when visits == 0; no reader inspects
    // cached_q in that case (all v_mix), so the fallback is a placeholder.
    inline void update_cached_q(double contempt) {
        cached_q = (visits > 0)
                 ? static_cast<float>((w_sum - l_sum + contempt * d_sum) / visits)
                 : 0.0f;
    }

    // v_mix over this node's children. Only edges with a materialised child
    // whose visits > 0 contribute -- unvisited edges are excluded, same as
    // 0-visit children were excluded under the old scheme.
    double calculate_v_mix(double contempt) const;
};