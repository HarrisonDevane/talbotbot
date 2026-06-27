#pragma once
#include <cstdint>

// Pure time-allocation logic: clock in, {soft target, hard limit} out.
// Knows nothing about UCI/network/GPU/MCTS. Unit-testable with no engine.
// Early-stopping (visit-based smart pruning) lives in the search layer (Phase 2).

struct ClockState {
    int64_t time_left_ms  = 0;   // our remaining time
    int64_t increment_ms  = 0;   // our per-move increment
    int     moves_to_go   = 0;   // 0 == not provided
    int     ply           = 1;   // 1-based; reserved for future phase logic
};

struct TimeBudget {
    int64_t target_ms     = 0;   // soft: aim here; early-stop allowed against it
    int64_t hard_limit_ms = 0;   // hard: search must never exceed this
};

// All fields loaded from play_uci.yaml `time_control:` -- no defaults, every field required.
struct TimeControlConfig {
    double  move_horizon;
    double  increment_fraction;
    double  base_fraction;
    double  hard_multiplier;
    double  max_time_fraction;
    int64_t move_overhead_ms;
    int64_t min_think_ms;
    double  nps_ewma_alpha; // smoothing for the trailing nodes/sec estimate (used by MCTS, not TimeControl)
    double  nps_ewma;
};

class TimeControl {
public:
    explicit TimeControl(TimeControlConfig cfg) : cfg_(cfg) {}
    TimeBudget allocate(const ClockState& clock) const;
    const TimeControlConfig& config() const { return cfg_; }

private:
    TimeControlConfig cfg_;
};