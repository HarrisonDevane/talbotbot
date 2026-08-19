#include "time_control.hpp"
#include <algorithm>

namespace {
inline int64_t clamp64(int64_t v, int64_t lo, int64_t hi) {
    return std::max(lo, std::min(v, hi));
}
}  // namespace

TimeBudget TimeControl::allocate(const ClockState& clock) const {
    TimeBudget out;

    // Usable time after lag reserve. At/below overhead -> scramble: move instantly.
    const int64_t remaining = clock.time_left_ms - cfg_.move_overhead_ms;
    if (remaining <= cfg_.min_think_ms) {
        out.target_ms     = cfg_.min_think_ms;
        out.hard_limit_ms = std::max<int64_t>(
            cfg_.min_think_ms,
            clock.time_left_ms > 0 ? clock.time_left_ms / 2 : cfg_.min_think_ms);
        return out;
    }

    // Base: divide remaining by moves_to_go (if given) or an assumed horizon.
    double base;
    if (clock.moves_to_go > 0) {
        base = static_cast<double>(remaining) / clock.moves_to_go;
    } else {
        base = static_cast<double>(remaining) / cfg_.move_horizon;
    }
    base *= cfg_.base_fraction;

    // Add (most of) the increment -- refunded each move, so sustainable.
    double target = base + cfg_.increment_fraction * static_cast<double>(clock.increment_ms);

    // Hard limit: bounded overspend on critical moves, capped by safety ceiling.
    const int64_t ceiling = static_cast<int64_t>(remaining * cfg_.max_time_fraction);
    int64_t hard = std::min<int64_t>(static_cast<int64_t>(target * cfg_.hard_multiplier), ceiling);

    // Clamp to 1 <= target <= hard <= remaining.
    int64_t tgt = clamp64(static_cast<int64_t>(target), cfg_.min_think_ms, ceiling);
    hard        = clamp64(hard, tgt, remaining);

    out.target_ms     = tgt;
    out.hard_limit_ms = hard;
    return out;
}
