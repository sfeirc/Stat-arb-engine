#include "arb/risk_manager.hpp"
#include <cmath>
#include <cstdlib>

bool RiskManager::check_signal(const Signal& sig) const noexcept {
    // Kill switch: early-out on the rare hot path (branch predicted not-taken).
    if (__builtin_expect(killed_.load(std::memory_order_acquire), 0)) return false;

    // Load shared state once — relaxed is sufficient here; the kill-switch
    // acquire above provides the necessary ordering for the common case.
    const double nav      = nav_.load(std::memory_order_relaxed);
    const double pair_exp = pair_exposure_[sig.pair_id & (MAX_PAIRS - 1)]
                                .load(std::memory_order_relaxed);
    const double gross    = gross_exposure_.load(std::memory_order_relaxed);

    // Branchless evaluation of all checks — bitwise AND avoids short-circuit
    // branches so the CPU can execute all comparisons in parallel.

    // Check 1: z-score sanity
    const bool z_ok  = (std::abs(sig.z_score) <= limits_.max_z_score_abs);

    // Check 2: size_factor in [0, 1]
    const bool sf_ok = (sig.size_factor >= 0.0) & (sig.size_factor <= 1.0);

    // Check 3: single-pair concentration (skip when side == 0 / exit signal)
    // When side == 0 we want this check to be vacuously true.
    const double projected = pair_exp
                             + static_cast<double>(sig.side) * sig.size_factor * 1000.0;
    const bool pair_ok = (sig.side == 0)
                       | (nav <= 0.0)
                       | (std::abs(projected) <= nav * limits_.max_single_pair_frac);

    // Check 4: total gross concentration
    const bool gross_ok = (nav <= 0.0)
                        | (gross / nav < limits_.max_concentration);

    return z_ok & sf_ok & pair_ok & gross_ok;
}

void RiskManager::record_fill(
    uint32_t pair_id,
    double   pnl,
    double   delta_exposure
) noexcept {
    // Update portfolio P&L
    const double prev_pnl = portfolio_pnl_.fetch_add(pnl, std::memory_order_acq_rel);
    const double new_pnl  = prev_pnl + pnl;

    // Update peak
    double peak = peak_pnl_.load(std::memory_order_acquire);
    while (new_pnl > peak) {
        if (peak_pnl_.compare_exchange_weak(peak, new_pnl,
                std::memory_order_release, std::memory_order_relaxed)) {
            break;
        }
        // peak was updated by another thread — retry with new value
    }

    // Kill switch: intraday drawdown
    const double updated_peak = peak_pnl_.load(std::memory_order_acquire);
    const double drawdown = updated_peak - new_pnl;
    const double nav      = nav_.load(std::memory_order_acquire);
    if (nav > 0.0 && drawdown / nav >= limits_.max_drawdown_kill) {
        killed_.store(true, std::memory_order_release);
    }

    // Update pair exposure
    if (pair_id < MAX_PAIRS) {
        pair_exposure_[pair_id].fetch_add(delta_exposure, std::memory_order_relaxed);
    }

    // Update gross exposure
    gross_exposure_.fetch_add(std::abs(delta_exposure), std::memory_order_relaxed);
}

double RiskManager::pair_exposure(uint32_t pair_id) const noexcept {
    if (pair_id >= MAX_PAIRS) return 0.0;
    return pair_exposure_[pair_id].load(std::memory_order_relaxed);
}
