#pragma once
#include "sim_exchange.hpp"
#include <atomic>
#include <array>
#include <cstdint>

/**
 * PositionManager
 *
 * Tracks real-time positions and P&L across up to 1024 pairs.
 * Receives fills from SimExchange and updates atomic state.
 *
 * Thread-safe: on_fill() may be called from the exchange callback thread;
 * getters may be called from any thread.
 */
class PositionManager {
public:
    static constexpr std::size_t MAX_PAIRS = 1024;

    struct PairState {
        alignas(64) std::atomic<double>  net_qty{0.0};          // signed position
        alignas(64) std::atomic<double>  avg_entry_price{0.0};
        alignas(64) std::atomic<double>  realised_pnl{0.0};
        alignas(64) std::atomic<double>  unrealised_pnl{0.0};
        alignas(64) std::atomic<int64_t> last_fill_ns{0};
        alignas(64) std::atomic<int64_t> trade_count{0};
    };

    PositionManager() noexcept = default;
    PositionManager(const PositionManager&) = delete;
    PositionManager& operator=(const PositionManager&) = delete;

    /**
     * Process an incoming fill.  Updates net_qty, avg_entry_price,
     * realised_pnl.  Called from the exchange fill callback.
     */
    void on_fill(const Fill& fill) noexcept;

    /**
     * Update mark-to-market price for a pair.
     * Recomputes unrealised P&L.
     *
     * mid_price: current mid price in USDT.
     */
    void mark_to_market(uint32_t pair_id, double mid_price) noexcept;

    /** Net quantity for a pair (+ve = long, -ve = short). */
    [[nodiscard]] double net_qty(uint32_t pair_id) const noexcept;

    /** Unrealised P&L for a pair (USDT). */
    [[nodiscard]] double unrealised_pnl(uint32_t pair_id) const noexcept;

    /** Realised P&L for a pair (USDT). */
    [[nodiscard]] double realised_pnl(uint32_t pair_id) const noexcept;

    /** Total portfolio P&L (realised + unrealised) across all pairs. */
    [[nodiscard]] double total_pnl() const noexcept;

    /** Total portfolio realised P&L. */
    [[nodiscard]] double total_realised_pnl() const noexcept;

    /** Reset all positions to flat.  Not thread-safe — use only when quiesced. */
    void reset() noexcept;

private:
    std::array<PairState, MAX_PAIRS> pairs_{};
};
