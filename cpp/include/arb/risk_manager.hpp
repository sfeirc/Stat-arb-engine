#pragma once
#include <atomic>
#include <array>
#include <cstdint>

/**
 * Signal: output of the Python signal generator, serialised to C++.
 */
struct Signal {
    int64_t  timestamp_ns;   // nanoseconds since epoch
    uint32_t pair_id;
    int8_t   side;           // +1 long spread, -1 short spread, 0 exit
    double   z_score;
    double   size_factor;    // [0, 1] from HMM regime model
};

/**
 * Order: produced by OrderRouter after risk validation.
 */
struct Order {
    int64_t  timestamp_ns;
    uint32_t pair_id;
    uint32_t order_id;       // set by exchange on submit
    int8_t   side;           // +1 buy, -1 sell
    double   qty;            // quantity in USDT
    double   limit_price;    // 0.0 = market order
};

/**
 * RiskManager
 *
 * Pre-trade risk checks:
 *   1. Kill switch: hard stop if intraday drawdown >= max_drawdown_kill
 *   2. Single-pair concentration: max_single_pair_frac of portfolio
 *   3. Z-score sanity: reject signals with |z| > 10
 *   4. Size factor sanity: reject if size_factor outside [0, 1]
 *
 * check_signal() is designed to be called in the hot path (< 100 ns).
 */
class RiskManager {
public:
    struct Limits {
        double max_single_pair_frac = 0.20;   // 20% of portfolio per pair
        double max_drawdown_kill    = 0.05;   // 5% intraday drawdown kill switch
        double max_concentration    = 0.60;   // 60% total gross exposure
        double max_z_score_abs      = 10.0;   // sanity check on z-score
    };

    RiskManager() noexcept = default;
    explicit RiskManager(Limits limits) noexcept : limits_(limits) {}

    RiskManager(const RiskManager&) = delete;
    RiskManager& operator=(const RiskManager&) = delete;

    /**
     * Check whether a signal passes all pre-trade risk checks.
     *
     * Returns true if the signal may be converted into an order.
     * Thread-safe (all state is atomic).
     */
    [[nodiscard]] bool check_signal(const Signal& sig) const noexcept;

    /**
     * Record a fill result.  Updates portfolio PnL and checks kill switch.
     *
     * pnl    : fill-level realised P&L (USDT, can be negative)
     * pair_id: which pair
     * delta_exposure : change in gross exposure (positive = buying)
     */
    void record_fill(uint32_t pair_id, double pnl, double delta_exposure) noexcept;

    /** Update portfolio NAV (call daily / on significant moves). */
    void set_nav(double nav) noexcept {
        nav_.store(nav, std::memory_order_release);
    }

    /** Mark a new intraday high-water mark. */
    void set_peak_pnl(double peak) noexcept {
        peak_pnl_.store(peak, std::memory_order_release);
    }

    /** Manually engage the kill switch (e.g. from monitoring thread). */
    void engage_kill_switch() noexcept {
        killed_.store(true, std::memory_order_release);
    }

    /** Reset kill switch (operator intervention after review). */
    void reset_kill_switch() noexcept {
        killed_.store(false, std::memory_order_release);
    }

    [[nodiscard]] bool is_killed() const noexcept {
        return killed_.load(std::memory_order_acquire);
    }

    [[nodiscard]] double portfolio_pnl() const noexcept {
        return portfolio_pnl_.load(std::memory_order_acquire);
    }

    [[nodiscard]] double gross_exposure() const noexcept {
        return gross_exposure_.load(std::memory_order_acquire);
    }

    [[nodiscard]] double pair_exposure(uint32_t pair_id) const noexcept;

private:
    Limits limits_;

    alignas(64) std::atomic<double>   nav_{1'000'000.0};      // portfolio NAV (USDT)
    alignas(64) std::atomic<double>   portfolio_pnl_{0.0};
    alignas(64) std::atomic<double>   peak_pnl_{0.0};
    alignas(64) std::atomic<double>   gross_exposure_{0.0};
    alignas(64) std::atomic<bool>     killed_{false};

    static constexpr std::size_t MAX_PAIRS = 1024;
    std::array<std::atomic<double>, MAX_PAIRS> pair_exposure_{};
};
