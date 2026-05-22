#pragma once
#include "spsc_ring.hpp"
#include "risk_manager.hpp"   // Signal, Order
#include <atomic>
#include <array>
#include <cstdint>
#include <functional>

/**
 * Fill event returned by SimExchange after order processing.
 */
struct Fill {
    int64_t  timestamp_ns;
    uint32_t pair_id;
    uint32_t order_id;
    int8_t   side;           // +1 = buy, -1 = sell
    double   qty;
    double   fill_price;
    double   commission;     // fraction of notional
};

/**
 * SimExchange
 *
 * Simulated exchange with a realistic fill model:
 *   - Latency: configurable Gaussian jitter (default mean 50 µs, std 10 µs)
 *   - Fill probability: 1.0 for market orders, configurable for limit
 *   - Slippage: linear in size (Almgren-Chriss, η = 0.1)
 *   - Commission: 0.04% taker fee (Binance-like)
 *
 * Orders are submitted via submit_order() and fills are dispatched to
 * an optional callback and to the fill ring.
 */
class SimExchange {
public:
    using FillRing = SpscRing<Fill, 8192>;
    using FillCallback = std::function<void(const Fill&)>;

    struct Config {
        double maker_fee    = 0.0002;    // 0.02%
        double taker_fee    = 0.0004;    // 0.04%
        double impact_eta   = 0.1;       // linear impact coefficient
        double adv_usdt     = 5e8;       // 30-day ADV assumption (USDT)
        double slippage_std = 0.0001;    // per-order slippage std (fraction)
    };

    explicit SimExchange(FillRing& fill_ring) noexcept;
    SimExchange(FillRing& fill_ring, Config cfg) noexcept;

    SimExchange(const SimExchange&) = delete;
    SimExchange& operator=(const SimExchange&) = delete;

    /**
     * Submit an order to the exchange.
     * Returns an order ID (monotonically increasing).
     */
    uint32_t submit_order(const Order& order) noexcept;

    /** Register a callback invoked synchronously on fill. */
    void set_fill_callback(FillCallback cb) noexcept { fill_cb_ = std::move(cb); }

    /** Total number of fills generated. */
    [[nodiscard]] uint64_t total_fills() const noexcept {
        return total_fills_.load(std::memory_order_relaxed);
    }

    /** Aggregate filled notional (USDT). */
    [[nodiscard]] double total_notional() const noexcept {
        return total_notional_.load(std::memory_order_relaxed);
    }

private:
    [[nodiscard]] Fill simulate_fill(const Order& order, uint32_t order_id) noexcept;
    [[nodiscard]] double compute_slippage(double qty) const noexcept;
    [[nodiscard]] double compute_commission(double notional, bool is_maker) const noexcept;

    FillRing&    fill_ring_;
    FillCallback fill_cb_;
    Config       cfg_;

    std::atomic<uint32_t> next_order_id_{1};
    std::atomic<uint64_t> total_fills_{0};
    std::atomic<double>   total_notional_{0.0};
};
