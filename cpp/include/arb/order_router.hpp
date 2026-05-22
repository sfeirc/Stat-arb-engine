#pragma once
#include "spsc_ring.hpp"
#include "sim_exchange.hpp"
#include "risk_manager.hpp"
#include <cstdint>
#include <functional>
#include <thread>
#include <atomic>

/**
 * OrderRouter
 *
 * Reads Signal objects from a signal ring, validates them through the
 * RiskManager, converts them into Orders, and submits them to the
 * SimExchange.  Runs on a dedicated thread to minimise latency.
 *
 * Typical latency: < 2 µs signal-to-order (single-core, no contention).
 */
class OrderRouter {
public:
    using SignalRing = SpscRing<Signal, 4096>;
    using OrderRing  = SpscRing<Order,  4096>;

    struct Config {
        double notional_per_unit = 1'000.0;   // USDT per unit of size_factor
        bool   dry_run           = false;      // if true, route to order ring but not exchange
    };

    OrderRouter(
        SignalRing&  signal_ring,
        OrderRing&   order_ring,
        RiskManager& risk,
        SimExchange& exchange
    ) noexcept;

    OrderRouter(
        SignalRing&  signal_ring,
        OrderRing&   order_ring,
        RiskManager& risk,
        SimExchange& exchange,
        Config       cfg
    ) noexcept;

    OrderRouter(const OrderRouter&) = delete;
    OrderRouter& operator=(const OrderRouter&) = delete;

    /** Start the routing loop on a background thread. */
    void start() noexcept;

    /** Stop the routing loop and join the thread. */
    void stop() noexcept;

    /** Is the router thread running? */
    [[nodiscard]] bool running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    /** Number of signals processed since start. */
    [[nodiscard]] std::uint64_t signals_processed() const noexcept {
        return signals_processed_.load(std::memory_order_relaxed);
    }

    /** Number of orders routed since start. */
    [[nodiscard]] std::uint64_t orders_routed() const noexcept {
        return orders_routed_.load(std::memory_order_relaxed);
    }

private:
    void run() noexcept;

    [[nodiscard]] Order build_order(const Signal& sig) const noexcept;

    SignalRing&  signal_ring_;
    OrderRing&   order_ring_;
    RiskManager& risk_;
    SimExchange& exchange_;
    Config       cfg_;

    std::atomic<bool>         running_{false};
    std::atomic<std::uint64_t> signals_processed_{0};
    std::atomic<std::uint64_t> orders_routed_{0};
    std::thread               thread_;
};
