#include "arb/order_router.hpp"
#include <chrono>
#include <cmath>

OrderRouter::OrderRouter(
    SignalRing&  signal_ring,
    OrderRing&   order_ring,
    RiskManager& risk,
    SimExchange& exchange
) noexcept
    : signal_ring_(signal_ring)
    , order_ring_(order_ring)
    , risk_(risk)
    , exchange_(exchange)
    , cfg_{}
{}

OrderRouter::OrderRouter(
    SignalRing&  signal_ring,
    OrderRing&   order_ring,
    RiskManager& risk,
    SimExchange& exchange,
    Config       cfg
) noexcept
    : signal_ring_(signal_ring)
    , order_ring_(order_ring)
    , risk_(risk)
    , exchange_(exchange)
    , cfg_(cfg)
{}

void OrderRouter::start() noexcept {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        return;  // already running
    }
    thread_ = std::thread([this]{ run(); });
}

void OrderRouter::stop() noexcept {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void OrderRouter::run() noexcept {
    Signal sig;
    while (running_.load(std::memory_order_acquire)) {
        if (!signal_ring_.pop(sig)) {
            // Spin-wait with a yield to avoid burning the CPU
            std::this_thread::yield();
            continue;
        }

        signals_processed_.fetch_add(1, std::memory_order_relaxed);

        if (!risk_.check_signal(sig)) {
            continue;
        }

        Order ord = build_order(sig);

        if (!cfg_.dry_run) {
            ord.order_id = exchange_.submit_order(ord);
        }

        // Also push to order ring for monitoring / audit
        static_cast<void>(order_ring_.push(ord));
        orders_routed_.fetch_add(1, std::memory_order_relaxed);
    }
}

Order OrderRouter::build_order(const Signal& sig) const noexcept {
    // Convert size_factor [0,1] to USDT notional
    const double qty = sig.size_factor * cfg_.notional_per_unit;

    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    return Order{
        .timestamp_ns = now_ns,
        .pair_id      = sig.pair_id,
        .order_id     = 0,        // set by exchange on submit
        .side         = sig.side,
        .qty          = qty,
        .limit_price  = 0.0,      // market order
    };
}
