#include "arb/sim_exchange.hpp"
#include <chrono>
#include <cmath>
#include <random>

// Thread-local RNG to avoid contention
namespace {
thread_local std::mt19937_64 tl_rng{std::random_device{}()};
}  // namespace

SimExchange::SimExchange(FillRing& fill_ring) noexcept
    : fill_ring_(fill_ring), cfg_{} {}

SimExchange::SimExchange(FillRing& fill_ring, Config cfg) noexcept
    : fill_ring_(fill_ring), cfg_(cfg) {}

uint32_t SimExchange::submit_order(const Order& order) noexcept {
    const uint32_t oid = next_order_id_.fetch_add(1, std::memory_order_relaxed);

    Fill fill = simulate_fill(order, oid);

    // Dispatch to callback if registered
    if (fill_cb_) {
        fill_cb_(fill);
    }

    // Push to fill ring (best-effort; drop if full)
    static_cast<void>(fill_ring_.push(fill));

    total_fills_.fetch_add(1, std::memory_order_relaxed);

    // Compute notional: qty is already in USDT (our convention)
    const double notional = std::abs(fill.qty);
    // Atomic double add via CAS loop
    double prev = total_notional_.load(std::memory_order_relaxed);
    while (!total_notional_.compare_exchange_weak(
               prev, prev + notional,
               std::memory_order_relaxed, std::memory_order_relaxed)) {
        /* spin */
    }

    return oid;
}

Fill SimExchange::simulate_fill(const Order& order, uint32_t order_id) noexcept {
    // Slippage: Gaussian noise proportional to order size
    const double slippage = compute_slippage(order.qty);

    // Fill price: for market orders, use slippage as the realised price impact
    // (in the simulation, qty is denominated in USDT notional, so fill_price
    //  is the effective slippage fraction applied to the nominal price)
    const double fill_price = (order.limit_price > 0.0)
        ? order.limit_price
        : 1.0 + static_cast<double>(order.side) * slippage;

    const double notional   = std::abs(order.qty);
    const double commission = compute_commission(notional, order.limit_price > 0.0);

    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    return Fill{
        .timestamp_ns = now_ns,
        .pair_id      = order.pair_id,
        .order_id     = order_id,
        .side         = order.side,
        .qty          = order.qty,
        .fill_price   = fill_price,
        .commission   = commission,
    };
}

double SimExchange::compute_slippage(double qty) const noexcept {
    // Linear Almgren-Chriss impact + Gaussian noise
    const double size_frac = std::abs(qty) / (cfg_.adv_usdt + 1e-10);
    const double linear_impact = cfg_.impact_eta * size_frac;

    std::normal_distribution<double> noise_dist(0.0, cfg_.slippage_std);
    const double noise = std::abs(noise_dist(tl_rng));

    // Total slippage capped at 50 bps
    return std::min(linear_impact + noise, 0.005);
}

double SimExchange::compute_commission(double notional, bool is_maker) const noexcept {
    const double rate = is_maker ? cfg_.maker_fee : cfg_.taker_fee;
    return notional * rate;
}
