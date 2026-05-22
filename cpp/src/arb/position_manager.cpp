#include "arb/position_manager.hpp"
#include <cmath>
#include <chrono>

void PositionManager::on_fill(const Fill& fill) noexcept {
    if (fill.pair_id >= MAX_PAIRS) return;

    PairState& ps = pairs_[fill.pair_id];

    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    ps.last_fill_ns.store(now_ns, std::memory_order_relaxed);
    ps.trade_count.fetch_add(1, std::memory_order_relaxed);

    const double fill_qty     = fill.qty * static_cast<double>(fill.side);
    const double fill_price   = fill.fill_price;
    const double commission   = fill.commission;

    // --- Update position via CAS on net_qty ---
    double old_qty = ps.net_qty.load(std::memory_order_acquire);
    double new_qty;
    do {
        new_qty = old_qty + fill_qty;
    } while (!ps.net_qty.compare_exchange_weak(
                 old_qty, new_qty,
                 std::memory_order_release, std::memory_order_relaxed));

    // --- Update average entry price ---
    // FIFO-style: for additions, blend in fill_price; for reductions, keep entry.
    if ((old_qty >= 0.0 && fill_qty > 0.0) || (old_qty <= 0.0 && fill_qty < 0.0)) {
        // Position increasing in same direction → update VWAP
        double old_price = ps.avg_entry_price.load(std::memory_order_acquire);
        double new_price;
        do {
            const double abs_old = std::abs(old_qty);
            const double abs_new = std::abs(fill_qty);
            const double total   = abs_old + abs_new;
            new_price = (total > 1e-10)
                ? (old_price * abs_old + fill_price * abs_new) / total
                : fill_price;
        } while (!ps.avg_entry_price.compare_exchange_weak(
                     old_price, new_price,
                     std::memory_order_release, std::memory_order_relaxed));
    } else {
        // Position reducing or reversing → book realised P&L
        const double entry_price = ps.avg_entry_price.load(std::memory_order_acquire);
        const double closed_qty  = std::min(std::abs(fill_qty), std::abs(old_qty));
        const double pnl_per_unit = (old_qty > 0.0)
            ? (fill_price - entry_price)
            : (entry_price - fill_price);
        const double realised = pnl_per_unit * closed_qty - commission;

        ps.realised_pnl.fetch_add(realised, std::memory_order_relaxed);

        // If position fully closed or reversed, reset entry price
        if (std::abs(new_qty) < 1e-10) {
            ps.avg_entry_price.store(0.0, std::memory_order_relaxed);
        } else if ((old_qty > 0.0) == (new_qty > 0.0)) {
            // Partial close — entry price unchanged
        } else {
            // Reversal — new entry price is the fill price
            ps.avg_entry_price.store(fill_price, std::memory_order_relaxed);
        }
    }

    // Commission always reduces P&L
    ps.realised_pnl.fetch_add(-commission, std::memory_order_relaxed);
}

void PositionManager::mark_to_market(uint32_t pair_id, double mid_price) noexcept {
    if (pair_id >= MAX_PAIRS) return;

    PairState& ps = pairs_[pair_id];
    const double qty         = ps.net_qty.load(std::memory_order_acquire);
    const double entry_price = ps.avg_entry_price.load(std::memory_order_acquire);

    const double unreal = qty * (mid_price - entry_price);
    ps.unrealised_pnl.store(unreal, std::memory_order_release);
}

double PositionManager::net_qty(uint32_t pair_id) const noexcept {
    if (pair_id >= MAX_PAIRS) return 0.0;
    return pairs_[pair_id].net_qty.load(std::memory_order_acquire);
}

double PositionManager::unrealised_pnl(uint32_t pair_id) const noexcept {
    if (pair_id >= MAX_PAIRS) return 0.0;
    return pairs_[pair_id].unrealised_pnl.load(std::memory_order_acquire);
}

double PositionManager::realised_pnl(uint32_t pair_id) const noexcept {
    if (pair_id >= MAX_PAIRS) return 0.0;
    return pairs_[pair_id].realised_pnl.load(std::memory_order_acquire);
}

double PositionManager::total_pnl() const noexcept {
    double total = 0.0;
    for (std::size_t i = 0; i < MAX_PAIRS; ++i) {
        total += pairs_[i].realised_pnl.load(std::memory_order_relaxed);
        total += pairs_[i].unrealised_pnl.load(std::memory_order_relaxed);
    }
    return total;
}

double PositionManager::total_realised_pnl() const noexcept {
    double total = 0.0;
    for (std::size_t i = 0; i < MAX_PAIRS; ++i) {
        total += pairs_[i].realised_pnl.load(std::memory_order_relaxed);
    }
    return total;
}

void PositionManager::reset() noexcept {
    for (auto& ps : pairs_) {
        ps.net_qty.store(0.0, std::memory_order_relaxed);
        ps.avg_entry_price.store(0.0, std::memory_order_relaxed);
        ps.realised_pnl.store(0.0, std::memory_order_relaxed);
        ps.unrealised_pnl.store(0.0, std::memory_order_relaxed);
        ps.last_fill_ns.store(0, std::memory_order_relaxed);
        ps.trade_count.store(0, std::memory_order_relaxed);
    }
}
