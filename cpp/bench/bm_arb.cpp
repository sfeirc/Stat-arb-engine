/**
 * bm_arb.cpp
 *
 * Google Benchmark microbenchmarks for the statistical arbitrage engine.
 *
 * Benchmarks:
 *   BM_SignalToOrder            — full signal → risk → order, RDTSC-timed
 *   BM_SignalToOrder_Throughput — sustained batch throughput
 *   BM_SpscRingPushPop          — SPSC ring push+pop baseline
 *   BM_RiskCheckOnly            — isolated risk::check_signal() cost
 *   BM_FillToPosition           — SimExchange → PositionManager fill chain
 */
#include <benchmark/benchmark.h>
#include "arb/spsc_ring.hpp"
#include "arb/risk_manager.hpp"
#include "arb/sim_exchange.hpp"
#include "arb/position_manager.hpp"
#include "arb/order_router.hpp"
#include <chrono>
#include <cstdint>

// ---------------------------------------------------------------------------
// RDTSC helpers
// ---------------------------------------------------------------------------

static inline uint64_t rdtsc() noexcept {
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi) :: "memory");
    return (uint64_t(hi) << 32) | lo;
}

// Calibrate TSC: measure ticks over a 10 ms steady-clock interval and return
// nanoseconds per tick.  Called once at static-init time.
static double tsc_ns_per_tick() {
    using clk = std::chrono::steady_clock;
    auto   t0 = clk::now();
    uint64_t c0 = rdtsc();
    while (std::chrono::duration<double>(clk::now() - t0).count() < 0.01) {}
    uint64_t c1 = rdtsc();
    auto   t1 = clk::now();
    const double ns = std::chrono::duration<double>(t1 - t0).count() * 1e9;
    return ns / static_cast<double>(c1 - c0);
}

static const double NS_PER_TICK = tsc_ns_per_tick();

// ---------------------------------------------------------------------------
// BM_SignalToOrder: end-to-end signal → risk check → order construction
// Timed with RDTSC to eliminate ~40-80 ns chrono::now() overhead per iter.
// ---------------------------------------------------------------------------

static void BM_SignalToOrder(benchmark::State& state) {
    SpscRing<Signal, 4096> signal_ring;
    SpscRing<Order,  4096> order_ring;
    RiskManager risk;
    Signal sig{0, 1, 1, 2.5, 0.8};
    Order dummy;

    for (auto _ : state) {
        const uint64_t t0 = rdtsc();

        static_cast<void>(signal_ring.push(sig));
        Signal recv;
        if (signal_ring.pop(recv)) {
            if (risk.check_signal(recv)) {
                Order ord{recv.timestamp_ns, recv.pair_id, 0,
                          recv.side, recv.size_factor * 1000.0, 0.0};
                static_cast<void>(order_ring.push(ord));
            }
        }
        static_cast<void>(order_ring.pop(dummy));

        const uint64_t t1 = rdtsc();
        state.SetIterationTime(
            static_cast<double>(t1 - t0) * NS_PER_TICK * 1e-9
        );
    }

    state.SetLabel("signal->risk->order (RDTSC)");
    benchmark::DoNotOptimize(dummy);
}

BENCHMARK(BM_SignalToOrder)
    ->Unit(benchmark::kNanosecond)
    ->UseManualTime()
    ->Iterations(10'000'000);


// ---------------------------------------------------------------------------
// BM_SpscRingPushPop: raw throughput of the SPSC ring
// ---------------------------------------------------------------------------

static void BM_SpscRingPushPop(benchmark::State& state) {
    SpscRing<Signal, 4096> ring;
    Signal sig{0, 1, 1, 2.5, 0.8};

    for (auto _ : state) {
        static_cast<void>(ring.push(sig));
        Signal out;
        static_cast<void>(ring.pop(out));
        benchmark::DoNotOptimize(out);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) * sizeof(Signal) * 2
    );
}

BENCHMARK(BM_SpscRingPushPop)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(5'000'000);


// ---------------------------------------------------------------------------
// BM_RiskCheckOnly: isolated cost of check_signal()
// ---------------------------------------------------------------------------

static void BM_RiskCheckOnly(benchmark::State& state) {
    RiskManager risk;
    risk.set_nav(1'000'000.0);

    Signal sig{0, 1, 1, 2.5, 0.8};
    bool result = false;

    for (auto _ : state) {
        result = risk.check_signal(sig);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_RiskCheckOnly)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(10'000'000);


// ---------------------------------------------------------------------------
// BM_FillToPosition: SimExchange submit + PositionManager on_fill chain
// ---------------------------------------------------------------------------

static void BM_FillToPosition(benchmark::State& state) {
    SpscRing<Fill, 8192> fill_ring;
    SimExchange exchange(fill_ring);
    PositionManager pos_mgr;

    // Wire fill callback to position manager
    exchange.set_fill_callback([&](const Fill& fill) {
        pos_mgr.on_fill(fill);
    });

    Order order{
        .timestamp_ns = 0,
        .pair_id      = 42,
        .order_id     = 0,
        .side         = 1,
        .qty          = 10'000.0,
        .limit_price  = 0.0,
    };

    for (auto _ : state) {
        uint32_t oid = exchange.submit_order(order);
        benchmark::DoNotOptimize(oid);

        // Also exercise mark-to-market
        pos_mgr.mark_to_market(42, 50'000.0);
        benchmark::DoNotOptimize(pos_mgr.total_pnl());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

BENCHMARK(BM_FillToPosition)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500'000);


// ---------------------------------------------------------------------------
// BM_SignalToOrder_Throughput: sustained batch throughput (items/sec)
// Fills the ring with BATCH signals, drains everything, measures items/s.
// ---------------------------------------------------------------------------

static void BM_SignalToOrder_Throughput(benchmark::State& state) {
    SpscRing<Signal, 65536> sig_ring;
    SpscRing<Order,  65536> ord_ring;
    RiskManager risk;
    Signal sig{0, 1, 1, 2.5, 0.8};

    constexpr int BATCH = 1000;
    for (auto _ : state) {
        for (int i = 0; i < BATCH; ++i) static_cast<void>(sig_ring.push(sig));
        int processed = 0;
        Signal recv;
        Order ord;
        while (sig_ring.pop(recv)) {
            if (risk.check_signal(recv)) {
                ord = {0, recv.pair_id, 0, recv.side,
                       recv.size_factor * 1000.0, 0.0};
                static_cast<void>(ord_ring.push(ord));
                ++processed;
            }
        }
        while (ord_ring.pop(ord)) {}
        benchmark::DoNotOptimize(processed);
    }
    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * BATCH
    );
}

BENCHMARK(BM_SignalToOrder_Throughput)->Unit(benchmark::kNanosecond);


BENCHMARK_MAIN();
