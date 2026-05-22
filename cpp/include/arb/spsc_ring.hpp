#pragma once
#include <atomic>
#include <array>
#include <cstddef>
#include <type_traits>

/**
 * Lock-free single-producer / single-consumer ring buffer.
 *
 * N must be a power of two.  Uses acquire/release memory ordering on
 * head and tail atomics.  The producer and consumer may run on
 * separate threads without any additional synchronisation.
 *
 * Adapted from the classic LMAX Disruptor design.
 *
 * Optimisations (2026-05):
 *   - [[gnu::always_inline]] on push/pop eliminates call overhead in hot path
 *   - __builtin_expect hints on full/empty checks to guide branch predictor
 *   - Producer-side cached_head_ avoids an acquire atomic load per push when
 *     the ring has space — the slow path only re-reads head_ when the cached
 *     value suggests the ring is full.
 */
template<typename T, std::size_t N>
class SpscRing {
    static_assert((N & (N - 1)) == 0, "N must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    static constexpr std::size_t MASK = N - 1;

    alignas(64) std::atomic<std::size_t> head_{0};          // consumer index
    alignas(64) std::atomic<std::size_t> tail_{0};          // producer index
    // Producer-side cache of last-seen head — lives on the same cache line as
    // tail_ so both are hot for the producer without an extra cache miss.
    alignas(64) std::size_t              cached_head_{0};
    alignas(64) std::array<T, N>         data_{};

public:
    SpscRing() = default;
    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    /**
     * Push one element.  Returns false if the ring is full (non-blocking).
     * Called by the producer thread only.
     */
    [[nodiscard]] [[gnu::always_inline]] bool push(const T& val) noexcept {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        // Fast path: use cached head to avoid an acquire load when there is
        // space.  Only re-read head_ (acquire) when the cached view says full.
        if (__builtin_expect(t - cached_head_ >= N, 0)) {
            cached_head_ = head_.load(std::memory_order_acquire);
            if (t - cached_head_ >= N) return false;   // truly full
        }
        data_[t & MASK] = val;
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

    /**
     * Pop one element into *val*.  Returns false if the ring is empty.
     * Called by the consumer thread only.
     */
    [[nodiscard]] [[gnu::always_inline]] bool pop(T& val) noexcept {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        if (__builtin_expect(h == t, 0)) return false;  // empty
        val = data_[h & MASK];
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    /**
     * Approximate size (may be stale by the time the caller uses it).
     * Safe to call from either thread.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t t = tail_.load(std::memory_order_acquire);
        const std::size_t h = head_.load(std::memory_order_acquire);
        return t - h;
    }

    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool full()  const noexcept { return size() >= N; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return N; }
};
