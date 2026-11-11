// A lock-free, single-producer single-consumer (SPSC) ring buffer with cache-line padding
// (alignas(64)) to completely eliminate CPU false sharing.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

namespace lobx {

template <typename T, std::size_t Capacity>
class SpscRingBuffer {
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
  static_assert(Capacity >= 2, "Capacity must be >= 2");
  static constexpr std::size_t kMask = Capacity - 1;

 public:
  SpscRingBuffer() = default;
  ~SpscRingBuffer() {
    while (try_pop().has_value()) {
    }
  }

  SpscRingBuffer(const SpscRingBuffer&) = delete;
  SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

  template <typename... Args>
  bool emplace(Args&&... args) {
    const uint64_t current_head = head_.load(std::memory_order_relaxed);
    const uint64_t current_tail = tail_.load(std::memory_order_acquire);
    if (current_head - current_tail >= Capacity) return false;  // Full

    ::new (static_cast<void*>(&storage_[current_head & kMask])) T(std::forward<Args>(args)...);
    head_.store(current_head + 1, std::memory_order_release);
    return true;
  }

  bool try_push(const T& item) { return emplace(item); }
  bool try_push(T&& item) { return emplace(std::move(item)); }

  std::optional<T> try_pop() {
    const uint64_t current_tail = tail_.load(std::memory_order_relaxed);
    const uint64_t current_head = head_.load(std::memory_order_acquire);
    if (current_tail == current_head) return std::nullopt;  // Empty

    T* slot = reinterpret_cast<T*>(&storage_[current_tail & kMask]);
    T val = std::move(*slot);
    slot->~T();
    tail_.store(current_tail + 1, std::memory_order_release);
    return val;
  }

  [[nodiscard]] constexpr std::size_t capacity() const noexcept { return Capacity; }
  [[nodiscard]] bool empty() const noexcept {
    return tail_.load(std::memory_order_relaxed) == head_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::size_t size() const noexcept {
    const uint64_t head = head_.load(std::memory_order_relaxed);
    const uint64_t tail = tail_.load(std::memory_order_relaxed);
    return (head >= tail) ? (head - tail) : 0;
  }

 private:
  // Separate cache lines for head and tail to eliminate CPU false sharing (64-byte alignment)
  alignas(64) std::atomic<uint64_t> head_{0};
  alignas(64) std::atomic<uint64_t> tail_{0};

  alignas(alignof(T)) std::aligned_storage_t<sizeof(T), alignof(T)> storage_[Capacity];
};

}  // namespace lobx
