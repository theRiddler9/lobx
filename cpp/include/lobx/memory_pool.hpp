// A zero-allocation fixed-size memory pool (slab allocator) with an intrusive freelist.
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace lobx {

template <typename T, std::size_t Capacity>
class FixedPool {
 public:
  static constexpr uint32_t kNullIndex = 0xFFFFFFFF;

  FixedPool() {
    for (uint32_t i = 0; i < Capacity - 1; ++i) {
      slots_[i].next_free = i + 1;
    }
    slots_[Capacity - 1].next_free = kNullIndex;
  }

  ~FixedPool() = default;

  FixedPool(const FixedPool&) = delete;
  FixedPool& operator=(const FixedPool&) = delete;

  template <typename... Args>
  T* allocate(Args&&... args) {
    if (free_head_ == kNullIndex) return nullptr;
    uint32_t idx = free_head_;
    free_head_ = slots_[idx].next_free;
    ++active_count_;
    return ::new (static_cast<void*>(&slots_[idx].storage)) T(std::forward<Args>(args)...);
  }

  void deallocate(T* ptr) noexcept {
    if (!ptr) return;
    ptr->~T();
    auto* slot = reinterpret_cast<Slot*>(ptr);
    uint32_t idx = static_cast<uint32_t>(slot - slots_);
    slot->next_free = free_head_;
    free_head_ = idx;
    --active_count_;
  }

  [[nodiscard]] constexpr std::size_t capacity() const noexcept { return Capacity; }
  [[nodiscard]] std::size_t active_count() const noexcept { return active_count_; }

 private:
  union Slot {
    alignas(alignof(T)) std::byte storage[sizeof(T)];
    uint32_t next_free;
  };

  Slot slots_[Capacity];
  uint32_t free_head_{0};
  std::size_t active_count_{0};
};

}  // namespace lobx
