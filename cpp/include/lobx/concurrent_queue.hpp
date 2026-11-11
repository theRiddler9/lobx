// A lean, thread-safe bounded/unbounded queue using atomics, mutex, and condition variables.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

namespace lobx {

template <typename T>
class ConcurrentQueue {
 public:
  explicit ConcurrentQueue(std::size_t max_capacity = 0) : capacity_(max_capacity) {}
  ~ConcurrentQueue() { close(); }

  ConcurrentQueue(const ConcurrentQueue&) = delete;
  ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

  // Blocking push: waits if bounded queue is full until space is freed or closed.
  bool push(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock,
                   [this] { return closed_.load(std::memory_order_relaxed) || !is_full_locked(); });
    if (closed_.load(std::memory_order_relaxed)) return false;
    queue_.push(std::move(item));
    total_pushed_.fetch_add(1, std::memory_order_relaxed);
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Non-blocking push: returns false if full or closed.
  bool try_push(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (closed_.load(std::memory_order_relaxed) || is_full_locked()) return false;
    queue_.push(std::move(item));
    total_pushed_.fetch_add(1, std::memory_order_relaxed);
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Blocking pop: waits until an item is available or queue is closed.
  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock,
                    [this] { return closed_.load(std::memory_order_relaxed) || !queue_.empty(); });
    return extract_locked(lock);
  }

  // Non-blocking pop.
  std::optional<T> try_pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    return extract_locked(lock);
  }

  // Blocking pop with timeout.
  template <typename Rep, typename Period>
  std::optional<T> wait_pop_for(const std::chrono::duration<Rep, Period>& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!not_empty_.wait_for(lock, timeout, [this] {
          return closed_.load(std::memory_order_relaxed) || !queue_.empty();
        }))
      return std::nullopt;
    return extract_locked(lock);
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_.exchange(true, std::memory_order_release)) return;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  bool is_closed() const noexcept { return closed_.load(std::memory_order_acquire); }
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }
  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }
  std::size_t total_pushed() const noexcept {
    return total_pushed_.load(std::memory_order_relaxed);
  }
  std::size_t total_popped() const noexcept {
    return total_popped_.load(std::memory_order_relaxed);
  }

 private:
  bool is_full_locked() const noexcept { return capacity_ > 0 && queue_.size() >= capacity_; }

  std::optional<T> extract_locked(std::unique_lock<std::mutex>& lock) {
    if (queue_.empty()) return std::nullopt;
    T val = std::move(queue_.front());
    queue_.pop();
    total_popped_.fetch_add(1, std::memory_order_relaxed);
    lock.unlock();
    not_full_.notify_one();
    return val;
  }

  const std::size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::queue<T> queue_;
  std::atomic<bool> closed_{false};
  std::atomic<std::size_t> total_pushed_{0};
  std::atomic<std::size_t> total_popped_{0};
};

}  // namespace lobx
