// A clean, robust, and educational C++17 thread pool.
//
// CONCURRENCY DESIGN RATIONALE:
// In quantitative simulation and strategy evaluation, running hundreds of simulation seeds
// serially (e.g. 200 seeds of 5-minute trading sessions) takes minutes of CPU time.
//
// Because each simulation run is a pure function:
//     (config, seed) -> event_log
// with ZERO shared mutable state between distinct seeds, simulation workloads are
// embarrassingly parallel.
//
// 1. ThreadPool Architecture:
//    - A fixed set of N worker threads (typically hardware concurrency).
//    - A task queue guarded by std::mutex.
//    - A std::condition_variable that puts idle worker threads to sleep when the task queue
//      is empty, consuming 0% CPU until new work is submitted.
//    - Tasks return std::future<ReturnType>, allowing callers to non-blockingly collect results.
//
// 2. Determinism Guarantee:
//    Multi-threaded parallel execution across independent seeds MUST produce byte-identical
//    event logs to single-threaded serial runs.
#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace lobx {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency()) {
    if (num_threads == 0) num_threads = 1;
    workers_.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
  }

  // Graceful RAII shutdown: notify all workers to drain the queue and exit
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // Enqueue any callable (lambda, function pointer, functor) with arbitrary arguments.
  // Returns a std::future holding the return value of the callable.
  template <typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::invoke_result<F, Args...>::type> {
    using ReturnType = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<ReturnType> result = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_) {
        throw std::runtime_error("enqueue called on stopped ThreadPool");
      }
      tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return result;
  }

  [[nodiscard]] std::size_t num_workers() const noexcept { return workers_.size(); }

 private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        // Wait until there is a task to execute or the thread pool is shutting down
        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

        if (stop_ && tasks_.empty()) {
          return;
        }

        task = std::move(tasks_.front());
        tasks_.pop();
      }
      // Execute task outside the lock so other workers can dequeue in parallel
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  bool stop_{false};
};

}  // namespace lobx
