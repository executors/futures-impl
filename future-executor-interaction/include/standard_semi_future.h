#ifndef STD_EXPERIMENTAL_FUTURES_V1_STANDARD_SEMIFUTURE_H
#define STD_EXPERIMENTAL_FUTURES_V1_STANDARD_SEMIFUTURE_H

#include <future_completion_token.h>
#include <standard_promise_shared_state.h>
#include <experimental/execution>
#include <executors_helpers.h>

#include <chrono>

namespace std {
namespace experimental {
inline namespace futures_v1 {

template<class T, class Executor>
class standard_future;

template<class T>
class standard_semi_future {
private:
  struct HelperF {
    T operator()(T val) { return val; }
  };

public:
  using value_type = T;

  standard_semi_future(const standard_semi_future&) = default;
  standard_semi_future(standard_semi_future&&) = default;
  standard_semi_future& operator=(const standard_semi_future&) = default;
  standard_semi_future& operator=(standard_semi_future&&) = default;

  void wait() {
    detail::CVStruct cv;

    // Callback does not consume the value from this core so try_get will still
    // be valid
    core_->set_task([&cv](T&& /* not used */) mutable {
      cv.ready = true;
      cv.cv.notify_one();
    });

    std::unique_lock<std::mutex> lck{cv.cv_mutex};
    cv.cv.wait(lck, [&cv](){return cv.ready == true;});
  }

  template<class Clock, class Duration>
  void wait_until(const chrono::time_point<Clock, Duration>& abs_time) {
    // condition variable heap allocated because we will return from this scope
    // if waiting on CV times out and it must stay alive to be satisfied later
    // by already enqueued callback.
    auto cv = std::make_shared<detail::CVStruct>();

    // Create a new future in the chain that will be in a callback free state
    // again.
    auto chainedCore =
      std::make_shared<detail::no_executor_promise_shared_state<T>>();
    auto nextSemiFuture = standard_semi_future<T>(chainedCore);

    // Callback does not consume the value from this core so try_get will still
    // be valid
    core_->set_task(
      [cv, chainedCore = std::move(chainedCore)](T&& val) mutable {
        cv->ready = true;
        cv->cv.notify_one();
        chainedCore->set_value(std::move(val));
      });

    std::unique_lock<std::mutex> lck{cv->cv_mutex};
    cv->cv.wait_until(lck, abs_time, [cv](){return cv->ready == true;});

    // Replace this future with sf so that timeout still gives us a valid state
    *this = std::move(nextSemiFuture);
  }

  T get() {
    // Signalling pattern for shared state is owned by future
    if(auto* value = core_->try_get()) {
      // If the core has already completed, then there should be a value
      return std::move(*value);
    }
    // Otherwise wait and return value
    wait();
    auto* value = core_->try_get();
    return std::move(*value);
  }

  // Allow via to extract future type from then_executor
  //
  // If the executors are the same, the future type will not change so there
  // is no need to enqueue the cost (and recursion risk) of calling then
  // under the hood.
  // If the executor is one-way, then the future type cannot change.
  template<class NextExecutor>
  auto via(
    NextExecutor&& exec,
    typename enable_if<
        experimental::execution::is_oneway_executor_v<NextExecutor> &&
        !experimental::execution::is_then_executor_v<NextExecutor>>::type* = 0
    ) && -> standard_future<T, NextExecutor>;

  // Allow via to extract future type from then_executor
  //
  // If the executor types are different and the executor is a then_executor,
  // the future type might change.
  template<class NextExecutor>
  auto via(
    NextExecutor&& exec,
    typename enable_if<
        experimental::execution::is_then_executor_v<NextExecutor>>::type* = 0,
      int a = 0) &&
      -> decltype(std::declval<std::decay_t<NextExecutor>>().then_execute(
        std::declval<HelperF>(),
        std::move(*this)));

  // Allow via to extract future type from then_executor
  //
  // If the executor types are different and the executor is a then_executor,
  // the future type might change.
  // Version of via that uses an executor with attached promise
  template<class NextExecutor>
  auto via_with_executor_promise(
    NextExecutor&& exec,
    typename enable_if<
        experimental::execution::is_then_executor_v<NextExecutor>>::type* = 0,
      int a = 0) &&
      -> decltype(std::declval<std::decay_t<NextExecutor>>().then_execute(
        std::declval<HelperF>(),
        std::declval<typename NextExecutor::template Promise<T>>().get_future()));


  // Should be called only by executor implementations
  // Callback should perform only trivial work to let the executor know
  // how to proceed.
  void set_callback(
      std::experimental::execution::future_completion_token<T>&& token){
    core_->set_task(std::move(token));
  }

private:
  template<class PromiseType>
  friend class standard_promise;
  template<class FutureType, class ExecutorType>
  friend class standard_future;

  standard_semi_future() = delete;
  standard_semi_future(std::shared_ptr<detail::promise_shared_state<T>> core) :
      core_{std::move(core)} {
  }

  std::shared_ptr<detail::promise_shared_state<T>> core_;
};

} // fuures_v1
} // experimental
} // std

#endif // STD_EXPERIMENTAL_FUTURES_V1_STANDARD_SEMIFUTURE_H
