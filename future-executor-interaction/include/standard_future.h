#ifndef STD_EXPERIMENTAL_FUTURES_V1_STANDARD_FUTURE_H
#define STD_EXPERIMENTAL_FUTURES_V1_STANDARD_FUTURE_H

#include <future_completion_token.h>
#include <executors_helpers.h>
#include <experimental/execution>

#include <chrono>
#include <type_traits>

namespace std {
namespace experimental {
inline namespace futures_v1 {

namespace detail {
template<typename ResultT>
struct generatedHelper {
  std::decay_t<ResultT> operator()();
};

template<class F, class ParamT>
struct unary_function {
  result_of_t<F(ParamT&&)> operator()(ParamT&&);
};
}

template<class T, class Executor>
class standard_future{
private:
  struct HelperF {
    T operator()(T val) { return val; }
  };

public:
  using value_type = T;
  using executor_type = Executor;

  standard_future(const standard_future&) = default;
  standard_future(standard_future&&) = default;
  standard_future& operator=(const standard_future&) = default;
  standard_future& operator=(standard_future&&) = default;

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

  // Then with one_way executor
  template<class F, class Exec = Executor>
  auto then(
    F&& continuation,
    typename enable_if<
      experimental::execution::is_oneway_executor_v<Exec>>::type* = 0) &&
    -> standard_future<std::result_of_t<F(T&&)>, Exec>;

  // Then with then_executor
  template<class F, class Exec = Executor>
  auto then(
    F&& continuation,
    typename enable_if<
        experimental::execution::is_then_executor_v<Exec>>::type* = 0,
    int a = 0) &&
    -> decltype(std::declval<Exec>().then_execute(
      std::declval<F>(),
      std::move(*this)));

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
  template<class SemiFutureType>
  friend class standard_semi_future;
  template<class FutureType, class ExecutorType>
  friend class standard_future;

  standard_future() = delete;
  standard_future(
      std::shared_ptr<detail::promise_shared_state<T>> core,
      Executor ex) :
      core_{std::move(core)},
      executor_{std::move(ex)} {
  }






  std::shared_ptr<detail::promise_shared_state<T>> core_;
  Executor executor_;
};

} // fuures_v1
} // experimental
} // std

#endif // STD_EXPERIMENTAL_FUTURES_V1_STANDARD_FUTURE_H
