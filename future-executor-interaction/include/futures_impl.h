#ifndef STD_EXPERIMENTAL_FUTURES_V1_FUTURES_IMPL_H
#define STD_EXPERIMENTAL_FUTURES_V1_FUTURES_IMPL_H

#include <standard_semi_future.h>
#include <standard_future.h>
#include <standard_promise_shared_state.h>

namespace std {
namespace experimental {
inline namespace futures_v1 {

template<class T>
template<class Executor>
standard_future<T, Executor> standard_semi_future<T>::via(Executor&& exec) && {
  return standard_future<T, Executor>{
    std::move(core_), std::forward<Executor>(exec)};
}

template<class T, class Executor>
template<class NextExecutor>
auto standard_future<T, Executor>::via(
    NextExecutor exec) && -> standard_future<T, NextExecutor>{
  return standard_future<T, NextExecutor>{
    std::move(core_), std::move(exec)};
}

template<class T, class Executor>
template<class F, class Exec>
auto standard_future<T, Executor>::then(
  F&& continuation,
  typename enable_if<
    experimental::execution::is_oneway_executor_v<Exec>>::type*) &&
  -> standard_future<std::result_of_t<F(T&&)>, Exec> {
  // Prepare next in chain
  auto chainedCore =
    std::make_shared<detail::no_executor_promise_shared_state<T>>();
  auto nextFuture = standard_future<T, Executor>(chainedCore, executor_);

  // Add task
  core_->set_task(
    [chainedCore,
     executor = std::move(executor_),
     continuation = std::move(continuation)](T&& val) mutable {
      executor.execute(
        [chainedCore,
         val = std::move(val),
         continuation = std::move(continuation)]() mutable {
          auto nextVal = continuation(std::move(val));
          chainedCore->set_value(std::move(nextVal));
        });
      });

  return nextFuture;
}

template<class T, class Executor>
template<class F, class Exec>
auto standard_future<T, Executor>::then(
  F&& continuation,
  typename enable_if<
      experimental::execution::is_then_executor_v<Exec>>::type*,
  int /* unused */) &&
  -> decltype(std::declval<Exec>().then_execute(
    std::declval<F>(),
    std::move(*this))) {
  return executor_.then_execute(std::move(continuation), std::move(*this));
}


} // fuures_v1
} // experimental
} // std

#endif // STD_EXPERIMENTAL_FUTURES_V1_FUTURES_IMPL_H
