#ifndef STD_EXPERIMENTAL_FUTURES_V1_FUTURES_IMPL_H
#define STD_EXPERIMENTAL_FUTURES_V1_FUTURES_IMPL_H

#include <standard_semi_future.h>
#include <standard_future.h>
#include <standard_promise_shared_state.h>

namespace std {
namespace experimental {
inline namespace futures_v1 {

// Specialised on oneway executor.
// This just modifies the standard_future to carry the new executor without
// making use of any special knowledge the executor may have.
template<class T>
template<class NextExecutor>
auto standard_semi_future<T>::via(
    NextExecutor&& exec, typename enable_if<
        experimental::execution::is_oneway_executor_v<NextExecutor> &&
        !experimental::execution::is_then_executor_v<NextExecutor>>::type*) &&
        -> standard_future<T, NextExecutor>{

  return standard_future<T, NextExecutor>{
    std::move(this->core_), std::forward<NextExecutor>(exec)};
}

// Specialised on then_executor.
// This ensures that the returned future is specialised for the executor with
// any necessary synchronization primitives it is aware of.
template<class T>
template<class NextExecutor>
auto standard_semi_future<T>::via(
    NextExecutor&& exec, typename enable_if<
        experimental::execution::is_then_executor_v<NextExecutor>>::type*, int /*unused*/) &&
        -> decltype(std::declval<std::decay_t<NextExecutor>>().then_execute(
          std::declval<HelperF>(),
          std::move(*this))){

  return standard_future<T, NextExecutor>{
    std::move(this->core_), std::forward<NextExecutor>(exec)}.then(
      HelperF{});
}

template<class T>
template<class NextExecutor>
auto standard_semi_future<T>::via_with_executor_promise(
    NextExecutor&& exec,
    typename enable_if<
        experimental::execution::is_then_executor_v<NextExecutor>>::type*,
    int /*unused*/) &&
        -> decltype(std::declval<std::decay_t<NextExecutor>>().then_execute(
          std::declval<HelperF>(),
          std::declval<typename NextExecutor::template Promise<T>>().get_future())) {

  // Get promise/future pair and use that with then_execute for chaining
  auto p = exec.template get_promise<T>();
  auto f = p.get_future();
  auto retFuture = exec.then_execute([](auto&& val){return std::move(val);}, f);
  core_->set_task([p = std::move(p)](auto&& val) mutable {p.set_value(std::move(val));});
  return retFuture;
}


// Specialised on oneway executor.
// This just modifies the standard_future to carry the new executor without
// making use of any special knowledge the executor may have.
template<class T, class Executor>
template<class NextExecutor>
auto standard_future<T, Executor>::via(
    NextExecutor&& exec, typename enable_if<
        experimental::execution::is_oneway_executor_v<NextExecutor> &&
        !experimental::execution::is_then_executor_v<NextExecutor>>::type*) &&
        -> standard_future<T, NextExecutor>{

  return standard_future<T, NextExecutor>{
    std::move(this->core_), std::forward<NextExecutor>(exec)};
}

// Specialised on then_executor.
// This ensures that the returned future is specialised for the executor with
// any necessary synchronization primitives it is aware of.
template<class T, class Executor>
template<class NextExecutor>
auto standard_future<T, Executor>::via(
    NextExecutor&& exec,
    typename enable_if<
        experimental::execution::is_then_executor_v<NextExecutor>>::type*,
    int /*unused*/) &&
        -> decltype(std::declval<std::decay_t<NextExecutor>>().then_execute(
          std::declval<HelperF>(),
          std::move(*this))){

  return standard_future<T, NextExecutor>{
    std::move(this->core_), std::forward<NextExecutor>(exec)}.then(
      HelperF{});
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
