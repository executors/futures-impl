// Copyright (c)      2018 NVIDIA Corporation 
//                         (Bryce Adelstein Lelbach <brycelelbach@gmail.com>)
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(FUTURES_EXECUTOR_TRAITS_HPP)
#define FUTURES_EXECUTOR_TRAITS_HPP

#include <utility>

///////////////////////////////////////////////////////////////////////////////

template <typename Executor, typename T>
struct executor_future
{
  using type = typename Executor::template future<T>;
};

template <typename Executor, typename T>
using executor_future_t = typename executor_future<Executor, T>::type;

template <typename Executor, typename T>
struct executor_promise
{
  using type = typename Executor::template promise<T>;
};

template <typename Executor, typename T>
using executor_promise_t = typename executor_promise<Executor, T>::type;

template <typename T, typename Executor>
std::pair<executor_promise_t<Executor, T>, executor_future_t<Executor, T>>
make_promise(Executor exec)
{ 
  return exec.template make_promise<T>();
} 

///////////////////////////////////////////////////////////////////////////////

#endif // FUTURES_EXECUTOR_TRAITS_HPP

