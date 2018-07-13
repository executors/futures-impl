// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(FUTURES_CUDA_FUTURE_NEW_HPP)
#define FUTURES_CUDA_FUTURE_NEW_HPP

#include "executor_traits.hpp"
#include "type_deduction.hpp"

#include "cuda_error_handling.hpp"
#include "cuda_stream.hpp"
#include "cuda_memory.hpp"

#include <atomic>

#include <cassert>

///////////////////////////////////////////////////////////////////////////////

// Lexicon: Task == Promise == Receiver

template <typename Function>
struct promise_t
{
  Function function_;

  __host__ __device__
  void operator()() &&
  {
    MV(function_)();
  }
};

template <typename Function>
auto promise(Function&& function)
{
  return promise_t<Function>{FWD(function)};
}

template <
  typename Executor
, typename ElementFunction
, typename Shape
, typename SharedFactory
>
struct bulk_promise_t
{
  Executor        executor_;
  ElementFunction element_function_;
  Shape           shape_;
  SharedFactory   shared_factory_;

  void operator()() &&
  {
    auto s = MV(shared_factory_)();

    auto element_function = MV(element_function_);

    for (auto i : MV(shape_))
      executor_.execute(
        promise([element_function, MVCAP(i), s]{ element_function(i, s); })
      );
  }

  ElementFunction&       element_function()       RETURNS(element_function_);
  ElementFunction const& element_function() const RETURNS(element_function_);

  Shape&       shape()       RETURNS(shape_);
  Shape const& shape() const RETURNS(shape_);

  SharedFactory&       shared_factory()       RETURNS(shared_factory_);
  SharedFactory const& shared_factory() const RETURNS(shared_factory_);
};

template <typename Executor, typename ElementFunction, typename Shape, typename SharedFactory>
auto bulk_promise(
  Executor&&        executor
, ElementFunction&& element_function
, Shape&&           shape
, SharedFactory&&   shared_factory
  )
{
  return bulk_promise_t<Executor, ElementFunction, Shape, SharedFactory>{
    FWD(executor), FWD(element_function), FWD(shape), FWD(shared_factory)
  };
}

///////////////////////////////////////////////////////////////////////////////

struct inline_executor final
{
  template <typename Promise>
  void execute(Promise&& p)
  {
    FWD(p)();
  }
};

template <typename T, typename NextExecutor>
struct ready_future final
{
private:
  T            content_;
  NextExecutor next_executor_;

  ready_future(T&& content, NextExecutor next_executor)
    : content_(MV(content)), next_executor_(MV(next_executor))
  {}

public:
  template <typename UExecutor>
  auto via(UExecutor&& uexec)
  AUTOQUALRETURNS(&&, ready_future<T, UExecutor>(MV(content_), FWD(uexec)));

  template <typename ValuePromise>
  void submit(ValuePromise&& p) &&
  {
    MV(next_executor_).execute(promise(
      [=] __host__ __device__ ()
      {
        FWD(p)(MV(content_));
      }
    ));
  }
};

template <typename T>
struct ready_semi_future final
{
private:
  T content_;

public:
  ready_semi_future(T&& content)
    : content_(MV(content))
  {}

  ready_semi_future(T const& content)
    : content_(content)
  {}

  template <typename UExecutor>
  auto via(UExecutor&& uexec)
  AUTOQUALRETURNS(&&, ready_future<T, UExecutor>(MV(content_), FWD(uexec)));
};

template <typename T>
auto ready(T&& content) AUTORETURNS(ready_semi_future<T>(FWD(content)));

///////////////////////////////////////////////////////////////////////////////

template <typename Promise>
__global__
void execute_impl(Promise p)
{
  MV(p)();
}

struct cuda_executor final
{
  cuda_stream stream_;

  // Internal -> Internal Dependent Execution.
  template <typename F>
  void execute(promise_t<F>&& p)
  {
    execute_impl<<<1, 1, 0, stream_.get()>>>(MV(p));
    THROW_ON_CUDA_RT_ERROR(cudaGetLastError());
  }

  template <typename ElementFunction, typename Shape, typename SharedFactory>
  void execute(bulk_promise_t<cuda_executor, ElementFunction, Shape, SharedFactory>&& p)
  {
    int block_size;
    int min_grid_size;

    auto shared_state = MV(p.shared_factory())();

    auto element_function = MV(p.element_function());

    auto element_promise =
      [=] __device__ ()
      {
        int const idx = blockIdx.x * blockDim.x + threadIdx.x;
        element_function(idx, shared_state);
      };

    THROW_ON_CUDA_RT_ERROR(
      cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, execute_impl<decltype(element_promise)>, 0, 0
      )
    );

    // Round up according to array size.
    int grid_size = (p.shape().size() + block_size - 1) / block_size;

    execute_impl<<<grid_size, block_size, 0, stream_.get()>>>(MV(element_promise));
    THROW_ON_CUDA_RT_ERROR(cudaGetLastError());
  }

  void wait()
  {
    THROW_ON_CUDA_RT_ERROR(cudaStreamSynchronize(stream_.get()));
  }
};

///////////////////////////////////////////////////////////////////////////////

// Internal -> External Dependent Execution
template <typename T, typename NextExecutor>
struct cuda_future final
{
private:
  cuda_managed_unique_ptr<T> content_;
  cuda_executor              this_executor_;
  NextExecutor               next_executor_;

  template <typename UContent, typename UThisExecutor, typename UNextExecutor>
  cuda_future(UContent&& c, UThisExecutor&& te, UNextExecutor&& ne)
    : content_(FWD(c)), this_executor_(FWD(te)), next_executor_(FWD(ne))
  {}

public:
  template <typename UExecutor>
  auto via(UExecutor&& uexec)
  AUTOQUALRETURNS(&&, cuda_future<T, UExecutor>(MV(content_), FWD(uexec)));

  template <typename ValuePromise>
  void submit(ValuePromise&& p) &&
  {
    auto outer_tmp = std::make_unique<
      std::tuple<NextExecutor, ValuePromise, cuda_managed_unique_ptr<T>>
    >(
      MV(next_executor_), FWD(p), MV(content_)
    );

    THROW_ON_CUDA_RT_ERROR(
      cudaStreamAddCallback(
        this_executor_.stream_.get(),
        [] (CUstream_st*, cudaError_t, void* tmp_ptr)
        {
          using tmp_type         = decltype(outer_tmp);
          using tmp_element_type = typename tmp_type::element_type;
          tmp_type inner_tmp(reinterpret_cast<tmp_element_type*>(tmp_ptr));

          MV(std::get<0>(*inner_tmp)).execute(promise(
            [MVCAP(inner_tmp)]
            {
              MV(std::get<1>(*inner_tmp))(MV(*std::get<2>(*inner_tmp)));
            }
          ));
        },
        outer_tmp.get(),
        0
      )
    );

    std::get<2>(outer_tmp).release(); // Release content.
    outer_tmp.release();
  }
};

// Internal -> Internal Dependent Execution
template <typename T>
struct cuda_future<T, cuda_executor> final
{
private:
  cuda_managed_unique_ptr<T> content_;
  cuda_executor              this_executor_;
  cuda_executor              next_executor_;

  template <typename UContent, typename UThisExecutor, typename UNextExecutor>
  cuda_future(UContent&& c, UThisExecutor&& te, UNextExecutor&& ne)
    : content_(FWD(c)), this_executor_(FWD(te)), next_executor_(FWD(ne))
  {}

public:
  template <typename UExecutor>
  auto via(UExecutor&& uexec)
  AUTOQUALRETURNS(&&, cuda_future<T, UExecutor>(MV(content_), FWD(uexec)));

  template <typename ValuePromise>
  void submit(ValuePromise&& p) &&
  {
    // TODO: Multi-device.
    // TODO: Keep alive.

    auto* raw_content = content_.get();

    MV(next_executor_).execute(promise(
      [=] __host__ __device__ ()
      {
        FWD(p)(MV(*raw_content));
      }
    ));
  }
};

#endif // FUTURES_CUDA_FUTURE_NEW_HPP

