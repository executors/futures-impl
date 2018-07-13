// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(FUTURES_CUDA_FUTURE_HPP)
#define FUTURES_CUDA_FUTURE_HPP

#include "executor_traits.hpp"
#include "type_deduction.hpp"

#include "cuda_error_handling.hpp"
#include "cuda_stream.hpp"
#include "cuda_memory.hpp"

#include <atomic>

#include <cassert>

template <typename T>
struct cuda_async_value;

template <typename T, typename Executor>
struct cuda_executor_future;

template <typename T>
struct cuda_executor_promise;

struct cuda_executor;

///////////////////////////////////////////////////////////////////////////////

template <typename Operation>
__global__
void launch_impl(Operation op)
{
  MV(op)();
}

struct cuda_executor final
{
  template <typename T>
  using future = cuda_executor_future<T, cuda_executor>;

  template <typename T>
  using promise = cuda_executor_promise<T>;

private:
  template <typename Operation>
  void launch(Operation&& op, cudaStream_t stream)
  {
    launch_impl<<<1, 1, 0, stream>>>(op);
    THROW_ON_CUDA_RT_ERROR(cudaGetLastError());
  }

public:
  template <typename Operation>
  void execute(Operation&& op)
  {
    launch(FWD(op), nullptr);
  }

  template <typename Operation>
  cuda_executor_future<RETOF(Operation), cuda_executor>
  twoway_execute(Operation&& op)
  {
    using U = RETOF(Operation);

    // Create a future with a new stream.
    auto ss = std::make_shared<cuda_async_value<U>>();
    cuda_executor_future<U, cuda_executor> f(ss, *this);

    auto data = ss->data();

    launch([=] __device__ { *data = op(); }, ss->stream());

    return MV(f);
  }

  // Internal -> Internal Dependent Execution.
  template <typename Operation, typename T>
  cuda_executor_future<RETOF(Operation, T), cuda_executor>
  then_execute(Operation&& op, cuda_executor_future<T, cuda_executor>&& f)
  {
    using U = RETOF(Operation, T);

    auto&& fss = MV(f.shared_state());

    // Get a copy of the pointer to the data, because we're about to move from
    // `fss`.
    auto fdata = fss->data();

    // Create a new future that uses the same stream as `f` and keeps the
    // shared state of `f` alive.
    auto ss = std::make_shared<cuda_async_value<U>>(
      MV(fss->shared_stream()), MV(fss)
    );
    cuda_executor_future<U, cuda_executor> g(MV(ss), MV(f.executor()));

    auto const& gss = g.shared_state();

    auto gdata = gss->data();

    launch([=] __device__ { *gdata = op(*fdata); }, gss->stream());
    THROW_ON_CUDA_RT_ERROR(cudaGetLastError());

    return MV(g);
  }

  // Internal -> External Dependent Execution.
  template <typename Operation, typename T, typename Executor>
  cuda_executor_future<RETOF(Operation, T), Executor>
  then_execute(Operation&& op, cuda_executor_future<T, Executor>&& f)
  {
    using U = RETOF(Operation, T);

    auto p_g = make_promise<U>(f.executor());
    auto& p  = p_g.first;
    auto& g  = p_g.second;

    auto h = f.executor().then_execute(FWD(op), MV(g));

    auto const& fss = f.shared_state();
    THROW_ON_CUDA_RT_ERROR(
      cudaStreamAddCallback(
        fss->stream(),
        [fss = MV(fss), p = MV(p)] { MV(p).set_value(MV(fss->data())); }
      )
    );

    return MV(h);
  }

  template <typename T>
  std::pair<cuda_executor_promise<T>, cuda_executor_future<T, cuda_executor>>
  make_promise()
  {
    auto ss = std::make_shared<cuda_async_value<T>>(
      cuda_async_value<T>::make_semaphore()
    );
    cuda_executor_future<T, cuda_executor> f(ss, *this);
    cuda_executor_promise<T> p(ss);

    auto const& fss = f.shared_state();
    THROW_ON_CUDA_DRV_ERROR(
      cuStreamWaitValue32(
        fss->stream(), fss->semaphore(), true, CU_STREAM_WAIT_VALUE_EQ
      )
    );

    return {MV(p), MV(f)};
  }

  template <typename T, typename Executor>
  void wait(cuda_executor_future<T, Executor>&& f)
  {
    auto const& fss = f.shared_state();
    THROW_ON_CUDA_RT_ERROR(cudaStreamSynchronize(fss->stream()));
  }
};

///////////////////////////////////////////////////////////////////////////////

struct cuda_async_value_base
{
  using semaphore_type
    = cuda_host_pinned_unique_ptr<std::atomic<std::uint32_t> volatile>;

  using keep_alive_type = std::shared_ptr<cuda_async_value_base>;

protected:
  cuda_stream     stream_;
  keep_alive_type keep_alive_;
  semaphore_type  semaphore_;

public:
  // Constructs a `cuda_async_value_base` with a new stream.
  cuda_async_value_base()
    : stream_(), keep_alive_(), semaphore_()
  {}

  // Constructs a `cuda_async_value_base` that takes ownership of its
  // predecessor's stream and holds a reference to the predecessor's shared
  // state to keep it alive.
  cuda_async_value_base(cuda_stream&& stream, keep_alive_type&& keep_alive)
    : stream_(MV(stream)), keep_alive_(MV(keep_alive)), semaphore_()
  {}

  // Constructs a `cuda_async_value_base` with a new stream and a new semaphore.
  // Remarks: Only used by `make_promise` for external dependencies.
  cuda_async_value_base(semaphore_type&& semaphore)
    : stream_(), keep_alive_(), semaphore_(MV(semaphore))
  {}

  virtual ~cuda_async_value_base() {}

  static semaphore_type make_semaphore()
  {
    return cuda_make_host_pinned_unique<typename semaphore_type::element_type>(0);
  }

  CUstream_st* stream() const
  {
    return stream_.get();
  }

  cuda_stream& shared_stream()
  {
    return stream_;
  }

  cuda_stream const& shared_stream() const
  {
    return stream_;
  }

  CUdeviceptr semaphore() const
  {
    return reinterpret_cast<CUdeviceptr>(semaphore_.get());
  }
};

template <typename T>
struct cuda_async_value final : cuda_async_value_base
{
private:
  cuda_managed_unique_ptr<T> content_;

public:
  // Constructs a `cuda_async_value` with a new default-constructed content
  // and a new stream.
  cuda_async_value()
    : cuda_async_value_base()
    , content_(cuda_make_managed_default_unique<T>())
  {}

  // Constructs a `cuda_async_value` with a new default-constructed
  // content that takes ownership of its predecessor's stream and holds a
  // reference to the predecessor's shared state to keep it alive.
  cuda_async_value(cuda_stream&& stream, keep_alive_type&& keep_alive)
    : cuda_async_value_base(MV(stream), MV(keep_alive))
    , content_(cuda_make_managed_default_unique<T>())
  {}

  // Constructs a `cuda_async_value` with a new default-constructed content, a
  // new stream, a new semaphore and a new default-constructed content.
  // Remarks: Only used by `make_promise` for external dependencies.
  cuda_async_value(semaphore_type&& semaphore)
    : cuda_async_value_base(MV(semaphore))
    , content_(cuda_make_managed_default_unique<T>())
  {}

  // Remarks: Only used by `make_promise` for external dependencies.
  // Precondition: This `cuda_async_value` was created by `make_promise`, e.g.
  // `semaphore_` is `true`.
  void set_value(T&& v)
  {
    // TODO: Should this be rvalue-ref qualified? It is on promise.
    assert(semaphore_);
    *content_ = FWD(v);
    semaphore_->store(true, std::memory_order_release);
  }

  T* data() const
  {
    return content_.get();
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct cuda_executor_promise final
{
  friend struct cuda_executor;

  using shared_state_type = std::shared_ptr<cuda_async_value<T>>;

private:
  shared_state_type ss_;

  cuda_executor_promise(shared_state_type const& s)
    : ss_(s)
  {}

  cuda_executor_promise(shared_state_type&& s)
    : ss_(MV(s))
  {}

public:
  void set_value(T&& value) &&
  {
    ss_->set_value(FWD(value));
    ss_.reset();
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename Executor>
struct cuda_executor_future final
{
  friend struct cuda_executor;

  using shared_state_type = std::shared_ptr<cuda_async_value<T>>;

private:
  shared_state_type ss_;
  Executor          exec_;

  cuda_executor_future(shared_state_type const& s, Executor e)
    : ss_(s), exec_(e)
  {}

  cuda_executor_future(shared_state_type&& s, Executor e)
    : ss_(MV(s)), exec_(e)
  {}

  shared_state_type&       shared_state()       RETURNS(ss_);
  shared_state_type const& shared_state() const RETURNS(ss_);

  Executor&       executor()       RETURNS(exec_);
  Executor const& executor() const RETURNS(exec_);

public:
  template <typename UExecutor>
  cuda_executor_future<T, UExecutor> via(UExecutor&& exec) &&
  {
    return cuda_executor_future<T, UExecutor>(MV(ss_), FWD(exec_));
  }

  template <typename Operation>
  executor_future_t<Executor, RETOF(Operation, T)> then(Operation&& op) &&
  {
    return exec_.then_execute(FWD(op), MV(*this));
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct cuda_semi_future final
{
  using shared_state_type = std::shared_ptr<cuda_async_value<T>>;

private:
  shared_state_type ss_;

public:
  template <typename UExecutor>
  cuda_executor_future<T, UExecutor> via(UExecutor&& exec) &&
  {
    return cuda_executor_future<T, UExecutor>(MV(ss_), FWD(exec));
  }
};

///////////////////////////////////////////////////////////////////////////////

#endif // FUTURES_CUDA_FUTURE_HPP

