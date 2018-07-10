#include "executor_traits.hpp"
#include "type_deduction.hpp"

#include "cuda_error_handling.hpp"
#include "cuda_stream.hpp"
#include "cuda_memory.hpp"

#include <atomic>

#include <cstdio>

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

    auto&& fss = MV(f).shared_state(); 

    // Get a copy of the pointer to the data, because we're about to move from
    // `fss`.
    auto fdata = fss->data();

    // Create a new future that uses the same stream as `f` and keeps the
    // shared state of `f` alive.
    auto ss = std::make_shared<cuda_async_value<U>>(
      MV(fss->shared_stream()), MV(fss)
    );
    cuda_executor_future<U, cuda_executor> g(MV(ss), MV(f).executor());

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
    auto ss = std::make_shared<cuda_async_value<T>>();
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
  cuda_stream stream_;
  semaphore_type semaphore_;
  keep_alive_type keep_alive_;

public:
  cuda_async_value_base()
    : stream_(), semaphore_(0), keep_alive_()
  {}

  cuda_async_value_base(cuda_stream&& s, keep_alive_type&& k)
    : stream_(MV(s)), semaphore_(0), keep_alive_(k)
  {}

  virtual ~cuda_async_value_base() {}

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
  cuda_async_value()
    : cuda_async_value_base(), content_()
  {}

  cuda_async_value(cuda_stream&& s, keep_alive_type&& k)
    : cuda_async_value_base(MV(s), MV(k))
  {}

  void set_value(T&& v)
  {
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

  cuda_executor_promise(shared_state_type s)
    : ss_(s)
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

  cuda_executor_future(shared_state_type s, Executor e)
    : ss_(s), exec_(e)
  {}

  shared_state_type&       shared_state() &      RETURNS(ss_)
  shared_state_type const& shared_state() const& RETURNS(ss_)
  shared_state_type&&      shared_state() &&     RETURNS(MV(ss_))

  Executor&       executor() &      RETURNS(exec_)
  Executor const& executor() const& RETURNS(exec_)
  Executor&&      executor() &&     RETURNS(MV(exec_))

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
  shared_state_type cuda_;

public:
  template <typename UExecutor>
  cuda_executor_future<T, UExecutor> via(UExecutor&& exec) &&
  {
    return cuda_executor_future<T, UExecutor>(MV(cuda_), FWD(exec));
  }
};

///////////////////////////////////////////////////////////////////////////////

int main()
{
  {
    cuda_executor exec;

    auto p_f = make_promise<int>(exec);
    cuda_executor_promise<int>&               p = p_f.first;
    cuda_executor_future<int, cuda_executor>& f = p_f.second;

    auto g = MV(f).then(
      [] __host__ __device__ (int x) { printf("%u\n", x); return x + 1; });

    auto h = MV(g).then(
      [] __host__ __device__ (int x) { printf("%u\n", x); return x + 2; });

    MV(p).set_value(1);

    exec.wait(MV(h));
  }

  {
    cuda_executor exec;

    auto f = exec.twoway_execute(
      [] __host__ __device__ { printf("0\n"); return 17; });

    auto g = MV(f).then(
      [] __host__ __device__ (int x) { printf("%u\n", x); return 3.14; }); 

    auto h = MV(g).then(
      [] __host__ __device__ (double x) { printf("%g\n", x); return 42; }); 

    exec.wait(MV(h));
  }
}

