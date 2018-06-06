#include <utility>
#include <memory>
#include <string>
#include <atomic>
#include <type_traits>
#include <cassert>

#include <iostream>
#include <iomanip>

#include <cuda.h>

using std::exception;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::atomic;
using std::uint32_t;

struct cuda_rt_exception : exception
{
  cuda_rt_exception(cudaError_t error_, char const* message_)
  // {{{
    : error(error_)
    , message(
        string(cudaGetErrorName(error_)) + ": "
      + cudaGetErrorString(error_) + ": "
      + message_
      )
  {}
  // }}}

  cudaError_t code() const
  { // {{{
    return error;
  } // }}}

  virtual const char* what() const noexcept
  { // {{{
    return message.c_str();
  } // }}}

private:
  cudaError_t error;
  string message;
};

struct cuda_drv_exception : exception
{
  cuda_drv_exception(CUresult error_, char const* message_)
  // {{{
    : error(error_)
  {
    const char* str = nullptr;
    cuGetErrorName(error_, &str);
    message = str;
    message += ": ";
    cuGetErrorString(error_, &str);
    message += str;
    message += ": ";
    message += message_;
  }
  // }}}

  CUresult code() const
  { // {{{
    return error;
  } // }}}

  virtual const char* what() const noexcept
  { // {{{
    return message.c_str();
  } // }}}

private:

  CUresult error;
  string message;
};

#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
  #define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
  #define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
  #define CURRENT_FUNCTION __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
  #define CURRENT_FUNCTION __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
  #define CURRENT_FUNCTION __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
  #define CURRENT_FUNCTION __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
  #define CURRENT_FUNCTION __func__
#else
  #define CURRENT_FUNCTION "(unknown)"
#endif

__host__
inline void throw_on_cuda_rt_error(cudaError_t error, char const* message)
{ // {{{
  if (cudaSuccess != error)
    throw cuda_rt_exception(error, message);
} // }}}

#define THROW_ON_CUDA_RT_ERROR(error) throw_on_cuda_rt_error(error, CURRENT_FUNCTION)

__host__
inline void throw_on_cuda_drv_error(CUresult error, char const* message)
{ // {{{
  if (CUDA_SUCCESS != error)
    throw cuda_drv_exception(error, message);
} // }}}

#define THROW_ON_CUDA_DRV_ERROR(error) throw_on_cuda_drv_error(error, CURRENT_FUNCTION)

struct cuda_stream_deleter final
{
  __host__
  inline void operator()(cudaStream_t s) const
  { // {{{
    if (nullptr != s)
      THROW_ON_CUDA_RT_ERROR(cudaStreamDestroy(s));
  } // }}}
};

struct cuda_stream final {
  cuda_stream() {
    CUstream_st* s;
    THROW_ON_CUDA_RT_ERROR(cudaStreamCreate(&s));
    ptr.reset(s, cuda_stream_deleter{});
  }

  cuda_stream(cuda_stream const&) = default;
  cuda_stream(cuda_stream&&) = default;

  CUstream_st* operator->() const {
    return ptr.get();
  }

  CUstream_st* get() const {
    return ptr.get();
  }

private:
  shared_ptr<CUstream_st> ptr;
};

template <typename T>
struct cuda_free_deleter final
{
  __host__
  inline void operator()(T* p) const
  { // {{{
    if (nullptr != p)
    {
      p->~T();
      THROW_ON_CUDA_RT_ERROR(cudaFree(p));
    }
  } // }}}
};

template <typename T>
struct cuda_free_host_deleter final
{
  __host__
  inline void operator()(T* p) const
  { // {{{
    if (nullptr != p)
    {
      p->~T();
      THROW_ON_CUDA_RT_ERROR(cudaFreeHost((void*)(p)));
    }
  } // }}}
};

template <typename T>
struct cuda_host_pinned_unique_ptr final {
  cuda_host_pinned_unique_ptr() {
    T* r;
    THROW_ON_CUDA_RT_ERROR(cudaHostAlloc(&r, sizeof(T), 0));
    new (const_cast<std::remove_cv_t<T>*>(r)) T;
    ptr.reset(r);
  }

  cuda_host_pinned_unique_ptr(T&& t) {
    T* r;
    THROW_ON_CUDA_RT_ERROR(cudaHostAlloc(&r, sizeof(T), 0));
    new (const_cast<std::remove_cv_t<T>*>(r)) T(move(t));
    ptr.reset(r);
  }

  cuda_host_pinned_unique_ptr(T const& t) {
    T* r;
    THROW_ON_CUDA_RT_ERROR(cudaHostAlloc(&r, sizeof(T), 0));
    new (const_cast<std::remove_cv_t<T>*>(r)) T(t);
    ptr.reset(r);
  }

  cuda_host_pinned_unique_ptr(cuda_host_pinned_unique_ptr const&) = default;
  cuda_host_pinned_unique_ptr(cuda_host_pinned_unique_ptr&&) = default;

  T* operator->() const {
    return ptr.get();
  }

  T* get() const {
    return ptr.get();
  }

private:
  unique_ptr<T, cuda_free_host_deleter<T>> ptr;
};

///////////////////////////////////////////////////////////////////////////////

struct share_stream_t final {};

share_stream_t share_stream{};

struct not_ready_t final {};

not_ready_t not_ready{};

template <typename T>
struct shared_state final
{ // {{{
  int device;
  cuda_stream stream;
  cuda_host_pinned_unique_ptr<atomic<uint32_t> volatile> ready;
  unique_ptr<T, cuda_free_deleter<T>> value;

  shared_state()
    : stream()
    , ready(0)
  { // {{{
    THROW_ON_CUDA_RT_ERROR(cudaGetDevice(&device));

    T* v;
    THROW_ON_CUDA_RT_ERROR(cudaMallocManaged(&v, sizeof(T)));
    new (v) T;
    value.reset(v);
  } // }}}

  shared_state(share_stream_t, shared_state const& other)
    : stream(other.stream)
    , ready(0)
  { // {{{
    THROW_ON_CUDA_RT_ERROR(cudaGetDevice(&device));

    T* v;
    THROW_ON_CUDA_RT_ERROR(cudaMallocManaged(&v, sizeof(T)));
    new (v) T;
    value.reset(v);
  } // }}}
}; // }}}

template <typename Continuation, typename U, typename T>
__global__
void continuation_kernel(Continuation c, U* u, T* t)
{ // {{{
  *u = c(*t);
} // }}}

struct cuda_executor final
{
  template <typename T>
  struct future;

  template <typename T>
  struct promise final
  { // {{{
    std::shared_ptr<shared_state<T>> ss;

    promise() {}

    promise(future<T> const& f) : ss(f.ss) {}

    void set_value(T&& value) 
    { // {{{
      *ss->value = std::forward<T>(value);
      ss->ready->store(true, std::memory_order_release); 
    } // }}}
  }; // }}}

  template <typename T>
  struct future final
  { // {{{
    using value_type = T;

    std::shared_ptr<shared_state<T>> ss;

    future()
    // {{{
      : ss(std::make_shared<shared_state<T>>())
    {}
    // }}}

    future(not_ready_t)
    // {{{
      : ss(std::make_shared<shared_state<T>>())
    {
      THROW_ON_CUDA_DRV_ERROR(cuStreamWaitValue32(stream(), semaphore(),
                                                  true, CU_STREAM_WAIT_VALUE_EQ));
    }
    // }}}

    future(share_stream_t, future const& other)
    // {{{
      : ss(std::make_shared<shared_state<T>>(share_stream, *other.ss))
    {}
    // }}}

    CUstream_st* stream() const
    { // {{{
      return ss->stream.get();
    } // }}}

    CUdeviceptr semaphore() const
    { // {{{
      return reinterpret_cast<CUdeviceptr>(ss->ready.get());
    } // }}}

    T* content() const
    { // {{{
      return ss->value.get();
    } // }}}

    int device() const
    { // {{{
      return ss->device;
    } // }}}

    template <typename Executor>
    auto via(Executor&& exec)
    { // {{{
      auto pf = std::forward<Executor>(exec).make_promise();

      THROW_ON_CUDA_RT_ERROR(cudaStreamAddCallback(stream(),
        [=] (CUstream_st*, cudaError_t, void*)
        { pf.first.set_value(*ss->value); }, nullptr, 0));

      return pf.second;
    } // }}}

    template <typename Continuation>
    auto then(Continuation&& c)
    { // {{{
      return cuda_executor{}.then_execute(std::forward<Continuation>(c));
    } // }}}
  }; // }}}

  template <typename T, typename Continuation>
  auto then_execute(future<T> const& prev, Continuation c)
  { // {{{
    using U = decltype(std::declval<Continuation>()(std::declval<T>()));

    int device = 0;
    THROW_ON_CUDA_RT_ERROR(cudaGetDevice(&device));

    if (prev.device() == device)
    {
      future<U> next(share_stream, prev);

      continuation_kernel<<<1, 1, 0, next.stream()>>>(c, next.content(), prev.content());
      THROW_ON_CUDA_RT_ERROR(cudaGetLastError());

      THROW_ON_CUDA_DRV_ERROR(cuStreamWriteValue32(next.stream(), next.semaphore(),
                                                   true, 0));

      return next;
    }

    else
    {
      future<U> next;

      THROW_ON_CUDA_DRV_ERROR(cuStreamWaitValue32(next.stream(), prev.semaphore(),
                                                  true, CU_STREAM_WAIT_VALUE_EQ));

      continuation_kernel<<<1, 1, 0, next.stream()>>>(c, next.content(), prev.content());
      THROW_ON_CUDA_RT_ERROR(cudaGetLastError());

      THROW_ON_CUDA_DRV_ERROR(cuStreamWriteValue32(next.stream(), next.semaphore(),
                                                   true, 0));

      return next;
    }
  } // }}}

  template <typename T>
  std::pair<promise<T>, future<T>> make_promise()
  { // {{{
    future<T>  f(not_ready);
    promise<T> p(f);
    return {std::move(p), std::move(f)};
  } // }}}
};

int main()
{
  cuda_executor exec;

  auto pf = exec.make_promise<int>();
  cuda_executor::promise<int>& p = pf.first;
  cuda_executor::future<int>&  f = pf.second;

  auto g = exec.then_execute(f, [] __host__ __device__ (int x)
                                { printf("%u\n", x); return x + 42; });

  auto h = exec.then_execute(g, [] __host__ __device__ (int x)
                                { printf("%u\n", x); return x + 15; });

  p.set_value(17);

  THROW_ON_CUDA_RT_ERROR(cudaDeviceSynchronize());
}

