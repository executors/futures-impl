// Copyright 2018 NVIDIA Corporation
// Copyright 2017 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
// Copyright 2002 Peter Dimov and Multi Media Ltd (`CURRENT_FUNCTION`)
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#include <utility>
#include <memory>
#include <string>
#include <type_traits>
#include <cassert>

#include <iostream>
#include <iomanip>

struct cuda_exception : std::exception
{
  cuda_exception(cudaError_t error_, char const* message_)
  // {{{
    : error(error_)
    , message(
        std::string(cudaGetErrorName(error_)) + ": "
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
  std::string message;
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
inline void throw_on_cuda_error(cudaError_t error, char const* message)
{ // {{{
  if (cudaSuccess != error)
    throw cuda_exception(error, message);
} // }}}

#define THROW_ON_CUDA_ERROR(error) throw_on_cuda_error(error, CURRENT_FUNCTION)

struct cuda_event_deleter final
{
  __host__
  inline void operator()(CUevent_st* e) const
  { // {{{
    if (nullptr != e)
      THROW_ON_CUDA_ERROR(cudaEventDestroy(e));
  } // }}}
};

using cuda_event = std::unique_ptr<CUevent_st, cuda_event_deleter>;

struct cuda_stream_deleter final
{
  __host__
  inline void operator()(CUstream_st* s) const
  { // {{{
    if (nullptr != s)
      THROW_ON_CUDA_ERROR(cudaStreamDestroy(s));
  } // }}}
};

using cuda_stream = std::unique_ptr<CUstream_st, cuda_stream_deleter>;

template <typename T>
struct cuda_free_deleter final
{
  __host__
  inline void operator()(T* p) const
  { // {{{
    if (nullptr != p)
    {
      p->~T();
      THROW_ON_CUDA_ERROR(cudaFree(p));
    }
  } // }}}
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, cuda_free_deleter<T> >;

template <typename F, typename T, typename U>
__global__
void continuation_kernel(F f, T* t, U* u)
{ // {{{
  // Consume the value from our dependency by moving it out of it's
  // `asynchronous_value`. The storage for said value will not be destroyed
  // until our `asynchronous_value` is destroyed.
  *t = f(static_cast<typename std::remove_reference<U>::type&&>(*u));
} // }}}

template <typename T>
struct asynchronous_value final
{

private:

  cuda_event         event;
  cuda_stream        stream;
  cuda_unique_ptr<T> value;
  bool               event_recorded;
  bool               value_consumed;

public:

  __host__
  asynchronous_value()
  // {{{
    : event{}
    , stream{}
    , value{}
    , event_recorded(false)
    , value_consumed(false)
  {
    CUevent_st* e;
    THROW_ON_CUDA_ERROR(cudaEventCreate(&e));
    event.reset(e);

    CUstream_st* s;
    THROW_ON_CUDA_ERROR(cudaStreamCreate(&s));
    stream.reset(s);

    T* p;
    THROW_ON_CUDA_ERROR(cudaMallocManaged(&p, sizeof(T)));
    new (p) T;
    value.reset(p);
  }
  // }}}

  // Immediate value constructor.
  template <typename U>
  __host__
  asynchronous_value(U&& u)
  // {{{
    : event{}
    , stream{}
    , value{}
    , event_recorded(false)
    , value_consumed(false)
  {
    T* p;
    THROW_ON_CUDA_ERROR(cudaMallocManaged(&p, sizeof(T)));
    new (p) T(std::forward<U>(u));
    value.reset(p);
  }
  // }}}

  __host__
  bool immediate() const
  { // {{{
    return !event;
  } // }}}

  __host__
  bool valid() const
  { // {{{
    if (immediate()) return true;
    else             return event_recorded;
  } // }}}

  __host__
  bool value_ready() const
  { // {{{
    if (immediate())
      return true;

    if (valid())
    {
      cudaError_t err = cudaEventQuery(event.get());

      if      (cudaSuccess       == err)
        return true;
      else if (cudaErrorNotReady == err)
        return false;
      else
        THROW_ON_CUDA_ERROR(err);
    }

    return false;
  } // }}}

  __host__
  bool continuation_ready() const
  { // {{{
    return event_recorded;
  } // }}}

  __host__
  bool consumed() const
  { // {{{
    return value_consumed;
  } // }}}

  template <typename U, typename F>
  __host__
  void set_continuation(asynchronous_value<U>& dep, F f)
  { // {{{
    static_assert(std::is_trivially_copyable<F>::value, "");

    assert(!immediate());
    assert(!valid());
    assert(dep.valid() && !dep.consumed());

    if (!dep.immediate())
      // If `dep` is not an immediate value, make our stream depend on the
      // completion of `dep`'s event.
      THROW_ON_CUDA_ERROR(cudaStreamWaitEvent(stream.get(), dep.event.get(), 0));

    // Launch a kernel that evaluates `f` on our stream.
    T* t = value.get();
    U* u = dep.value.get();
    void* args[] = { &f, &t, &u };
    void const* k = reinterpret_cast<void const*>(continuation_kernel<F, T, U>);
    THROW_ON_CUDA_ERROR(cudaLaunchKernel(k,
                        dim3{1}, dim3{1}, args, 0, stream.get()));

    THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());

    // Mark `dep`'s value as consumed. Its storage will be freed later.
    dep.value_consumed = true;

    // Record our event, which will be ready once `f`'s evaluation is complete.
    THROW_ON_CUDA_ERROR(cudaEventRecord(event.get(), stream.get()));
    event_recorded = true;
  } // }}} 

  __host__
  void wait() const
  { // {{{
    assert(valid());

    if (!immediate())
      THROW_ON_CUDA_ERROR(cudaEventSynchronize(event.get()));
  } // }}}

  __host__
  T get()
  { // {{{
    assert(valid() && !consumed());

    wait();

    // Consume the value by moving it out of the storage.
    T tmp(std::move(*value));

    // Free the storage. 
    value.reset();

    // Mark the value as consumed.
    value_consumed = true;

    return std::move(tmp);
  } // }}}
};

#define TEST_EQ(a, b) assert(a == b)

int main()
{
  std::cout << std::setbase(2);

  { // Create a default constructed `asynchronous_value`.
    asynchronous_value<int> a;

    TEST_EQ(a.valid(),              false);
    TEST_EQ(a.immediate(),          false);
    TEST_EQ(a.value_ready(),        false);
    TEST_EQ(a.continuation_ready(), false);
    TEST_EQ(a.consumed(),           false);
  }

  { // Create an immediate `asynchronous_value`, then consume it.
    asynchronous_value<int> a(42);

    TEST_EQ(a.valid(),              true);
    TEST_EQ(a.immediate(),          true);
    TEST_EQ(a.value_ready(),        true);
    TEST_EQ(a.continuation_ready(), false);
    TEST_EQ(a.consumed(),           false);

    int a_val = a.get();

    TEST_EQ(a_val,                  42);

    TEST_EQ(a.valid(),              true);
    TEST_EQ(a.immediate(),          true);
    TEST_EQ(a.value_ready(),        true);
    TEST_EQ(a.continuation_ready(), false);
    TEST_EQ(a.consumed(),           true);
  }

  { // Create an immediate `asynchronous_value`, then attach a dependency to it.
    asynchronous_value<int> a(42);

    TEST_EQ(a.valid(),              true);
    TEST_EQ(a.immediate(),          true);
    TEST_EQ(a.value_ready(),        true);
    TEST_EQ(a.continuation_ready(), false);
    TEST_EQ(a.consumed(),           false);

    asynchronous_value<int> b;

    b.set_continuation(a, [] __device__ (int i) { return i + 17; });

    TEST_EQ(a.valid(),              true);
    TEST_EQ(a.immediate(),          true);
    TEST_EQ(a.value_ready(),        true);
    TEST_EQ(a.continuation_ready(), false);
    TEST_EQ(a.consumed(),           true);

    TEST_EQ(b.valid(),              true);
    TEST_EQ(b.immediate(),          false);
    // We don't test `b.value_ready()` here because the result is unspecified -
    // the kernel may or may not have asynchronously launched and completed by
    // now.
    TEST_EQ(b.continuation_ready(), true);
    TEST_EQ(b.consumed(),           false);

    int b_val = b.get();

    TEST_EQ(b_val,                  59);

    TEST_EQ(b.valid(),              true);
    TEST_EQ(b.immediate(),          false);
    TEST_EQ(b.value_ready(),        true);
    TEST_EQ(b.continuation_ready(), true);
    TEST_EQ(b.consumed(),           true);
  }

  // TODO: Chaining different types.
}

