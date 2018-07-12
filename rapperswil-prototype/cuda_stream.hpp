// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(FUTURES_CUDA_STREAM_HPP)
#define FUTURES_CUDA_STREAM_HPP

#include <memory>

#include <cuda_runtime.h>

#include "cuda_error_handling.hpp"

///////////////////////////////////////////////////////////////////////////////

struct cuda_stream_deleter final
{
  void operator()(CUstream_st* s) const
  {
    if (nullptr != s)
      THROW_ON_CUDA_RT_ERROR(cudaStreamDestroy(s));
  }
};

struct cuda_stream final
{
private:
  std::shared_ptr<CUstream_st> ptr;

public:
  cuda_stream()
  {
    CUstream_st* s;
    THROW_ON_CUDA_RT_ERROR(cudaStreamCreate(&s));
    ptr.reset(s, cuda_stream_deleter{});
  }

  cuda_stream(cuda_stream const&) = delete;
  cuda_stream(cuda_stream&&) = default;

  CUstream_st& operator*() const
  {
    return *ptr.get();
  }

  CUstream_st* operator->() const
  {
    return ptr.get();
  }

  CUstream_st* get() const
  {
    return ptr.get();
  }
};

///////////////////////////////////////////////////////////////////////////////

#endif // FUTURES_CUDA_STREAM_HPP

