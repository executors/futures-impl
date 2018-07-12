// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(FUTURES_CUDA_MEMORY_HPP)
#define FUTURES_CUDA_MEMORY_HPP

#include <memory>
#include <type_traits>

#include <cuda_runtime.h>

#include "cuda_error_handling.hpp"

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct cuda_free_uninitialized_deleter final
{
  using pointer = T*;

  void operator()(pointer raw) const
  {
    if (nullptr != raw)
      THROW_ON_CUDA_RT_ERROR(cudaFree(const_cast<std::remove_cv_t<T>*>(raw)));
  }
};

template <typename T>
struct cuda_free_deleter final
{
  using pointer = T*;

  void operator()(pointer raw) const
  {
    if (nullptr != raw)
    {
      raw->~T();
      THROW_ON_CUDA_RT_ERROR(cudaFree(const_cast<std::remove_cv_t<T>*>(raw)));
    }
  }
};

template <typename T>
struct cuda_free_host_uninitialized_deleter final
{
  using pointer = T*;

  void operator()(pointer raw) const
  {
    if (nullptr != raw)
      THROW_ON_CUDA_RT_ERROR(cudaFreeHost(const_cast<std::remove_cv_t<T>*>(raw)));
  }
};

template <typename T>
struct cuda_free_host_deleter final
{
  using pointer = T*;

  void operator()(pointer raw) const
  {
    if (nullptr != raw)
    {
      raw->~T();
      THROW_ON_CUDA_RT_ERROR(cudaFreeHost(const_cast<std::remove_cv_t<T>*>(raw)));
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
auto cuda_make_host_pinned_uninitialized_unique()
{
  std::unique_ptr<T, cuda_free_host_uninitialized_deleter<T>> ptr;

  T* raw;
  THROW_ON_CUDA_RT_ERROR(cudaHostAlloc(&raw, sizeof(T), cudaHostAllocDefault));
  ptr.reset(raw); // This is `noexcept`.

  return ptr;
}

template <typename T>
using cuda_host_pinned_uninitialized_unique_ptr
  = std::unique_ptr<T, cuda_free_host_uninitialized_deleter<T>>;

template <typename T>
auto cuda_make_host_pinned_default_unique()
{
  std::unique_ptr<T, cuda_free_host_deleter<T>> ptr;

  auto uptr = cuda_make_host_pinned_uninitialized_unique<T>();
  ptr.reset(uptr.release()); // These are both `noexcept`.

  new (const_cast<std::remove_cv_t<T>*>(ptr.get())) T;

  return ptr;
}

template <typename T>
auto cuda_make_host_pinned_unique(T&& t)
{
  std::unique_ptr<T, cuda_free_host_deleter<T>> ptr;

  auto uptr = cuda_make_host_pinned_uninitialized_unique<T>();
  ptr.reset(uptr.release()); // These are both `noexcept`.

  new (const_cast<std::remove_cv_t<T>*>(ptr.get())) T(FWD(t));

  return ptr;
}

template <typename T>
using cuda_host_pinned_unique_ptr
  = std::unique_ptr<T, cuda_free_host_deleter<T>>;

///////////////////////////////////////////////////////////////////////////////

template <typename T>
auto cuda_make_managed_uninitialized_unique()
{
  std::unique_ptr<T, cuda_free_uninitialized_deleter<T>> ptr;

  T* raw;
  THROW_ON_CUDA_RT_ERROR(cudaMallocManaged(&raw, sizeof(T)));
  ptr.reset(raw); // This is `noexcept`.

  return ptr;
}

template <typename T>
using cuda_managed_uninitialized_unique_ptr
  = std::unique_ptr<T, cuda_free_uninitialized_deleter<T>>;

template <typename T>
auto cuda_make_managed_default_unique()
{
  std::unique_ptr<T, cuda_free_deleter<T>> ptr;

  auto uptr = cuda_make_managed_uninitialized_unique<T>();
  ptr.reset(uptr.release()); // These are both `noexcept`.

  new (const_cast<std::remove_cv_t<T>*>(ptr.get())) T;

  return ptr;
}

template <typename T>
auto cuda_make_managed_unique(T&& t)
{
  std::unique_ptr<T, cuda_free_deleter<T>> ptr;

  auto uptr = cuda_make_managed_uninitialized_unique<T>();
  ptr.reset(uptr.release()); // These are both `noexcept`.

  new (const_cast<std::remove_cv_t<T>*>(ptr.get())) T(FWD(t));

  return ptr;
}

template <typename T>
using cuda_managed_unique_ptr
  = std::unique_ptr<T, cuda_free_deleter<T>>;

///////////////////////////////////////////////////////////////////////////////

#endif // FUTURES_CUDA_MEMORY_HPP

