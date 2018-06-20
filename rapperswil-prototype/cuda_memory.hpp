// Copyright (c)      2018 NVIDIA Corporation 
//                         (Bryce Adelstein Lelbach <brycelelbach@gmail.com>)
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
struct cuda_free_deleter final
{
  void operator()(T* p) const
  {
    if (nullptr != p)
    {
      p->~T();
      THROW_ON_CUDA_RT_ERROR(cudaFree(const_cast<std::remove_cv_t<T>*>(p)));
    }
  }
};

template <typename T>
struct cuda_free_host_deleter final
{
  void operator()(T* p) const
  {
    if (nullptr != p)
    {
      p->~T();
      THROW_ON_CUDA_RT_ERROR(cudaFreeHost(const_cast<std::remove_cv_t<T>*>(p)));
    }
  }
};

template <typename T>
struct cuda_host_pinned_unique_ptr final
{
private:
  std::unique_ptr<T, cuda_free_host_deleter<T>> ptr;

public:
  cuda_host_pinned_unique_ptr()
  {
    T* v;
    THROW_ON_CUDA_RT_ERROR(cudaHostAlloc(&v, sizeof(T), 0));
    new (const_cast<std::remove_cv_t<T>*>(v)) T;
    ptr.reset(v);
  }

  cuda_host_pinned_unique_ptr(T&& t)
  {
    T* v;
    THROW_ON_CUDA_RT_ERROR(cudaHostAlloc(&v, sizeof(T), 0));
    new (const_cast<std::remove_cv_t<T>*>(v)) T(move(t));
    ptr.reset(v);
  }

  cuda_host_pinned_unique_ptr(T const& t)
  {
    T* v;
    THROW_ON_CUDA_RT_ERROR(cudaHostAlloc(&v, sizeof(T), 0));
    new (const_cast<std::remove_cv_t<T>*>(v)) T(t);
    ptr.reset(v);
  }

  cuda_host_pinned_unique_ptr(cuda_host_pinned_unique_ptr const&) = delete;
  cuda_host_pinned_unique_ptr(cuda_host_pinned_unique_ptr&&) = default;

  T& operator*() const
  {
    return *ptr.get();
  }

  T* operator->() const
  {
    return ptr.get();
  }

  T* get() const
  {
    return ptr.get();
  }
};

template <typename T>
struct cuda_managed_unique_ptr final
{
private:
  std::unique_ptr<T, cuda_free_deleter<T>> ptr;

public:
  cuda_managed_unique_ptr()
  {
    T* v;
    THROW_ON_CUDA_RT_ERROR(cudaMallocManaged(&v, sizeof(T)));
    new (const_cast<std::remove_cv_t<T>*>(v)) T;
    ptr.reset(v);
  }

  cuda_managed_unique_ptr(T&& t)
  {
    T* v;
    THROW_ON_CUDA_RT_ERROR(cudaMallocManaged(&v, sizeof(T)));
    new (const_cast<std::remove_cv_t<T>*>(v)) T(move(t));
    ptr.reset(v);
  }

  cuda_managed_unique_ptr(T const& t)
  {
    T* v;
    THROW_ON_CUDA_RT_ERROR(cudaMallocManaged(&v, sizeof(T)));
    new (const_cast<std::remove_cv_t<T>*>(v)) T(t);
    ptr.reset(v);
  }

  cuda_managed_unique_ptr(cuda_managed_unique_ptr const&) = delete;
  cuda_managed_unique_ptr(cuda_managed_unique_ptr&&) = default;

  T& operator*() const
  {
    return *ptr.get();
  }

  T* operator->() const
  {
    return ptr.get();
  }

  T* get() const
  {
    return ptr.get();
  }
};

///////////////////////////////////////////////////////////////////////////////

#endif // FUTURES_CUDA_MEMORY_HPP

