// Copyright (c)      2018 NVIDIA Corporation 
//                         (Bryce Adelstein Lelbach <brycelelbach@gmail.com>)
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(FUTURES_CUDA_ERROR_HANDLING_HPP)
#define FUTURES_CUDA_ERROR_HANDLING_HPP

#include <string>
#include <exception>

#include <cuda.h>
#include <cuda_runtime.h>

#include "preprocessor.hpp"

///////////////////////////////////////////////////////////////////////////////

struct cuda_rt_exception : std::exception
{
  cuda_rt_exception(cudaError_t error_, char const* message_)
    : error(error_)
    , message(
        std::string(cudaGetErrorName(error_)) + ": "
      + cudaGetErrorString(error_) + ": "
      + message_
      )
  {}

  cudaError_t code() const
  {
    return error;
  }

  virtual char const* what() const noexcept
  {
    return message.c_str();
  } 

private:
  cudaError_t error;
  std::string message;
};

struct cuda_drv_exception : std::exception
{
  cuda_drv_exception(CUresult error_, char const* message_)
    : error(error_)
  {
    char const* str = nullptr;
    cuGetErrorName(error_, &str);
    message = str;
    message += ": ";
    cuGetErrorString(error_, &str);
    message += str;
    message += ": ";
    message += message_;
  }

  CUresult code() const
  {
    return error;
  } 

  virtual char const* what() const noexcept
  { 
    return message.c_str();
  } 

private:
  CUresult error;
  std::string message;
};

inline void throw_on_cuda_rt_error(cudaError_t error, char const* message)
{
  if (cudaSuccess != error)
    throw cuda_rt_exception(error, message);
} 

#define THROW_ON_CUDA_RT_ERROR(error)                                         \
  throw_on_cuda_rt_error(error, CURRENT_FUNCTION)                             \
  /**/

inline void throw_on_cuda_drv_error(CUresult error, char const* message)
{
  if (CUDA_SUCCESS != error)
    throw cuda_drv_exception(error, message);
}

#define THROW_ON_CUDA_DRV_ERROR(error)                                        \
  throw_on_cuda_drv_error(error, CURRENT_FUNCTION)                            \
  /**/

///////////////////////////////////////////////////////////////////////////////

#endif // FUTURES_CUDA_ERROR_HANDLING_HPP

