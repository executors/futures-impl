// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#include "cuda_future.hpp"

#include <cstdio>

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

