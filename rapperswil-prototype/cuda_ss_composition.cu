// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#include "cuda_future.hpp"
#include "ss_future.hpp"

#include <cstdio>

struct A{};
struct B{};
struct C{};
struct D{};

int main()
{
  {
    cuda_executor cuda;
    ss_executor   ss;

    auto f = cuda.twoway_execute(
      [] __host__ __device__ { printf("0\n"); return 17; }
    );

    auto g = MV(f).via(ss).then(
      [] __host__ __device__ (int x) { printf("%u\n", x); return 3.14; }
    );

    ss.wait(MV(g));
  }

  {
    cuda_executor cuda;
    ss_executor   ss;

    auto f = cuda.twoway_execute(
      [] __host__ __device__ { printf("0\n"); return 17; }
    );

    auto g = MV(f).via(ss).then(
      [] __host__ __device__ (int x) { printf("%u\n", x); return 3.14; }
    );

    auto h = MV(g).via(cuda).then(
      [] __host__ __device__ (double x) { printf("%g\n", x); return 42; }
    );

    cuda.wait(MV(h));
  }

  {
    cuda_executor cuda;
    ss_executor   ss;

    auto f = cuda.twoway_execute(
      [] __host__ __device__ { printf("0\n"); return 17; }
    );

    auto g = MV(f).via(ss).then(
      [] __host__ __device__ (int x) { printf("%u\n", x); return 3.14; }
    );

    auto h = MV(g).via(cuda).then(
      [] __host__ __device__ (double x) { printf("%g\n", x); return 42; }
    );

    auto i = MV(h).via(ss).then(
      [] __host__ __device__ (int x) { printf("%u\n", x); return true; }
    );

    ss.wait(MV(i));
  }

  {
    cuda_executor cuda;
    ss_executor   ss;

    auto f = cuda.twoway_execute(
      [] __host__ __device__ { printf("f\n"); return A{}; }
    );

    auto g = MV(f).via(ss).then(
      [] __host__ __device__ (A) { printf("g\n"); return B{}; }
    );

    auto h = MV(g).via(cuda).then(
      [] __host__ __device__ (B) { printf("h\n"); return C{}; }
    );

    cuda.wait(MV(h));
  }
}

