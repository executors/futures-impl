// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#include "cuda_future_new.hpp"

#include <cstdio>

int main()
{
  {
    cuda_executor exec;

    exec.execute(promise([] __host__ __device__ () { printf("a\n"); }));
    exec.execute(promise([] __host__ __device__ () { printf("b\n"); }));
    exec.execute(promise([] __host__ __device__ () { printf("c\n"); }));

    exec.wait();
  }
}

