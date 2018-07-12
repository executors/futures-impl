// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#include "ss_future.hpp"

#include <cstdio>

int main()
{
  {
    ss_executor exec;

    auto p_f = make_promise<int>(exec);
    ss_executor_promise<int>&             p = p_f.first;
    ss_executor_future<int, ss_executor>& f = p_f.second;

    auto g = MV(f).then(
      [] (int x) { printf("%u\n", x); return x + 1; });

    auto h = MV(g).then(
      [] (int x) { printf("%u\n", x); return x + 2; });

    MV(p).set_value(1);

    exec.wait(MV(h));
  }

  {
    ss_executor exec;

    auto f = exec.twoway_execute(
      [] { printf("0\n"); return 17; });

    auto g = MV(f).then(
      [] (int x) { printf("%u\n", x); return 3.14; }); 

    auto h = MV(g).then(
      [] (double x) { printf("%g\n", x); return 42; }); 

    exec.wait(MV(h));
  }
}

