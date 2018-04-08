// This example is meant to demonstrate that the legacy .sync_execute() and .bulk_sync_execute() execution functions
// are made obsolete by always_ready_future. Observe that the performance of all the SAXPY variants in this example program
// are similar because always_ready_future introduces negligible overhead.

#include <iostream>
#include <chrono>
#include <tuple>
#include <cassert>
#include <vector>

#include "inline_executor.hpp"

void for_loop_saxpy(float a,
                    const std::vector<float>& x,
                    const std::vector<float>& y,
                    std::vector<float>& z)
{
  for(size_t i = 0; i < x.size(); ++i)
  {
    z[i] = a * x[i] + y[i];
  }
}

void sync_execute_saxpy(float a,
                        const std::vector<float>& x,
                        const std::vector<float>& y,
                        std::vector<float>& z)
{
  inline_executor exec;

  for(size_t i = 0; i < x.size(); ++i)
  {
    exec.sync_execute([&]
    {
      z[i] = a * x[i] + y[i];
    });
  }
}

void twoway_execute_saxpy(float a,
                          const std::vector<float>& x,
                          const std::vector<float>& y,
                          std::vector<float>& z)
{
  inline_executor exec;

  for(size_t i = 0; i < x.size(); ++i)
  {
    exec.twoway_execute([&]
    {
      z[i] = a * x[i] + y[i];
    }).wait();
  }
}

void bulk_sync_execute_saxpy(float a,
                             const std::vector<float>& x,
                             const std::vector<float>& y,
                             std::vector<float>& z)
{
  inline_executor exec;

  exec.bulk_sync_execute([&](size_t i, auto, auto)
  {
    z[i] = a * x[i] + y[i];
  },
  x.size(),
  []{ return std::ignore; },
  []{ return std::ignore; }
  );
}

void bulk_twoway_execute_saxpy(float a,
                               const std::vector<float>& x,
                               const std::vector<float>& y,
                               std::vector<float>& z)
{
  inline_executor exec;

  exec.bulk_twoway_execute([&](size_t i, auto, auto)
  {
    z[i] = a * x[i] + y[i];
  },
  x.size(),
  []{ return std::ignore; },
  []{ return std::ignore; }
  ).wait();
}

int main()
{
  size_t num_trials = 100;

  size_t n = 1 << 25;

  float a = 42;
  std::vector<float> x(n, 7);
  std::vector<float> y(n, 13);
  std::vector<float> z(n);

  std::vector<float> reference(n, 42 * 7 + 13);

  std::cout << "SAXPY problem size: " << n << std::endl;

  {
    // make sure it works
    for_loop_saxpy(a, x, y, z);
    assert(z == reference);

    // warm-up
    for_loop_saxpy(a, x, y, z);

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < num_trials; ++i)
    {
      for_loop_saxpy(a, x, y, z);
    }

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

    double seconds = elapsed.count() / num_trials;
    double gigabytes = double(3 * n * sizeof(float)) / (1 << 30);
    double bandwidth = gigabytes / seconds;

    std::cout << "for_loop_saxpy (reference): " << bandwidth << " GB/s" << std::endl;
  }

  {
    // make sure it works
    sync_execute_saxpy(a, x, y, z);
    assert(z == reference);

    // warm-up
    sync_execute_saxpy(a, x, y, z);

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < num_trials; ++i)
    {
      sync_execute_saxpy(a, x, y, z);
    }

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

    double seconds = elapsed.count() / num_trials;
    double gigabytes = double(3 * n * sizeof(float)) / (1 << 30);
    double bandwidth = gigabytes / seconds;

    std::cout << "sync_execute_saxpy: " << bandwidth << " GB/s" << std::endl;
  }

  {
    // make sure it works
    twoway_execute_saxpy(a, x, y, z);
    assert(z == reference);

    // warm-up
    twoway_execute_saxpy(a, x, y, z);

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < num_trials; ++i)
    {
      twoway_execute_saxpy(a, x, y, z);
    }

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

    double seconds = elapsed.count() / num_trials;
    double gigabytes = double(3 * n * sizeof(float)) / (1 << 30);
    double bandwidth = gigabytes / seconds;

    std::cout << "twoway_execute_saxpy: " << bandwidth << " GB/s" << std::endl;
  }

  {
    // make sure it works
    bulk_sync_execute_saxpy(a, x, y, z);
    assert(z == reference);

    // warm-up
    bulk_sync_execute_saxpy(a, x, y, z);

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < num_trials; ++i)
    {
      bulk_sync_execute_saxpy(a, x, y, z);
    }

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

    double seconds = elapsed.count() / num_trials;
    double gigabytes = double(3 * n * sizeof(float)) / (1 << 30);
    double bandwidth = gigabytes / seconds;

    std::cout << "bulk_sync_execute_saxpy: " << bandwidth << " GB/s" << std::endl;
  }

  {
    // make sure it works
    bulk_twoway_execute_saxpy(a, x, y, z);
    assert(z == reference);

    // warm-up
    bulk_twoway_execute_saxpy(a, x, y, z);

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < num_trials; ++i)
    {
      bulk_twoway_execute_saxpy(a, x, y, z);
    }

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

    double seconds = elapsed.count() / num_trials;
    double gigabytes = double(3 * n * sizeof(float)) / (1 << 30);
    double bandwidth = gigabytes / seconds;

    std::cout << "bulk_twoway_execute_saxpy: " << bandwidth << " GB/s" << std::endl;
  }

  return 0;
}

