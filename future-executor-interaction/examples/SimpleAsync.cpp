#include <iostream>

#include "../include/futures_static_thread_pool.h"
#include <futures.h>

// Example taken from executors to ensure modified thread pool future still
// works

namespace execution = std::experimental::execution;
using std::experimental::futures_static_thread_pool;

template <class Executor, class Function>
auto async(Executor ex, Function f)
{
  return execution::require(ex, execution::twoway).twoway_execute(std::move(f));
}

int main() {
  futures_static_thread_pool pool{1};
  auto f = async(pool.executor(), []{ return 42; });
  std::cout << "result is " << f.get() << "\n";

  return 0;
}
