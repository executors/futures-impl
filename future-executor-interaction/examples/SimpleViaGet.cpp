#include <iostream>

#include "TestHelper.h"
#include "../include/futures_static_thread_pool.h"
#include <futures.h>

namespace execution = std::experimental::execution;
using std::experimental::futures_static_thread_pool;

struct inline_executor
{
public:
  friend bool operator==(
    const inline_executor&, const inline_executor&) noexcept { return true; }
  friend bool operator!=(
    const inline_executor&, const inline_executor&) noexcept { return false; }
  template<class Function> void execute(Function f) const noexcept { f(); }
  constexpr bool query(execution::oneway_t) { return true; }
  constexpr bool query(execution::single_t) { return true; }
};

int main() {
  const std::string TESTNAME = "Simple via and get with inline executor";
  const int EXPECTED = 3;

  std::experimental::standard_promise<int> promise;

  auto f = promise.get_semi_future().via(inline_executor{});

  std::thread setter{[&promise](){
    promise.set_value(3);
  }};

  auto result = f.get();
  setter.join();
  std::cout << TESTNAME << "\t" << check(result, EXPECTED) << "\n";

  return 0;
}
