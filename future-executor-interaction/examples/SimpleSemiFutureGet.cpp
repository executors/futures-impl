#include <iostream>

#include "TestHelper.h"
#include <futures.h>
#include <thread>

int main() {
  {
    const std::string TESTNAME = "Simple get";
    const int EXPECTED = 3;
    std::experimental::standard_promise<int> promise;

    auto s = promise.get_semi_future();

    std::thread setter{[&promise](){
      promise.set_value(3);
    }};

    auto result = s.get();
    setter.join();
    std::cout << TESTNAME << "\t" << check(result, EXPECTED) << "\n";
  }
  {
    const std::string TESTNAME = "Simple wait and get";
    const int EXPECTED = 4;
    std::experimental::standard_promise<int> promise;

    auto s = promise.get_semi_future();

    std::thread setter{[&promise](){
      promise.set_value(4);
    }};

    s.wait();
    auto result = s.get();

    setter.join();
    std::cout << TESTNAME << "\t" << check(result, EXPECTED) << "\n";
  }
  {
    const std::string TESTNAME = "Simple wait until and get";
    const int EXPECTED = 5;
    std::experimental::standard_promise<int> promise;

    auto s = promise.get_semi_future();


    s.wait_until(
      std::chrono::steady_clock::now() + std::chrono::milliseconds{100});
    s.wait_until(
      std::chrono::steady_clock::now() + std::chrono::milliseconds{100});
    // Now create thread to satisfy promise concurrently
    std::thread setter{[&promise](){
      promise.set_value(5);
    }};
    s.wait();
    auto result = s.get();

    setter.join();
    std::cout << TESTNAME << "\t" << check(result, EXPECTED) << "\n";
  }

  return 0;
}
