#ifndef STD_EXPERIMENTAL_FUTURES_V1_STANDARD_PROMISE_H
#define STD_EXPERIMENTAL_FUTURES_V1_STANDARD_PROMISE_H

#include <mutex>
#include <standard_promise_shared_state.h>
#include <standard_semi_future.h>

namespace std {
namespace experimental {
inline namespace futures_v1 {

template<class T>
class standard_promise {
public:
  standard_promise() {
    auto core = std::make_shared<detail::no_executor_promise_shared_state<T>>();
    core_ = std::static_pointer_cast<detail::promise_shared_state<T>>(
      std::move(core));
  }

  standard_semi_future<T> get_semi_future() const {
    return standard_semi_future<T>(core_);
  }

  void set_value(T&& value) {
    core_->set_value(std::move(value));
  }

  void swap(standard_promise& other) {
    std::swap(core_, other.core_);
  }

private:
  std::shared_ptr<detail::promise_shared_state<T>> core_;
};

} // fuures_v1
} // experimental
} // std

#endif // STD_EXPERIMENTAL_FUTURES_V1_STANDARD_PROMISE_H
