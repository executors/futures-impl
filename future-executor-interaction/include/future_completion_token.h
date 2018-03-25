#ifndef STD_EXPERIMENTAL_FUTURES_V1_COMPLETION_TOKEN_H
#define STD_EXPERIMENTAL_FUTURES_V1_COMPLETION_TOKEN_H

namespace std {
namespace experimental {
inline namespace executors_v1 {
namespace execution {

template<class T>
class future_completion_token {
public:
  future_completion_token() = default;
  future_completion_token(const future_completion_token&) = default;
  future_completion_token(future_completion_token&&) = default;
  future_completion_token(std::function<void(T&&)> func)
      : function_{std::move(func)} {}

  future_completion_token& operator=(const future_completion_token&) = default;
  future_completion_token& operator=(future_completion_token&&) = default;

  void operator()(T&& value) {
    function_(std::move(value));
  }

private:
  std::function<void(T&&)> function_;
};

} // execution
} // executors_v1
} // experimental
} // std

#endif // STD_EXPERIMENTAL_FUTURES_V1_COMPLETION_TOKEN_H
