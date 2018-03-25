#ifndef STD_EXPERIMENTAL_FUTURES_V1_EXECUTORS_HELPERS_H
#define STD_EXPERIMENTAL_FUTURES_V1_EXECUTORS_HELPERS_H

// Temporary helpers for missing features of executors_impl

#include <experimental/bits/is_executor.h>

namespace std {
namespace experimental {
inline namespace executors_v1 {
namespace execution {

struct then_t{};

namespace is_then_executor_impl {

template<class...>
struct type_check
{
  typedef void type;
};

struct dummy {};

struct unary_function
{
  template<class P>
  dummy operator()(P&&) { return {}; }
};

// This should work until we define the future concept to check against
struct trivial_future
{
  using value_type = dummy;
  dummy get() {
    return {};
  }
};

template<class T, class = void>
struct eval : std::false_type {};

template<class T>
struct eval<T,
  typename type_check<
    decltype(static_cast<const dummy&>(std::declval<const T&>().then_execute(std::declval<unary_function>(), std::declval<trivial_future>()).get())),
    decltype(static_cast<const dummy&>(std::declval<const T&>().then_execute(std::declval<unary_function&>(), std::declval<trivial_future>()).get())),
    decltype(static_cast<const dummy&>(std::declval<const T&>().then_execute(std::declval<const unary_function&>(), std::declval<trivial_future>()).get())),
    decltype(static_cast<const dummy&>(std::declval<const T&>().then_execute(std::declval<unary_function&&>(), std::declval<trivial_future>()).get()))
	>::type> : is_executor_impl::eval<T> {};

} // namespace is_oneway_executor_impl

template<class Executor>
struct is_then_executor : is_then_executor_impl::eval<Executor> {};

template<class Executor>
constexpr bool is_then_executor_v = is_then_executor<Executor>::value;

} // namespace execution
} // inline namespace executors_v1
} // namespace experimental
} // namespace std

#endif // STD_EXPERIMENTAL_FUTURES_V1_EXECUTORS_HELPERS_H
