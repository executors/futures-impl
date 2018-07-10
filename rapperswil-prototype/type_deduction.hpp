// Copyright (c)      2018 NVIDIA Corporation 
//                         (Bryce Adelstein Lelbach <brycelelbach@gmail.com>)
// Copyright (c) 2013-2018 Eric Niebler (`RETURNS`)
// Copyright (c) 2016-2018 Casey Carter (`RETURNS`)
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(FUTURES_TYPE_DEDUCTION_HPP)
#define FUTURES_TYPE_DEDUCTION_HPP

#include <utility>

#include "preprocessor.hpp"

///////////////////////////////////////////////////////////////////////////////

/// \def FWD(x)
/// \brief Performs universal forwarding of a universal reference.
///
#define FWD(x) ::std::forward<decltype(x)>(x)

/// \def MV(x)
/// \brief Moves `x`.
///
#define MV(x) ::std::move(x)

/// \def RETOF(invocable, ...)
/// \brief Expands to the type returned by invoking an instance of the invocable
///        type \a invocable with parameters of type <tt>__VA_ARGS__</tt>.
///
#define RETOF(...)   PP_DISPATCH(RETOF, __VA_ARGS__)
#define RETOF1(C)    decltype(::std::declval<C>()())
#define RETOF2(C, V) decltype(::std::declval<C>()(::std::declval<V>()))

/// \def RETURNS(...)
/// \brief Expands to a function definition that returns the expression
///        <tt>__VA_ARGS__</tt>.
///
#define RETURNS(...)                                                          \
  noexcept(noexcept(decltype(__VA_ARGS__)(__VA_ARGS__)))                      \
  { return (__VA_ARGS__); }                                                   \
  /**/

/// \def AUTORETURNS(...)
/// \brief Expands to a function definition, including a trailing returning
///        type, that returns the expression <tt>__VA_ARGS__</tt>.
///
#define AUTORETURNS(...)                                                      \
  -> decltype(__VA_ARGS__)                                                    \
  noexcept(noexcept(decltype(__VA_ARGS__)(__VA_ARGS__)))                      \
  { return (__VA_ARGS__); }                                                   \
  /**/

///////////////////////////////////////////////////////////////////////////////

#endif // FUTURES_TYPE_DEDUCTION_HPP

