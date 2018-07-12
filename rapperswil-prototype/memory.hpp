// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(FUTURES_MEMORY_HPP)
#define FUTURES_MEMORY_HPP

#include <utility>
#include <new>
#include <memory>
#include <iterator>

#include "type_deduction.hpp"

///////////////////////////////////////////////////////////////////////////////

template <typename ForwardIt>
ForwardIt constexpr destroy(ForwardIt first, ForwardIt last)
{
  for (; first != last; ++first)
    destroy_at(std::addressof(*first));

  return first;
}

template <typename Allocator, typename ForwardIt>
ForwardIt constexpr destroy(Allocator&& alloc, ForwardIt first, ForwardIt last)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  using traits = typename std::allocator_traits<
    std::remove_cv_t<std::remove_reference_t<Allocator>>
  >::template rebind_traits<T>;
  auto alloc_T = typename traits::allocator_type(FWD(alloc));

  for (; first != last; ++first)
    destroy_at(alloc_T, std::addressof(*first));

  return first;
}

template <typename ForwardIt, typename Size>
ForwardIt constexpr destroy_n(ForwardIt first, Size n)
{
  for (; n > 0; (void) ++first, --n)
    destroy_at(std::addressof(*first));

  return first;
}

template <typename Allocator, typename ForwardIt, typename Size>
ForwardIt constexpr destroy_n(Allocator&& alloc, ForwardIt first, Size n)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  using traits = typename std::allocator_traits<
    std::remove_cv_t<std::remove_reference_t<Allocator>>
  >::template rebind_traits<T>;
  auto alloc_T = typename traits::allocator_type(FWD(alloc));

  for (; n > 0; (void) ++first, --n)
    destroy_at(alloc_T, std::addressof(*first));

  return first;
}

template <typename T>
void constexpr destroy_at(T* location) 
{
  location->~T();
}

template <typename Allocator, typename T>
void constexpr destroy_at(Allocator&& alloc, T* location) 
{
  using traits = typename std::allocator_traits<
    std::remove_cv_t<std::remove_reference_t<Allocator>>
  >::template rebind_traits<T>;
  auto alloc_T = typename traits::allocator_type(FWD(alloc));
 
  traits::destroy(FWD(alloc), location);
}

template <typename ForwardIt, typename... Args>
void uninitialized_construct(
  ForwardIt first, ForwardIt last, Args const&... args
)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;

  ForwardIt current = first;
  try {
    for (; current != last; ++current)
      ::new (static_cast<void*>(std::addressof(*current))) T(args...);
  } catch (...) {
    destroy(first, current);
    throw;
  }
}

template <typename Allocator, typename ForwardIt, typename... Args>
void uninitialized_allocator_construct(
  Allocator&& alloc, ForwardIt first, ForwardIt last, Args const&... args
)
{
  using T  = typename std::iterator_traits<ForwardIt>::value_type;
  using traits = typename std::allocator_traits<
    std::remove_cv_t<std::remove_reference_t<Allocator>>
  >::template rebind_traits<T>;
  auto alloc_T = typename traits::allocator_type(FWD(alloc));

  ForwardIt current = first;
  try {
    for (; current != last; ++current)
      traits::construct(alloc_T, std::addressof(*current), args...);
  } catch (...) {
    destroy(alloc_T, first, current);
    throw;
  }
}

template <typename ForwardIt, typename Size, typename... Args>
void uninitialized_construct_n(
  ForwardIt first, Size n, Args const&... args
)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;

  ForwardIt current = first;
  try {
    for (; n > 0; (void) ++current, --n)
      ::new (static_cast<void*>(std::addressof(*current))) T(args...);
  } catch (...) {
    destroy(first, current);
    throw;
  }
}

template <typename Allocator, typename ForwardIt, typename Size, typename... Args>
void uninitialized_allocator_construct_n(
  Allocator&& alloc, ForwardIt first, Size n, Args const&... args
)
{
  using T = typename std::iterator_traits<ForwardIt>::value_type;
  using traits = typename std::allocator_traits<
    std::remove_cv_t<std::remove_reference_t<Allocator>>
  >::template rebind_traits<T>;
  auto alloc_T = typename traits::allocator_type(FWD(alloc));

  ForwardIt current = first;
  try {
    for (; n > 0; (void) ++current, --n)
      traits::construct(alloc_T, std::addressof(*current), args...);
  } catch (...) {
    destroy(alloc_T, first, current);
    throw;
  }
}

///////////////////////////////////////////////////////////////////////////////

// wg21.link/p0316r0
template <typename T, typename Allocator>
struct allocator_delete final
{
  using allocator_type
    = typename std::remove_cv_t<std::remove_reference_t<Allocator>>::template
      rebind<T>::other; 
  using pointer = typename std::allocator_traits<allocator_type>::pointer;

  template <typename UAllocator>
  allocator_delete(UAllocator&& other) noexcept
    : alloc_(FWD(other))
  {}

  template <typename U, typename UAllocator>
  allocator_delete(
      allocator_delete<U, UAllocator> const& other
    ) noexcept
    : alloc_(other.get_allocator())
  {}
  template <typename U, typename UAllocator>
  allocator_delete(
      allocator_delete<U, UAllocator>&& other
    ) noexcept
    : alloc_(MV(other.get_allocator()))
  {}

  template <typename U, typename UAllocator>
  allocator_delete& operator=(
    allocator_delete<U, UAllocator> const& other
  ) noexcept
  {
    alloc_ = other.get_allocator();
    return *this;
  }
  template <typename U, typename UAllocator>
  allocator_delete& operator=(
    allocator_delete<U, UAllocator>&& other
  ) noexcept
  {
    alloc_ = MV(other.get_allocator());
    return *this;
  }

  void operator()(pointer p) 
  {
    using traits = std::allocator_traits<allocator_type>;

    if (nullptr != p)
    {
      traits::destroy(get_allocator(), p);
      traits::deallocate(get_allocator(), p, 1);
    }
  }

  allocator_type& get_allocator() noexcept { return alloc_; }
  allocator_type const& get_allocator() const noexcept { return alloc_; }

  void swap(allocator_delete& other) noexcept
  {
    using std::swap;
    swap(alloc_, other.alloc_);
  }

private:
  allocator_type alloc_;
};

template <typename T, typename Allocator>
struct allocator_array_delete final
{
  using allocator_type
    = typename std::remove_cv_t<std::remove_reference_t<Allocator>>::template
      rebind<T>::other; 
  using pointer = typename std::allocator_traits<allocator_type>::pointer;

  template <typename UAllocator>
  allocator_array_delete(UAllocator&& other, std::size_t n) noexcept
    : alloc_(FWD(other)), count_(n)
  {}

  template <typename U, typename UAllocator>
  allocator_array_delete(
      allocator_array_delete<U, UAllocator> const& other
    ) noexcept
    : alloc_(other.get_allocator()), count_(other.count_)
  {}
  template <typename U, typename UAllocator>
  allocator_array_delete(
      allocator_array_delete<U, UAllocator>&& other
    ) noexcept
    : alloc_(MV(other.get_allocator())), count_(other.count_)
  {}

  template <typename U, typename UAllocator>
  allocator_array_delete& operator=(
    allocator_array_delete<U, UAllocator> const& other
  ) noexcept
  {
    alloc_ = other.get_allocator();
    count_ = other.count_;
    return *this;
  }
  template <typename U, typename UAllocator>
  allocator_array_delete& operator=(
    allocator_array_delete<U, UAllocator>&& other
  ) noexcept
  {
    alloc_ = MV(other.get_allocator());
    count_ = other.count_;
    return *this;
  }

  void operator()(pointer p) 
  {
    using traits = std::allocator_traits<allocator_type>;

    if (nullptr != p)
    {
      destroy_n(get_allocator(), p, count_);
      traits::deallocate(get_allocator(), p, count_);
    }
  }

  allocator_type& get_allocator() noexcept { return alloc_; }
  allocator_type const& get_allocator() const noexcept { return alloc_; }

  void swap(allocator_array_delete& other) noexcept
  {
    using std::swap;
    swap(alloc_, other.alloc_);
    swap(count_, other.count_);
  }

private:
  allocator_type alloc_;
  std::size_t    count_;
};

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename Allocator, typename... Args>
auto allocate_unique(
  Allocator&& alloc, Args&&... args
)
{
  using traits = typename std::allocator_traits<
    std::remove_cv_t<std::remove_reference_t<Allocator>>
  >::template rebind_traits<T>;
  auto alloc_T = typename traits::allocator_type(FWD(alloc));

  auto hold_deleter = [&alloc_T] (auto p) {
    traits::deallocate(alloc_T, p, 1);
  };
  using hold_t = std::unique_ptr<T, decltype(hold_deleter)>;
  auto hold = hold_t(traits::allocate(alloc_T, 1), hold_deleter);

  traits::construct(alloc_T, hold.get(), FWD(args)...);
  auto deleter = allocator_delete<T, decltype(alloc_T)>(alloc_T);
  return std::unique_ptr<T, decltype(deleter)>(hold.release(), move(deleter));
}

template <typename T, typename Allocator>
auto uninitialized_allocate_unique(
  Allocator&& alloc
)
{
  using traits = typename std::allocator_traits<
    std::remove_cv_t<std::remove_reference_t<Allocator>>
  >::template rebind_traits<T>;
  auto alloc_T = typename traits::allocator_type(FWD(alloc));

  auto hold_deleter = [&alloc_T] (auto p) {
    traits::deallocate(alloc_T, p, 1);
  };
  using hold_t = std::unique_ptr<T, decltype(hold_deleter)>;
  auto hold = hold_t(traits::allocate(alloc_T, 1), hold_deleter);

  auto deleter = allocator_delete<T, decltype(alloc_T)>(alloc_T);
  return std::unique_ptr<T, decltype(deleter)>(hold.release(), move(deleter));
}

template <typename T, typename Allocator, typename Size, typename... Args>
auto allocate_unique_n(
  Allocator&& alloc, Size n, Args&&... args
)
{
  using traits = typename std::allocator_traits<
    std::remove_cv_t<std::remove_reference_t<Allocator>>
  >::template rebind_traits<T>;
  auto alloc_T = typename traits::allocator_type(FWD(alloc));

  auto hold_deleter = [n, &alloc_T] (auto p) {
    traits::deallocate(alloc_T, p, n);
  };
  using hold_t = std::unique_ptr<T, decltype(hold_deleter)>;
  auto hold = hold_t(traits::allocate(alloc_T, n), hold_deleter);

  uninitialized_allocator_construct_n(alloc_T, hold.get(), n, FWD(args)...);
  auto deleter = allocator_array_delete<T, Allocator>(alloc_T, n);
  return std::unique_ptr<T, decltype(deleter)>(hold.release(), move(deleter));
}

template <typename T, typename Allocator, typename Size>
auto uninitialized_allocate_unique_n(
  Allocator&& alloc, Size n
)
{
  using traits = typename std::allocator_traits<
    std::remove_cv_t<std::remove_reference_t<Allocator>>
  >::template rebind_traits<T>;
  auto alloc_T = typename traits::allocator_type(FWD(alloc));

  auto hold_deleter = [n, &alloc_T] (auto p) {
    traits::deallocate(alloc_T, p, n);
  };
  using hold_t = std::unique_ptr<T[], decltype(hold_deleter)>;
  auto hold = hold_t(traits::allocate(alloc_T, n), hold_deleter);

  auto deleter = allocator_array_delete<T, Allocator>(alloc_T, n);
  return std::unique_ptr<T[], decltype(deleter)>(hold.release(), move(deleter));
}

///////////////////////////////////////////////////////////////////////////////

#endif // FUTURES_MEMORY_HPP

