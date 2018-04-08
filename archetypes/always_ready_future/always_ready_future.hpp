// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <exception>
#include <type_traits>
#include <utility>
#include <future>
#include "variant.hpp" // my compiler doesn't have <variant> yet
#include "optional.hpp" // my compiler doesn't have <optional> yet

// always_ready_future is intended to be a lightweight wrapper around
// a ready value that fulfills the basic requirements of the Future concept
// (whatever that may be someday)
//
// Executors which always block their client can use always_ready_future as their
// associated future and still expose two-way asynchronous execution functions like .twoway_execute()

template<class T>
class always_ready_future
{
  public:
    // Default constructor creates an invalid always_ready_future
    // Postcondition: !valid()
    always_ready_future() = default;

    always_ready_future(const T& value) : state_(value) {}

    always_ready_future(T&& value) : state_(std::move(value)) {}

    always_ready_future(std::exception_ptr e) : state_(e) {}

    always_ready_future(always_ready_future&& other)
      : state_()
    {
      using std::swap;
      swap(state_, other.state_);
    }

    always_ready_future& operator=(always_ready_future&& other)
    {
      state_.reset();

      using std::swap;
      swap(state_, other.state_);

      return *this;
    }

    constexpr static bool is_ready()
    {
      return true;
    }

  private:
    struct get_ptr_visitor
    {
      T* operator()(T& result) const
      {
        return &result;
      }

      T* operator()(std::exception_ptr& e) const
      {
        std::rethrow_exception(e);
        return nullptr;
      }
    };

    T& get_ref()
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      return *std::experimental::visit(get_ptr_visitor(), *state_);
    }

  public:
    T get()
    {
      T result = std::move(get_ref());

      invalidate();

      return result;
    }

    void wait() const
    {
      // wait() is a no-op: *this is always ready
    }

    bool valid() const
    {
      return state_.has_value();
    }

  private:
    void invalidate()
    {
      state_ = std::experimental::nullopt;
    }

    std::experimental::optional<std::experimental::variant<T,std::exception_ptr>> state_;
};


template<>
class always_ready_future<void>
{
  public:
    // XXX the default constructor creates a ready, valid future, but we may wish
    //     to redefine the default constructor to create an invalid future
    //     in such a scheme, we would need to distinguish another constructor for
    //     creating a ready (void) result.
    //     We could create an emplacing constructor distinguished with an in_place_t parameter
    always_ready_future() : valid_(true) {}

    always_ready_future(std::exception_ptr e) : exception_(e), valid_(true) {}

    always_ready_future(always_ready_future&& other)
      : exception_(), valid_(false)
    {
      using std::swap;
      swap(exception_, other.exception_);
      swap(valid_, other.valid_);
    }

    always_ready_future& operator=(always_ready_future&& other)
    {
      exception_.reset();
      valid_ = false;

      using std::swap;
      swap(exception_, other.exception_);
      swap(valid_, other.valid_);
      return *this;
    }

    constexpr static bool is_ready()
    {
      return true;
    }

  public:
    void get()
    {
      if(!valid())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      invalidate();

      if(exception_)
      {
        std::rethrow_exception(exception_.value());
      }
    }

    void wait() const
    {
      // wait() is a no-op: this is always ready
    }

    bool valid() const
    {
      return valid_;
    }

  private:
    void invalidate()
    {
      valid_ = false;
    }

    std::experimental::optional<std::exception_ptr> exception_;

    bool valid_;
};


template<class T>
always_ready_future<std::decay_t<T>> make_always_ready_future(T&& value)
{
  return always_ready_future<std::decay_t<T>>(std::forward<T>(value));
}

inline always_ready_future<void> make_always_ready_future()
{
  return always_ready_future<void>();
}


template<class T>
always_ready_future<T> make_always_ready_exceptional_future(std::exception_ptr e)
{
  return always_ready_future<T>(e);
}


namespace detail
{


template<class Function,
         class Result = std::result_of_t<std::decay_t<Function>()>,
         class = std::enable_if_t<
           !std::is_void<Result>::value
         >>
always_ready_future<Result>
try_invoke(Function&& f)
{
  try
  {
    return always_ready_future<Result>(std::forward<Function>(f)());
  }
  catch(...)
  {
    return always_ready_future<Result>(std::current_exception());
  }
}


template<class Function,
         class Result = std::result_of_t<std::decay_t<Function>()>,
         class = std::enable_if_t<
           std::is_void<Result>::value
         >>
always_ready_future<void>
try_invoke(Function&& f)
{
  try
  {
    std::forward<Function>(f)();
    return always_ready_future<void>();
  }
  catch(...)
  {
    return always_ready_future<void>(std::current_exception());
  }
}


}

