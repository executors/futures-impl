#pragma once

#include "always_ready_future.hpp"
#include <type_traits>

// inline_executor is an executor which always creates execution "inline"
// Therefore, the execution it creates always blocks the execution of its client.

class inline_executor
{
  public:
    bool operator==(const inline_executor&) { return true; }
    bool operator!=(const inline_executor&) { return false;}
    const inline_executor& context() const { return *this; }

    template<class T>
    using future = always_ready_future<T>;

    template<class Function>
    auto sync_execute(Function&& f) const
    {
      std::forward<Function>(f)();
    }

    template<class Function>
    future<std::result_of_t<std::decay_t<Function>()>>
    twoway_execute(Function&& f) const
    {
      using result_type = std::result_of_t<std::decay_t<Function>()>;

      try
      {
        return detail::try_invoke([&]
        {
          return sync_execute(std::forward<Function>(f));
        });
      }
      catch(...)
      {
        return always_ready_future<result_type>(std::current_exception());
      }
    }

    template<class Function, class Factory1, class Factory2>
    auto bulk_sync_execute(Function f, size_t n, Factory1 result_factory, Factory2 shared_factory) const
    {
      auto result = result_factory();
      auto shared = shared_factory();

      for(size_t i = 0; i < n; ++i)
      {
        f(i, result, shared);
      }

      return result;
    }

    template<class Function, class Factory1, class Factory2>
    future<std::result_of_t<Factory1()>>
    bulk_twoway_execute(Function f, size_t n, Factory1 result_factory, Factory2 shared_factory) const
    {
      using result_type = std::result_of_t<Factory1()>;

      try
      {
        return detail::try_invoke([&]
        {
          return bulk_sync_execute(f, n, result_factory, shared_factory);
        });
      }
      catch(...)
      {
        return always_ready_future<result_type>(std::current_exception());
      }
    }
};

