// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#if !defined(FUTURES_SS_FUTURE_HPP)
#define FUTURES_SS_FUTURE_HPP

#include "executor_traits.hpp"
#include "type_deduction.hpp"

#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

#include <typeinfo>

template <typename T>
struct ss_async_value;

template <typename T, typename Executor>
struct ss_executor_future;

template <typename T>
struct ss_semi_future;

template <typename T>
struct ss_executor_promise;

struct ss_executor;

///////////////////////////////////////////////////////////////////////////////

struct ss_executor final
{
  template <typename T>
  using future = ss_executor_future<T, ss_executor>;

  template <typename T>
  using promise = ss_executor_promise<T>;

  template <typename Operation>
  void execute(Operation&& op) const
  {
    std::thread t(FWD(op));
    t.detach();
  }

  template <typename Operation>
  ss_executor_future<RETOF(Operation), ss_executor>
  twoway_execute(Operation&& op) const
  {
    using U = RETOF(Operation);

    auto ss = std::make_shared<ss_async_value<U>>();
    ss_executor_future<U, ss_executor> f(ss, *this, *this);
    ss_executor_promise<U> p(ss);

    execute([op = FWD(op), p = MV(p)] () mutable { MV(p).set_value(op()); });

    return MV(f);
  }

  // Internal -> Internal Dependent Execution.
  template <typename Operation, typename T>
  ss_executor_future<RETOF(Operation, T), ss_executor>
  then_execute(Operation&& op, ss_executor_future<T, ss_executor>&& f) const
  {
    using U = RETOF(Operation, T);

    auto ss = std::make_shared<ss_async_value<U>>();
    ss_executor_future<U, ss_executor> g(ss, f.next_executor(), *this);
    ss_executor_promise<U> p(ss);

    auto&& fexec = MV(f.next_executor());
    auto&& fss = MV(f.shared_state());
    fss->set_trigger(
      [op = FWD(op), p = MV(p), fexec = MV(fexec)] (T v) mutable
      {
        fexec.execute(
          [op = FWD(op), p = MV(p), v = MV(v)] () mutable
          { MV(p).set_value(op(v)); }
        );
      }
    );

    return MV(g);
  }

  // Internal -> External Dependent Execution.
  template <typename Operation, typename T, typename Executor>
  executor_future_t<Executor, RETOF(Operation, T)>
  then_execute(Operation&& op, ss_executor_future<T, Executor>&& f) const
  {
    using U = RETOF(Operation, T);

    auto p_g = ::make_promise<T>(f.next_executor());
    auto& p  = p_g.first;
    auto& g  = p_g.second;

    auto h = MV(f.next_executor()).then_execute(FWD(op), MV(g));

    auto&& fss = MV(f.shared_state());
    fss->set_trigger([p = MV(p)] (T v) mutable { MV(p).set_value(MV(v)); });
    fss.reset();

    return MV(h);
  }

  template <typename T>
  std::pair<ss_executor_promise<T>, ss_executor_future<T, ss_executor>>
  make_promise() const
  {
    auto ss = std::make_shared<ss_async_value<T>>();
    ss_executor_future<T, ss_executor> f(ss, *this, *this);
    ss_executor_promise<T> p(ss);
    return {MV(p), MV(f)};
  }

  template <typename T, typename Executor>
  void wait(ss_executor_future<T, Executor>&& f) const
  {
    std::condition_variable cv;

    auto&& fss = MV(f).shared_state();

    then_execute([&cv] (auto&& v) { cv.notify_one(); return FWD(v); }, MV(f));

    std::unique_lock<std::mutex> l(fss->mutex());
    cv.wait(l, [fss = MV(fss), &l] { return fss->done(l); });
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct ss_async_value final
{
private:
  std::mutex mutable       mutex_;
  T                        content_;
  std::function<void(T&&)> trigger_;
  bool                     has_content_;
  bool                     has_trigger_;

public:
  ss_async_value()
    : mutex_(), content_(), trigger_(), has_content_(false), has_trigger_(false)
  {}

  void set_value(T&& v)
  {
    bool run_trigger = false;
    decltype(trigger_) t;

    {
      std::unique_lock<std::mutex> l(mutex_);
      if (has_trigger_)
        { t = MV(trigger_); run_trigger = true; }
      else
        { content_ = FWD(v); has_content_ = true; }
    }

    if (run_trigger)
    {
      MV(t)(FWD(v));

      std::unique_lock<std::mutex> l(mutex_);
      has_content_ = true;
    }
  }

  template <typename Trigger>
  void set_trigger(Trigger&& t)
  {
    bool run_trigger = false;
    T v; // TODO: Should be an `optional`.

    {
      std::unique_lock<std::mutex> l(mutex_);
      if (has_content_)
        { v = MV(content_); run_trigger = true; }
      else
        { trigger_ = FWD(t); has_trigger_ = true; }
    }

    if (run_trigger)
    {
      FWD(t)(MV(v));

      std::unique_lock<std::mutex> l(mutex_);
      has_trigger_ = true;
    }
  }

  std::mutex& mutex() const
  {
    return mutex_;
  }

  bool done() const
  {
    std::unique_lock<std::mutex> l(mutex_);
    return has_content_ && has_trigger_;
  }

  bool done(std::unique_lock<std::mutex>& l) const
  {
    if (!l) l.lock();
    return has_content_ && has_trigger_;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct ss_executor_promise final
{
  friend struct ss_executor;

  using shared_state_type = std::shared_ptr<ss_async_value<T>>;

private:
  shared_state_type ss_;

  ss_executor_promise(std::shared_ptr<ss_async_value<T>> s)
    : ss_(s)
  {}

public:
  void set_value(T&& value) &&
  {
    ss_->set_value(FWD(value));
    ss_.reset();
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename NextExecutor>
struct ss_executor_future final
{
  friend struct ss_executor;

  template <typename U, typename UNextExecutor>
  friend struct ss_executor_future;

  template <typename U>
  friend struct ss_semi_future;

  using shared_state_type = std::shared_ptr<ss_async_value<T>>;

private:
  shared_state_type ss_;
  ss_executor       this_executor_;
  NextExecutor      next_executor_;

  ss_executor_future(
      shared_state_type const& s, ss_executor te, NextExecutor ne
    )
    : ss_(s), this_executor_(te), next_executor_(MV(ne))
  {}

  ss_executor_future(
      shared_state_type&& s, ss_executor te, NextExecutor ne
    )
    : ss_(MV(s)), this_executor_(te), next_executor_(MV(ne))
  {}

  shared_state_type&       shared_state()       RETURNS(ss_);
  shared_state_type const& shared_state() const RETURNS(ss_);

  ss_executor&       this_executor()       RETURNS(this_executor_);
  ss_executor const& this_executor() const RETURNS(this_executor_);

  NextExecutor&       next_executor()       RETURNS(next_executor_);
  NextExecutor const& next_executor() const RETURNS(next_executor_);

public:
  template <typename UNextExecutor>
  auto via(UNextExecutor&& exec) &&
  {
    return ss_executor_future<
      T, std::remove_cv_t<std::remove_reference_t<UNextExecutor>>
    >(MV(ss_), MV(this_executor_), FWD(exec));
  }

  template <typename Operation>
  executor_future_t<NextExecutor, RETOF(Operation, T)> then(Operation&& op) &&
  {
    return this_executor_.then_execute(FWD(op), MV(*this));
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct ss_semi_future final
{
  using shared_state_type = std::shared_ptr<ss_async_value<T>>;

private:
  shared_state_type ss_;

public:
  template <typename UExecutor>
  auto via(UExecutor&& exec) &&
  {
    return ss_executor_future<
      T, std::remove_cv_t<std::remove_reference_t<UExecutor>>
    >(MV(ss_), FWD(exec));
  }
};

///////////////////////////////////////////////////////////////////////////////

#endif // FUTURES_SS_FUTURE_HPP

