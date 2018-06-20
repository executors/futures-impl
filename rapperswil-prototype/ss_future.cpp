#include "executor_traits.hpp"
#include "type_deduction.hpp"

#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

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
    ss_executor_future<U, ss_executor> f(ss, *this);
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
    ss_executor_future<U, ss_executor> g(ss, *this);
    ss_executor_promise<U> p(ss);

    auto&& fexec = MV(f).executor();
    auto&& fss = MV(f).shared_state();
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
  ss_executor_future<RETOF(Operation, T), Executor>
  then_execute(Operation&& op, ss_executor_future<T, Executor>&& f) &&
  {
    using U = RETOF(Operation, T);

    auto p_g = make_promise<U>(f.exec);
    auto& p  = p_g.first;
    auto& g  = p_g.second;

    auto h = f.exec.then_execute(FWD(op), MV(g));

    auto&& fss = MV(f).shared_state();
    fss->set_trigger([p = MV(p)] (T v) mutable { MV(p).set_value(MV(v)); });
    fss.reset();

    return MV(h);
  }

  template <typename T>
  std::pair<ss_executor_promise<T>, ss_executor_future<T, ss_executor>>
  make_promise() const
  {
    auto ss = std::make_shared<ss_async_value<T>>();
    ss_executor_future<T, ss_executor> f(ss, *this);
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

template <typename T, typename Executor>
struct ss_executor_future final
{
  friend struct ss_executor;

  using shared_state_type = std::shared_ptr<ss_async_value<T>>;

private:
  shared_state_type ss_;
  Executor          exec_;

  ss_executor_future(shared_state_type s, Executor e)
    : ss_(s), exec_(e)
  {}

  shared_state_type&       shared_state() &      RETURNS(ss_)
  shared_state_type const& shared_state() const& RETURNS(ss_)
  shared_state_type&&      shared_state() &&     RETURNS(MV(ss_))

  Executor&       executor() &      RETURNS(exec_)
  Executor const& executor() const& RETURNS(exec_)
  Executor&&      executor() &&     RETURNS(MV(exec_))

public:
  template <typename UExecutor>
  ss_executor_future<T, UExecutor> via(UExecutor&& exec) &&
  {
    return ss_executor_future<T, UExecutor>(MV(ss_), FWD(exec));
  }

  template <typename Operation>
  executor_future_t<Executor, RETOF(Operation, T)> then(Operation&& op) &&
  {
    return exec_.then_execute(FWD(op), MV(*this));
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
  ss_executor_future<T, UExecutor> via(UExecutor&& exec) &&
  {
    return ss_executor_future<T, UExecutor>(MV(ss_), FWD(exec));
  }
};

///////////////////////////////////////////////////////////////////////////////

int main()
{
  {
    ss_executor exec;

    auto p_f = make_promise<int>(exec);
    ss_executor_promise<int>&             p = p_f.first;
    ss_executor_future<int, ss_executor>& f = p_f.second;

    auto g = MV(f).then(
      [] (int x) { printf("%u\n", x); return x + 1; });

    auto h = MV(g).then(
      [] (int x) { printf("%u\n", x); return x + 2; });

    MV(p).set_value(1);

    exec.wait(MV(h));
  }

  {
    ss_executor exec;

    auto f = exec.twoway_execute(
      [] { printf("0\n"); return 17; });

    auto g = MV(f).then(
      [] (int x) { printf("%u\n", x); return 3.14; }); 

    auto h = MV(g).then(
      [] (double x) { printf("%g\n", x); return 42; }); 

    exec.wait(MV(h));
  }
}

