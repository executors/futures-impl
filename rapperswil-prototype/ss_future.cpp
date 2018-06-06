#include <utility>
#include <functional>
#include <optional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <iostream>

using std::move;
using std::declval;
using std::pair;
using std::function;
using std::optional;
using std::shared_ptr;
using std::make_shared;
using std::mutex;
using std::condition_variable;
using std::unique_lock;
using std::thread;
using std::cout;
using std::endl;

#define FWD(x) std::forward<decltype(x)>(x)
#define RETOF(C, V) decltype(std::declval<C>()(std::declval<V>()))
#define NRETOF(C) decltype(std::declval<C>()())

template <typename Executor, typename T>
struct executor_future {
  using type = typename Executor::future;
};

template <typename Executor, typename T>
using executor_future_t = typename executor_future<Executor, T>::type;

template <typename T>
struct ss_asynchronous_value;

template <typename T, typename Executor>
struct ss_future;

template <typename T>
struct ss_semi_future;

template <typename T>
struct ss_promise;

struct ss_executor;

template <typename T>
pair<ss_promise<T>, ss_future<T, ss_executor>>
make_promise_contract(ss_executor exec);

///////////////////////////////////////////////////////////////////////////////

struct ss_executor {
  template <typename Continuation, typename T>
  ss_future<RETOF(Continuation, T), ss_executor>
  then_execute(Continuation&& c, ss_future<T, ss_executor> f) const {
    auto [p, g] = make_promise_contract<RETOF(Continuation, T)>(f.next_exec);
    f.ss->set_trigger(
      [f = move(f), c = FWD(c), p = move(p)] (T v) mutable {
        f.next_exec.execute(
          [p = move(p), c = FWD(c), v = move(v)] () mutable
          { move(p).set_value(c(v)); }
        );
      }
    );
    return move(g);
  }

  template <typename F>
  void execute(F&& f) const {
    thread t(FWD(f));
    t.detach();
  }

  template <typename F>
  ss_future<NRETOF(F), ss_executor> twoway_execute(F&& f) const {
    auto [p, g] = make_promise_contract<NRETOF(F)>(*this);
    execute([p = move(p), f = FWD(f)] () mutable { move(p).set_value(f()); });
    return move(g);
  }
};

template <typename T>
struct ss_asynchronous_value {
  void set_value(T&& v) {
    bool run_trigger = false;
    decltype(trig) t;

    {
      unique_lock l(mtx);
      if (has_trigger)
        { swap(t, trig); run_trigger = true; }
      else
        { value = FWD(v); has_value = true; }   
    }

    if (run_trigger)
    {
      move(t)(FWD(v));

      unique_lock l(mtx);
      done = true;
    }
  }
  
  template <typename Trigger>
  void set_trigger(Trigger&& t) {
    bool run_trigger = false;
    optional<T> v;

    {
      unique_lock l(mtx);
      if (has_value)
        { v = move(value); run_trigger = true; }
      else
        { trig = FWD(t); has_trigger = true; }
    }

    if (run_trigger)
    {
      FWD(t)(move(*v));

      unique_lock l(mtx);
      done = true;
    }
  }

  ss_asynchronous_value()
    : mtx(), value(), trig(), has_value(false), has_trigger(false), done(false)
  {}

  mutex mutable       mtx;
  T                   value;
  function<void(T&&)> trig;
  bool                has_value;
  bool                has_trigger;
  bool                done;
};

template <typename T, typename Executor>
struct ss_future {
  template <typename UExecutor>
  ss_future<T, UExecutor> via(UExecutor&& exec) && {
    ss_future<T, UExecutor> tmp{ss, move(exec)};
    ss.reset();
    return tmp;
  }

  template <typename Continuation>
  executor_future_t<Executor, RETOF(Continuation, T)>
  then(Continuation&& c) && {
    auto [p, g] = make_promise_contract<RETOF(Continuation, T)>(next_exec);
    auto h = next_exec.then_execute(FWD(c), move(g));
    ss->set_trigger([p = move(p)] (T v) mutable { move(p).set_value(v); });
    ss.reset();
    return move(h);
  }
  
  shared_ptr<ss_asynchronous_value<T>> ss;
  Executor                             next_exec;
};

template <typename T>
struct ss_future<T, ss_executor> {
  template <typename UExecutor>
  ss_future<T, UExecutor> via(UExecutor&& exec) && {
    ss_future<T, UExecutor> tmp{ss, FWD(exec)};
    ss.reset();
    return move(tmp);
  }

  template <typename Continuation>
  ss_future<RETOF(Continuation, T), ss_executor>
  then(Continuation&& c) && {
    auto g = next_exec.then_execute(FWD(c), *this);
    ss.reset();
    return move(g);
  }
  
  shared_ptr<ss_asynchronous_value<T>> ss;
  ss_executor                          next_exec;
};


template <typename T>
struct ss_semi_future {
  template <typename UExecutor>
  ss_future<T, UExecutor> via(UExecutor&& exec) && {
    ss_future<T, UExecutor> tmp{ss, FWD(exec)};
    ss.reset();
    return move(tmp);
  }
  
  shared_ptr<ss_asynchronous_value<T>> ss;
};

template <typename T>
struct ss_promise {
  void set_value(T&& value) && {
    ss->set_value(FWD(value));
    ss.reset();
  }
    
  shared_ptr<ss_asynchronous_value<T>> ss;
};

template <typename T>
pair<ss_promise<T>, ss_future<T, ss_executor>>
make_promise_contract(ss_executor exec) {
  ss_future<T, ss_executor> f{make_shared<ss_asynchronous_value<T>>(), exec};
  ss_promise<T> p{f.ss};
  return {p, f};
}

template <typename T, typename Executor>
void future_wait(ss_future<T, Executor>&& f) {
  condition_variable cv;

  auto ss = f.ss;

  move(f).then([&cv] (auto&& v) { cv.notify_one(); return FWD(v); });

  unique_lock l(ss->mtx);
  cv.wait(l, [ss] { return ss->done; });
}

///////////////////////////////////////////////////////////////////////////////

int main() {
  ss_executor exec;

  auto f = exec.twoway_execute([] { cout << "0" << endl; return 17; });

  auto g = move(f).then([] (int v) { cout << v << endl; return 3.14; }); 

  auto h = move(g).then([] (double v) { cout << v << endl; return 42; }); 

  future_wait(move(h));
}

