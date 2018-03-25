#ifndef STD_EXPERIMENTAL_FUTURES_V1_STANDARD_PROMISE_SHARED_STATE_H
#define STD_EXPERIMENTAL_FUTURES_V1_STANDARD_PROMISE_SHARED_STATE_H

#include <mutex>
#include <condition_variable>

namespace std {
namespace experimental {
inline namespace futures_v1 {
namespace detail {

// Future type defined synchronization primitive
struct CVStruct {
  std::condition_variable cv;
  std::mutex cv_mutex;
  std::atomic<bool> ready{false};
};

template<class T>
class promise_shared_state {
public:
  // set_value will set the value if there is no task set
  // it will call the task if there is
  virtual void set_value(T&&) = 0;
  // set_task will store the task if value is not already satisfied
  // it will call the continuation if it is
  virtual void set_task(std::function<void(T&&)>&&) = 0;

  // Return null if the value is not currently set, or a pointer to the value
  // Does not block.
  virtual T* try_get() = 0;
};

template<class T>
class no_executor_promise_shared_state : public promise_shared_state<T> {
public:
  virtual void set_value(T&& value) {
    std::lock_guard<std::mutex> lck(m_);
    if(has_value_) {
      throw std::logic_error("set_value called twice on promise");
    }
    if(task_) {
      auto t = std::move(task_);
      // Store and move may mean a double move, but it means that the core
      // maintains the state in case the callback is only involved in waiting
      // not consuming the value.
      value_ = std::move(value);
      t(std::move(value_));
    } else {
      value_ = std::move(value);
    }
    has_value_ = true;
  }

  virtual void set_task(std::function<void(T&&)>&& task) {
    std::lock_guard<std::mutex> lck(m_);
    if(task_) {
      throw std::logic_error("set_task called twice on promise");
    }
    if(value_) {
      auto t = std::move(task);
      t(std::move(value_));
    } else {
      task_ = std::move(task);
    }
  }

  virtual T* try_get() {
    std::lock_guard<std::mutex> lck(m_);
    if(has_value_) {
      return &value_;
    } else {
      return nullptr;
    }
  }

private:
  std::mutex m_;
  bool has_value_ = false;
  T value_{};
  std::function<void(T&&)> task_;
};

// May not need this at all - may be able to make shared state non-virtual
// by hiding the exector in the callback
#if 0
template<class T, class Executor>
class executor_promise_shared_state {
public:
  virtual void set_value(T&&) {
    throw std::logic_error("set_value not implemented");
  }
  virtual void set_task(std::function<void(T&&)>&) {
    throw std::logic_error("set_task not implemented");
  }

private:
  std::mutex m_;
  bool has_value_ = false;
  T value_{};
  std::function<void(T&&)> task_;
  //Executor exec_;
};
#endif
} // detail
} // fuures_v1
} // experimental
} // std

#endif // STD_EXPERIMENTAL_FUTURES_V1_STANDARD_PROMISE_SHARED_STATE_H
