// This is an example executor and future implementation that is intended for 
// single-threaded a synchronous networking uses. It avoids all synchronization
// overhead by not being shared between threads. Technically, since it uses
// shared_ptr, the reference counts are updated with atomic RMW operations,
// although that could easily be solved by using boost::intrusive_ptr, or
// rolling your own.
//
// This is based on the conceptual design of nim-lang's asyncdispatch and
// asyncfutures libraries:
// https://github.com/nim-lang/Nim/blob/devel/lib/pure/asyncdispatch.nim
// https://github.com/nim-lang/Nim/blob/devel/lib/pure/asyncfutures.nim
// It also shares some structure (but no code!) with the Future implementation
// used by MongoDB due to me being the author of both.

#include <functional>
#include <deque>
#include <memory>
#include <optional>
#include <exception>
#include <stdexcept>
#include <chrono>
#include <cassert>

class single_threaded_executor {
private:
    struct shared_state_base : public std::enable_shared_from_this<shared_state_base> {
        virtual ~shared_state_base() = default;

        void transition_to_ready() {
            if (continuation)
                exec->execute([self = shared_from_this()] { self->continuation(self); } );
        }

        void set_error(std::exception_ptr&& err) {
            this->err = std::move(err);
            transition_to_ready();
        }

        single_threaded_executor* exec;
        std::exception_ptr err;
        std::function<void(std::shared_ptr<shared_state_base>)> continuation; // invoked with this.
        // TODO can't use std::function with move-only callbacks.
    };

    template <typename T>
    struct shared_state : shared_state_base {
        bool ready() const {
            return err || val;
        }

        void set_val(T val) {
            this->val = std::move(val);
            transition_to_ready();
        }

        std::optional<T> val;
    };

public:
    using Task = std::function<void()>; // TODO move-only callbacks...

    template <typename T>
    class future;

    template <typename T>
    class promise {
    public:
        using value_type = T;

        ~promise() {
            if (_tookFuture && !_state->ready())
                _state->set_error(std::make_exception_ptr(std::runtime_error("broken promise")));
        }

        future<T> get_future() noexcept {
            assert(!_tookFuture);
            _tookFuture = true;
            return future(_state);
        }

        void set_error(std::exception_ptr err) noexcept {
            _state->set_error(std::move(err));
        }
        
        void set_val(T val) noexcept {
            _state->set_val(std::move(val));
        }

        template <typename Func>
        void set_with(Func&& func) noexcept {
            try {
                _state->set_val(func());
            } catch(...) {
                _state->set_error(std::current_exception());
            }
        }

    private:
        friend class single_threaded_executor;
        explicit promise(std::shared_ptr<shared_state<T>> state) : _state(std::move(state)) {}

        bool _tookFuture = false;
        std::shared_ptr<shared_state<T>> _state = std::make_shared<shared_state<T>>();
    };
    
    template <typename T>
    class [[nodiscard]] future {
    public:
        using value_type = T;

        bool ready() {
            return _state->ready();
        }

        T& get() & {
            while (!ready()) {
                _state->exec->poll();
            }
            if (_state->err) std::rethrow_exception(_state->err);
            return *_state->val;
        }
        T get() && {
            return std::move(get()); // calls Lvalue overload of get().
        }

        template <typename Func>
        auto then(Func&& func) && {
            using RawReturn = decltype(func(std::move(*this)));
            auto phase1 = _state->exec->then_execute(std::move(*this), std::forward<Func>(func));
            if constexpr(!is_future_v<RawReturn>) {
                return std::move(phase1);
            } else {
                auto out_state = std::make_shared<shared_state<typename RawReturn::value_type>>();
                out_state->exec = _state->exec;

                phase1._state->continuation = [out_state] (const auto& ssb) noexcept {
                    shared_state<RawReturn>* phase1_state = static_cast<shared_state<RawReturn>*>(ssb.get());
                    assert(phase1_state->ready());
                    if (phase1_state->err) {
                        out_state->err = std::move(phase1_state->err);
                        out_state->transition_to_ready();
                        return;
                    }

                    auto& phase2 = *phase1_state->val;
                    if (phase2.ready()) {
                        out_state->val = std::move(phase2._state->val);
                        out_state->err = std::move(phase2._state->err);
                        out_state->transition_to_ready();
                    } else {
                        phase2._state->continuation = [out_state](const auto& ssb) {
                            auto state = static_cast<decltype(phase2._state.get())>(ssb.get());
                            out_state->val = std::move(state->val);
                            assert(state->ready());
                            out_state->err = std::move(state->err);
                            out_state->transition_to_ready();
                        };
                    }
                };
                return future(out_state);
            }
        }

    private:
        friend single_threaded_executor;
        explicit future(std::shared_ptr<shared_state<T>> state) : _state(std::move(state)) {}
        std::shared_ptr<shared_state<T>> _state;
    };

    template <typename T>
    struct is_future {
        template <typename U>
        static std::true_type check(future<U>&&);
        static std::false_type check(...);

        static constexpr bool value = decltype(check(std::declval<T>()))();
    };
    template <typename T>
    static constexpr bool is_future_v = is_future<T>::value;

    void execute(Task task) {
        _queue.push_back(std::move(task));
    }

    template <typename GeneralTask>
    auto twoway_execute(GeneralTask&& task) {
        // Easy case - task() returns a normal value.
        auto state = std::make_shared<shared_state<decltype(task())>>();
        state->exec = this;
        execute([state, task = std::forward<GeneralTask>(task)] {
            promise<decltype(task())>(state).set_with(task);
        });
        return future(state);
    }

    template <typename T, typename GeneralTask>
    auto then_execute(future<T>&& f, GeneralTask&& task) {
        using Result = decltype(task(f));
        auto out_state = std::make_shared<shared_state<Result>>();
        out_state->exec = this;
        f._state->continuation = [out_state, task = std::forward<GeneralTask>(task)] (const auto& ssb) {
            auto state = std::static_pointer_cast<shared_state<T>>(ssb->shared_from_this());
            assert(state->ready());
            promise<Result>(out_state).set_with([&] {
                // TODO update this to reflect however we decide to pass result to task.
                // TODO std::invoke?
                return task(future(state));
            });
        };
        return future(out_state);
    }

    void poll(std::chrono::milliseconds timeout = std::chrono::milliseconds(500)) {
        // poll for network events and add ready work to _queue. (exercise for reader)
        
        while (!_queue.empty()) {
            [&]() noexcept {
                _queue.front()(); // May append to _queue.
            }();
            _queue.pop_front();
        }
    }
    
private:
    using TaskQueue = std::deque<Task>;
    TaskQueue _queue;
};

template <typename T> using promise = single_threaded_executor::promise<T>;
template <typename T> using future = single_threaded_executor::future<T>;
template <typename T> constexpr bool is_future_v = single_threaded_executor::is_future_v<T>;

int main() {
    single_threaded_executor exec;
    static_assert(!is_future_v<int>);
    static_assert(is_future_v<future<int>>);

    {
        auto fut1 = exec.twoway_execute([]{ return 1; });
        assert(fut1.get() == 1);
    }

    {
        auto fut1 = exec.twoway_execute([]{ return 1; });
        auto fut2 = exec.then_execute(std::move(fut1), [](auto&& fut1) { return fut1.get() + 1; });
        assert(fut2.get() == 2);
    }

    {
        auto fut = exec
            .twoway_execute([]{ return 1; })
            .then([](auto&& fut1) { return fut1.get() + 1; });
        assert(fut.get() == 2);
    }

    {
        auto fut = exec
            .twoway_execute([]{ return 1; })
            .then([&exec](auto&& fut1) {
                return exec.twoway_execute([x = fut1.get()]{ return x + 1; });
            })
            .then([](auto&& fut2) { return fut2.get() + 1; });
        assert(fut.get() == 3);
    }
}
