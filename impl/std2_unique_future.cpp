#include <atomic>
#include <functional>
#include <variant>
#include <exception>
#include <cassert>

namespace std2
{

template <typename T>
struct asynchronous_value
{
    enum state_type
    {
        V1 = 0b01,      // Value Set
        C1 = 0b10,      // Continuation Set

        C0V0 = 0b00,    // Continuation Unset, Value Unset
        C0V1 = 0b01,    // Continuation Unset, Value Set
        C1V0 = 0b10,    // Continuation Set,   Value Unset
        C1V1 = 0b11     // Continuation Set,   Value Set
    };

    std::atomic<state_type> state;
    std::function<void(T)>  continuation;
    T                       value;

    bool value_ready()
    {
        return state.load(std::memory_order_seq_cst) & V1;
    }

    bool continuation_ready()
    {
        return state.load(std::memory_order_seq_cst) & C1;
    }

    template <typename U>
    void set_value(U&& u)
    {
        state_type expected = state.load(std::memory_order_seq_cst);

        if (expected & V1) throw std::runtime_error("value already set.");

        value = std::forward<U>(u);

        state_type desired = expected | V1;

        while (!state.compare_exchange_weak(expected, desired,
                                            std::memory_order_seq_cst))
        {
            if (expected & V1) throw std::runtime_error("value already set.");
            desired = expected | V1;
        }

        if (desired | C1)
            std::move(continuation)(std::move(value));
    }

    template <typename F>
    void set_continuation(F&& f)
    {
        state_type expected = state.load(std::memory_order_seq_cst);

        if (expected & C1) throw std::runtime_error("value already set.");

        continuation = std::forward<F>(f);

        state_type desired = expected | C1;

        while (!state.compare_exchange_weak(expected, desired,
                                            std::memory_order_seq_cst))
        {
            if (expected & C1) throw std::runtime_error("value already set.");
            desired = expected | C1;
        }

        if (desired | V1)
            std::move(continuation)(std::move(value));
    }
};

}

int main()
{
}

