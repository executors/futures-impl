#include <atomic>
#include <thread>
#include <functional>
#include <map>
#include <exception>
#include <iostream>
#include <iomanip>

#include <cassert>

#include <boost/core/lightweight_test.hpp>

template <typename Integral>
std::string as_binary(Integral i)
{
    Integral constexpr base = 2;

    std::string buf;

    bool is_negative = i < 0;

    if constexpr (std::is_signed_v<Integral>) 
        i = std::abs(i);

    do
    {
        Integral digit = i % base;

        char c = [](Integral d) {
            switch (d)
            {
                case 0: return '0';
                case 1: return '1';
            }
            assert(false);
        }(digit);

        buf.push_back(c);

        i = (i - digit) / base;
    } while (i != 0);

    buf += (is_negative ? "b0-" : "b0");

    std::reverse(buf.begin(), buf.end());

    return buf;
}

namespace std2
{

template <typename T>
struct asynchronous_value
{
    enum state_type
    {
        UNSET = 0,

        VC = 0b10000, // Value        Changing
        VR = 0b01000, // Value        Ready

        CC = 0b00100, // Continuation Changing
        CR = 0b00010, // Continuation Ready
        CX = 0b00001, // Continuation Executed
    };

    std::atomic<state_type> state;
    std::function<void(T)>  continuation;
    T                       value;

    constexpr asynchronous_value() noexcept
      : state{}
      , continuation{}
      , value{}
    {}

    static void check_state_invariants(state_type s)
    {
        // No VC0_VR1         (If VR is set, VC must be set)
        if (s & VR) assert(s & VC);
        // No CC0_CR1         (If CR is set, CC must be set)
        if (s & CR) assert(s & CC);
        // No CR0_CX1         (If CX is set, CR and CC must be set)
        // No VR0_CC1_CR1_CX1 (If CX is set, VC and VR must be set)
        if (s & CX) assert((s & CR) && (s & CC) && (s & CR) && (s & CC));
    }

    bool value_ready()
    {
        return state.load(std::memory_order_seq_cst) & VR;
    }

    bool continuation_ready()
    {
        return state.load(std::memory_order_seq_cst) & CR;
    }

    bool consumed()
    {
        return state.load(std::memory_order_seq_cst) & CX;
    }

    template <typename U>
    void set_value(U&& u)
    {
        state_type expected = state.load(std::memory_order_seq_cst);

        check_state_invariants(expected);

        // Value should not be set yet.
        assert(!(expected & VR));
        assert(!(expected & VC));

        state_type desired = UNSET;

        ///////////////////////////////////////////////////////////////////////
        // First, attempt to acquire the value "lock" by setting the "value is
        // being changed" bit (VC).

        { // Introduce a new scope to prevent misuse of the lambda below.
            auto const acquire_mask_update =
                [] (state_type s)
                {
                    return state_type(s | VC);
                };

            desired = acquire_mask_update(expected);

            while (!state.compare_exchange_weak(expected, desired,
                                                std::memory_order_seq_cst))
            {
                // No one else should be setting the value.
                assert(!(expected & VR));
                assert(!(expected & VC));

                // The continuation should not have run yet.
                assert(!(expected & CX));

                desired = acquire_mask_update(expected);
            }
        }

        // The continuation should now be changing.
        assert(desired & VC);

        ///////////////////////////////////////////////////////////////////////
        // We either set the VC bit or raised an error; now we can write to the
        // value variable.

        value = std::forward<U>(u);

        ///////////////////////////////////////////////////////////////////////
        // Now we need to release the value "lock" (VC), indicate that the
        // value is ready, and determine if we need to run the continuation.

        // CAS doesn't update expected when it succeeds, so expected is not up
        // to date.
        expected = desired;

        // The "value is being changed" bit (VC) should be set. 
        assert(expected & VC);

        // The continuation should not have run yet; we've set the value, but
        // we haven't signalled that it is set.
        assert(!(expected & CX));

        { // Introduce a new scope to prevent misuse of the lambda below.
            auto const release_mask_update =
                [] (state_type s)
                {
                    if (s & CR)
                        // If the continuation is ready, we'll run it.
                        return state_type(s | CX | VR);
                    else
                        return state_type(s | VR);
                };

            desired = release_mask_update(expected);

            while (!state.compare_exchange_weak(expected, desired,
                                                std::memory_order_seq_cst))
            {
                // No one else should be setting the value.
                assert(!(expected & VR));

                // The continuation should not have run yet.
                assert(!(expected & CX));

                desired = release_mask_update(expected);
            }

        }

        // The value should now be ready.
        assert(desired & VR);

        ///////////////////////////////////////////////////////////////////////
        // Execute the continuation if needed (e.g. if we set the CX bit in the
        // release CAS loop).

        if (desired & CX)
        {
            // The continuation should not be empty.
            assert(continuation);

            std::move(continuation)(std::move(value));
        }
    }

    template <typename F>
    void set_continuation(F&& f)
    {
        state_type expected = state.load(std::memory_order_seq_cst);

        check_state_invariants(expected);

        // Continuation should not be set yet.
        assert(!(expected & CR));
        assert(!(expected & CC));

        state_type desired = UNSET;

        ///////////////////////////////////////////////////////////////////////
        // First, attempt to acquire the continuation "lock" by setting the
        // "continuation is being changed" bit (CC).

        { // Introduce a new scope to prevent misuse of the lambda below.
            auto const acquire_mask_update =
                [] (state_type s)
                {
                    return state_type(s | CC);
                };

            desired = acquire_mask_update(expected);

            while (!state.compare_exchange_weak(expected, desired,
                                                std::memory_order_seq_cst))
            {
                // No one else should be setting the continuation.
                assert(!(expected & CR));
                assert(!(expected & CC));
                assert(!(expected & CX));

                desired = acquire_mask_update(expected);
            }
        }

        // The continuation should now be changing.
        assert(desired & CC);

        ///////////////////////////////////////////////////////////////////////
        // We either set the CC bit or raised an error; now we can write to the
        // continuation variable.

        continuation = std::forward<F>(f);

        ///////////////////////////////////////////////////////////////////////
        // Now we need to release the continuation "lock" (CC), indicate that
        // the continuation is ready, and determine if we need to run the
        // continuation.

        // CAS doesn't update expected when it succeeds, so expected is not up
        // to date.
        expected = desired;

        // The "continuation is being changed" bit (VC) should be set. 
        assert(expected & CC);

        // The continuation should not have run yet; we've set the
        // continuation, but we haven't signalled that it is set.
        assert(!(expected & CX));

        { // Introduce a new scope to prevent misuse of the lambda below.
            auto const release_mask_update =
                [] (state_type s)
                {
                    if (s & VR)
                        // If the value is ready, we'll run the continuation.
                        return state_type(s | CX | CR);
                    else
                        return state_type(s | CR);
                };

            desired = release_mask_update(expected);

            while (!state.compare_exchange_weak(expected, desired,
                                                std::memory_order_seq_cst))
            {
                // No one else should be setting the continuation.
                assert(!(expected & CR));
                assert(!(expected & CX));

                desired = release_mask_update(expected);
            }
        }

        // The continuation should now be ready.
        assert(desired & CR);

        ///////////////////////////////////////////////////////////////////////
        // Execute the continuation if needed (e.g. if we set the CX bit in the
        // release CAS loop).

        if (desired & CX)
        {
            // The continuation should not be empty.
            assert(continuation);

            std::move(continuation)(std::move(value));
        }
    }
};

}

int main()
{
    std::cout << std::setbase(2);

    { // Set value, then set continuation.
        std2::asynchronous_value<int> a;

        BOOST_TEST_EQ(a.value_ready(),        false);
        BOOST_TEST_EQ(a.continuation_ready(), false);

        a.set_value(42);

        BOOST_TEST_EQ(a.value_ready(),        true);
        BOOST_TEST_EQ(a.continuation_ready(), false);

        int a_val = 0;

        a.set_continuation([&a_val] (int v) { a_val = v; }); 

        BOOST_TEST_EQ(a.value_ready(),        true);
        BOOST_TEST_EQ(a.continuation_ready(), true);

        BOOST_TEST_EQ(a_val, 42);
    }

    { // Set continuation, then set value.
        std2::asynchronous_value<int> a;

        BOOST_TEST_EQ(a.value_ready(),        false);
        BOOST_TEST_EQ(a.continuation_ready(), false);

        int a_val = 0;

        a.set_continuation([&a_val] (int v) { a_val = v; }); 

        BOOST_TEST_EQ(a_val, 0);

        BOOST_TEST_EQ(a.value_ready(),        false);
        BOOST_TEST_EQ(a.continuation_ready(), true);

        a.set_value(42);

        BOOST_TEST_EQ(a.value_ready(),        true);
        BOOST_TEST_EQ(a.continuation_ready(), true);

        BOOST_TEST_EQ(a_val, 42);
    }

    std::map<std::thread::id, unsigned> scoreboard;

    for (int i = 0; i < 64; ++i)
    {
        //std::cout << "\n";

        std2::asynchronous_value<int> a;

        std::atomic<int> go_flag(false);

        auto const barrier =
            [&] () 
            {
                go_flag.fetch_add(1, std::memory_order_seq_cst);

                while (go_flag.load(std::memory_order_seq_cst) < 2)
                    ; // Spin.
            };

        std::thread t(
            [&] ()
            {
                barrier();

                a.set_value(42);
            }
        );

        barrier();        

        int a_val = 0;

        a.set_continuation(
            [&] (int v)
            {
                ++scoreboard[std::this_thread::get_id()];
                //std::cout << "Running on " << std::this_thread::get_id() << "\n";
                a_val = v;
            }
        ); 
   
        t.join();

        //std::cout << as_binary(unsigned(a.state.load(std::memory_order_seq_cst)))
        //          << "\n";

        BOOST_TEST_EQ(a.value_ready(),        true);
        BOOST_TEST_EQ(a.continuation_ready(), true);

        BOOST_TEST_EQ(a_val, 42);
    }

    for (auto&& [id, count] : scoreboard)
    {
        if (id == std::this_thread::get_id())
            std::cout << "Consumer thread";
        else
            std::cout << "Producer thread";

        std::cout << " : " << count << "\n";
    }

    return boost::report_errors();
}

