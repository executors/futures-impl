#include <atomic>
#include <condition_variable>
#include <thread>
#include <functional>
#include <map>
#include <exception>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <optional>

#include <cassert>

#include <boost/core/lightweight_test.hpp>

namespace std2
{

template <typename Sig>
struct fire_once;

template <typename T>
struct emplace_as {};

template <typename R, typename... Args>
struct fire_once<R(Args...)>
{
  private:

    std::unique_ptr<void, void(*)(void*)> ptr{nullptr, +[] (void*) {}};
    void(*invoke)(void*, Args...) = nullptr;

  public:

    constexpr fire_once() = default;

    constexpr fire_once(fire_once&&) = default;
    constexpr fire_once& operator=(fire_once&&) = default;

    template <
        typename F
        // {{{ SFINAE
      , std::enable_if_t<!std::is_same_v<std::decay_t<F>, fire_once>, int> = 0
      , std::enable_if_t<
               std::is_convertible_v<
                 std::result_of_t<std::decay_t<F>&(Args...)>, R
               >
            || std::is_same_v<R, void>
          , int
        > = 0
        // }}}
    >
    fire_once(F&& f)
    // {{{
      : fire_once(emplace_as<std::decay_t<F>>{}, std::forward<F>(f))
    {}
    // }}}

    template <typename F, typename...FArgs>
    fire_once(emplace_as<F>, FArgs&&...fargs)
    { // {{{
        rebind<F>(std::forward<FArgs>(fargs)...);
    } // }}}

    R operator()(Args... args) &&
    { // {{{
        try {
            if constexpr (std::is_same_v<R, void>)
            {
                invoke(ptr.get(), std::forward<Args>(args)...);
                clear();
            }
            else
            {
                R ret = invoke(ptr.get(), std::forward<Args>(args)...);
                clear();
                return ret;
            }
        } catch (...) {
            clear();
            throw;
        }
    } // }}}

    void clear()
    { // {{{
        invoke = nullptr;
        ptr.reset();
    } // }}}

    explicit operator bool() const
    { // {{{
        return bool(ptr);
    } // }}}

    template <typename F, typename...FArgs>
    void rebind(FArgs&&... fargs)
    { // {{{
        clear();
        auto pf = std::make_unique<F>(std::forward<FArgs>(fargs)...);
        invoke =
            +[](void* pf, Args...args) -> R
            {
                return (*reinterpret_cast<F*>(pf))(std::forward<Args>(args)...);
            };
        ptr = { pf.release(), [] (void* pf) { delete (F*)(pf); } };
    } // }}}
};

struct semaphore
{
    semaphore()
    // {{{
      : count{0}
    {}
    // }}}

    void notify(std::size_t i = 1)
    { // {{{
        {
            std::unique_lock<std::mutex> lock(mutex);
            count += i;
        }

        while (i--)
            condition.notify_one();
    } // }}}

    void wait()
    { // {{{
        std::unique_lock<std::mutex> lock(mutex);

        while (!count)
            condition.wait(lock);

        --count;
    } // }}}

    bool try_wait()
    { // {{{
        std::unique_lock<std::mutex> lock(mutex);

        if (!count)
            return false;

        --count;

        return true;
    } // }}}

  private:
    std::mutex               mutex;
    std::condition_variable  condition;
    std::atomic<std::size_t> count;
};

template <typename T>
struct asynchronous_value
{
    enum state_type
    { // {{{
        UNSET = 0,

        VC = 0b10000, // Value        Changing
        VR = 0b01000, // Value        Ready

        CC = 0b00100, // Continuation Changing
        CR = 0b00010, // Continuation Ready
        CX = 0b00001, // Continuation Executed
    }; // }}}

  private:

    std::atomic<state_type> state;
    fire_once<void(T)>      continuation;
    T                       value;

    static void check_state_invariants(state_type s)
    { // {{{
        // No VC0_VR1         (If VR is set, VC must be set)
        if (s & VR) assert(s & VC);
        // No CC0_CR1         (If CR is set, CC must be set)
        if (s & CR) assert(s & CC);
        // No CR0_CX1         (If CX is set, CR and CC must be set)
        // No VR0_CC1_CR1_CX1 (If CX is set, VC and VR must be set)
        if (s & CX) assert((s & CR) && (s & CC) && (s & CR) && (s & CC));
    } // }}}

  public:

    constexpr asynchronous_value() noexcept
    // {{{
      : state{}
      , continuation{}
      , value{}
    {}
    // }}}

    state_type status() const
    { // {{{
        return state.load(std::memory_order_acquire);
    } // }}}

    bool value_ready() const
    { // {{{
        return state.load(std::memory_order_acquire) & VR;
    } // }}}

    bool continuation_ready() const
    { // {{{
        return state.load(std::memory_order_acquire) & CR;
    } // }}}

    bool consumed() const
    { // {{{
        return state.load(std::memory_order_acquire) & CX;
    } // }}}

    template <typename U>
    void set_value(U&& u)
    { // {{{
        state_type expected = state.load(std::memory_order_acquire);

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
                                                std::memory_order_acq_rel))
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
                                                std::memory_order_acq_rel))
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
    } // }}}

    template <typename Executor, typename F>
    void set_continuation(Executor&& exec, F&& f)
    { // {{{
        state_type expected = state.load(std::memory_order_acquire);

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
                                                std::memory_order_acq_rel))
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

        continuation =
            [exec = std::forward<Executor>(exec), f = std::forward<F>(f)]
            (auto&&... args) mutable
            {
                exec.execute(std::move(f), std::forward<decltype(args)>(args)...);
            };

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
                                                std::memory_order_acq_rel))
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
    } // }}}

    std::optional<T> try_get() 
    { // {{{
        state_type expected = state.load(std::memory_order_acquire);

        check_state_invariants(expected);

        // Value has not been set yet.
        if (!(expected & VR))
            return {};

        // No continuation should be set (or be in the process of being set),
        // and the value should not have been consumed.
        assert(expected & CC || expected && CR || expected && CX);

        { // Introduce a new scope to prevent misuse of the lambda below.
            auto const consume_mask_update =
                [] (state_type s) { return state_type(s | CX); };

            state_type desired = consume_mask_update(expected);

            while (!state.compare_exchange_weak(expected, desired,
                                                std::memory_order_acq_rel))
            {
                // The value should not have been consumed.
                assert(!(expected & CX));

                desired = consume_mask_update(expected);
            }
        }

        return { std::move(value) };
    } // }}}
};

struct default_executor;

template <typename... Ts>
struct unique_future;

template <typename... Ts>
struct unique_promise
{
  private:
    std::shared_ptr<asynchronous_value<std::tuple<Ts...>>> data;
    bool future_retrieved; // If !data, this must be false.

    void deferred_data_allocation()
    { // {{{
        if (!data)
        {
            assert(!future_retrieved);
            data = std::make_shared<asynchronous_value<std::tuple<Ts...>>>();
        }
    } // }}}

  public:
    // DefaultConstructible
    constexpr unique_promise() noexcept
    // {{{
      : data{}
      , future_retrieved{}
    {}
    // }}}

    // MoveAssignable 
    constexpr unique_promise(unique_promise&&) noexcept = default;
    constexpr unique_promise& operator=(unique_promise&&) noexcept = default;

    // Not CopyAssignable
    unique_promise(unique_promise const&) = delete;
    unique_promise& operator=(unique_promise const&) = delete;

    unique_future<Ts...> get_future()
    { // {{{
        // Exits via error if the future has been retrieved.
        deferred_data_allocation();
        future_retrieved = true; 
        return unique_future<Ts...>(data);
    } // }}}

    template <typename... Us>
    void set(Us&&... us)
    { // {{{
        deferred_data_allocation();
        data->set_value(std::tuple<Ts...>(std::forward<Us>(us)...));
    } // }}}
};

template <typename... Ts>
struct unique_future
{
    std::shared_ptr<asynchronous_value<std::tuple<Ts...>>> data;

  private:
    unique_future(std::shared_ptr<asynchronous_value<std::tuple<Ts...>>> ptr)
    // {{{
      : data(ptr)
    {}
    // }}}

    friend class unique_promise<Ts...>;

  public:

    // DefaultConstructible
    constexpr unique_future() noexcept = default;

    // MoveAssignable 
    constexpr unique_future(unique_future&&) noexcept = default;
    constexpr unique_future& operator=(unique_future&&) noexcept = default;

    // Not CopyAssignable
    unique_future(unique_future const&) = delete;
    unique_future& operator=(unique_future const&) = delete;

    template <typename Executor, typename F>
    // {{{ unique_future<...>
    std::conditional_t<
        std::is_same_v<decltype(std::declval<F>()(std::declval<Ts>()...)), void>
      , unique_future<>
      , unique_future<decltype(std::declval<F>()(std::declval<Ts>()...))>
    >
    // }}}
    then(Executor&& exec, F&& f)
    { // {{{
        using promise_type = std::conditional_t<
            std::is_same_v<
                decltype(std::declval<F>()(std::declval<Ts>()...)), void
            >
          , unique_promise<>
          , unique_promise<decltype(std::declval<F>()(std::declval<Ts>()...))>
        >;
        assert(data);
        promise_type p;
        auto g = p.get_future();
        data->set_continuation(
            std::forward<Executor>(exec)
          , [f = std::forward<F>(f), p = std::move(p)] 
            (std::tuple<Ts...> v) mutable
            {
                if constexpr (std::is_same_v<promise_type, unique_promise<>>)
                {
                    std::apply(f, std::move(v));
                    p.set();
                }
                else
                    p.set(std::apply(std::forward<F>(f), std::move(v)));
            }
        );
        return g;
    } // }}}

    template <typename F>
    auto then(F&& f)
    { // {{{
        return then(default_executor{}, std::forward<F>(f));
    } // }}}

    std::optional<std::tuple<Ts...>> try_get()
    { // {{{
        if (data)
            return data->try_get();
        else
            return {};
    } // }}}
};

struct default_executor
{
    template <typename F, typename... Args>
    void execute(F&& f, Args&&... args)
    { // {{{
        std::forward<F>(f)(std::forward<Args>(args)...);
    } // }}}

    template <typename F, typename... Args>
    auto async(F&& f, Args&&... args)
        -> unique_future<decltype(std::declval<F>(std::declval<Args>(args)...))>
    { // {{{
        using promise_type = std::conditional_t<
            std::is_same_v<
                decltype(std::declval<F>()(std::declval<Args>()...)), void
            >
          , unique_promise<>
          , unique_promise<decltype(std::declval<F>()(std::declval<Args>()...))>
        >;
        promise_type p;
        auto g = p.get_future();
        std::thread t(
            [ f = std::forward<F>(f)
            , args = std::forward_as_tuple(std::forward<Args>(args)...)
            , p = std::move(p)]
            () mutable
            {
                if constexpr (std::is_same_v<promise_type, unique_promise<>>)
                {
                    std::apply(f, std::move(args));
                    p.set();
                }
                else
                    p.set(std::apply(std::forward<F>(f), std::move(args)));
            }
        );
        return g;
    } // }}}

    template <typename... Ts>
    auto depend(unique_future<Ts...> f)
    { // {{{
        semaphore sem;

        std::optional<std::tuple<Ts...>> v;

        f.then(
            [&] (auto&&... args)
            {
                v = std::make_tuple(std::forward<decltype(args)>(args)...);
                sem.notify();
            }
        );

        sem.wait();

        return std::move(v);
    } // }}}
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

        a.set_continuation(
            std2::default_executor{}
          , [&a_val] (int v) { a_val = v; }
        ); 

        BOOST_TEST_EQ(a.value_ready(),        true);
        BOOST_TEST_EQ(a.continuation_ready(), true);

        BOOST_TEST_EQ(a_val, 42);
    }

    { // Set continuation, then set value.
        std2::asynchronous_value<int> a;

        BOOST_TEST_EQ(a.value_ready(),        false);
        BOOST_TEST_EQ(a.continuation_ready(), false);

        int a_val = 0;

        a.set_continuation(
            std2::default_executor{}
          , [&a_val] (int v) { a_val = v; }
        ); 

        BOOST_TEST_EQ(a_val, 0);

        BOOST_TEST_EQ(a.value_ready(),        false);
        BOOST_TEST_EQ(a.continuation_ready(), true);

        a.set_value(42);

        BOOST_TEST_EQ(a.value_ready(),        true);
        BOOST_TEST_EQ(a.continuation_ready(), true);

        BOOST_TEST_EQ(a_val, 42);
    }

    unsigned producer_count = 0;
    unsigned consumer_count = 0;

    for (int i = 0; i < 128; ++i)
    {
        std2::asynchronous_value<std::string> a;

        std::atomic<int> go_flag(false);

        auto const barrier =
            [&] () 
            {
                go_flag.fetch_add(1, std::memory_order_relaxed);

                while (go_flag.load(std::memory_order_relaxed) < 2)
                    ; // Spin.
            };

        std::string a_val = "";

        std::thread producer(
            [&] ()
            {
                barrier();

                a.set_value(
                    "foo foo foo foo foo foo foo foo foo foo "
                    "foo foo foo foo foo foo foo foo foo foo"
                );
            }
        );

        barrier();        

        a.set_continuation(
            std2::default_executor{}
          , [&] (std::string v)
            {
                a_val = v;

                if (std::this_thread::get_id() == producer.get_id())
                    ++producer_count;
                else
                    ++consumer_count;
            }
        );
   
        producer.join();

        //std::cout << std::bitset<6>(a.status()) << "\n";

        BOOST_TEST_EQ(a.value_ready(),        true);
        BOOST_TEST_EQ(a.continuation_ready(), true);

        BOOST_TEST_EQ(
            a_val
          , "foo foo foo foo foo foo foo foo foo foo "
            "foo foo foo foo foo foo foo foo foo foo"
        );
    }

    std::cout << "Consumer thread ran the continuation in "
              << consumer_count
              << " trials\n";
    std::cout << "Producer thread ran the continuation in "
              << producer_count
              << " trials\n";

    { // Set value, then set continuation.
        std2::unique_promise<int> p;

        p.set(42);

        std2::unique_future<int> f = p.get_future();

        int a_val = 0;
        f.then([&a_val] (int v) { a_val = v; }); 

        BOOST_TEST_EQ(a_val, 42);
    }

    { // Set continuation, then set value.
        std2::unique_promise<int> p;

        std2::unique_future<int> f = p.get_future();

        int a_val = 0;
        f.then([&a_val] (int v) { a_val = v; }); 

        BOOST_TEST_EQ(a_val, 0);

        p.set(42);

        BOOST_TEST_EQ(a_val, 42);
    }

    { // Set value, then set continuation.
        std2::unique_promise<int> p;

        p.set(42);

        auto [a_val] = (*p.get_future().try_get());

        BOOST_TEST_EQ(a_val, 42);
    }

    { // Set continuation, then set value.
        std2::unique_promise<int> p;

        std2::unique_future<int> f = p.get_future();

        std::optional<std::tuple<int>> maybe_a_val = f.try_get();

        BOOST_TEST_EQ(maybe_a_val.has_value(), false);

        p.set(42);

        auto [a_val] = (*p.get_future().try_get());

        BOOST_TEST_EQ(a_val, 42);
    }

    return boost::report_errors();
}

