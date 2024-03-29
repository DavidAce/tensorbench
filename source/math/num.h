#pragma once

#include <cmath>
#include <complex>
#include <exception>
#include <iterator>
#include <numeric>
#include <vector>

/*!
 *  \namespace num
 *  \brief Small convenience-type num functions, like modulo
 *  \tableofcontents
 */

namespace num {
#if defined(NDEBUG)
    static constexpr bool ndebug = true;
#else
    static constexpr bool ndebug = false;
#endif
    namespace internal {
        template<typename T>
        struct is_reference_wrapper : std::false_type {};

        template<typename T>
        struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_reference_wrapper_v = is_reference_wrapper<T>::value;
    }

    // Safe integer comparison functions from C++20

    template<class T, class U>
    [[nodiscard]] constexpr bool cmp_equal(T t, U u) noexcept {
        if constexpr(internal::is_reference_wrapper_v<T>)
            return cmp_equal(t.get(), u);
        else if constexpr(internal::is_reference_wrapper_v<U>)
            return cmp_equal(t, u.get());
        else if constexpr(std::is_same_v<T, U>)
            return t == u;
        else if constexpr(std::is_floating_point_v<T> and std::is_floating_point_v<U>)
            return t == u;
        else if constexpr(std::is_floating_point_v<T>)
            return static_cast<T>(t) == static_cast<T>(u);
        else if constexpr(std::is_floating_point_v<U>)
            return static_cast<U>(t) == static_cast<U>(u);
        else {
            using UT = std::make_unsigned_t<T>;
            using UU = std::make_unsigned_t<U>;
            if constexpr(std::is_signed_v<T> == std::is_signed_v<U>)
                return t == u;
            else if constexpr(std::is_signed_v<T>)
                return t < 0 ? false : UT(t) == u;
            else
                return u < 0 ? false : t == UU(u);
        }
    }

    template<class T, class U>
    [[nodiscard]] constexpr bool cmp_not_equal(T t, U u) noexcept {
        return !cmp_equal(t, u);
    }

    template<class T, class U>
    [[nodiscard]] constexpr bool cmp_less(T t, U u) noexcept {
        if constexpr(internal::is_reference_wrapper_v<T>)
            return cmp_less(t.get(), u);
        else if constexpr(internal::is_reference_wrapper_v<U>)
            return cmp_less(t, u.get());
        else if constexpr(std::is_same_v<T, U>)
            return t < u;
        else if constexpr(std::is_floating_point_v<T> and std::is_floating_point_v<U>)
            return t < u;
        else if constexpr(std::is_floating_point_v<T>)
            return static_cast<T>(t) < static_cast<T>(u);
        else if constexpr(std::is_floating_point_v<U>)
            return static_cast<U>(t) < static_cast<U>(u);
        else {
            using UT = std::make_unsigned_t<T>;
            using UU = std::make_unsigned_t<U>;
            if constexpr(std::is_signed_v<T> == std::is_signed_v<U>)
                return t < u;
            else if constexpr(std::is_signed_v<T>)
                return t < 0 ? true : UT(t) < u;
            else
                return u < 0 ? false : t < UU(u);
        }
    }

    template<class T, class U>
    [[nodiscard]] constexpr bool cmp_greater(T t, U u) noexcept {
        return cmp_less(u, t);
    }

    template<class T, class U>
    [[nodiscard]] constexpr bool cmp_less_equal(T t, U u) noexcept {
        return !cmp_greater(t, u);
    }

    template<class T, class U>
    [[nodiscard]] constexpr bool cmp_greater_equal(T t, U u) noexcept {
        return !cmp_less(t, u);
    }

    /*! \brief MatLab-style modulo operator
     *   \param x first number
     *   \param y second number
     *   \return modulo of x and y. Example, <code> mod(7,2)  = 1 </code> but <code> mod(-0.5,10)  = 9.5 </code>, instead of <code> -0.5 </code>  as given by
     * x%y.
     */
    template<typename T>
    [[nodiscard]] inline T mod(const T x, const T y) {
        if constexpr(!ndebug)
            if(y == 0) throw std::logic_error("num::mod(x,y): divisor y == 0");
        if constexpr(std::is_integral_v<T>) {
            if constexpr(std::is_unsigned_v<T>)
                return x >= y ? x % y : x;
            else { return x >= y ? x % y : (x < 0 ? (x % y + y) % y : x); }

        }
        //            return (x % y + y) % y;
        else
            return std::fmod((std::fmod(x, y) + y), y);
    }

    /*! \brief Similar to mod but faster for use with periodic boundary condition
     *   \param x first number
     *   \param y second number
     *   \return modulo of x and y. Example, <code> mod(7,2)  = 1 </code> but <code> mod(-0.5,10)  = 9.5 </code>, instead of <code> -0.5 </code>  as given by
     * x%y.
     */
    template<typename T>
    [[nodiscard]] inline T pbc(const T x, const T y) {
        if constexpr(!ndebug)
            if(y == 0) throw std::logic_error("num::pbc(x,y): divisor y == 0");
        if constexpr(std::is_signed_v<T>) {
            if(x >= 0 and x < y) return x;
            if(x < 0 and x >= -2 * y) return x + y;
        } else {
            if(x < y) return x;
        }
        if(x >= y and x < 2 * y) return x - y;
        return num::mod(x, y);
    }

    template<typename T>
    [[nodiscard]] int sign(const T val) noexcept {
        if(val > 0) return +1;
        if(val < 0) return -1;
        return 0;
    }

    template<typename T>
    [[nodiscard]] bool between(const T &value, const T &low, const T &high) noexcept {
        return value >= low and value <= high;
    }

    /*! \brief Python-style range generator, i.e. not-including "last"
     *   \return Range of T's. Example, <code> range(0,8,2) </code> gives a std::vector<int>: <code> [0,2,4,6] </code>
     */
    namespace internal {
        template<typename TA, typename TB>
        using int_or_dbl = typename std::conditional<std::is_floating_point_v<TA> or std::is_floating_point_v<TB>, double, int>::type;
    }

    template<typename T = int, typename T1, typename T2, typename T3 = internal::int_or_dbl<T1, T2>>
    [[nodiscard]] std::vector<T> range(T1 first, T2 last, T3 step = static_cast<T3>(1)) noexcept {
        if(step == 0) return {static_cast<T>(first)};
        auto num_steps =
            static_cast<size_t>(std::abs((static_cast<double>(last) - static_cast<double>(first) + static_cast<double>(step)) / static_cast<double>(step)));
        std::vector<T> vec;
        vec.reserve(num_steps);
        auto val = static_cast<T>(first);
        if(cmp_less(first, last)) {
            while(cmp_less(val, last)) {
                vec.emplace_back(val);
                val += cmp_less(step, 0) ? -static_cast<T>(step) : static_cast<T>(step);
            }
        } else {
            while(cmp_greater(val, last)) {
                vec.emplace_back(val);
                val -= cmp_less(step, 0) ? -static_cast<T>(step) : static_cast<T>(step);
            }
        }
        if constexpr(std::is_signed_v<T3>) {
            if(step < 0) return {vec.rbegin(), vec.rend()};
        }
        return vec;
    }

    /*! \brief MatLab-style linearly spaced array
     *   \param num number of linearly spaced values
     *   \param a first value in range
     *   \param b last value in range
     *   \return std::vector<T2>. Example,  <code> Linspaced(5,1,5) </code> gives a std::vector<int>: <code> [1,2,3,4,5] </code>
     */
    [[nodiscard]] inline std::vector<double> LinSpaced(std::size_t N, double a, double b) {
        double              h = (b - a) / static_cast<double>(N - 1);
        std::vector<double> xs(N);
        double              val = a;
        for(auto &x : xs) {
            x = val;
            val += h;
        }
        return xs;
    }

    [[nodiscard]] inline std::vector<double> LogSpaced(std::size_t N, double a, double b, double base = 10.0) {
        if(a <= 0) throw std::range_error("a must be positive");
        if(b <= 0) throw std::range_error("b must be positive");
        double              loga   = std::log(a) / std::log(base);
        double              logb   = std::log(b) / std::log(base);
        double              h      = (logb - loga) / static_cast<double>(N - 1);
        double              factor = std::pow(base, h);
        double              val    = std::pow(base, loga);
        std::vector<double> xs(N);
        for(auto &x : xs) {
            x = val;
            val *= factor;
        }
        return xs;
    }

    /*! \brief Sum operator for containers such as vector
     *   \param in a vector, array or any 1D container with "<code> .data() </code>" method.
     *   \param from first element to add (default == 0)
     *   \param to last element to add (default == -1: from - size)
     *   \return sum of elements with type Input::value_type .
     *   \example Let <code> v = {1,2,3,4}</code>. Then <code> sum(v,0,3) = 10 </code>.
     */
    template<typename Input>
    [[nodiscard]] auto sum(const Input &in, long from = 0, long num = -1) {
        if(num < 0) num = in.size();
        num = std::min<long>(num, static_cast<long>(in.size()) - from);
        return std::accumulate(std::begin(in) + from, std::begin(in) + from + num, static_cast<typename Input::value_type>(0));
    }

    /*! \brief Product operator for containers such as vector
     *   \param in a vector, array or any 1D container with "<code> .data() </code>" method.
     *   \param from first element to multiply (default == 0)
     *   \param to last element to multiply (default == -1: from - size)
     *   \return product of elements with type Input::value_type .
     *   \example Let <code> v = {1,2,3,4}</code>. Then <code> prod(v,0,3) = 24 </code>.
     */
    template<typename Input>
    [[nodiscard]] auto prod(const Input &in, long from = 0, long num = -1) {
        if(num < 0) num = in.size();
        num = std::min<long>(num, static_cast<long>(in.size()) - from);
        return std::accumulate(std::begin(in) + from, std::begin(in) + from + num, 1, std::multiplies<>());
    }


    /*! \brief Cumulative sum operator for containers such as vector
     *   \param in a vector, array or any 1D container with "<code> .data() </code>" method.
     *   \param from first element to add (default == 0)
     *   \param to last element to add (default == -1: from - size)
     *   \return cumulative sum of elements with type Input::value_type .
     *   \example Let <code> v = {1,2,3,4}</code>. Then <code> cumsum(v,0,3) = {1,3,6,10} </code>.
     */
    template<typename Input>
    [[nodiscard]] auto cumsum(const Input &in, long from = 0, long num = -1) {
        if(num < 0) num = in.size();
        num = std::min<long>(num, static_cast<long>(in.size()) - from);
        typename Input::value_type sum = 0;
        Input out;
        out.reserve(num);
        for (auto it = std::begin(in) + from; it != std::begin(in) + from + num; it++ ){
            sum += *it;
            out.emplace_back(sum);
        }
        return out;
    }
    /*! \brief Displacements for containers such as vector
     *   \param in a vector, array or any 1D container with "<code> .data() </code>" method.
     *   \param from first element to add (default == 0)
     *   \param to last element to add (default == -1: from - size)
     *   \return displacements of elements with type Input::value_type .
     *   \example Let <code> v = {1,2,3,4}</code>. Then <code> disps(v,0,3) = {0,1,3,6} </code>.
     */
    template<typename Input>
    [[nodiscard]] auto disps(const Input &in, long from = 0, long num = -1) {
        if(num < 0) num = in.size();
        num = std::min<long>(num, static_cast<long>(in.size()) - from);
        typename Input::value_type sum = 0;
        Input out;
        out.reserve(num);
        for (auto it = std::begin(in) + from; it != std::begin(in) + from + num; it++ ){
            out.emplace_back(sum);
            sum += *it;
        }
        return out;
    }


    /*! \brief Checks if multiple values are equal to each other
     *   \param args any number of values
     *   \return bool, true if all args are equal
     */
    template<typename First, typename... T>
    [[nodiscard]] bool all_equal(First &&first, T &&...t) noexcept {
        return ((first == t) && ...);
    }

    template<typename R, typename T>
    [[nodiscard]] R next_power_of_two(T val) {
        return static_cast<R>(std::pow<long>(2, static_cast<long>(std::ceil(std::log2(std::real(val))))));
    }
    template<typename R, typename T>
    [[nodiscard]] R prev_power_of_two(T val) {
        return static_cast<R>(std::pow<long>(2, static_cast<long>(std::floor(std::log2(std::real(val - 1))))));
    }

    template<typename R, typename T>
    [[nodiscard]] inline R next_multiple(const T num, const T mult) {
        if(mult == 0) return num;
        return (num + mult) - mod(num, mult);
    }
    template<typename R, typename T>
    [[nodiscard]] inline R prev_multiple(const T num, const T mult) {
        if(mult == 0) return num;
        auto m = mod(num, mult);
        if(m == 0) return prev_multiple<R>(num - 1, mult);
        return num - m;
    }

    template<typename T>
    [[nodiscard]] inline T round_to_multiple_of(const T number, const T multiple) {
        T result = number + multiple / 2;
        result -= num::mod(result, multiple);
        return result;
    }

    template<typename T>
    [[nodiscard]] inline T round_up_to_multiple_of(const T number, const T multiple) {
        if(multiple == 0) return number;
        auto remainder = num::mod(std::abs(number), multiple);
        if(remainder == 0) return number;
        if(number < 0)
            return -(std::abs(number) - remainder);
        else
            return number + multiple - remainder;
    }
}
