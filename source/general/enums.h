#pragma once

#include <h5pp/details/h5ppEnums.h>
#include <spdlog/common.h>
#include <stdexcept>
#include <string>
#include <string_view>

enum class tb_mode { eigen1, eigen2, eigen3, cutensor, xtensor, tblis, cyclops };
enum class tb_type { fp32, fp64, cplx };

template<class>
inline constexpr bool unrecognized_type_v = false;

template<typename T>
constexpr std::string_view enum2sv(const T &item) {
    static_assert(std::is_enum_v<T> and "enum2sv<T>: T must be an enum");
    if constexpr(std::is_same_v<T, tb_mode>) {
        switch(item) {
            case tb_mode::eigen1: return "eigen1";
            case tb_mode::eigen2: return "eigen2";
            case tb_mode::eigen3: return "eigen3";
            case tb_mode::cutensor: return "cutensor";
            case tb_mode::xtensor: return "xtensor";
            case tb_mode::tblis: return "tblis";
            case tb_mode::cyclops: return "cyclops";
            default: throw std::runtime_error("enum2sv: invalid enum case for tb_mode");
        }
    } else if constexpr(std::is_same_v<T, tb_type>) {
        switch(item) {
            case tb_type::fp32: return "fp32";
            case tb_type::fp64: return "fp64";
            case tb_type::cplx: return "cplx";
            default: throw std::runtime_error("enum2sv: invalid enum case for tb_type");
        }
    } else if constexpr(std::is_same_v<T, spdlog::level::level_enum>) {
        switch(item) {
            case spdlog::level::level_enum::trace: return "trace";
            case spdlog::level::level_enum::debug: return "debug";
            case spdlog::level::level_enum::info: return "info";
            case spdlog::level::level_enum::warn: return "warn";
            case spdlog::level::level_enum::err: return "err";
            case spdlog::level::level_enum::critical: return "critical";
            case spdlog::level::level_enum::off: return "off";
            default: throw std::runtime_error("enum2sv: invalid enum case for spdlog::level::level_enum");
        }
    } else if constexpr(std::is_same_v<T, h5pp::FileAccess>) {
        switch(item) {
            case h5pp::FileAccess::READONLY: return "READONLY";
            case h5pp::FileAccess::COLLISION_FAIL: return "COLLISION_FAIL";
            case h5pp::FileAccess::RENAME: return "RENAME";
            case h5pp::FileAccess::READWRITE: return "READWRITE";
            case h5pp::FileAccess::BACKUP: return "BACKUP";
            case h5pp::FileAccess::REPLACE: return "REPLACE";
            default: throw std::runtime_error("enum2sv: invalid enum case for h5pp::FileAccess");
        }
    } else
        static_assert(unrecognized_type_v<T>);

    throw std::runtime_error("enum2sv: invalid enum item");
}

template<typename T>
constexpr auto sv2enum(std::string_view item) {
    if constexpr(std::is_same_v<T, tb_mode>) {
        if(item == "eigen1") return tb_mode::eigen1;
        if(item == "eigen2") return tb_mode::eigen2;
        if(item == "eigen3") return tb_mode::eigen3;
        if(item == "cutensor") return tb_mode::cutensor;
        if(item == "xtensor") return tb_mode::xtensor;
        if(item == "tblis") return tb_mode::tblis;
        if(item == "cyclops") return tb_mode::cyclops;
    } else if constexpr(std::is_same_v<T, tb_type>) {
        if(item == "fp32") return tb_type::fp32;
        if(item == "fp64") return tb_type::fp64;
        if(item == "cplx") return tb_type::cplx;
    } else if constexpr(std::is_same_v<T, spdlog::level::level_enum>) {
        if(item == "trace") return spdlog::level::level_enum::trace;
        if(item == "debug") return spdlog::level::level_enum::debug;
        if(item == "info") return spdlog::level::level_enum::info;
        if(item == "warn") return spdlog::level::level_enum::warn;
        if(item == "warning") return spdlog::level::level_enum::warn;
        if(item == "err") return spdlog::level::level_enum::err;
        if(item == "error") return spdlog::level::level_enum::err;
        if(item == "crit") return spdlog::level::level_enum::critical;
        if(item == "critical") return spdlog::level::level_enum::critical;
        if(item == "off") return spdlog::level::level_enum::off;
    } else if constexpr(std::is_same_v<T, h5pp::FileAccess>) {
        if(item == "READONLY") return h5pp::FileAccess::READONLY;
        if(item == "COLLISION_FAIL") return h5pp::FileAccess::COLLISION_FAIL;
        if(item == "RENAME") return h5pp::FileAccess::RENAME;
        if(item == "READWRITE") return h5pp::FileAccess::READWRITE;
        if(item == "BACKUP") return h5pp::FileAccess::BACKUP;
        if(item == "REPLACE") return h5pp::FileAccess::REPLACE;
    } else {
        static_assert(unrecognized_type_v<T>);
    }

    throw std::runtime_error("sv2enum given invalid string item: " + std::string(item));
}

template<typename T, auto num>
using enumarray_t = std::array<std::pair<std::string, T>, num>;

template<typename T, typename... Args>
constexpr auto mapStr2Enum(Args... names) {
    constexpr auto num     = sizeof...(names);
    auto           pairgen = [](const std::string &name) -> std::pair<std::string, T> { return {name, sv2enum<T>(name)}; };
    return enumarray_t<T, num>{pairgen(names)...};
}

template<typename T, typename... Args>
constexpr auto mapEnum2Str(Args... enums) {
    constexpr auto num     = sizeof...(enums);
    auto           pairgen = [](const T &e) -> std::pair<std::string, T> { return {std::string(enum2sv(e)), e}; };
    return enumarray_t<T, num>{pairgen(enums)...};
}

template<typename T, auto num>
constexpr auto mapPairs2Str(const enumarray_t<T, num> &arrayPairEnumStr) -> std::array<std::string, num> {
    std::array<std::string, num> arrstr;
    for(size_t idx = 0; idx < num; ++idx) arrstr[idx] = arrayPairEnumStr[idx].first;
    return arrstr;
}