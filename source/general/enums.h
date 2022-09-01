#pragma once
#include <string>
#include <string_view>
enum class tb_mode { eigen1, eigen1_split, eigen2, eigen3, cute, acro, xtensor, tblis };

/* clang-format off */
template<typename T>
constexpr std::string_view enum2sv(const T &item) {
    static_assert(std::is_enum_v<T> and "enum2sv<T>: T must be an enum");
    if constexpr(std::is_same_v<T, tb_mode>) {
        if(item == tb_mode::eigen1)               return "eigen1";
        if(item == tb_mode::eigen1_split)         return "eigen1_split";
        if(item == tb_mode::eigen2)               return "eigen2";
        if(item == tb_mode::eigen3)               return "eigen3";
        if(item == tb_mode::cute)                 return "cute";
        if(item == tb_mode::acro)                 return "acro";
        if(item == tb_mode::xtensor)              return "xtensor";
        if(item == tb_mode::tblis)                return "tblis";
    }
    throw std::runtime_error("Given invalid enum item");

}


template<typename T>
constexpr auto sv2enum(std::string_view item) {
    if constexpr(std::is_same_v<T, tb_mode>) {
        if(item == "eigen1")                      return tb_mode::eigen1;
        if(item == "eigen1_split")                return tb_mode::eigen1_split;
        if(item == "eigen2")                      return tb_mode::eigen2;
        if(item == "eigen3")                      return tb_mode::eigen3;
        if(item == "cute")                        return tb_mode::cute;
        if(item == "acro")                        return tb_mode::acro;
        if(item == "xtensor")                     return tb_mode::xtensor;
        if(item == "tblis")                       return tb_mode::tblis;
    }

    throw std::runtime_error("sv2enum given invalid string item: " + std::string(item));

}