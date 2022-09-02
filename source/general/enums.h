#pragma once
#include <string>
#include <string_view>
enum class tb_mode { eigen1, eigen2, eigen3, cute, acro, xtensor, tblis, cyclops };

/* clang-format off */
template<typename T>
constexpr std::string_view enum2sv(const T &item) {
    static_assert(std::is_enum_v<T> and "enum2sv<T>: T must be an enum");
    if constexpr(std::is_same_v<T, tb_mode>) {
        switch(item){
            case tb_mode::eigen1         :     return "eigen1";
            case tb_mode::eigen2         :     return "eigen2";
            case tb_mode::eigen3         :     return "eigen3";
            case tb_mode::cute           :     return "cute";
            case tb_mode::acro           :     return "acro";
            case tb_mode::xtensor        :     return "xtensor";
            case tb_mode::tblis          :     return "tblis";
            case tb_mode::cyclops        :     return "cyclops";
        }

    }
    throw std::runtime_error("Given invalid enum item");

}


template<typename T>
constexpr auto sv2enum(std::string_view item) {
    if constexpr(std::is_same_v<T, tb_mode>) {
        if(item == "eigen1")                      return tb_mode::eigen1;
        if(item == "eigen2")                      return tb_mode::eigen2;
        if(item == "eigen3")                      return tb_mode::eigen3;
        if(item == "cute")                        return tb_mode::cute;
        if(item == "acro")                        return tb_mode::acro;
        if(item == "xtensor")                     return tb_mode::xtensor;
        if(item == "tblis")                       return tb_mode::tblis;
        if(item == "cyclops")                     return tb_mode::cyclops;
    }

    throw std::runtime_error("sv2enum given invalid string item: " + std::string(item));

}