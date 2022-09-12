//
// Created by david on 2021-10-12.
//

#include "config.h"
#include <CLI/CLI.hpp>
#include <omp.h>
#include <spdlog/common.h>
namespace config {
    template<class>
    inline constexpr bool unrecognized_type_v = false;

    template<typename T>
    constexpr auto sv2enum(std::string_view item) {
        if constexpr(std::is_same_v<T, spdlog::level::level_enum>) {
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
        } else {
            static_assert(unrecognized_type_v<T>);
        }
        throw std::logic_error("Unrecognized type");
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
}

// MWE: https://godbolt.org/z/jddxod53d
int config::parse(int argc, char **argv) {
    auto s2e_log = mapStr2Enum<spdlog::level::level_enum>("trace", "debug", "info", "warn", "error", "critical", "off");

    //    std::string omp_max_threads_default = "1";
    //
#if defined(_OPENMP)
    // This will honor env variable OMP_NUM_THREADS
    v_nomp = {omp_get_max_threads()};
#endif

    CLI::App app;
    app.description("Tensorbench: Benchmark tensor contractions");
    app.get_formatter()->column_width(140);
    app.option_defaults()->always_capture_default();
    app.allow_extras(false);
    /* clang-format off */
    app.add_flag("-v"                                , [](int count) { count == 1 ? config::loglevel = spdlog::level::debug : config::loglevel = spdlog::level::trace; }
                                                                              , "Set log level to debug (-vv for trace)");
    app.add_option("--loglevel"                      , config::loglevel       , "Log level of tensorbench")->transform(CLI::CheckedTransformer(s2e_log, CLI::ignore_case))
                                                                              ->option_text("    ENUM:{trace, debug, info, warn, error, critical, off}  [info]");
    app.add_option("--logh5pp"                       , config::loglevel_h5pp  , "Log level of h5pp library")->transform(CLI::CheckedTransformer(s2e_log, CLI::ignore_case))
                                                                              ->option_text("     ENUM:{trace, debug, info, warn, error, critical, off}  [info]");
    app.add_option("-i,--iters"                      , config::n_iter         , "Number of times to run each benchmark");
    app.add_option("-n,--nomps"                      , config::v_nomp         , "Number(s) of OpenMP threads to test");
    app.add_option("-D,--spindims"                   , config::v_spin         , "Size(s) of MPS spin dimensions to test");
    app.add_option("-M,--mpobonds"                   , config::v_mpod         , "Size(s) of MPO bond dimensions to test");
    app.add_option("-B,--mpsbonds"                   , config::v_bond         , "Size(s) of MPS bond dimensions to test");
    auto mpsbondL = app.add_option("-L,--mpsbondL"   , config::v_bondL        , "Size(s) of MPS left  bond dimensions to test")->excludes("--mpsbonds");
    auto mpsbondR = app.add_option("-R,--mpsbondR"   , config::v_bondR        , "Size(s) of MPS right bond dimensions to test")->excludes("--mpsbonds");
    /* clang-format on */
    mpsbondL->needs(mpsbondR);
    mpsbondR->needs(mpsbondL);
    app.get_formatter()->label("--loglevel", "trace");
    try {
        app.parse(argc, argv);
    } catch(const CLI::ParseError &e) { std::exit(app.exit(e)); }
    return 0;
}