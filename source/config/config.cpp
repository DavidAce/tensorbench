//
// Created by david on 2021-10-12.
//

#include "config.h"
#include "general/enums.h"
#include <CLI/CLI.hpp>
#include <fmt/ranges.h>
#include <omp.h>
#include <spdlog/common.h>

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
std::string config::getCurrentDateTime() {
    time_t    now     = time(nullptr);
    struct tm tstruct = {};
    char      buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%F", &tstruct);
    return buf;
}

// MWE: https://godbolt.org/z/jddxod53d
int config::parse(int argc, char **argv) {
    auto s2e_log = mapStr2Enum<spdlog::level::level_enum>("trace", "debug", "info", "warn", "error", "critical", "off");
    auto s2e_acc = mapStr2Enum<h5pp::FileAccess>("READONLY", "COLLISION_FAIL", "RENAME", "READWRITE", "BACKUP", "REPLACE");
    auto s2e_tbm = mapStr2Enum<tb_mode>("eigen1", "eigen2", "eigen3", "cute", "xtensor", "tblis", "cyclops");
    auto s2e_tbt = mapStr2Enum<tb_type>("fp32", "fp64", "cplx");

    // Set defaults
    config::tb_types = {tb_type::fp32, tb_type::fp64, tb_type::cplx};
    config::tb_modes = {tb_mode::eigen1, tb_mode::eigen2, tb_mode::eigen3, tb_mode::cute, tb_mode::xtensor, tb_mode::tblis, tb_mode::cyclops};

#if defined(_OPENMP)
    // This will honor env variable OMP_NUM_THREADS
    v_nomp = {omp_get_max_threads()};
#endif

    CLI::App app;
    app.description("Tensorbench: Benchmark tensor contractions");
    app.get_formatter()->column_width(100);
    app.option_defaults()->always_capture_default();
    app.allow_extras(false);

    app.add_flag(
        "-v", [](int count) { count == 1 ? config::loglevel = spdlog::level::debug : config::loglevel = spdlog::level::trace; },
        "Set log level to debug (-vv for trace)");
    app.add_option("--loglevel", config::loglevel, "Log level of tensorbench")
        ->transform(CLI::CheckedTransformer(s2e_log, CLI::ignore_case))
        ->option_text(fmt::format("ENUM:{}", mapPairs2Str(s2e_log)));
    app.add_option("--logh5pp", config::loglevel_h5pp, "Log level of h5pp library")
        ->transform(CLI::CheckedTransformer(s2e_log, CLI::ignore_case))
        ->option_text(fmt::format("ENUM:{}", mapPairs2Str(s2e_log)));
    app.add_option("-m,--modes", config::tb_modes, "List of benchmark modes (libraries)")
        ->transform(CLI::CheckedTransformer(s2e_tbm, CLI::ignore_case))
        ->option_text(fmt::format("ENUM:{}", mapPairs2Str(s2e_tbm)));
    app.add_option("-t,--types", config::tb_types, "List of benchmark types (arithmetic)")
        ->transform(CLI::CheckedTransformer(s2e_tbt, CLI::ignore_case))
        ->option_text(fmt::format("ENUM:{}", mapPairs2Str(s2e_tbt)));
    app.add_option("-a,--facc", config::tb_fileaccess, "File access to the output file")
        ->transform(CLI::CheckedTransformer(s2e_acc, CLI::ignore_case))
        ->option_text(fmt::format("ENUM:{}", mapPairs2Str(s2e_acc)));

    app.add_option("-d,--dname", config::tb_dsetname, "Name (or path) to the dataset in the resulting HDF5 output file");
    app.add_option("-f,--fname", config::tb_filename, "Path to the resulting HDF5 output file");
    app.add_option("-i,--iters", config::n_iter, "Number of times to run each benchmark");
    app.add_option("-n,--nomps", config::v_nomp, "Number(s) of OpenMP threads to test");
    app.add_option("-D,--spindims", config::v_spin, "Size(s) of MPS spin dimensions to test");
    app.add_option("-M,--mpobonds", config::v_mpoD, "Size(s) of MPO bond dimensions to test");
    app.add_option("-B,--mpsbonds", config::v_chi, "Size(s) of MPS bond dimensions to test");
    auto mpsbondL = app.add_option("-L,--mpsbondL", config::v_chiL, "Size(s) of MPS left  bond dimensions to test")->excludes("--mpsbonds");
    auto mpsbondR = app.add_option("-R,--mpsbondR", config::v_chiR, "Size(s) of MPS right bond dimensions to test")->excludes("--mpsbonds");

    mpsbondL->needs(mpsbondR);
    mpsbondR->needs(mpsbondL);
    app.get_formatter()->label("--loglevel", "trace");
    try {
        app.parse(argc, argv);
    } catch(const CLI::ParseError &e) { std::exit(app.exit(e)); }

    return 0;
}