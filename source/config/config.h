#pragma once

#include "benchmark/benchmark.h"
#include <cstddef>
#include <fmt/core.h>
#include <h5pp/details/h5ppEnums.h>
#include <spdlog/fwd.h>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

enum class tb_mode;
enum class tb_type;

namespace config {
    // Get current date/time, format is YYYY-MM-DD.HH:mm:ss
    std::string getCurrentDateTime();
    void        initializeCuda(int gpun);
    std::string getCpuName();
    void        showCpuName();
    void        showGpuInfo();
    int         parse(int argc, char *argv[]);

    inline auto                 loglevel      = static_cast<spdlog::level::level_enum>(2); /*!<    */
    inline auto                 loglevel_h5pp = static_cast<spdlog::level::level_enum>(2); /*!<    */
    inline std::string          tb_dsetname   = "tbdb";
    inline std::string          tb_filename   = fmt::format("tbdb-{}.h5", getCurrentDateTime());
    inline h5pp::FileAccess     tb_fileaccess = h5pp::FileAccess::READWRITE;
    inline unsigned int         n_iter        = 1;    /*!<    */
    inline std::vector<int>     v_nomp        = {1};  /*!<    */
    inline std::vector<long>    v_spin        = {2};  /*!<    */
    inline std::vector<long>    v_mpoD        = {2};  /*!<    */
    inline std::vector<long>    v_chi         = {16}; /*!<    */
    inline std::vector<long>    v_chiL        = {-1}; /*!<    */
    inline std::vector<long>    v_chiR        = {-1}; /*!<    */
    inline std::vector<tb_mode> tb_modes      = {};   /*!<    */
    inline std::vector<tb_type> tb_types      = {};
    inline std::vector<int>     v_gpun        = {0};  /*!< There may be more gpus on this machine */
}