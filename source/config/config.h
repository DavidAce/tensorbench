#pragma once
#include <spdlog/fwd.h>
#include <vector>
namespace config {
    int parse(int argc, char *argv[]);

    inline auto              loglevel      = static_cast<spdlog::level::level_enum>(2); /*!<    */
    inline auto              loglevel_h5pp = static_cast<spdlog::level::level_enum>(2); /*!<    */
    inline unsigned int      n_iter        = 1;                                         /*!<    */
    inline std::vector<int>  v_nomp        = {1};                                      /*!<    */
    inline std::vector<long> v_spin        = {2};                                       /*!<    */
    inline std::vector<long> v_mpod        = {2};                                       /*!<    */
    inline std::vector<long> v_bond        = {16};                                      /*!<    */
    inline std::vector<long> v_bondL       = {-1};                                      /*!<    */
    inline std::vector<long> v_bondR       = {-1};                                      /*!<    */
}