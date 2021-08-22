#pragma once
#include <memory>
#include <string_view>
#include <vector>
class class_tic_toc;

namespace tools::prof{
    void init_profiling();
    void reset_profiling();
    inline std::unique_ptr<class_tic_toc> t_total;
    inline std::unique_ptr<class_tic_toc> t_eigen1;
    inline std::unique_ptr<class_tic_toc> t_eigen2;
    inline std::unique_ptr<class_tic_toc> t_eigen3;
    inline std::unique_ptr<class_tic_toc> t_cuda;
    inline std::unique_ptr<class_tic_toc> t_acro;
    inline std::unique_ptr<class_tic_toc> t_cute;
    inline std::unique_ptr<class_tic_toc> t_xtensor;
    inline std::unique_ptr<class_tic_toc> t_tblis;


    extern void print_mem_usage();
    extern double mem_usage_in_mb(std::string_view name);
    extern double mem_rss_in_mb();
    extern double mem_hwm_in_mb();
    extern double mem_vm_in_mb();
}