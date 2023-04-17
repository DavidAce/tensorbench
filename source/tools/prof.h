#pragma once
#include <memory>
#include <string_view>
#include <vector>

namespace tools::prof {
    extern void   print_mem_usage();
    extern double mem_usage_in_mb(std::string_view name);
    extern double mem_rss_in_mb();
    extern double mem_hwm_in_mb();
    extern double mem_vm_in_mb();
}