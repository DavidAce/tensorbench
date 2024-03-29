#include "info.h"
#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <unistd.h>

std::string debug::hostname() {
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);
    return hostname;
}

std::string debug::cpu_info(std::string_view info) {
    std::ifstream filestream("/proc/cpuinfo");
    std::string   line;
    while(std::getline(filestream, line)) {
        std::istringstream is_line(line);
        std::string        key;
        if(std::getline(is_line, key, ':')) {
            if(key == info) {
                std::string value_str;
                if(std::getline(is_line, value_str)) { return value_str; }
            }
        }
    }
    return {};
}

double debug::mem_usage_in_mb(std::string_view name) {
    std::ifstream filestream("/proc/self/status");
    std::string   line;
    while(std::getline(filestream, line)) {
        std::istringstream is_line(line);
        std::string        key;
        if(std::getline(is_line, key, ':')) {
            if(key == name) {
                std::string value_str;
                if(std::getline(is_line, value_str)) {
                    // Filter non-digit characters
                    value_str.erase(std::remove_if(value_str.begin(), value_str.end(), [](auto const &c) -> bool { return not std::isdigit(c); }),
                                    value_str.end());
                    // Extract the number
                    long long value = 0;
                    try {
                        std::string::size_type sz; // alias of size_t
                        value = std::stoll(value_str, &sz);
                    } catch(const std::exception &ex) {
                        std::fprintf(stderr, "Could not read mem usage from /proc/self/status: Failed to parse string [%s]: %s", value_str.c_str(), ex.what());
                    }
                    // Now we have the value in kb
                    return static_cast<double>(value) / 1024.0;
                }
            }
        }
    }
    return -1.0;
}

double debug::mem_rss_in_mb() { return mem_usage_in_mb("VmRSS"); }
double debug::mem_hwm_in_mb() { return mem_usage_in_mb("VmHWM"); }
double debug::mem_vm_in_mb() { return mem_usage_in_mb("VmPeak"); }

void debug::print_mem_usage() {
    std::printf("%-30s%10.1f MB\n", "Memory RSS", mem_rss_in_mb());
    std::printf("%-30s%10.1f MB\n", "Memory Peak", mem_hwm_in_mb());
    std::printf("%-30s%10.1f MB\n", "Memory Vm", mem_vm_in_mb());
}
