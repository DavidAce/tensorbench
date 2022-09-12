#include "tid.h"
#include <fmt/format.h>
#include <fmt/ranges.h>
namespace tid {
    token::token(ur &t_) : t(t_) { t.tic(); }
    token::token(ur &t_, std::string_view prefix_) : t(t_) {
        temp_prefix = std::string(prefix_);
        tic();
    }

    token::~token() noexcept {
        try {
            if(t.is_measuring) t.toc();
            if(not tid::internal::current_scope.empty() and tid::internal::current_scope.back() == temp_prefix) {
                tid::internal::current_scope.pop_back();
                fmt::print("token popped [{}] from scope | current scope {}\n", temp_prefix, tid::internal::current_scope);
            }
        } catch(const std::exception &ex) { fprintf(stderr, "tid: error in token destructor for tid::ur [%s]: %s", t.get_label().c_str(), ex.what()); }
    }

    void token::tic() noexcept {
        tid::internal::current_scope.emplace_back(temp_prefix);
        fmt::print("token appended [{}] to scope| current scope {}\n", temp_prefix, tid::internal::current_scope);
        t.tic();
    }
    void token::toc() noexcept {
        t.toc();
        if(not tid::internal::current_scope.empty() and tid::internal::current_scope.back() == temp_prefix) {
            tid::internal::current_scope.pop_back();
            fmt::print("token popped [{}] from scope | current scope {}\n", temp_prefix, tid::internal::current_scope);
        }
    }
    ur &token::ref() noexcept { return t; }
    ur *token::operator->() noexcept { return &t; }

}
