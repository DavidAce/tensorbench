#include "tid.h"
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <regex>
#include <string>
namespace tid {
    namespace internal {

        const ur *ur_ref_t::operator->() const { return &ref.get(); }

        [[nodiscard]] std::string ur_ref_t::str() const {
            return fmt::format(FMT_STRING("{0:<{1}} {2:>8.3f} s | sum {3:>8.3f} s | {4:>6.2f} % | avg {5:>8.2e} s | level {6} | count {7}"), key,
                               tree_max_key_size, ref.get().get_time(), sum, 100 * frac, ref.get().get_time_avg(), level2sv(ref.get().get_level()),
                               ref.get().get_tic_count());
        }

        template<typename T>
        T split(std::string_view strv, std::string_view delims) {
            T output;
            //            output.reserve(strv.length() / 4);
            auto first = strv.begin();
            while(first != strv.end()) {
                const auto second = std::find_first_of(first, std::cend(strv), std::cbegin(delims), std::cend(delims));
                const auto pos    = static_cast<std::string_view::size_type>(std::distance(strv.begin(), first));
                const auto len    = static_cast<std::string_view::size_type>(std::distance(first, second));
                if(first != second) output.emplace_back(strv.substr(pos, len));
                if(second == strv.end()) break;
                first = std::next(second);
            }
            //            output.shrink_to_fit();
            return output;
        }
        template std::vector<std::string_view> split(std::string_view strv, std::string_view delims);
        template std::deque<std::string_view>  split(std::string_view strv, std::string_view delims);
    }

    ur &get(std::deque<std::string_view> &keys, level l, ur &u) {
        if(keys.empty()) return u;
        auto &u_sub = u.insert(keys[0], l);
        keys.pop_front();
        return get(keys, l, u_sub);
    }
    ur &get(std::string_view key, level l) {
        /* This function searches the current tid database tid::internal::tid_db for ur objects with respect to the current scope.
         * Note that this function never updates the current scope
         * Example:
         *  Let tid_db have:
         *          main
         *          main.func1
         *          main.func1.opt
         *          main.func2.opt
         *          main.func2.step
         *
         *  Then
         *      if key == "opt" and current_scope == {"main", "func1"}: return existing ur with label "main.func1.opt"
         *      if key == "opt" and current_scope == {"main", "func1", "opt"}: create a new item "opt" in the current scope, to return "main.func1.opt.opt"
         *  Note that if key has at least one "." in it, we assume that the key is an "absolute path", so we do not prepend the current prefix.
         *
         *
         */
        std::string prefix_key;
        bool        has_dot = key.rfind('.', 0) != std::string_view::npos;
        if(has_dot) prefix_key = key; // Assume absolute scope
        else {
            // Create a key assuming relative scope
            if(not internal::current_scope.empty() and not key.empty()) {
                prefix_key = fmt::format("{}.{}", fmt::join(internal::current_scope, "."), key);
            } else if(not internal::current_scope.empty() and key.empty()) {
                prefix_key = fmt::format("{}", fmt::join(internal::current_scope, "."));
            } else if(internal::current_scope.empty() and not key.empty()) {
                prefix_key = key;
            }
        }
//        fmt::print("prefix_key {} | has_dot {} | current_scope: {}\n", prefix_key, has_dot, tid::internal::current_scope);
        // Now prefix_key is an "absolute path" to an ur that may or may not exist.
        // If it does not exist we insert it
//        fmt::print("tid_db before: [");
//        for(const auto &t : tid::internal::tid_db) fmt::print("{}, ", t.first);
//        fmt::print("]\n");
        auto result = tid::internal::tid_db.insert({prefix_key, tid::ur(prefix_key)});

        if(tid::internal::tid_db.find(prefix_key) == tid::internal::tid_db.end())
            throw std::runtime_error("Failed to insert " + prefix_key + " into std::unordered_map: tid::internal::tid_db");
//        fmt::print("tid_db after : [");
//        for(const auto &t : tid::internal::tid_db) fmt::print("{}, ", t.first);
//        fmt::print("]\n");
        auto &ur_ref = result.first->second;
        if(result.second and l != level::parent) ur_ref.set_level(l);
        return ur_ref;
    }

    token tic_token(std::string_view key, level l) { return tid::get(key, l).tic_token(); }

    token tic_scope(std::string_view key, level l) {
        return tid::get(key, l).tic_token(key);
    }

    void tic(std::string_view key, level l) { get(key, l).tic(); }

    void toc(std::string_view key, level l) { get(key, l).toc(); }

    void reset(std::string_view expr) {
        for(auto &[key, ur] : tid::internal::tid_db) {
            if(std::regex_match(std::string(key), std::regex(std::string(expr)))) ur.reset();
        }
    }

    void reset(const std::vector<std::string> &excl) {
        for(auto &[key, ur] : tid::internal::tid_db) {
            for(const auto &e : excl) {
                if(key == e) continue;
            }
            ur.reset();
        }
    }

    void set_scope(std::string_view prefix) {
        tid::internal::current_scope.clear();
        for(const auto &s : internal::split(prefix, ".")) internal::current_scope.emplace_back(s);
    }

    std::vector<internal::ur_ref_t> get_tree(const tid::ur &u, std::string_view prefix, level l) {
        std::string key;
        if(prefix.empty()) key = u.get_label();
        else
            key = fmt::format("{}.{}", prefix, u.get_label());

        std::vector<internal::ur_ref_t> tree = {internal::ur_ref_t{key, u, 0.0, 1.0}};
        for(const auto &un : u.ur_under) {
            tree.front().sum += un.second->get_time(); // Add times under
            for(const auto &t : get_tree(*un.second, key, l)) {
                //                if(un.second->get_time() == 0) {
                //                    // If the intermediate node did not measure time, add the times under it instead
                //                    tree.front().sum += t.sum;
                //                }
                tree.emplace_back(t);
            }
        }

        // Sort the tree
        if(tree.size() > 1)
            std::sort(std::next(tree.begin()), tree.end(), [](const internal::ur_ref_t &t1, const internal::ur_ref_t &t2) -> bool { return t1.key < t2.key; });

        // Prune the tree based on level
        tree.erase(std::remove_if(tree.begin(), tree.end(), [&l](auto &t) -> bool { return t->get_level() > l; }), tree.end());

        // Find the longest key in the tree
        auto   max_it       = std::max_element(tree.begin(), tree.end(), [](const auto &a, const auto &b) -> bool { return a.key.size() < b.key.size(); });
        size_t max_key_size = 0;
        if(max_it != tree.end()) max_key_size = max_it->key.size();

        // Calculate the fractions and set the maximum key size
        for(auto &t : tree) {
            t.tree_max_key_size = max_key_size;
            if(tree.front()->get_time() == 0) break;
            if(&t == &tree.front()) continue;
            auto t_parent = tree.front()->get_time() == 0.0 ? tree.front().sum : tree.front()->get_time();
            t.frac        = t->get_time() / t_parent;
        }

        return tree;
    }

    std::vector<internal::ur_ref_t> get_tree(std::string_view prefix, level l) {
        std::vector<internal::ur_ref_t> tree;
        for(const auto &[key, u] : tid::internal::tid_db) {
            if(key == prefix or prefix.empty()) {
                auto t = get_tree(u, "", l);
                tree.insert(tree.end(), t.begin(), t.end());
            }
        }
        // Find the longest key in the tree
        auto max_it = std::max_element(tree.begin(), tree.end(), [](const auto &a, const auto &b) -> bool { return a.key.size() < b.key.size(); });
        if(max_it != tree.end()) {
            // Set the max key size
            for(auto &t : tree) t.tree_max_key_size = max_it->key.size();
        }

        return tree;
    }

    std::vector<internal::ur_ref_t> search(const tid::ur &u, std::string_view match) {
        std::vector<internal::ur_ref_t> matches;
        for(const auto &t : get_tree(u, "", level::detailed)) {
            if(t.key.find(match) != std::string_view::npos) matches.push_back(t);
        }
        return matches;
    }

    std::vector<internal::ur_ref_t> search(std::string_view match) {
        std::vector<internal::ur_ref_t> matches;
        for(const auto &t : get_tree("", level::detailed)) {
            if(t.key.find(match) != std::string_view::npos) matches.push_back(t);
        }
        return matches;
    }

    void print_tree(const tid::ur &u, std::string_view prefix, level l) {
        for(const auto &t : tid::get_tree(u, prefix)) {
            if(t->get_level() <= l) fmt::print("{}\n", t.str());
        }
    }
    void print_tree(std::string_view prefix, level l) {
        for(const auto &t : tid::get_tree(prefix)) {
            if(t->get_level() <= l) fmt::print("{}\n", t.str());
        }
    }
}