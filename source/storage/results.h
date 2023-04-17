#include <h5pp/h5pp.h>
template<typename T>
struct tb_setup;

class tb_results {
    public:
    static inline h5pp::hid::h5t h5_type;
    struct table {
        using vlen_type      = h5pp::vstr_t;
        h5pp::vstr_t mode    = "";
        h5pp::vstr_t type    = "";
        int          nomp    = 0;
        int          nmpi    = 0;
        long         spin    = 0;
        long         chiL    = 0;
        long         chiR    = 0;
        long         mpoD    = 0;
        size_t       itrn    = 0;
        size_t       itrs    = 0;
        double       t_contr = 0;
        double       t_total = 0;
        template<typename T>
        table(const tb_setup<T> &tbs, size_t itrn, double t_contr, double t_total);
    };
    tb_results();
    static void register_table_type();
};
