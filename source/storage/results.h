#include <h5pp/h5pp.h>

class tb_results{
    public:
    static inline h5pp::hid::h5t h5_type;
    struct table {
        char   name[16] = "";
        int    iter     = 0;
        int    threads  = 0;
        long   chiL     = 0;
        long   chiR     = 0;
        long   mpod     = 0;
        long   spin     = 0;
        double t_iter   = 0;
        double t_total  = 0;
        table(std::string_view name_, int iter_, int threads_, long chiL_,long chiR_,long mpod_, long spin_, double t_iter_, double t_total_):
         iter(iter_),
         threads(threads_),
         chiL(chiL_),
         chiR(chiR_),
         mpod(mpod_),
         spin(spin_),
         t_iter(t_iter_),
         t_total(t_total_)
        {std::strcpy(name,name_.data());}
    };
    tb_results() { register_table_type(); }
    static void register_table_type() {
        if(h5_type.valid()) return;
        h5_type = H5Tcreate(H5T_COMPOUND, sizeof(table));
        // Create a type for the char array from the template H5T_C_S1
        // The template describes a string with a single char.
        // Set the size with H5Tset_size, or h5pp::hdf5::setStringSize(...)
        h5pp::hid::h5t h5t_string = H5Tcopy(H5T_C_S1);
        H5Tset_size(h5t_string, 16);
        // Optionally set the null terminator '\0'
        H5Tset_strpad(h5t_string, H5T_STR_NULLTERM);
        H5Tinsert(h5_type, "name", HOFFSET(table, name), h5t_string);
        H5Tinsert(h5_type, "iter", HOFFSET(table, iter), H5T_NATIVE_INT);
        H5Tinsert(h5_type, "threads", HOFFSET(table, threads), H5T_NATIVE_INT);
        H5Tinsert(h5_type, "chiL", HOFFSET(table, chiL), H5T_NATIVE_LONG);
        H5Tinsert(h5_type, "chiR", HOFFSET(table, chiR), H5T_NATIVE_LONG);
        H5Tinsert(h5_type, "mpod", HOFFSET(table, mpod), H5T_NATIVE_LONG);
        H5Tinsert(h5_type, "spin", HOFFSET(table, spin), H5T_NATIVE_LONG);
        H5Tinsert(h5_type, "t_iter", HOFFSET(table, t_iter), H5T_NATIVE_DOUBLE);
        H5Tinsert(h5_type, "t_total", HOFFSET(table, t_total), H5T_NATIVE_DOUBLE);
    }
};
