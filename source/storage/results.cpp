#include "results.h"
#include "benchmark/benchmark.h"
#include "general/enums.h"
template<typename T>
tb_results::table::table(const tb_setup<T> &tbs, size_t itrn, double t_contr, double t_total)
    : mode(enum2sv(tbs.mode)), type(enum2sv(tbs.type)), device(tbs.device), nomp(tbs.nomp), nmpi(tbs.nmpi), gpun(tbs.gpun), spin(tbs.psi.dimension(0)),
      chiL(tbs.psi.dimension(1)), chiR(tbs.psi.dimension(2)), mpoD(tbs.mpo.dimension(0)), itrn(itrn), itrs(tbs.iters), t_contr(t_contr), t_total(t_total) {}

template tb_results::table::table(const tb_setup<benchmark::fp32> &tbs, size_t itrn, double t_contr, double t_total);
template tb_results::table::table(const tb_setup<benchmark::fp64> &tbs, size_t itrn, double t_contr, double t_total);
template tb_results::table::table(const tb_setup<benchmark::cplx> &tbs, size_t itrn, double t_contr, double t_total);

tb_results::tb_results() { register_table_type(); }

void tb_results::register_table_type() {
    if(h5_type.valid()) return;
    h5_type = H5Tcreate(H5T_COMPOUND, sizeof(table));
    H5Tinsert(h5_type, "mode", HOFFSET(table, mode), h5pp::vstr_t::get_h5type());
    H5Tinsert(h5_type, "type", HOFFSET(table, type), h5pp::vstr_t::get_h5type());
    H5Tinsert(h5_type, "device", HOFFSET(table, device), h5pp::vstr_t::get_h5type());
    H5Tinsert(h5_type, "nomp", HOFFSET(table, nomp), H5T_NATIVE_INT);
    H5Tinsert(h5_type, "nmpi", HOFFSET(table, nmpi), H5T_NATIVE_INT);
    H5Tinsert(h5_type, "gpun", HOFFSET(table, gpun), H5T_NATIVE_INT);
    H5Tinsert(h5_type, "spin", HOFFSET(table, spin), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "chiL", HOFFSET(table, chiL), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "chiR", HOFFSET(table, chiR), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "mpoD", HOFFSET(table, mpoD), H5T_NATIVE_LONG);
    H5Tinsert(h5_type, "itrn", HOFFSET(table, itrn), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "itrs", HOFFSET(table, itrs), H5T_NATIVE_ULONG);
    H5Tinsert(h5_type, "t_contr", HOFFSET(table, t_contr), H5T_NATIVE_DOUBLE);
    H5Tinsert(h5_type, "t_total", HOFFSET(table, t_total), H5T_NATIVE_DOUBLE);
}