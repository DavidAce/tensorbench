#include "benchmark.h"
#include "general/enums.h"
#if defined(TB_TBLIS)
    #include "tid/tid.h"
    #include <complex>
    #include <tblis/tblis.h>
    #include <tblis/util/thread.h>
template<typename T, auto N>
tblis::short_vector<tblis::len_type, N> dim2len(const Eigen::DSizes<T, N> &dims) {
    tblis::short_vector<tblis::len_type, N> len = {};
    for(size_t i = 0; i < N; i++) len[i] = dims[i];
    return len;
}
template<typename T, auto N>
tblis::short_vector<tblis::stride_type, N> dim2stride(const Eigen::DSizes<T, N> &dims) {
    tblis::short_vector<tblis::stride_type, N> stride = {};
    stride[0]                                         = 1;
    for(size_t i = 1; i < N; i++) stride[i] = stride[i - 1] * dims[i - 1];
    return stride;
}

template<typename Scalar, auto NA, auto NB, auto NC>
void contract_tblis(const Eigen::Tensor<Scalar, NA> &ea, const Eigen::Tensor<Scalar, NB> &eb, Eigen::Tensor<Scalar, NC> &ec, const tblis::label_vector &la,
                    const tblis::label_vector &lb, const tblis::label_vector &lc) {
    tblis::len_vector da, db, dc;
    da.assign(ea.dimensions().begin(), ea.dimensions().end());
    db.assign(eb.dimensions().begin(), eb.dimensions().end());
    dc.assign(ec.dimensions().begin(), ec.dimensions().end());
    auto                         ta    = tblis::varray_view<const Scalar>(da, ea.data(), tblis::COLUMN_MAJOR);
    auto                         tb    = tblis::varray_view<const Scalar>(db, eb.data(), tblis::COLUMN_MAJOR);
    auto                         tc    = tblis::varray_view<Scalar>(dc, ec.data(), tblis::COLUMN_MAJOR);
    Scalar                       alpha = 1.0;
    Scalar                       beta  = 0.0;
    tblis::tblis_tensor          A_s(alpha, ta);
    tblis::tblis_tensor          B_s(tb);
    tblis::tblis_tensor          C_s(beta, tc);
    const tblis::tblis_config_s *tblis_config = tblis::tblis_get_config("zen");
    tblis_tensor_mult(nullptr, tblis_config, &A_s, la.c_str(), &B_s, lb.c_str(), &C_s, lc.c_str());
}
#endif

template<typename T>
benchmark::ResultType<T> benchmark::tensor_product_tblis([[maybe_unused]] const tb_setup<T> &tbs) {
#if defined(TB_TBLIS)
    #if defined(TCI_USE_OPENMP_THREADS) && defined(_OPENMP)
    tblis_set_num_threads(static_cast<unsigned int>(tbs.nomp));
    #endif

    auto                   t_complete = tid::tic_scope(enum2sv(tb_mode::tblis));
    Eigen::DSizes<long, 3> dsizes     = tbs.psi.dimensions();
    Eigen::Tensor<T, 3>    ham_sq_psi(dsizes);
    if(tbs.psi.dimension(1) >= tbs.psi.dimension(2)) {
        auto                t_con = tid::tic_token("contract", tid::level::detailed);
        Eigen::Tensor<T, 4> psi_envL(tbs.psi.dimension(0), tbs.psi.dimension(2), tbs.envL.dimension(1), tbs.envL.dimension(2));
        Eigen::Tensor<T, 4> psi_envL_mpo(tbs.psi.dimension(2), tbs.envL.dimension(1), tbs.mpo.dimension(1), tbs.mpo.dimension(3));
        contract_tblis(tbs.psi, tbs.envL, psi_envL, "afb", "fcd", "abcd");
        contract_tblis(psi_envL, tbs.mpo, psi_envL_mpo, "abcd", "diaj", "bcij");
        contract_tblis(psi_envL_mpo, tbs.envR, ham_sq_psi, "bcij", "bki", "jck");
    } else {
        auto                t_con = tid::tic_token("contract", tid::level::detailed);
        Eigen::Tensor<T, 4> psi_envR(tbs.psi.dimension(0), tbs.psi.dimension(1), tbs.envR.dimension(1), tbs.envR.dimension(2));
        Eigen::Tensor<T, 4> psi_envR_mpo(tbs.psi.dimension(1), tbs.envR.dimension(1), tbs.mpo.dimension(0), tbs.mpo.dimension(3));
        contract_tblis(tbs.psi, tbs.envR, psi_envR, "abf", "fcd", "abcd");
        contract_tblis(psi_envR, tbs.mpo, psi_envR_mpo, "qijk", "rkql", "ijrl");
        contract_tblis(psi_envR_mpo, tbs.envL, ham_sq_psi, "qkri", "qjr", "ijk");
    }
    return ham_sq_psi;
#else
    return {};
#endif
}

using fp32 = benchmark::fp32;
using fp64 = benchmark::fp64;
using cplx = benchmark::cplx;

template benchmark::ResultType<fp32> benchmark::tensor_product_tblis(const tb_setup<fp32> &tbs);
template benchmark::ResultType<fp64> benchmark::tensor_product_tblis(const tb_setup<fp64> &tbs);
template benchmark::ResultType<cplx> benchmark::tensor_product_tblis(const tb_setup<cplx> &tbs);
