#include "benchmark.h"
#include "debug/exceptions.h"
#include "general/enums.h"
#if defined(TB_TBLIS)
    #include "math/tenx.h"
    #include "tid/tid.h"
    #include <complex>
    #include <tblis/tblis.h>

template<typename Scalar, auto NA, auto NB, auto NC>
void contract_tblis(const Eigen::Tensor<Scalar, NA> &ea, const Eigen::Tensor<Scalar, NB> &eb, Eigen::Tensor<Scalar, NC> &ec, const tblis::label_vector &la,
                    const tblis::label_vector &lb, const tblis::label_vector &lc) {
    auto   ta    = tblis::tensor_wrapper(ea);
    auto   tb    = tblis::tensor_wrapper(eb);
    auto   tc    = tblis::tensor_wrapper(ec);
    Scalar alpha = 1.0;
    Scalar beta  = 0.0;
    tblis::mult(alpha, ta, la, tb, lb, beta, tc, lc);
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
        {
            auto t_con1 = tid::tic_token("contract1", tid::level::detailed);
            contract_tblis(tbs.psi, tbs.envL, psi_envL, "afb", "fcd", "abcd");
        }
        {
            auto t_con2 = tid::tic_token("contract2", tid::level::detailed);
            contract_tblis(psi_envL, tbs.mpo, psi_envL_mpo, "abcd", "diaj", "bcij");
        }
        {
            auto t_con3 = tid::tic_token("contract3", tid::level::detailed);
            contract_tblis(psi_envL_mpo, tbs.envR, ham_sq_psi, "bcij", "bki", "jck");
        }
    } else {
        auto                t_con = tid::tic_token("contract", tid::level::detailed);
        Eigen::Tensor<T, 4> psi_envR(tbs.psi.dimension(0), tbs.psi.dimension(1), tbs.envR.dimension(1), tbs.envR.dimension(2));
        Eigen::Tensor<T, 4> psi_envR_mpo(tbs.psi.dimension(1), tbs.envR.dimension(1), tbs.mpo.dimension(0), tbs.mpo.dimension(3));
        {
            auto t_con1 = tid::tic_token("contract1", tid::level::detailed);
            contract_tblis(tbs.psi, tbs.envR, psi_envR, "abf", "fcd", "abcd");
        }

        {
            auto t_con2 = tid::tic_token("contract2", tid::level::detailed);
            contract_tblis(psi_envR, tbs.mpo, psi_envR_mpo, "qijk", "rkql", "ijrl");
        }

        {
            auto t_con3 = tid::tic_token("contract3", tid::level::detailed);
            contract_tblis(psi_envR_mpo, tbs.envL, ham_sq_psi, "qkri", "qjr", "ijk");
        }
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
