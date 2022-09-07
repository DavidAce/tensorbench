#if defined(TB_TBLIS)

    #include <complex>
    #include "contract/contract.h"
    #include "tblis/tblis.h"
    #include "tblis/util/thread.h"
    #include "tools/class_tic_toc.h"
    #include "tools/log.h"
    #include "tools/prof.h"

long get_ops_tblis_L(long d, long chiL, long chiR, long m);
long get_ops_tblis_R(long d, long chiL, long chiR, long m) {
    // Same as L, just swap chiL and chiR
    if(chiR > chiL) return get_ops_tblis_L(d, chiR, chiL, m);
    else
        return get_ops_tblis_L(d, chiL, chiR, m);
}

long get_ops_tblis_L(long d, long chiL, long chiR, long m) {
    if(chiR > chiL) return get_ops_tblis_R(d, chiL, chiR, m);
    if(d > m) {
        // d first
        long step1 = chiL * chiL * chiR * m * m * d;
        long step2 = chiL * chiR * d * m * m * m * (d * m + 1);
        long step3 = chiL * chiR * d * m * (chiR * m * m * m + m * m + 1);
        return step1 + step2 + step3;
    } else {
        // m first
        long step1 = chiL * chiL * chiR * m * d;
        long step2 = chiL * chiR * d * m * m * (d * m + 1);
        long step3 = chiL * chiR * chiR * d * m * (m + 1);
        return step1 + step2 + step3;
    }
}

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
    auto   ta    = tblis::varray_view<const Scalar>(da, ea.data(), tblis::COLUMN_MAJOR);
    auto   tb    = tblis::varray_view<const Scalar>(db, eb.data(), tblis::COLUMN_MAJOR);
    auto   tc    = tblis::varray_view<Scalar>(dc, ec.data(), tblis::COLUMN_MAJOR);
    double alpha = 1.0;
    double beta  = 0.0;
    tblis::mult(alpha, ta, la.c_str(), tb, lb.c_str(), beta, tc, lc.c_str());
}

template<typename Scalar>
contract::ResultType<Scalar> contract::tensor_product_tblis(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 4> &mpo,
                                                            const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    auto                     dsizes = psi.dimensions();
    tools::prof::t_tblis->tic();
    Eigen::Tensor<Scalar, 3> ham_sq_psi(psi.dimensions());

    if (psi.dimension(1) >= psi.dimension(2)){
        Eigen::Tensor<Scalar, 4> psi_envL(psi.dimension(0), psi.dimension(2), envL.dimension(1), envL.dimension(2));
        Eigen::Tensor<Scalar, 4> psi_envL_mpo(psi.dimension(2), envL.dimension(1), mpo.dimension(1), mpo.dimension(3));
        contract_tblis(psi, envL, psi_envL, "afb", "fcd", "abcd");
        contract_tblis(psi_envL, mpo, psi_envL_mpo, "abcd", "diaj", "bcij");
        contract_tblis(psi_envL_mpo, envR, ham_sq_psi, "bcij", "bki", "jck");
    }
    else{
        Eigen::Tensor<Scalar, 4> psi_envR(psi.dimension(0), psi.dimension(1), envR.dimension(1), envR.dimension(2));
        Eigen::Tensor<Scalar, 4> psi_envR_mpo(psi.dimension(1), envR.dimension(1), mpo.dimension(0), mpo.dimension(3));
        contract_tblis(psi, envR, psi_envR, "abf", "fcd", "abcd");
        contract_tblis(psi_envR, mpo, psi_envR_mpo, "qijk", "rkql", "ijrl");
        contract_tblis(psi_envR_mpo, envL, ham_sq_psi, "qkri", "qjr", "ijk");
    }

    tools::prof::t_tblis->toc();
    return std::make_pair(ham_sq_psi, get_ops_tblis_L(dsizes[0], dsizes[1], dsizes[2], mpo.dimension(0)));
}

using cx64 = std::complex<double>;
using fp32 = float;
using fp64 = double;

// template contract::ResultType<cplx> contract::tensor_product_tblis(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
//                                                                           const Eigen::Tensor<cplx, 3> &envL, const Eigen::Tensor<cplx, 3> &envR);
// template contract::ResultType<fp32> contract::tensor_product_tblis(const Eigen::Tensor<fp32, 3> &psi_in, const Eigen::Tensor<fp32, 4> &mpo,
//                                                                           const Eigen::Tensor<fp32, 3> &envL, const Eigen::Tensor<fp32, 3> &envR);
template contract::ResultType<fp64> contract::tensor_product_tblis(const Eigen::Tensor<fp64, 3> &psi, const Eigen::Tensor<fp64, 4> &mpo,
                                                                   const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);

#endif