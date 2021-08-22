#if defined(TB_TBLIS)

    #include <complex>
    #include <contract/contract.h>
    #include <tblis/tblis.h>
    #include <tblis/util/thread.h>
    #include <tools/class_tic_toc.h>
    #include <tools/log.h>
    #include <tools/prof.h>

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

    auto ta = tblis::varray_view<const Scalar>(da, ea.data(), tblis::COLUMN_MAJOR);
    auto tb = tblis::varray_view<const Scalar>(db, eb.data(), tblis::COLUMN_MAJOR);
    auto tc = tblis::varray_view<Scalar>(dc, ec.data(), tblis::COLUMN_MAJOR);
    //    for(long i = 0; i < ea.size(); i++) { tools::log->info("ea i {:>5}: ea {:>24.16f} ta {:>24.16f}", i, ea.coeff(i), ta.data()[i]); }
    //    for(long i = 0; i < eb.size(); i++) { tools::log->info("eb i {:>5}: eb {:>24.16f} tb {:>24.16f}", i, eb.coeff(i), tb.data()[i]); }
    //    for(long i = 0; i < ec.size(); i++) { tools::log->info("ec i {:>5}: ec {:>24.16f} tc {:>24.16f}", i, ec.coeff(i), tc.data()[i]); }
    double alpha = 1.0;
    double beta  = 0.0;
    tblis::mult(alpha, ta, la, tb, lb, beta, tc, lc);
    //
    //    for(long i = 0; i < ea.size(); i++) { tools::log->info("ea j {:>5}: ea {:>24.16f} ta {:>24.16f}", i, ea.coeff(i), ta.data()[i]); }
    //    for(long i = 0; i < eb.size(); i++) { tools::log->info("eb j {:>5}: eb {:>24.16f} tb {:>24.16f}", i, eb.coeff(i), tb.data()[i]); }
    //    for(long i = 0; i < ec.size(); i++) { tools::log->info("ec j {:>5}: ec {:>24.16f} tc {:>24.16f}", i, ec.coeff(i), tc.data()[i]); }
    //
    //    for(long i = 0; i < ea.size(); i++) { tools::log->info("ea k {:>5}: ea {:>24.16f} ta {:>24.16f}", i, ea.coeff(i), ta.data()[i]); }
    //    for(long i = 0; i < eb.size(); i++) { tools::log->info("eb k {:>5}: eb {:>24.16f} tb {:>24.16f}", i, eb.coeff(i), tb.data()[i]); }
    //    for(long i = 0; i < ec.size(); i++) { tools::log->info("ec k {:>5}: ec {:>24.16f} tc {:>24.16f}", i, ec.coeff(i), tc.data()[i]); }
}

template<typename Scalar>
contract::ResultType<Scalar> contract::tensor_product_tblis(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 4> &mpo,
                                                            const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    auto                     dsizes = psi.dimensions();
    Eigen::Tensor<Scalar, 4> psienL(psi.dimension(0), psi.dimension(2), envL.dimension(1), envL.dimension(2));
    Eigen::Tensor<Scalar, 4> psienLmpo(psi.dimension(2), envL.dimension(1), mpo.dimension(1), mpo.dimension(3));
    Eigen::Tensor<Scalar, 3> ham_sq_psi(psi.dimensions());

    tools::prof::t_tblis->tic();
//    tools::log->info("tblis threads: {}",tblis_get_num_threads());
    contract_tblis(psi, envL, psienL, "afb", "fcd", "abcd");
    contract_tblis(psienL, mpo, psienLmpo, "qijr", "rkql", "ijkl");
    contract_tblis(psienLmpo, envR, ham_sq_psi, "qjri", "qkr", "ijk");

    tools::prof::t_tblis->toc();
    return std::make_pair(ham_sq_psi, get_ops_tblis_L(dsizes[0], dsizes[1], dsizes[2], mpo.dimension(0)));
}

using cplx = std::complex<double>;
using fp32 = float;
using fp64 = double;

// template contract::ResultType<cplx> contract::tensor_product_tblis(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
//                                                                           const Eigen::Tensor<cplx, 3> &envL, const Eigen::Tensor<cplx, 3> &envR);
// template contract::ResultType<fp32> contract::tensor_product_tblis(const Eigen::Tensor<fp32, 3> &psi_in, const Eigen::Tensor<fp32, 4> &mpo,
//                                                                           const Eigen::Tensor<fp32, 3> &envL, const Eigen::Tensor<fp32, 3> &envR);
template contract::ResultType<fp64> contract::tensor_product_tblis(const Eigen::Tensor<fp64, 3> &psi, const Eigen::Tensor<fp64, 4> &mpo,
                                                                   const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);

#endif