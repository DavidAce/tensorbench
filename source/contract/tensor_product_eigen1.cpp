#if defined(TB_EIGEN1)

    #include <complex>
    #include <contract/contract.h>
    #include <tools/class_tic_toc.h>
    #include <tools/prof.h>

long get_ops_eigen1_L(long d, long chiL, long chiR, long m);
long get_ops_eigen1_R(long d, long chiL, long chiR, long m) {
    // Same as L, just swap chiL and chiR
    if(chiR > chiL) return get_ops_eigen1_L(d, chiR, chiL, m);
    else
        return get_ops_eigen1_L(d, chiL, chiR, m);
}

long get_ops_eigen1_L(long d, long chiL, long chiR, long m) {
    if(chiR > chiL) return get_ops_eigen1_R(d, chiL, chiR, m);
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

template<typename Scalar>
contract::ResultType<Scalar> contract::tensor_product_eigen1(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 4> &mpo,
                                                             const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    Eigen::DSizes<long, 3>   dsizes = psi.dimensions();
    Eigen::Tensor<Scalar, 3> ham_sq_psi(dsizes);
    tools::prof::t_eigen1->tic();
    if(psi.dimension(1) >= psi.dimension(2)) {
        ham_sq_psi.device(tenx::omp::getDevice()) = psi.contract(envL, tenx::idx({1}, {0}))
                                                        .contract(mpo, tenx::idx({3, 0}, {0, 2}))
                                                        .contract(envR, tenx::idx({0, 2}, {0, 2}))
                                                        .shuffle(tenx::array3{1, 0, 2});
    } else {
        ham_sq_psi.device(tenx::omp::getDevice()) = psi.contract(envR, tenx::idx({2}, {0}))
                                                        .contract(mpo, tenx::idx({3, 0}, {1, 2}))
                                                        .contract(envL, tenx::idx({0, 2}, {0, 2}))
                                                        .shuffle(tenx::array3{1, 2, 0});
    }

    tools::prof::t_eigen1->toc();
    return std::make_pair(ham_sq_psi, get_ops_eigen1_L(dsizes[0], dsizes[1], dsizes[2], mpo.dimension(0)));
}

using cx64 = std::complex<double>;
using fp32 = float;
using fp64 = double;

// template contract::ResultType<cplx> contract::tensor_product_eigen1(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
//                                                                           const Eigen::Tensor<cplx, 3> &envL, const Eigen::Tensor<cplx, 3> &envR);
// template contract::ResultType<fp32> contract::tensor_product_eigen1(const Eigen::Tensor<fp32, 3> &psi_in, const Eigen::Tensor<fp32, 4> &mpo,
//                                                                           const Eigen::Tensor<fp32, 3> &envL, const Eigen::Tensor<fp32, 3> &envR);
template contract::ResultType<fp64> contract::tensor_product_eigen1(const Eigen::Tensor<fp64, 3> &psi_in, const Eigen::Tensor<fp64, 4> &mpo,
                                                                    const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);

#endif