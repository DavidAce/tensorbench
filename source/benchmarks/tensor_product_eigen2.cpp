#if defined(TB_EIGEN2)

    #include "benchmarks/benchmarks.h"
    #include "math/tenx.h"
    #include "tools/class_tic_toc.h"
    #include "tools/log.h"
    #include "tools/prof.h"
    #include <complex>

long get_ops_eigen2_L(long d, long chiL, long chiR, long m);
long get_ops_eigen2_R(long d, long chiL, long chiR, long m) {
    // Same as L, just swap chiL and chiR
    if(chiR > chiL) return get_ops_eigen2_L(d, chiR, chiL, m);
    else
        return get_ops_eigen2_L(d, chiL, chiR, m);
}

long get_ops_eigen2_L(long d, long chiL, long chiR, long m) {
    if(chiR > chiL) return get_ops_eigen2_R(d, chiL, chiR, m);
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
contract::ResultType<Scalar> contract::tensor_product_eigen2(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 4> &mpo,
                                                             const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    Eigen::DSizes<long, 3>   dsizes = psi.dimensions();
    Eigen::Tensor<Scalar, 3> ham_sq_psi(dsizes);
    //    Eigen::Tensor<Scalar, 4> psi_envL(psi.dimension(0), psi.dimension(2), envL.dimension(1), envL.dimension(2));
    //    Eigen::Tensor<Scalar, 4> psi_envL_mpo(psi.dimension(2), envL.dimension(1), mpo.dimension(1), mpo.dimension(3));
    //    ham_sq_psi.setZero();
    //    psi_envL.setZero();
    //    psi_envL_mpo.setZero();
    //    auto dim_envL         = envL.dimensions();
    //    auto dim_envR         = envR.dimensions();
    //    auto dim_psi          = psi.dimensions();
    //    auto dim_mpo          = mpo.dimensions();
    //    auto dim_psi_envL     = psi_envL.dimensions();
    //    auto dim_psi_envL_mpo = psi_envL_mpo.dimensions();
    //
    //    for(long x = 0; x < dim_envL[1]; x++) {
    //        for(long y = 0; y < dim_envL[2]; y++) {
    //            #pragma omp parallel for collapse(3)
    //            for(long a = 0; a < dim_psi[2]; a++) {
    //                for(long b = 0; b < dim_psi[0]; b++) {
    //                    for(long i = 0; i < dim_psi[1]; i++) {
    //                        //                            #pragma omp atomic
    //                        psi_envL(b, a, x, y) += psi(b, i, a) * envL(i, x, y);
    //                    }
    //                }
    //            }
    //        }
    //    }
    //
    //    auto step1 = tools::prof::t_eigen2->get_last_time_interval();
    //    tools::log->info(" step1: {:.8f}", step1);
    //
    //    for(long y = 0; y < dim_mpo[3]; y++) {
    //        for(long x = 0; x < dim_mpo[1]; x++) {
    //            #pragma omp parallel for collapse(3)
    //            for(long b = 0; b < dim_psi_envL[1]; b++) {
    //                for(long a = 0; a < dim_psi_envL[2]; a++) {
    //                    for(long j = 0; j < dim_psi_envL[0]; j++) {
    //                        for(long i = 0; i < dim_psi_envL[3]; i++) { psi_envL_mpo(b, a, x, y) += psi_envL(j, b, a, i) * mpo(i, x, j, y); }
    //                    }
    //                }
    //            }
    //        }
    //    }
    //

    //
    //    auto step2 = tools::prof::t_eigen2->get_last_time_interval();
    //    tools::log->info(" step2: {:.8f} ({:.8f})", step2 - step1, step2);
    //
    //    for(long y = 0; y < dim_envR[1]; y++) {
    //        #pragma omp parallel for collapse(4)
    //        for(long x = 0; x < dim_psi_envL_mpo[3]; x++) {
    //            for(long b = 0; b < dim_psi_envL_mpo[1]; b++) {
    //                for(long a = 0; a < dim_psi_envL_mpo[2]; a++) {
    //                    for(long i = 0; i < dim_psi_envL_mpo[0]; i++) { ham_sq_psi(x, b, y) += psi_envL_mpo(i, b, a, x) * envR(i, y, a); }
    //                }
    //            }
    //        }
    //    }
    //
    //
    //
    //    auto step3 = tools::prof::t_eigen2->get_last_time_interval();
    //    tools::log->info(" step3: {:.8f} ({:.8f})", step3 - step2, step3);
    tools::prof::t_eigen2->tic();

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
    tools::prof::t_eigen2->toc();
    return std::make_pair(ham_sq_psi, get_ops_eigen2_R(dsizes[0], dsizes[1], dsizes[2], mpo.dimension(0)));
}

using cplx = std::complex<double>;
using fp32 = float;
using fp64 = double;

// template contract::ResultType<cplx> contract::tensor_product_eigen2(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
//                                                                           const Eigen::Tensor<cplx,3> &envL, const Eigen::Tensor<cplx,3> &envR);
// template contract::ResultType<fp32> contract::tensor_product_eigen2(const Eigen::Tensor<fp32, 3> &psi_in, const Eigen::Tensor<fp32, 4> &mpo,
//                                                                           const Eigen::Tensor<fp32,3> &envL, const Eigen::Tensor<fp32,3> &envR);
template contract::ResultType<fp64> contract::tensor_product_eigen2(const Eigen::Tensor<fp64, 3> &psi_in, const Eigen::Tensor<fp64, 4> &mpo,
                                                                    const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);
#endif