
#if defined(TB_XTENSOR)
    #undef EIGEN_USE_MKL_ALL
//    #undef EIGEN_USE_LAPACKE_STRICT

    #ifndef HAVE_CBLAS
        #define HAVE_CBLAS 1
    #endif
    #include <complex>
    #include <xtensor-blas/xlinalg.hpp>
    #include <xtensor/xadapt.hpp>
    #include <xtensor/xtensor.hpp>

// include blas first
    #include "benchmarks/benchmarks.h"
    #include "math/tenx.h"
    #include "tools/class_tic_toc.h"
    #include "tools/prof.h"

long get_ops_xtensor(long d, long chiL, long chiR, long m) {
    if(chiR > chiL) return get_ops_xtensor(d, chiR, chiL, m);
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
contract::ResultType<Scalar> contract::tensor_product_xtensor(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 4> &mpo,
                                                              const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    Eigen::DSizes<long, 3>   dsizes = psi.dimensions();
    Eigen::Tensor<Scalar, 3> ham_sq_psi(dsizes);
    std::vector<size_t>      psi_shape(psi.dimensions().begin(), psi.dimensions().end());
    std::vector<size_t>      mpo_shape(mpo.dimensions().begin(), mpo.dimensions().end());
    std::vector<size_t>      envL_shape(envL.dimensions().begin(), envL.dimensions().end());
    std::vector<size_t>      envR_shape(envR.dimensions().begin(), envR.dimensions().end());
    auto                     psi_xt = xt::adapt<xt::layout_type::column_major>(psi.data(), psi.size(), xt::no_ownership(), psi_shape);
    auto                     mpo_xt = xt::adapt<xt::layout_type::column_major>(mpo.data(), mpo.size(), xt::no_ownership(), mpo_shape);
    auto                     enL_xt = xt::adapt<xt::layout_type::column_major>(envL.data(), envL.size(), xt::no_ownership(), envL_shape);
    auto                     enR_xt = xt::adapt<xt::layout_type::column_major>(envR.data(), envR.size(), xt::no_ownership(), envR_shape);
    tools::prof::t_xtensor->tic();
    auto                                                  psienL_xt       = xt::linalg::tensordot(psi_xt, enL_xt, {1}, {0});
    auto                                                  psienLmpo_xt    = xt::linalg::tensordot(psienL_xt, mpo_xt, {0, 3}, {2, 0});
    auto                                                  psienLmpoenR_xt = xt::linalg::tensordot(psienLmpo_xt, enR_xt, {0, 2}, {0, 2});
    xt::xtensor<Scalar, 3, xt::layout_type::column_major> hamsqpsi        = xt::transpose(psienLmpoenR_xt, {1, 0, 2});
    tools::prof::t_xtensor->toc();
    ham_sq_psi = Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>(hamsqpsi.data(), dsizes);

    //    ham_sq_psi.device(*tenx::omp::dev) =
    //        psi_in
    //            .contract(envL, tenx::idx({1}, {0}))
    //            .contract(mpo, tenx::idx({0, 3}, {2, 0}))
    //            .contract(envR, tenx::idx({0, 2}, {0, 2}))
    //            .shuffle(tenx::array3{1, 0, 2});
    return std::make_pair(ham_sq_psi, get_ops_xtensor(dsizes[0], dsizes[1], dsizes[2], mpo.dimension(0)));
}

using cx64 = std::complex<double>;
using fp32 = float;
using fp64 = double;

// template contract::ResultType<cplx> contract::tensor_product_xtensor(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
//                                                                        const Eigen::Tensor<cplx, 3> &envL, const Eigen::Tensor<cplx, 3> &envR);
// template contract::ResultType<fp32> contract::tensor_product_xtensor(const Eigen::Tensor<fp32, 3> &psi_in, const Eigen::Tensor<fp32, 4> &mpo,
//                                                                        const Eigen::Tensor<fp32, 3> &envL, const Eigen::Tensor<fp32, 3> &envR);
template contract::ResultType<fp64> contract::tensor_product_xtensor(const Eigen::Tensor<fp64, 3> &theta_in, const Eigen::Tensor<fp64, 4> &mpo,
                                                                     const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);

#endif