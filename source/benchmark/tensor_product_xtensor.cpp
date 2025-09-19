
#if defined(TB_XTENSOR)
    #ifndef HAVE_CBLAS
        #define HAVE_CBLAS 1
    #endif
    #include <complex>
    #include <xtensor-blas/xlinalg.hpp>
    #include <xtensor.hpp>
#endif
// include blas first
#include "benchmark.h"
#include "general/enums.h"
#include "tid/tid.h"

template<typename T>
benchmark::ResultType<T> benchmark::tensor_product_xtensor([[maybe_unused]] const tb_setup<T> &tbs) {
#if defined(TB_XTENSOR)
    auto t_complete      = tid::tic_scope(enum2sv(tb_mode::xtensor));
    auto dsizes          = tbs.psi.dimensions();
    auto psi_shape       = std::vector<size_t>(tbs.psi.dimensions().begin(), tbs.psi.dimensions().end());
    auto mpo_shape       = std::vector<size_t>(tbs.mpo.dimensions().begin(), tbs.mpo.dimensions().end());
    auto envL_shape      = std::vector<size_t>(tbs.envL.dimensions().begin(), tbs.envL.dimensions().end());
    auto envR_shape      = std::vector<size_t>(tbs.envR.dimensions().begin(), tbs.envR.dimensions().end());
    auto mpo_xt          = xt::adapt<xt::layout_type::column_major>(tbs.mpo.data(), static_cast<size_t>(tbs.mpo.size()), xt::no_ownership(), mpo_shape);
    auto enL_xt          = xt::adapt<xt::layout_type::column_major>(tbs.envL.data(), static_cast<size_t>(tbs.envL.size()), xt::no_ownership(), envL_shape);
    auto enR_xt          = xt::adapt<xt::layout_type::column_major>(tbs.envR.data(), static_cast<size_t>(tbs.envR.size()), xt::no_ownership(), envR_shape);
    auto t_contract      = tid::tic_token("contract");
    auto psi_xt          = xt::adapt<xt::layout_type::column_major>(tbs.psi.data(), static_cast<size_t>(tbs.psi.size()), xt::no_ownership(), psi_shape);
    auto psienL_xt       = xt::linalg::tensordot(psi_xt, enL_xt, {1}, {0});
    auto psienLmpo_xt    = xt::linalg::tensordot(psienL_xt, mpo_xt, {0, 3}, {2, 0});
    auto psienLmpoenR_xt = xt::linalg::tensordot(psienLmpo_xt, enR_xt, {0, 2}, {0, 2});
    xt::xtensor<T, 3, xt::layout_type::column_major> hamsqpsi = xt::transpose(psienLmpoenR_xt, {1, 0, 2});
    return Eigen::TensorMap<Eigen::Tensor<T, 3>>(hamsqpsi.data(), dsizes);
#else
    return {};
#endif
}

using fp32 = benchmark::fp32;
using fp64 = benchmark::fp64;
using cplx = benchmark::cplx;
template benchmark::ResultType<fp32> benchmark::tensor_product_xtensor(const tb_setup<fp32> &tbs);
template benchmark::ResultType<fp64> benchmark::tensor_product_xtensor(const tb_setup<fp64> &tbs);
template benchmark::ResultType<cplx> benchmark::tensor_product_xtensor(const tb_setup<cplx> &tbs);
