#include "benchmark.h"
#if defined(TB_EIGEN3)
    #include "general/enums.h"
    #include "math/tenx.h"
    #include "tid/tid.h"
    #include <complex>
#endif
template<typename T>
benchmark::ResultType<T> benchmark::tensor_product_eigen3([[maybe_unused]] const tb_setup<T> &tbs) {
#if defined(TB_EIGEN3)
    tenx::threads::setNumThreads(static_cast<unsigned int>(tbs.num_threads));
    auto t_complete = tid::tic_scope(enum2sv(tb_mode::eigen3));
    auto dsizes     = tbs.psi.dimensions();
    auto ham_sq_psi = Eigen::Tensor<T, 3>(dsizes);
    auto enL_sh     = Eigen::Tensor<T, 3>(tbs.envL.shuffle(tenx::array3{0, 2, 1}));
    auto enR_sh     = Eigen::Tensor<T, 3>(tbs.envR.shuffle(tenx::array3{0, 2, 1}));
    auto mpo_sh     = Eigen::Tensor<T, 4>(tbs.mpo.shuffle(tenx::array4{0, 2, 1, 3}));
    auto t_contract = tid::tic_token("contract");
    auto psi_sh     = Eigen::Tensor<T, 3>(tbs.psi.shuffle(tenx::array3{1, 0, 2}));

    ham_sq_psi.device(*tenx::threads::dev) = psi_sh.contract(enL_sh, tenx::idx({0}, {0}))
                                                 .contract(mpo_sh, tenx::idx({2, 0}, {0, 1}))
                                                 .contract(enR_sh, tenx::idx({0, 2}, {0, 1}))
                                                 .shuffle(tenx::array3{1, 0, 2});
    return ham_sq_psi;
#else
    return {};
#endif
}

using fp32 = benchmark::fp32;
using fp64 = benchmark::fp64;
using cplx = benchmark::cplx;

template benchmark::ResultType<fp32> benchmark::tensor_product_eigen3(const tb_setup<fp32> &tbs);
template benchmark::ResultType<fp64> benchmark::tensor_product_eigen3(const tb_setup<fp64> &tbs);
template benchmark::ResultType<cplx> benchmark::tensor_product_eigen3(const tb_setup<cplx> &tbs);
