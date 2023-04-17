#include "benchmark.h"
#if defined(TB_EIGEN2)
    #include "general/enums.h"
    #include "math/tenx.h"
    #include "tid/tid.h"
    #include <complex>
#endif

template<typename T>
benchmark::ResultType<T> benchmark::tensor_product_eigen2([[maybe_unused]] const tb_setup<T> &tbs) {
#if defined(TB_EIGEN2)
    tenx::threads::setNumThreads(static_cast<unsigned int>(tbs.num_threads));
    auto t_complete = tid::tic_scope(enum2sv(tb_mode::eigen2));
    auto dsizes     = tbs.psi.dimensions();
    auto ham_sq_psi = Eigen::Tensor<T, 3>(dsizes);
    auto t_contract = tid::tic_token("contract");

    if(tbs.psi.dimension(1) >= tbs.psi.dimension(2)) {
        ham_sq_psi.device(tenx::threads::getDevice()) = tbs.psi.contract(tbs.envL, tenx::idx({1}, {0}))
                                                            .contract(tbs.mpo, tenx::idx({3, 0}, {0, 2}))
                                                            .contract(tbs.envR, tenx::idx({0, 2}, {0, 2}))
                                                            .shuffle(tenx::array3{1, 0, 2});
    } else {
        ham_sq_psi.device(tenx::threads::getDevice()) = tbs.psi.contract(tbs.envR, tenx::idx({2}, {0}))
                                                            .contract(tbs.mpo, tenx::idx({3, 0}, {1, 2}))
                                                            .contract(tbs.envL, tenx::idx({0, 2}, {0, 2}))
                                                            .shuffle(tenx::array3{1, 2, 0});
    }
    return ham_sq_psi;
#else
    return {};
#endif
}

using fp32 = benchmark::fp32;
using fp64 = benchmark::fp64;
using cplx = benchmark::cplx;
template benchmark::ResultType<fp32> benchmark::tensor_product_eigen2(const tb_setup<fp32> &tbs);
template benchmark::ResultType<fp64> benchmark::tensor_product_eigen2(const tb_setup<fp64> &tbs);
template benchmark::ResultType<cplx> benchmark::tensor_product_eigen2(const tb_setup<cplx> &tbs);