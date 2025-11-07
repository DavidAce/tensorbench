#include "benchmark.h"
#if defined(TB_EIGEN1)
    #include "general/enums.h"
    #include "math/tenx.h"
    #include "tid/tid.h"
    #include <complex>
#endif

template<typename T>
benchmark::ResultType<T> benchmark::tensor_product_eigen1([[maybe_unused]] const tb_setup<T> &tbs) {
#if defined(TB_EIGEN1)
    auto dsizes     = tbs.psi.dimensions();
    auto ham_sq_psi = Eigen::Tensor<T, 3>(dsizes);
    tenx::threads::setNumThreads(tbs.nomp);
    auto &threads = tenx::threads::get();
    std::printf("threads: %d getNumThreads(): %d\n", threads->dev->numThreads(), tenx::threads::getNumThreads());
    auto t_complete = tid::tic_scope(enum2sv(tb_mode::eigen1));
    auto t_contract = tid::tic_token("contract");
    if(tbs.psi.dimension(1) >= tbs.psi.dimension(2)) {
        Eigen::Tensor<T, 4> psi_envL(tbs.psi.dimension(0), tbs.psi.dimension(2), tbs.envL.dimension(1), tbs.envL.dimension(2));
        Eigen::Tensor<T, 4> psi_envL_mpo(tbs.psi.dimension(2), tbs.envL.dimension(1), tbs.mpo.dimension(1), tbs.mpo.dimension(3));

        {
            auto t_contract1               = tid::tic_token("contract1");
            psi_envL.device(*threads->dev) = tbs.psi.contract(tbs.envL, tenx::idx({1}, {0}));
        }
        {
            auto t_contract2                   = tid::tic_token("contract2");
            psi_envL_mpo.device(*threads->dev) = psi_envL.contract(tbs.mpo, tenx::idx({3, 0}, {0, 2}));
        }
        {
            auto t_contract3                 = tid::tic_token("contract3");
            ham_sq_psi.device(*threads->dev) = psi_envL_mpo.contract(tbs.envR, tenx::idx({0, 2}, {0, 2})).shuffle(tenx::array3{1, 0, 2});
        }
        // ham_sq_psi.device(*threads->dev) = tbs.psi.contract(tbs.envL, tenx::idx({1}, {0}))
        // .contract(tbs.mpo, tenx::idx({3, 0}, {0, 2}))
        // .contract(tbs.envR, tenx::idx({0, 2}, {0, 2}))
        // .shuffle(tenx::array3{1, 0, 2});
    } else {
        ham_sq_psi.device(*threads->dev) = tbs.psi.contract(tbs.envR, tenx::idx({2}, {0}))
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

template benchmark::ResultType<fp32> benchmark::tensor_product_eigen1(const tb_setup<fp32> &tbs);
template benchmark::ResultType<fp64> benchmark::tensor_product_eigen1(const tb_setup<fp64> &tbs);
template benchmark::ResultType<cplx> benchmark::tensor_product_eigen1(const tb_setup<cplx> &tbs);
