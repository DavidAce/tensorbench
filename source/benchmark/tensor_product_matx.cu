#include "benchmark.h"
#include "general/enums.h"
#include <matx.h>
#include "tid/tid.h"

template<typename T>
benchmark::ResultType<T> benchmark::tensor_product_matx([[maybe_unused]] tb_setup<T> &tbs) {
    // std::array<int, 3> psi_dims  = ;
    // std::array<int, 4> mpo_dims  = ;
    // std::array<int, 3> envL_dims = ;
    // std::array<int, 3> envR_dims = ;

    auto                t_matx = tid::tic_scope("matx");
    cudaStream_t        stream = nullptr;
    matx::cudaExecutor  exec{stream};
    Eigen::Tensor<T, 3> res_host(tbs.psi.dimensions());

    long I              = tbs.psi.dimension(0);
    long J              = tbs.psi.dimension(1);
    long K              = tbs.psi.dimension(2);
    long L              = tbs.envL.dimension(1);
    long M              = tbs.envL.dimension(2);
    long N              = tbs.envR.dimension(2);
    long P              = tbs.envR.dimension(1);
    long O              = tbs.mpo.dimension(3);
    auto matx_psi_host  = matx::make_tensor<T>(tbs.psi.data(), {I, J, K});
    auto matx_mpo_host  = matx::make_tensor<T>(tbs.mpo.data(), {M, N, I, O});
    auto matx_envL_host = matx::make_tensor<T>(tbs.envL.data(), {J, L, M});
    auto matx_envR_host = matx::make_tensor<T>(tbs.envR.data(), {K, P, N});
    auto matx_res_host  = matx::make_tensor<T>(res_host.data(), {I, J, K});

    auto matx_psi_dev  = matx::make_tensor<T>({I, J, K}, matx::MATX_DEVICE_MEMORY);
    auto matx_mpo_dev  = matx::make_tensor<T>({M, N, I, O}, matx::MATX_DEVICE_MEMORY);
    auto matx_envL_dev = matx::make_tensor<T>({J, L, M}, matx::MATX_DEVICE_MEMORY);
    auto matx_envR_dev = matx::make_tensor<T>({K, P, N}, matx::MATX_DEVICE_MEMORY);
    auto matx_res_dev  = matx::make_tensor<T>({I, J, K}, matx::MATX_DEVICE_MEMORY);
    //
    (matx_psi_dev  = matx_psi_host).run(exec);
    (matx_mpo_dev  = matx_mpo_host).run(exec);
    (matx_envL_dev = matx_envL_host).run(exec);
    (matx_envR_dev = matx_envR_host).run(exec);
    auto t_contract = tid::tic_scope("contract");
    (matx_res_host  = matx::cutensor::einsum("ijk,jlm,mnio,kpn->olp", matx_psi_dev, matx_envL_dev, matx_mpo_dev, matx_envR_dev)).run(exec);
    t_contract.toc();
    (matx_res_host = matx_res_dev).run(exec);
    return res_host;
}

using fp32 = benchmark::fp32;
using fp64 = benchmark::fp64;
using cplx = benchmark::cplx;

template benchmark::ResultType<fp32> benchmark::tensor_product_matx(tb_setup<fp32> &tbs);
template benchmark::ResultType<fp64> benchmark::tensor_product_matx(tb_setup<fp64> &tbs);
template benchmark::ResultType<cplx> benchmark::tensor_product_matx(tb_setup<cplx> &tbs);