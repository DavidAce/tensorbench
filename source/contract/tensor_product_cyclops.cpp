
#if defined(TB_CYCLOPS)
    #include <ctf.hpp>
// ctf must come first to avoid collision with blas headers

    #include "contract/contract.h"
    #include "math/num.h"
    #include "mpi/mpi-tools.h"
    #include "tools/class_tic_toc.h"
    #include "tools/log.h"
    #include "tools/prof.h"
    #include <complex>

long get_ops_cyclops_L(long d, long chiL, long chiR, long m);
long get_ops_cyclops_R(long d, long chiL, long chiR, long m) {
    // Same as L, just swap chiL and chiR
    if(chiR > chiL) return get_ops_cyclops_L(d, chiR, chiL, m);
    else
        return get_ops_cyclops_L(d, chiL, chiR, m);
}

long get_ops_cyclops_L(long d, long chiL, long chiR, long m) {
    if(chiR > chiL) return get_ops_cyclops_R(d, chiL, chiR, m);
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

template<typename Scalar, auto rank>
CTF::Tensor<Scalar> get_ctf_tensor(const Eigen::Tensor<Scalar, rank> &tensor, CTF::World &w) {
    auto dims = tensor.dimensions();
    mpi::bcast(dims, 0);
    auto order = static_cast<int>(rank);
    auto len   = dims.data();

    auto    ctf = CTF::Tensor<Scalar>(order, len, w);
    int64_t nglobal;
    Scalar *data;
    ctf.get_all_data(&nglobal, &data);
    for(int64_t i = 0; i < nglobal; ++i) data[i] = i < tensor.size() ? tensor(i) : 0.0;
    auto global_idx = num::range<int64_t>(0, nglobal);
    ctf.write(nglobal, global_idx.data(), data);
    delete[] data;
    return ctf;
}

template<typename Scalar, auto rank>
Eigen::Tensor<Scalar, rank> get_eigen_tensor(const CTF::Tensor<Scalar> &ctf) {
    int64_t  nlocal;
    int64_t *idx_loc;
    Scalar  *data_loc;
    ctf.get_local_data(&nlocal, &idx_loc, &data_loc);

    auto idx_local  = tenx::span(idx_loc, nlocal);
    auto data_local  = tenx::span(data_loc, nlocal);
    auto idx_global = std::vector<int64_t>();
    auto data_global = std::vector<Scalar>();

    mpi::gatherv(idx_global, idx_local, 0);
    mpi::gatherv(data_global, data_local, 0);
    mpi::barrier();
    free(idx_loc);
    delete[] data_loc;

    if(mpi::world.id == 0) {
        auto dims = std::array<long, rank>();
        for(int i = 0; i < ctf.order; ++i) dims[i] = ctf.lens[i];
        Eigen::Tensor<Scalar, rank> tensor(dims);
        if(idx_global.size() != static_cast<size_t>(tensor.size())) throw std::runtime_error("get_eigen_tensor: size mismatch");
        for(int64_t i = 0; i < tensor.size(); ++i) tensor(idx_global[i]) = data_global[i];
        return tensor;
    } else
        return {};
}

template<typename Scalar>
contract::ResultType<Scalar> contract::tensor_product_cyclops(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 4> &mpo,
                                                              const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
#pragma message "omp_set_num_threads(1) for cyclops"
    omp_set_num_threads(1); // Make sure we don't use local threads!
    auto dsizes = psi.dimensions();
    auto world  = CTF::World(MPI_COMM_WORLD);

    CTF::Tensor<Scalar> mpo_ctf  = get_ctf_tensor(mpo, world);
    CTF::Tensor<Scalar> envL_ctf = get_ctf_tensor(envL, world);
    CTF::Tensor<Scalar> envR_ctf = get_ctf_tensor(envR, world);
    tools::prof::t_cyclops->tic();
    CTF::Tensor<Scalar> psi_ctf  = get_ctf_tensor(psi, world);
    auto res_ctf = CTF::Tensor<Scalar>(psi_ctf.order, psi_ctf.lens, world); // Same dims as psi_ctf
    mpi::barrier();

    res_ctf["ijk"] = psi_ctf["abc"] * envL_ctf["bjd"] * mpo_ctf["deai"] * envR_ctf["cke"];
    auto res = get_eigen_tensor<Scalar, 3>(res_ctf);
    tools::prof::t_cyclops->toc();

    return {res, get_ops_cyclops_L(dsizes[0], dsizes[1], dsizes[2], mpo.dimension(0))};
}

using cplx = std::complex<double>;
using fp32 = float;
using fp64 = double;
template contract::ResultType<fp64> contract::tensor_product_cyclops(const Eigen::Tensor<fp64, 3> &psi, const Eigen::Tensor<fp64, 4> &mpo,
                                                                     const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);

#endif