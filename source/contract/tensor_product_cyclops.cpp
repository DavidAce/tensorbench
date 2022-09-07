
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
    auto t_bca = class_tic_toc(mpi::world.id == 0, 5, "bcast");
    auto t_get = class_tic_toc(mpi::world.id == 0, 5, "get_global_data");
    auto t_cpy = class_tic_toc(mpi::world.id == 0, 5, "copy");
    auto t_wrt = class_tic_toc(mpi::world.id == 0, 5, "write");
    auto t_del = class_tic_toc(mpi::world.id == 0, 5, "del");

    auto dims = tensor.dimensions();
    t_bca.tic();
    mpi::bcast(dims, 0);
    t_bca.toc();

    auto order = static_cast<int>(rank);
    auto len   = dims.data();

    auto    ctf = CTF::Tensor<Scalar>(order, len, w);
    int64_t nglobal;
    Scalar *data;
    t_get.tic();
    ctf.get_all_data(&nglobal, &data);
    t_get.toc();
    t_cpy.tic();
    for(int64_t i = 0; i < nglobal; ++i) data[i] = i < tensor.size() ? tensor(i) : 0.0;
    t_cpy.toc();
    auto global_idx = num::range<int64_t>(0, nglobal);
    t_wrt.tic();
    ctf.write(nglobal, global_idx.data(), data);
    t_wrt.toc();
    t_del.tic();
    delete[] data;
    t_del.toc();

    if(mpi::world.id == 0) {
        tools::log->info("  t_bca {:.6f} s", t_bca.get_measured_time());
        tools::log->info("  t_get {:.6f} s", t_get.get_measured_time());
        tools::log->info("  t_cpy {:.6f} s", t_cpy.get_measured_time());
        tools::log->info("  t_wrt {:.6f} s", t_wrt.get_measured_time());
        tools::log->info("  t_del {:.6f} s", t_del.get_measured_time());
        tools::log->info("  t_sum {:.6f} s\n", t_bca.get_measured_time() + t_get.get_measured_time() + t_cpy.get_measured_time() + t_wrt.get_measured_time() +
                                                   t_del.get_measured_time());
    }
    return ctf;
}

template<typename Scalar, auto rank>
CTF::Tensor<Scalar> get_ctf_tensor_v2(const Eigen::Tensor<Scalar, rank> &tensor, CTF::World &w) {
    auto t_bca = class_tic_toc(mpi::world.id == 0, 5, "bcast");
    auto t_get = class_tic_toc(mpi::world.id == 0, 5, "get_global_data");
    auto t_gth = class_tic_toc(mpi::world.id == 0, 5, "gatherv");
    auto t_cpy = class_tic_toc(mpi::world.id == 0, 5, "copy");
    auto t_sct = class_tic_toc(mpi::world.id == 0, 5, "scatterv");
    auto t_wrt = class_tic_toc(mpi::world.id == 0, 5, "write");
    auto t_del = class_tic_toc(mpi::world.id == 0, 5, "del");

    auto dims = tensor.dimensions();
    t_bca.tic();
    mpi::bcast(dims, 0);
    t_bca.toc();
    if(mpi::world.id == 0) tools::log->info("  t_bca {:.6f} s", t_bca.get_measured_time());
    auto order = static_cast<int>(rank);
    auto len   = dims.data();

    auto     ctf = CTF::Tensor<Scalar>(order, len, w);
    int64_t  nlocal;
    int64_t *idx;
    Scalar  *data;
    t_get.tic();
    ctf.get_local_data(&nlocal, &idx, &data);
    t_get.toc();
    if(mpi::world.id == 0) tools::log->info("  t_get {:.6f} s", t_get.get_measured_time());

    tenx::span<int64_t>  idx_local(idx, nlocal);
    tenx::span<Scalar>   data_local(data, nlocal);
    std::vector<int64_t> idx_global;
    std::vector<Scalar>  data_global;
    t_gth.tic();
    mpi::gatherv(idx_local, idx_global, 0);
    t_gth.toc();
    if(mpi::world.id == 0) tools::log->info("  t_gth {:.6f} s", t_gth.get_measured_time());
    // Now id 0 has the global list of scrambled indices in idx_global
    if(mpi::world.id == 0) {
        // We can make a new tensor with elements sorted according to idx_global,
        // then scatter contiguous chunks to each process.
        t_cpy.tic();
        if(idx_global.size() != static_cast<size_t>(tensor.size())) throw std::logic_error("idx_global.size() != tensor.size()");
        data_global.reserve(idx_global.size());
        for(const auto &i : idx_global) data_global.push_back(tensor(i));
        t_cpy.toc();
        tools::log->info("  t_cpy {:.6f} s", t_cpy.get_measured_time());
    }
    t_sct.tic();
    mpi::scatterv(data_global, data_local, 0);
    t_sct.toc();
    if(mpi::world.id == 0) tools::log->info("  t_sct {:.6f} s", t_sct.get_measured_time());
    t_wrt.tic();
    ctf.write(nlocal, idx, data);
    t_wrt.toc();
    if(mpi::world.id == 0) tools::log->info("  t_wrt {:.6f} s", t_wrt.get_measured_time());
    t_del.tic();
    free(idx);
    delete[] data;
    t_del.toc();
    if(mpi::world.id == 0) tools::log->info("  t_del {:.6f} s", t_del.get_measured_time());
    if(mpi::world.id == 0) {
        tools::log->info("  t_sum {:.6f} s\n", t_bca.get_measured_time() + t_get.get_measured_time() + t_gth.get_measured_time() + t_cpy.get_measured_time() +
                                                   t_sct.get_measured_time() + t_wrt.get_measured_time() + t_del.get_measured_time());
    }

    return ctf;
}

template<typename Scalar, auto rank>
CTF::Tensor<Scalar> get_ctf_tensor_v3(const Eigen::Tensor<Scalar, rank> &tensor, CTF::World &w) {
    auto t_bca = class_tic_toc(mpi::world.id == 0, 5, "bcast");
    auto t_get = class_tic_toc(mpi::world.id == 0, 5, "get_global_data");
    auto t_gth = class_tic_toc(mpi::world.id == 0, 5, "gatherv");
    auto t_cpy = class_tic_toc(mpi::world.id == 0, 5, "copy");
    auto t_sct = class_tic_toc(mpi::world.id == 0, 5, "scatterv");
    auto t_del = class_tic_toc(mpi::world.id == 0, 5, "del");

    auto dims = tensor.dimensions();
    t_bca.tic();
    mpi::bcast(dims, 0);
    t_bca.toc();
    if(mpi::world.id == 0) tools::log->info("  t_bca {:.6f} s", t_bca.get_measured_time());
    auto order = static_cast<int>(rank);
    auto len   = dims.data();

    auto     ctf = CTF::Tensor<Scalar>(order, len, w);
    int64_t  nlocal, nsize;
    int64_t *idx;
    Scalar *data;
    t_get.tic();
    ctf.get_local_data(&nlocal, &idx, &data);
    delete[] data; // We only need the indices from here
    data = ctf.get_raw_data(&nsize);
    t_get.toc();
    if(mpi::world.id == 0) tools::log->info("  t_get {:.6f} s", t_get.get_measured_time());

    tenx::span<int64_t>  idx_local(idx, nlocal);
    tenx::span<Scalar>   data_local(data, nlocal); // Raw padded data
    std::vector<int64_t> idx_global;
    std::vector<Scalar>  data_global;
    t_gth.tic();
    mpi::gatherv(idx_local, idx_global, 0);
    t_gth.toc();
    if(mpi::world.id == 0) tools::log->info("  t_gth {:.6f} s", t_gth.get_measured_time());
    // Now id 0 has the global list of scrambled indices in idx_global
    if(mpi::world.id == 0) {
        // We can make a new tensor with elements sorted according to idx_global,
        // then scatter contiguous chunks to each process.
        t_cpy.tic();
        if(idx_global.size() != static_cast<size_t>(tensor.size())) throw std::logic_error("idx_global.size() != tensor.size()");
        data_global.reserve(idx_global.size());
        for(const auto &i : idx_global) data_global.push_back(tensor(i));
        t_cpy.toc();
        tools::log->info("  t_cpy {:.6f} s", t_cpy.get_measured_time());
    }
    t_sct.tic();
    mpi::scatterv(data_global, data_local, 0);
    t_sct.toc();
    if(mpi::world.id == 0) tools::log->info("  t_sct {:.6f} s", t_sct.get_measured_time());
    t_del.tic();
    free(idx);
    t_del.toc();
    if(mpi::world.id == 0) tools::log->info("  t_del {:.6f} s", t_del.get_measured_time());
    if(mpi::world.id == 0) {
        tools::log->info("  t_sum {:.6f} s\n", t_bca.get_measured_time() + t_get.get_measured_time() + t_gth.get_measured_time() + t_cpy.get_measured_time() +
                                                   t_sct.get_measured_time() + t_del.get_measured_time());
    }

    return ctf;
}

template<typename Scalar, auto rank>
Eigen::Tensor<Scalar, rank> get_eigen_tensor(const CTF::Tensor<Scalar> &ctf) {
    int64_t  nlocal;
    int64_t *idx_loc;
    Scalar  *data_loc;

    ctf.get_local_data(&nlocal, &idx_loc, &data_loc);

    auto idx_local   = tenx::span(idx_loc, nlocal);
    auto data_local  = tenx::span(data_loc, nlocal);
    auto idx_global  = std::vector<int64_t>();
    auto data_global = std::vector<Scalar>();

    mpi::gatherv(idx_local, idx_global, 0);
    mpi::gatherv(data_local, data_global, 0);
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
    auto t_e2c  = class_tic_toc(true, 5, "eigen2ctf");
    auto t_c2e  = class_tic_toc(true, 5, "ctf2eigen");
    auto t_con  = class_tic_toc(true, 5, "contract");

    if(mpi::world.id == 0) t_e2c.tic();
    CTF::Tensor<Scalar> mpo_ctf  = get_ctf_tensor_v3(mpo, world);
    CTF::Tensor<Scalar> envL_ctf = get_ctf_tensor_v3(envL, world);
    CTF::Tensor<Scalar> envR_ctf = get_ctf_tensor_v3(envR, world);
    tools::prof::t_cyclops->tic();
    CTF::Tensor<Scalar> psi_ctf = get_ctf_tensor_v3(psi, world);
    if(mpi::world.id == 0) t_e2c.toc();
    if(mpi::world.id == 0) tools::log->info("eigen 2 ctf: {:.5f}s", t_e2c.get_measured_time());
    if(mpi::world.id == 0) tools::log->info("psi 2 ctf: {:.5f}s", tools::prof::t_cyclops->get_last_time_interval());

    auto res_ctf = CTF::Tensor<Scalar>(psi_ctf.order, psi_ctf.lens, world); // Same dims as psi_ctf
    t_con.tic();
    res_ctf["ijk"] = psi_ctf["abc"] * envL_ctf["bjd"] * mpo_ctf["deai"] * envR_ctf["cke"];
    t_con.toc();
    if(mpi::world.id == 0) tools::log->info("contraction: {:.5f}s", t_con.get_measured_time());
    t_c2e.tic();
    auto res = get_eigen_tensor<Scalar, 3>(res_ctf);
    t_c2e.toc();
    if(mpi::world.id == 0) tools::log->info("psi 2 eigen: {:.5f}s", t_c2e.get_measured_time());
    tools::prof::t_cyclops->toc();
    return {res, get_ops_cyclops_L(dsizes[0], dsizes[1], dsizes[2], mpo.dimension(0))};
}

using cx64 = std::complex<double>;
using fp32 = float;
using fp64 = double;
template contract::ResultType<fp64> contract::tensor_product_cyclops(const Eigen::Tensor<fp64, 3> &psi, const Eigen::Tensor<fp64, 4> &mpo,
                                                                     const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);

#endif