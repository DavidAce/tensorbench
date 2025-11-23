
#if defined(TB_CYCLOPS)
    #include <ctf.hpp>
// ctf must come first to avoid collision with blas headers
    #include "mpi/mpi-tools.h"
#endif

#include "benchmark/benchmark.h"
#include "math/num.h"
#include "math/tenx.h"
#include "tid/tid.h"
#include "tools/class_tic_toc.h"
#include "tools/log.h"
#include "tools/prof.h"
#include <complex>

#if defined(TB_CYCLOPS)
template<typename T, auto rank>
CTF::Tensor<T> get_ctf_tensor(const Eigen::Tensor<T, rank> &tensor, CTF::World &w, std::string_view tag) {
    auto t_get = tid::tic_scope(fmt::format("get_ctf_tensor()", tag));
    auto t_sct = class_tic_toc(mpi::world.id == 0, 5, "scatterv");
    auto t_del = class_tic_toc(mpi::world.id == 0, 5, "del");

    auto dims = tensor.dimensions();
    mpi::bcast(dims, 0);
    auto order = static_cast<int>(rank);
    auto len   = dims.data();

    auto     ctf = CTF::Tensor<T>(order, len, w);
    int64_t  nlocal, nsize;
    int64_t *idx;
    T       *data;
    {
        auto t_gld = tid::tic_token("get_local_data");
        ctf.get_local_data(&nlocal, &idx, &data);
        delete[] data; // We only need the indices from here
        data = ctf.get_raw_data(&nsize);
    }
    tenx::span<int64_t>  idx_local(idx, nlocal);
    tenx::span<T>        data_local(data, nlocal); // Raw padded data
    std::vector<int64_t> idx_global;
    std::vector<T>       data_global;
    mpi::gatherv(idx_local, idx_global, 0);
    free(idx);

    // Now id 0 has the global list of scrambled indices in idx_global
    if(mpi::world.id == 0) {
        // We can make a new tensor with elements sorted according to idx_global,
        // then scatter contiguous chunks to each process.
        auto t_cpy = tid::tic_token("copy");
        if(idx_global.size() != static_cast<size_t>(tensor.size())) throw std::logic_error("idx_global.size() != tensor.size()");
        data_global.reserve(idx_global.size());
        for(const auto &i : idx_global) data_global.push_back(tensor(i));
    }
    mpi::scatterv(data_global, data_local, 0);
    return ctf;
}

template<typename T, auto rank>
Eigen::Tensor<T, rank> get_eigen_tensor(const CTF::Tensor<T> &ctf) {
    int64_t  nlocal;
    int64_t *idx_loc;
    T       *data_loc;

    ctf.get_local_data(&nlocal, &idx_loc, &data_loc);

    auto idx_local   = tenx::span(idx_loc, nlocal);
    auto data_local  = tenx::span(data_loc, nlocal);
    auto idx_global  = std::vector<int64_t>();
    auto data_global = std::vector<T>();

    mpi::gatherv(idx_local, idx_global, 0);
    mpi::gatherv(data_local, data_global, 0);
    free(idx_loc);
    delete[] data_loc;

    if(mpi::world.id == 0) {
        auto dims = std::array<long, rank>();
        for(size_t i = 0; i < static_cast<size_t>(ctf.order); ++i) dims[i] = ctf.lens[i];
        Eigen::Tensor<T, rank> tensor(dims);
        if(idx_global.size() != static_cast<size_t>(tensor.size())) throw std::runtime_error("get_eigen_tensor: size mismatch");
        for(size_t i = 0; i < static_cast<size_t>(tensor.size()); ++i) tensor(idx_global[i]) = data_global[i];
        return tensor;
    } else
        return {};
}

#endif

template<typename T>
benchmark::ResultType<T> benchmark::tensor_product_cyclops([[maybe_unused]] const tb_setup<T> &tbs) {
#if defined(TB_CYCLOPS)
    auto t_cyclops = tid::tic_scope("cyclops");
    auto world  = CTF::World(MPI_COMM_WORLD);
    CTF::Tensor<T> mpo_ctf  = get_ctf_tensor(tbs.mpo, world, "mpo");
    CTF::Tensor<T> envL_ctf = get_ctf_tensor(tbs.envL, world, "envL");
    CTF::Tensor<T> envR_ctf = get_ctf_tensor(tbs.envR, world, "envR");
    auto t_contract = tid::tic_scope("contract");
    auto psi_ctf    = get_ctf_tensor(tbs.psi, world, "psi");
    auto res_ctf    = CTF::Tensor<T>(psi_ctf.order, psi_ctf.lens, world); // Same dims as psi_ctf
    res_ctf["ijk"]  = psi_ctf["abc"] * envL_ctf["bjd"] * mpo_ctf["deai"] * envR_ctf["cke"];
    return get_eigen_tensor<T, 3>(res_ctf);
#else
    return {};
#endif
}

using fp32 = benchmark::fp32;
using fp64 = benchmark::fp64;
using cplx = benchmark::cplx;

template benchmark::ResultType<fp32> benchmark::tensor_product_cyclops(const tb_setup<fp32> &tbs);
template benchmark::ResultType<fp64> benchmark::tensor_product_cyclops(const tb_setup<fp64> &tbs);
template benchmark::ResultType<cplx> benchmark::tensor_product_cyclops(const tb_setup<cplx> &tbs);
