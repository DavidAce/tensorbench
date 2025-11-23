//
// Created by david on 2023-04-15.
//
#include "benchmark.h"
#include "config/config.h"
#include "general/enums.h"
#include "math/stat.h"
#include "math/tenx.h"
#include "mpi/mpi-tools.h"
#include "storage/results.h"
#include "tid/tid.h"
#include "tools/log.h"
#include "tools/prof.h"
#include <env/environment.h>
#include <h5pp/h5pp.h>

#if defined(_OPENMP)
    #include <omp.h>
#endif

template<typename T>
tb_setup<T>::tb_setup(tb_mode mode, tb_type type, int nomp, int nmpi, int gpun, long spin, long chi, long chiL, long chiR, long mpoD, size_t iters)
    : mode(mode), type(type), nomp(nomp), nmpi(nmpi), gpun(gpun), iters(iters) {
    if(chiL == -1l) chiL = chi;
    if(chiR == -1l) chiR = chi;
    if(mpi::world.id == 0) {
        envL = Eigen::Tensor<T, 3>(chiL, chiL, mpoD);
        envR = Eigen::Tensor<T, 3>(chiR, chiR, mpoD);
        mpo  = Eigen::Tensor<T, 4>(mpoD, mpoD, spin, spin);
        psi  = Eigen::Tensor<T, 3>(spin, chiL, chiR);

        envL.setRandom();
        envR.setRandom();
        mpo.setRandom();
        psi.setRandom();
    }
    if(mode != tb_mode::cutensor) this->gpun = -1;
}
using fp32 = benchmark::fp32;
using fp64 = benchmark::fp64;
using cplx = benchmark::cplx;

template tb_setup<fp32>::tb_setup(tb_mode mode, tb_type type, int nomp, int nmpi, int gpun, long spin, long chi, long chiL, long chiR, long mpoD, size_t iters);
template tb_setup<fp64>::tb_setup(tb_mode mode, tb_type type, int nomp, int nmpi, int gpun, long spin, long chi, long chiL, long chiR, long mpoD, size_t iters);
template tb_setup<cplx>::tb_setup(tb_mode mode, tb_type type, int nomp, int nmpi, int gpun, long spin, long chi, long chiL, long chiR, long mpoD, size_t iters);

template<typename T>
std::string tb_setup<T>::string() const {
    return fmt::format("{} | {} | nomp {:2} | nmpi {:2} | gpun {:2} | psi {} = {} | mpo {} = {}", enum2sv(mode), enum2sv(type), nomp, nmpi, gpun,
                       psi.dimensions(), psi.size(), mpo.dimensions(), mpo.size());
}

template<typename T>
void benchmark::run_benchmark( tb_setup<T> &tbs) {
    mpi::barrier();
    auto t_run   = tid::tic_scope("run");
    auto t_vec   = std::vector<double>();
    auto tb      = std::vector<tb_results::table>();
    auto psi_out = benchmark::ResultType<T>();

#if defined(_OPENMP)
    omp_set_num_threads(tbs.nomp);
#endif
    if(tbs.mode == tb_mode::cutensor)
        tbs.device = config::getGpuName(tbs.gpun);
    else
        tbs.device = config::getCpuName();

    tenx::threads::setNumThreads(static_cast<unsigned int>(tbs.nomp));

    if(mpi::world.id == 0) {
        tools::log->info("Running Tensor Benchmark | mode {} | type {} | nomp {} | nmpi {} | device {} | iter {}", enum2sv(tbs.mode), enum2sv(tbs.type),
                         tbs.nomp, tbs.nmpi, tbs.device, tbs.iters);
        tbs.psi_check.resize(tbs.iters);
    }
    for(size_t iter = 0; iter < tbs.iters; iter++) {
        mpi::barrier();
        if(mpi::world.id == 0) {
            if(tbs.mode == tb_mode::eigen1) psi_out = benchmark::tensor_product_eigen1(tbs);
            if(tbs.mode == tb_mode::eigen2) psi_out = benchmark::tensor_product_eigen2(tbs);
            if(tbs.mode == tb_mode::eigen3) psi_out = benchmark::tensor_product_eigen3(tbs);
            if(tbs.mode == tb_mode::cutensor) psi_out = benchmark::tensor_product_cutensor(tbs);
            if(tbs.mode == tb_mode::xtensor) psi_out = benchmark::tensor_product_xtensor(tbs);
            if(tbs.mode == tb_mode::tblis) psi_out = benchmark::tensor_product_tblis(tbs);
            if(tbs.mode == tb_mode::matx) psi_out = benchmark::tensor_product_matx(tbs);
        }
        mpi::barrier();
        if(tbs.mode == tb_mode::cyclops) psi_out = benchmark::tensor_product_cyclops(tbs);
        if(mpi::world.id == 0) {
            double t_total = tid::get(fmt::format("{}", enum2sv(tbs.mode))).get_last_interval();
            double t_contr = tid::get(fmt::format("{}.contract", enum2sv(tbs.mode))).get_last_interval();
            size_t freq    = 1ul;
            if(tbs.iters > 20) freq = 5;
            if(tbs.iters > 100) freq = 10;
            if(tbs.iters > 1000) freq = 100;
            if(num::mod<size_t>(iter + 1, freq) == 0)
                tools::log->info("{} | iter {:>3}/{:<3} | time {:9.3e}s + {:9.3e}s overhead", tbs.string(), iter + 1, tbs.iters, t_contr, t_total - t_contr);

            if(tbs.psi_check[iter].size() == 0)
                tbs.psi_check[iter] = psi_out;
            else {
                auto   psi_out_vec = tenx::VectorMap(psi_out);
                auto   psi_chk_vec = tenx::VectorMap(tbs.psi_check[iter]);
                double overlap     = std::abs(psi_out_vec.normalized().dot(psi_chk_vec.normalized()));
                bool   approx      = psi_out_vec.isApprox(psi_chk_vec);
                if(std::abs(1.0 - overlap) > 1e-3 or not approx) tools::log->error("Mismatch | overlap {:.16f} | approx {}", overlap, approx);
            }
            tb.emplace_back(tbs, iter, t_contr, t_total);
            t_vec.emplace_back(t_contr);
        }
    }
    if(mpi::world.id == 0) {
        auto tbdb = h5pp::File(config::tb_filename, h5pp::FileAccess::READWRITE);
        tbdb.createTable(tb_results::h5_type, config::tb_dsetname, "TensorBenchmark", std::nullopt, 6);
        std::vector<double> ops;
        for(const auto &t : t_vec) ops.emplace_back(1.0 / t);
        tools::log->info(FMT_STRING("{} | time {:.4e}s avg {:.4e} +- {:.4e}s | op/s: {:.4f} +- {:.4f}"), enum2sv(tbs.mode), stat::sum(t_vec), stat::mean(t_vec),
                         stat::stdev(t_vec), stat::mean(ops), stat::stdev(ops));
        tbdb.appendTableRecords(tb, config::tb_dsetname);
    }
    mpi::barrier();
}

template void benchmark::run_benchmark(tb_setup<fp32> &tbs);
template void benchmark::run_benchmark(tb_setup<fp64> &tbs);
template void benchmark::run_benchmark(tb_setup<cplx> &tbs);

void benchmark::iterate_benchmarks() {
    if(mpi::world.id == 0) {
        // Initialize the output file
        auto tbdb = h5pp::File(config::tb_filename, config::tb_fileaccess);
        tb_results::register_table_type();
        tbdb.writeAttribute(env::git::branch, "/", "branch");
        tbdb.writeAttribute(env::git::commit_hash, "/", "commit_hash");
        tbdb.writeAttribute(env::git::revision, "/", "revision");

        tools::log->info("Starting benchmark");
        tools::log->info("spin  : {}", config::v_spin);
        tools::log->info("mpod  : {}", config::v_spin);
        tools::log->info("bond  : {}", config::v_chi);
        tools::log->info("bond L: {}", config::v_chiL);
        tools::log->info("bond R: {}", config::v_chiR);
    }
    auto nmpi = mpi::world.get_size<int>();
    auto t_tb = tid::tic_scope("tensorbench", tid::level::normal);
    for(auto type : config::tb_types) {
        for(auto mode : config::tb_modes) {
            for(auto chi : config::v_chi) {
                for(auto chiL : config::v_chiL) {
                    for(auto chiR : config::v_chiR) {
                        for(auto mpoD : config::v_mpoD) {
                            for(auto spin : config::v_spin) {
                                for(auto nomp : config::v_nomp) {
                                    omp_set_num_threads(nomp);
                                    for(auto gpun : config::v_gpun) {
                                        if(mode == tb_mode::cutensor or mode == tb_mode::matx) {
                                            if(nomp > 1) {
                                                tools::log->info("skipping benchmark in [{}] mode because nomp > 1", enum2sv(mode));
                                                continue;
                                            } else {
                                                config::initializeCuda(gpun);
                                            }
                                        }
                                        switch(type) {
                                            case tb_type::fp32: {
                                                auto tbs = tb_setup<fp32>(mode, type, nomp, nmpi, gpun, spin, chi, chiL, chiR, mpoD, config::n_iter);
                                                benchmark::run_benchmark(tbs);
                                                break;
                                            }
                                            case tb_type::fp64: {
                                                auto tbs = tb_setup<fp64>(mode, type, nomp, nmpi, gpun, spin, chi, chiL, chiR, mpoD, config::n_iter);
                                                benchmark::run_benchmark(tbs);
                                                break;
                                            }
                                            case tb_type::cplx: {
                                                auto tbs = tb_setup<cplx>(mode, type, nomp, nmpi, gpun, spin, chi, chiL, chiR, mpoD, config::n_iter);
                                                benchmark::run_benchmark(tbs);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
