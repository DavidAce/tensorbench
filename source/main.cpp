
#include "contract/contract.h"
#include "general/enums.h"
#include "math/stat.h"
#include "mpi/mpi-tools.h"
#include "storage/results.h"
#include "tblis/util/thread.h"
#include "tid/tid.h"
#include "tools/class_tic_toc.h"
#include "tools/log.h"
#include "tools/prof.h"
#include <cxxopts.hpp>
#include <getopt.h>
#include <gitversion.h>
#include <thread>

#if defined(TB_OPENBLAS)
    #include <openblas/cblas.h>
    #include <openblas/openblas_config.h>
#endif

#if defined(TB_MKL)
    #define MKL_Complex8 std::complex<float>
    #define MKL_Complex16 std::complex<double>
    #include <mkl.h>
    #include <mkl_service.h>
#endif

#if defined(TB_CUDA) && __has_include(<cuda.h>)
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif

template<typename T>
struct tb_setup {
    int                                      num_threads = 0;
    size_t                                   iters;
    std::string                              group;
    Eigen::Tensor<T, 3>                      envL, envR, psi;
    Eigen::Tensor<T, 4>                      mpo;
    mutable std::vector<Eigen::Tensor<T, 3>> psi_check;
};

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
std::string currentDateTime() {
    time_t    now     = time(nullptr);
    struct tm tstruct = {};
    char      buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%F", &tstruct);
    return buf;
}

template<typename T, tb_mode mode>
void run_benchmark(h5pp::File &tdb, const tb_setup<T> &tsp, class_tic_toc &tt) {
    mpi::barrier();
    auto t_run   = tid::get("run").tic_token();
    long chiL    = tsp.psi.dimension(1);
    long chiR    = tsp.psi.dimension(2);
    long mpod    = tsp.mpo.dimension(0);
    long spin    = tsp.psi.dimension(0);
    auto t_vec   = std::vector<double>();
    auto tb      = std::vector<tb_results::table>();
    auto psi_out = Eigen::Tensor<T, 3>();
    long ops     = 0;

    if(mpi::world.id == 0) {
        tools::log->info("Running benchmark {} | iters {} | threads {}", enum2sv(mode), tsp.iters, tsp.num_threads);
        tsp.psi_check.resize(tsp.iters);
        tdb.createTable(tb_results::h5_type, fmt::format("{}/{}_{}", tsp.group, enum2sv(mode), tsp.num_threads),
                        fmt::format("TensorBenchmark {}", enum2sv(mode)), 5, 3);
    }
    for(size_t iter = 0; iter < tsp.iters; iter++) {
        mpi::barrier();
        auto t_prod = tid::get(enum2sv(mode)).tic_token();
        if(mpi::world.id == 0) {
            if constexpr(mode == tb_mode::eigen1) std::tie(psi_out, ops) = contract::tensor_product_eigen1(tsp.psi, tsp.mpo, tsp.envL, tsp.envR);
            if constexpr(mode == tb_mode::eigen2) std::tie(psi_out, ops) = contract::tensor_product_eigen2(tsp.psi, tsp.mpo, tsp.envL, tsp.envR);
            if constexpr(mode == tb_mode::eigen3) std::tie(psi_out, ops) = contract::tensor_product_eigen3(tsp.psi, tsp.mpo, tsp.envL, tsp.envR);
            if constexpr(mode == tb_mode::cute) std::tie(psi_out, ops)   = contract::tensor_product_cute  (tsp.psi, tsp.mpo, tsp.envL, tsp.envR);
            if constexpr(mode == tb_mode::acro) throw std::runtime_error("not implemented?");
            if constexpr(mode == tb_mode::xtensor) std::tie(psi_out, ops) = contract::tensor_product_xtensor(tsp.psi, tsp.mpo, tsp.envL, tsp.envR);
            if constexpr(mode == tb_mode::tblis) std::tie(psi_out, ops) = contract::tensor_product_tblis(tsp.psi, tsp.mpo, tsp.envL, tsp.envR);
        }
        if constexpr(mode == tb_mode::cyclops) std::tie(psi_out, ops) = contract::tensor_product_cyclops(tsp.psi, tsp.mpo, tsp.envL, tsp.envR);
        t_prod.toc();
        if(mpi::world.id == 0) {
            tools::log->info("{} | threads {:2} | psi {} | mpo {} | iter {}/{} |  time {:8.4f} s | GOp {:<8.4f} | GOp/s {:<.4f}", tt.get_name(),
                             tsp.num_threads, psi_out.dimensions(), mpod, iter + 1, tsp.iters, tt.get_last_time_interval(), ops / 1e9,
                             ops / 1e9 / tt.get_last_time_interval());

            if(tsp.psi_check[iter].size() == 0) tsp.psi_check[iter] = psi_out;
            else {
                auto   psi_out_vec = tenx::VectorMap(psi_out);
                auto   psi_chk_vec = tenx::VectorMap(tsp.psi_check[iter]);
                double overlap     = psi_out_vec.normalized().dot(psi_chk_vec.normalized());
                bool   approx      = psi_out_vec.isApprox(psi_chk_vec);
                if(overlap < 0.999 or not approx) tools::log->error("Mismatch | overlap {:.16f} | approx {}", overlap, approx);
            }

            tb.emplace_back(enum2sv(mode), iter, tsp.num_threads, chiL, chiR, mpod, spin, ops, tt.get_last_time_interval(), tt.get_measured_time());
            t_vec.emplace_back(tt.get_last_time_interval());
        }
    }
    if(mpi::world.id == 0) {
        tools::log->info("{} | threads {:2} | total time {:.4f} s | min {:.4f} | max {:.4f} | avg {:.4f} |  stdev {:.4f}", tt.get_name(), tsp.num_threads,
                         tt.get_measured_time(), stat::min(t_vec), stat::max(t_vec), stat::mean(t_vec), stat::stdev(t_vec));
        tdb.appendTableRecords(tb, fmt::format("{}/{}_{}", tsp.group, enum2sv(mode), tsp.num_threads));
    }
    mpi::barrier();
}

/*
 * Resources explaining MPI/OpenMP Hybrid parameters
 *
 * https://www.admin-magazine.com/HPC/Articles/Processor-Affinity-for-OpenMP-and-MPI
 *
 * These mpirun flags seem to work well:
 * -n 4 --report-bindings --map-by l3cache:pe=4 --bind-to core -x OMP_PLACES=sockets -x OMP_PROC_BIND=true -x OMP_NUM_THREADS=4 -x OMP_DISPLAY_AFFINITY=true ...
 *
 */

int main(int argc, char *argv[]) {
    cxxopts::Options options("tensorbench", "Benchmarks of tensor contractions");
    /* clang-format off */

    std::string omp_max_threads_default = "1";

#if defined(_OPENMP)
        // This will honor env variable OMP_NUM_THREADS
        omp_max_threads_default = fmt::format("{}", omp_get_max_threads());
#endif
    options.add_options()
        ("h,help",     "Show help")
        ("B,bond",     "Bond dimension (Sets both chiL and chiR)",  cxxopts::value<std::vector<long>>()->default_value("16"))
        ("L,chiL",     "Bond dimension to the left",                cxxopts::value<std::vector<long>>()->default_value("-1"))
        ("R,chiR",     "Bond dimension to the right",               cxxopts::value<std::vector<long>>()->default_value("-1"))
        ("D,spin",     "Spin dimension",                            cxxopts::value<std::vector<long>>()->default_value("2"))
        ("M,mpod",     "MPO dimension",                             cxxopts::value<std::vector<long>>()->default_value("16"))
        ("n,threads",  "Number of threads",                         cxxopts::value<std::vector<int>>()->default_value(omp_max_threads_default))
        ("i,iters",    "Number of iterations",                      cxxopts::value<size_t>()->default_value("3"))
        ("v,verbose",  "Sets verbosity level",                      cxxopts::value<size_t>()->default_value("2"));
    /* clang-format on */

    auto in = options.parse(argc, argv);
    if(in["help"].count() > 0) {
        fmt::print(options.help());
        exit(0);
    }

    if(in["bond"].count() > 0 and (in["chiL"].count() > 0 or in["chiR"].count() > 0)) throw std::runtime_error("Argument error: Use EITHER bond or chiL/chiR");

    auto v_chi     = in["bond"].as<std::vector<long>>();
    auto v_chiL    = in["chiL"].as<std::vector<long>>();
    auto v_chiR    = in["chiR"].as<std::vector<long>>();
    auto v_spin    = in["spin"].as<std::vector<long>>();
    auto v_mpod    = in["mpod"].as<std::vector<long>>();
    auto v_threads = in["threads"].as<std::vector<int>>();
    auto iters     = in["iters"].as<size_t>();
    auto verbosity = in["verbose"].as<size_t>();

    tools::log = tools::Logger::setLogger("tensorbench", verbosity);
    auto t_tot = tid::get("tb").tic_token();
    tools::prof::init_profiling();
    mpi::init();
#if defined(_OPENMP)
    omp_set_num_threads(v_threads.front());
#endif
    for(int id = 0; id < mpi::world.size; ++id) {
        if(id == mpi::world.id) {
#pragma omp parallel
            {
                tools::log->info("Hello from mpi rank {} of {} | omp thread {} of {} | on thread {}", mpi::world.id, mpi::world.size, omp_get_thread_num(),
                                 omp_get_max_threads(), sched_getcpu());
            }
        }
        tools::log->flush();
        mpi::barrier();
    }

    // mpi::finalize();
    // exit(0);
    //  Set the number of threads to be used

#if defined(TB_CUDA) && __has_include(<cuda.h>)
    tools::log->info("Initializing CUDA");
    auto init_result = cudaSetDevice(0);
    //    auto init_result = cuInit(0);
    if(init_result != 0) throw std::runtime_error("cuInit returned " + std::to_string(init_result));
#endif

    using cplx   = std::complex<double>;
    using fp32   = float;
    using fp64   = double;
    using Scalar = fp64;

    h5pp::File tbdb;
    if(mpi::world.id == 0) {
        tbdb = h5pp::File(fmt::format("../output/tbdb-{}.h5", currentDateTime()), h5pp::FilePermission::REPLACE);
        tb_results::register_table_type();
        tbdb.writeAttribute(GIT::BRANCH, "/", "git_branch");
        tbdb.writeAttribute(GIT::COMMIT_HASH, "/", "git_commit");
        tbdb.writeAttribute(GIT::REVISION, "/", "git_revision");

#if defined(TB_EIGEN1) || defined(TB_EIGEN2) || defined(TB_EIGEN3)
        tbdb.writeAttribute(fmt::format("Eigen {}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION), "/", "eigen_version");
#endif
#if defined(TB_OPENBLAS)
        tbdb.writeAttribute(OPENBLAS_VERSION, "/", "openblas_version");
#endif
#if defined(TB_MKL)
        tbdb.writeAttribute(fmt::format("Intel MKL {}", INTEL_MKL_VERSION), "/", "intelmkl_version");
#endif
        tools::log->info("Starting benchmark");
        tools::log->info("chi : {}", v_chi);
        tools::log->info("chiL: {}", v_chiL);
        tools::log->info("chiR: {}", v_chiR);
        tools::log->info("mpod: {}", v_mpod);
        tools::log->info("spin: {}", v_spin);
    }

    tools::prof::t_total->tic();

    for(auto chi : v_chi)
        for(auto chiL : v_chiL)
            for(auto chiR : v_chiR)
                for(auto mpod : v_mpod)
                    for(auto spin : v_spin) {
                        if(chiL == -1) chiL = chi;
                        if(chiR == -1) chiR = chi;
                        tb_setup<Scalar> tbs;
                        tbs.iters = iters;
                        tbs.group = fmt::format("chiL_{}_chiR_{}_mpod_{}_spin_{}", chiL, chiR, mpod, spin);

                        if(mpi::world.id == 0) {
                            tbs.envL = Eigen::Tensor<Scalar, 3>(chiL, chiL, mpod);
                            tbs.envR = Eigen::Tensor<Scalar, 3>(chiR, chiR, mpod);
                            tbs.mpo  = Eigen::Tensor<Scalar, 4>(mpod, mpod, spin, spin);
                            tbs.psi  = Eigen::Tensor<Scalar, 3>(spin, chiL, chiR);

                            tbs.envL.setRandom();
                            tbs.envR.setRandom();
                            tbs.mpo.setRandom();
                            tbs.psi.setRandom();
                            tools::log->info("Starting benchmark: {}", tbs.group);
                        }
                        tools::prof::reset_profiling();

#if defined(TB_CUTE)
                        run_benchmark<Scalar, tb_mode::cute>(tbdb, tbs, *tools::prof::t_cute);
#endif

                        for(auto num_threads : v_threads) {
                            if(num_threads <= 0) throw std::runtime_error(fmt::format("Invalid num threads: {}", optarg));
                            tools::prof::reset_profiling();
#if defined(EIGEN_USE_THREADS)
                            tenx::omp::setNumThreads(num_threads);
                            tools::log->info("Using Eigen {}.{}.{} Tensor Module with {} threads", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION,
                                             EIGEN_MINOR_VERSION, tenx::omp::tp->NumThreads());
#endif
#if defined(TB_TBLIS)
                            tblis_set_num_threads(num_threads);
#endif
#if defined(_OPENMP)
                            omp_set_num_threads(num_threads);
                            tools::log->info("Using OpenMP with {} threads", omp_get_max_threads());
    #if defined(TB_OPENBLAS)
                            openblas_set_num_threads(num_threads);
                            tools::log->info(
                                "{} compiled with parallel mode {} for target {} with config {} | multithread threshold {} | running with {} threads",
                                OPENBLAS_VERSION, openblas_get_parallel(), openblas_get_corename(), openblas_get_config(), OPENBLAS_GEMM_MULTITHREAD_THRESHOLD,
                                openblas_get_num_threads());
    #endif

    #if defined(TB_MKL)
                            mkl_set_num_threads(num_threads);
                            tools::log->info("Using Intel MKL {} with {} threads", INTEL_MKL_VERSION, mkl_get_max_threads());
                                //                            mkl_verbose(1);
    #endif
#endif
                            tbs.num_threads = num_threads;

#if defined(TB_EIGEN1)
                            run_benchmark<Scalar, tb_mode::eigen1>(tbdb, tbs, *tools::prof::t_eigen1);
#endif
#if defined(TB_EIGEN2)
                            run_benchmark<Scalar, tb_mode::eigen2>(tbdb, tbs, *tools::prof::t_eigen2);
#endif
#if defined(TB_EIGEN3)
                            run_benchmark<Scalar, tb_mode::eigen3>(tbdb, tbs, *tools::prof::t_eigen3);

#endif
#if defined(TB_TBLIS)
                            run_benchmark<Scalar, tb_mode::tblis>(tbdb, tbs, *tools::prof::t_tblis);
#endif
#if defined(TB_XTENSOR)
                            run_benchmark<Scalar, tb_mode::xtensor>(tbdb, tbs, *tools::prof::t_xtensor);
#endif
#if defined(TB_CYCLOPS)
                            run_benchmark<Scalar, tb_mode::cyclops>(tbdb, tbs, *tools::prof::t_cyclops);
#endif
                        }
                    }

    tools::prof::t_total->toc();
    mpi::barrier();
    tools::log->info("total time {:.4f} s", tools::prof::t_total->get_measured_time());

    mpi::finalize();
}