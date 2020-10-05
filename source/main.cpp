
#include <contract/contract.h>
#include <getopt.h>
#include <storage/results.h>
#include <thread>
#include <tools/class_tic_toc.h>
#include <tools/log.h>
#include <tools/prof.h>

#ifdef OPENBLAS_AVAILABLE
    #include <cblas.h>
    #include <openblas_config.h>
#endif

#ifdef MKL_AVAILABLE
    #define MKL_Complex8 std::complex<float>
    #define MKL_Complex16 std::complex<double>
    #include <mkl.h>
    #include <mkl_service.h>
#endif

#if __has_include(<cuda.h>)
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif

void print_usage() {
    std::cout <<
        R"(
==========  cpp_merger  ============
Usage                       : ./cpp_merger [-option <value>].
-h                          : Help. Shows this text.
-i <iterations>             : Sets number of iterations
-n <num threads>            : Sets number of threads for OpenMP operations
-B <chi>                    : Sets both left and right bond dimension
-L <chiL>                   : Sets the left bond dimension
-R <chiR>                   : Sets the right bond dimension
-D <spin dim>               : Sets the aggregate spin dimension
-M <mpo dim>                : Sets the mpo dimension
-v <level>                  : Enables trace-level verbosity
)";
}

int main(int argc, char *argv[]) {
    // Here we use getopt to parse CLI input
    // Note that CLI input always override config-file values
    // wherever they are found (config file, h5 file)
    auto              log       = tools::Logger::setLogger("tensorbench", 2);
    size_t            verbosity = 2;
    int               iters     = 3;
    std::vector<int>  v_threads;
    std::vector<long> v_chi;
    std::vector<long> v_chiL;
    std::vector<long> v_chiR;
    std::vector<long> v_spin;
    std::vector<long> v_mpod;
    while(true) {
        char opt = static_cast<char>(getopt(argc, argv, "hB:L:R:D:M:i:n:v:"));
        if(opt == EOF) break;
        if(optarg == nullptr) log->info("Parsing input argument: -{}", opt);
        else
            log->info("Parsing input argument: -{} {}", opt, optarg);
        switch(opt) {
            case 'B':
                if(not v_chiL.empty()) throw std::runtime_error("Use EITHER -B or -L/R");
                if(not v_chiR.empty()) throw std::runtime_error("Use EITHER -B or -L/R");
                v_chi.push_back(std::strtol(optarg, nullptr, 10));
                continue;
            case 'L':
                if(not v_chi.empty()) throw std::runtime_error("Use EITHER -B or -L/R");
                v_chiL.push_back(std::strtol(optarg, nullptr, 10));
                continue;
            case 'R':
                if(not v_chi.empty()) throw std::runtime_error("Use EITHER -B or -L/R");
                v_chiR.push_back(std::strtol(optarg, nullptr, 10));
                continue;
            case 'D': v_spin.push_back(std::strtol(optarg, nullptr, 10)); continue;
            case 'M': v_mpod.push_back(std::strtol(optarg, nullptr, 10)); continue;
            case 'i': iters = std::strtol(optarg, nullptr, 10); continue;
            case 'n':
#if !defined(EIGEN_USE_THREADS)
                throw std::runtime_error("Threading option [-n:<num>] is invalid: Please define EIGEN_USE_THREADS");
#endif
#if !defined(_OPENMP)
                throw std::runtime_error("Threading option [-n:<num>] is invalid: Please define _OPENMP");
#endif
                v_threads.push_back(std::stoi(optarg, nullptr, 10));
                continue;
            case 'v': verbosity = std::strtoul(optarg, nullptr, 10); continue;
            case ':': log->error("Option -{} needs a value", opt); break;
            case 'h':
            case '?':
            default: print_usage(); exit(0);
            case -1: break;
        }
        break;
    }

    if(v_chi.empty() and v_chiL.empty() and v_chiR.empty()) {
        v_chi  = {256};
        v_chiL = {-1};
        v_chiR = {-1};
    } else if(not v_chi.empty()) {
        v_chiL = {-1};
        v_chiR = {-1};
    }
    if(v_chiL.empty()) v_chiL = {16};
    if(v_chiR.empty()) v_chiR = {16};
    if(v_spin.empty()) v_spin = {4};
    if(v_mpod.empty()) v_mpod = {5};
    if(v_threads.empty()) v_threads = {1};

    tools::prof::init_profiling();
    tools::log = tools::Logger::setLogger("tensorbench", verbosity);

    // Set the number of threads to be used

#if __has_include(<cuda.h>)
    tools::log->info("Initializing CUDA");
    auto init_result = cudaSetDevice(0);
    //    auto init_result = cuInit(0);
    if(init_result != 0) throw std::runtime_error("cuInit returned " + std::to_string(init_result));
#endif

    using cplx   = std::complex<double>;
    using fp32   = float;
    using fp64   = double;
    using Scalar = fp64;

    h5pp::File tbdb("tbdb.h5", h5pp::FilePermission::REPLACE);
    tb_results::register_table_type();
    tools::prof::t_total->tic();
    tools::log->info("Starting benchmark");

    for(auto chi : v_chi)
        for(auto chiL : v_chiL)
            for(auto chiR : v_chiR)
                for(auto mpod : v_mpod)
                    for(auto spin : v_spin) {
                        if(chiL == -1) chiL = chi;
                        if(chiR == -1) chiR = chi;
                        Eigen::Tensor<Scalar, 4> envL(chiL, chiL, mpod, mpod);
                        Eigen::Tensor<Scalar, 4> envR(chiR, chiR, mpod, mpod);
                        Eigen::Tensor<Scalar, 4> mpo(mpod, mpod, spin, spin);
                        Eigen::Tensor<Scalar, 3> psi(spin, chiL, chiR);
                        envL.setRandom();
                        envR.setRandom();
                        mpo.setRandom();
                        psi.setRandom();
                        std::string tb_basename = fmt::format("chiL_{}_chiR_{}/mpod_{}/spin_{}", chiL, chiR, mpod, spin);
                        tools::log->info("Starting benchmark mode: {}", tb_basename);
                        tools::prof::reset_profiling();
#if defined(TB_CUDA)
                        std::vector<tb_results::table> tb_cuda;
                        tbdb.createTable(tb_results::h5_type, tb_basename + "/cuda", "TensorBenchmark CUDA", 5, 3);
                        for(int iter = 0; iter < iters; iter++) {
                            auto psi_out = contract::hamiltonian_squared_dot_psi_cuda(psi, mpo, envL, envR);
                            tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.7f} s", tools::prof::t_ham_sq_psi_cuda->get_name(),
                                             psi_out.dimensions(), iter + 1, iters, tools::prof::t_ham_sq_psi_cuda->get_last_time_interval());
                            tb_cuda.emplace_back("cuda", iter, num_threads, chiL, chiR, mpod, spin, tools::prof::t_ham_sq_psi_cuda->get_last_time_interval(),
                                                 tools::prof::t_ham_sq_psi_cuda->get_measured_time())
                        }
                        tools::log->info("{} | total time {:.7f} s", tools::prof::t_ham_sq_psi_cuda->get_name(),
                                         tools::prof::t_ham_sq_psi_cuda->get_measured_time());
                        tbdb.appendTableRecords(tb_cuda, tb_basename + "/cuda");
#endif

#if defined(TB_ACRO)
                        std::vector<tb_results::table> tb_acro;
                        tbdb.createTable(tb_results::h5_type, tb_basename + "/acro", "TensorBenchmark AcroTensor", 5, 3);
                        for(int iter = 0; iter < iters; iter++) {
                            auto psi_out = contract::hamiltonian_squared_dot_psi_acro(psi, mpo, envL, envR);
                            tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_acro->get_name(),
                                             psi_out.dimensions(), iter + 1, iters, tools::prof::t_ham_sq_psi_acro->get_last_time_interval());
                            tb_acro.emplace_back("acro", iter, num_threads, chiL, chiR, mpod, spin, tools::prof::t_ham_sq_psi_acro->get_last_time_interval(),
                                                 tools::prof::t_ham_sq_psi_acro->get_measured_time());
                        }
                        tools::log->info("{} | total time {:.4f} s", tools::prof::t_ham_sq_psi_acro->get_name(),
                                         tools::prof::t_ham_sq_psi_acro->get_measured_time());
                        tbdb.appendTableRecords(tb_acro, tb_basename + "/acro");
#endif

#if defined(TB_CUTE)
                        std::vector<tb_results::table> tb_cute;
                        tbdb.createTable(tb_results::h5_type, tb_basename + "/cute", "cutensor", 5, 3);
                        for(int iter = 0; iter < iters; iter++) {
                            auto psi_out = contract::hamiltonian_squared_dot_psi_cute(psi, mpo, envL, envR);
                            tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_cute->get_name(),
                                             psi_out.dimensions(), iter + 1, iters, tools::prof::t_ham_sq_psi_cute->get_last_time_interval());
                            tb_cute.emplace_back("cute", iter, 1, chiL, chiR, mpod, spin, tools::prof::t_ham_sq_psi_cute->get_last_time_interval(),
                                                 tools::prof::t_ham_sq_psi_cute->get_measured_time());
                        }

                        tools::log->info("{} | total time {:.4f} s ", tools::prof::t_ham_sq_psi_cute->get_name(),
                                         tools::prof::t_ham_sq_psi_cute->get_measured_time());
                        tbdb.appendTableRecords(tb_cute, tb_basename + "/cute");
#endif

                        for(auto num_threads : v_threads) {
                            if(num_threads <= 0) throw std::runtime_error(fmt::format("Invalid num threads: {}", optarg));
                            tools::prof::reset_profiling();
#if defined(EIGEN_USE_THREADS)
                            Textra::omp::setNumThreads(num_threads);
                            tools::log->info("Using Eigen Tensor with {} threads", Textra::omp::tp->NumThreads());
#endif
#if defined(_OPENMP)
                            omp_set_num_threads(num_threads);
                            tools::log->info("Using OpenMP with {} threads", omp_get_max_threads());
    #ifdef OPENBLAS_AVAILABLE
                            openblas_set_num_threads(num_threads);
                            tools::log->info(
                                "{} compiled with parallel mode {} for target {} with config {} | multithread threshold {} | running with {} threads",
                                OPENBLAS_VERSION, openblas_get_parallel(), openblas_get_corename(), openblas_get_config(), OPENBLAS_GEMM_MULTITHREAD_THRESHOLD,
                                openblas_get_num_threads());
    #endif

    #ifdef MKL_AVAILABLE
                            mkl_set_num_threads(num_threads);
                            tools::log->info("Using Intel MKL with {} threads", mkl_get_max_threads());
                            mkl_verbose(1);
    #endif
#endif

#if defined(TB_CPU)
                            std::vector<tb_results::table> tb_cpu1;
                            tbdb.createTable(tb_results::h5_type, fmt::format("{}/cpu_v1_{}",tb_basename,num_threads), "TensorBenchmark Cpu version 1", 5, 3);
                            for(int iter = 0; iter < iters; iter++) {
                                auto psi_out = contract::hamiltonian_squared_dot_psi_v1(psi, mpo, envL, envR);
                                tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_v1->get_name(),
                                                 psi_out.dimensions(), iter + 1, iters, tools::prof::t_ham_sq_psi_v1->get_last_time_interval());
                                tb_cpu1.emplace_back("cpu1", iter, num_threads, chiL, chiR, mpod, spin, tools::prof::t_ham_sq_psi_v1->get_last_time_interval(),
                                                     tools::prof::t_ham_sq_psi_v1->get_measured_time());
                            }
                            tools::log->info("{} | total time {:.4f} s", tools::prof::t_ham_sq_psi_v1->get_name(),
                                             tools::prof::t_ham_sq_psi_v1->get_measured_time());
                            tbdb.appendTableRecords(tb_cpu1,fmt::format("{}/cpu_v1_{}",tb_basename,num_threads));

                            std::vector<tb_results::table> tb_cpu2;
                            tbdb.createTable(tb_results::h5_type, fmt::format("{}/cpu_v2_{}",tb_basename,num_threads), "TensorBenchmark Cpu version 2", 5, 3);
                            for(int iter = 0; iter < iters; iter++) {
                                auto psi_out = contract::hamiltonian_squared_dot_psi_v2(psi, mpo, envL, envR);
                                tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_v2->get_name(),
                                                 psi_out.dimensions(), iter + 1, iters, tools::prof::t_ham_sq_psi_v2->get_last_time_interval());
                                tb_cpu2.emplace_back("cpu2", iter, num_threads, chiL, chiR, mpod, spin, tools::prof::t_ham_sq_psi_v2->get_last_time_interval(),
                                                     tools::prof::t_ham_sq_psi_v2->get_measured_time());
                            }
                            tools::log->info("{} | total time {:.4f} s", tools::prof::t_ham_sq_psi_v2->get_name(),
                                             tools::prof::t_ham_sq_psi_v2->get_measured_time());
                            tbdb.appendTableRecords(tb_cpu2, fmt::format("{}/cpu_v2_{}",tb_basename,num_threads));

                            std::vector<tb_results::table> tb_cpu3;
                            tbdb.createTable(tb_results::h5_type, fmt::format("{}/cpu_v3_{}",tb_basename,num_threads), "TensorBenchmark Cpu version 3", 5, 3);
                            for(int iter = 0; iter < iters; iter++) {
                                auto psi_out = contract::hamiltonian_squared_dot_psi_v3(psi, mpo, envL, envR);
                                tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_v3->get_name(),
                                                 psi_out.dimensions(), iter + 1, iters, tools::prof::t_ham_sq_psi_v3->get_last_time_interval());
                                tb_cpu3.emplace_back("cpu3", iter, num_threads, chiL, chiR, mpod, spin, tools::prof::t_ham_sq_psi_v3->get_last_time_interval(),
                                                     tools::prof::t_ham_sq_psi_v3->get_measured_time());
                            }
                            tools::log->info("{} | total time {:.4f} s", tools::prof::t_ham_sq_psi_v3->get_name(),
                                             tools::prof::t_ham_sq_psi_v3->get_measured_time());
                            tbdb.appendTableRecords(tb_cpu3, fmt::format("{}/cpu_v3_{}",tb_basename,num_threads));
#endif
                        }
                    }

    tools::prof::t_total->toc();
    tools::log->info("total time {:.4f} s", tools::prof::t_total->get_measured_time());
}