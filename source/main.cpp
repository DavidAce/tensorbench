
#include <contract/contract.h>
#include <getopt.h>
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
    auto   log         = tools::Logger::setLogger("tensorbench", 2);
    size_t verbosity   = 2;
    int    num_threads = 1; // static_cast<int>(std::thread::hardware_concurrency());
    long chiL  = 256;
    long chiR  = 256;
    long spin  = 4;
    long mpod  = 5;
    int  iters = 3;
    while(true) {
        char opt = static_cast<char>(getopt(argc, argv, "hL:R:D:M:i:n:v:"));
        if(opt == EOF) break;
        if(optarg == nullptr) log->info("Parsing input argument: -{}", opt);
        else
            log->info("Parsing input argument: -{} {}", opt, optarg);
        switch(opt) {
            case 'L': chiL = std::strtol(optarg, nullptr, 10); continue;
            case 'R': chiR = std::strtol(optarg, nullptr, 10); continue;
            case 'D': spin = std::strtol(optarg, nullptr, 10); continue;
            case 'M': mpod = std::strtol(optarg, nullptr, 10); continue;
            case 'i': iters = std::strtol(optarg, nullptr, 10); continue;
            case 'n':
#if !defined(EIGEN_USE_THREADS)
                throw std::runtime_error("Threading option [-n:<num>] is invalid: Please define EIGEN_USE_THREADS");
#endif
#if !defined(_OPENMP)
                throw std::runtime_error("Threading option [-n:<num>] is invalid: Please define _OPENMP");
#endif
                num_threads = std::stoi(optarg, nullptr, 10);
                if(num_threads <= 0) throw std::runtime_error(fmt::format("Invalid num threads: {}", optarg));
                else
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

    tools::prof::init_profiling();
    tools::log = tools::Logger::setLogger("tensorbench", verbosity);
    Textra::omp::setNumThreads(num_threads);
    // Set the number of threads to be used

#if defined(EIGEN_USE_THREADS)
    Textra::omp::setNumThreads(num_threads);
    tools::log->info("Using Eigen Tensor with {} threads", Textra::omp::tp->NumThreads());
#endif
#if defined(_OPENMP)
    omp_set_num_threads(num_threads);
    tools::log->info("Using OpenMP with {} threads", omp_get_max_threads());
    #ifdef OPENBLAS_AVAILABLE
    openblas_set_num_threads(num_threads);
    tools::log->info("{} compiled with parallel mode {} for target {} with config {} | multithread threshold {} | running with {} threads", OPENBLAS_VERSION,
                     openblas_get_parallel(), openblas_get_corename(), openblas_get_config(), OPENBLAS_GEMM_MULTITHREAD_THRESHOLD, openblas_get_num_threads());
    #endif

    #ifdef MKL_AVAILABLE
    mkl_set_num_threads(num_threads);
    tools::log->info("Using Intel MKL with {} threads", mkl_get_max_threads());
    mkl_verbose(1);
    #endif
#endif
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
    Eigen::Tensor<Scalar, 4> envL(chiL, chiL, mpod, mpod);
    Eigen::Tensor<Scalar, 4> envR(chiR, chiR, mpod, mpod);
    Eigen::Tensor<Scalar, 4> mpo(mpod, mpod, spin, spin);
    Eigen::Tensor<Scalar, 3> psi(spin, chiL, chiR);
    envL.setRandom();
    envR.setRandom();
    mpo.setRandom();
    psi.setRandom();
    tools::prof::t_total->tic();
    tools::log->info("Starting benchmark");
#if defined(TB_CPU)
    for(int iter = 0; iter < iters; iter++) {
        auto psi_out = contract::hamiltonian_squared_dot_psi_v1(psi, mpo, envL, envR);
        tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_v1->get_name(), psi_out.dimensions(), iter, iters,
                         tools::prof::t_ham_sq_psi_v1->get_last_time_interval());
    }
    tools::log->info("{} | total time {:.4f} s", tools::prof::t_ham_sq_psi_v1->get_name(),
                     tools::prof::t_ham_sq_psi_v1->get_measured_time());

    for(int iter = 0; iter < iters; iter++) {
        auto psi_out = contract::hamiltonian_squared_dot_psi_v2(psi, mpo, envL, envR);
        tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_v2->get_name(), psi_out.dimensions(), iter, iters,
                         tools::prof::t_ham_sq_psi_v2->get_last_time_interval());
    }
    tools::log->info("{} | total time {:.4f} s", tools::prof::t_ham_sq_psi_v2->get_name(),
                     tools::prof::t_ham_sq_psi_v2->get_measured_time());

    for(int iter = 0; iter < iters; iter++) {
        auto psi_out = contract::hamiltonian_squared_dot_psi_v3(psi, mpo, envL, envR);
        tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_v3->get_name(), psi_out.dimensions(), iter, iters,
                         tools::prof::t_ham_sq_psi_v3->get_last_time_interval());
    }
    tools::log->info("{} | total time {:.4f} s", tools::prof::t_ham_sq_psi_v3->get_name(),
                     tools::prof::t_ham_sq_psi_v3->get_measured_time());
#endif

#if defined(TB_CUDA)
//    for(int iter = 0; iter < iters; iter++) {
//        auto psi_out = contract::hamiltonian_squared_dot_psi_cuda(psi, mpo, envL, envR);
//        tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.7f} s", tools::prof::t_ham_sq_psi_cuda->get_name(), psi_out.dimensions(), iter, iters,
//                         tools::prof::t_ham_sq_psi_cuda->get_last_time_interval());
//    }
//    tools::log->info("{} | total time {:.7f} s", tools::prof::t_ham_sq_psi_cuda->get_name(),
//                     tools::prof::t_ham_sq_psi_cuda->get_measured_time());
#endif


#if defined(TB_ACRO)
    for(int iter = 0; iter < iters; iter++) {
        auto psi_out = contract::hamiltonian_squared_dot_psi_acro(psi, mpo, envL, envR);
        tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_acro->get_name(), psi_out.dimensions(), iter, iters,
                         tools::prof::t_ham_sq_psi_acro->get_last_time_interval());
    }
    tools::log->info("{} | total time {:.4f} s", tools::prof::t_ham_sq_psi_acro->get_name(),
                     tools::prof::t_ham_sq_psi_acro->get_measured_time());
#endif


#if defined(TB_CUTE)
    for(int iter = 0; iter < iters; iter++) {
        auto psi_out = contract::hamiltonian_squared_dot_psi_cute(psi, mpo, envL, envR);
        tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_cute->get_name(), psi_out.dimensions(), iter, iters,
                         tools::prof::t_ham_sq_psi_cute->get_last_time_interval());
    }
    tools::log->info("{} | total time {:.4f} s", tools::prof::t_ham_sq_psi_cute->get_name(),
                     tools::prof::t_ham_sq_psi_cute->get_measured_time());
#endif


    tools::prof::t_total->toc();
    tools::log->info("total time {:.4f} s", tools::prof::t_total->get_measured_time());
}