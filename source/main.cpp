
#include <contract/contract.h>
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


int main() {
    tools::prof::init_profiling();
    size_t logLevel = 0;
    tools::log      = tools::Logger::setLogger("tensorbench", logLevel);
    int num_threads = 8;// static_cast<int>(std::thread::hardware_concurrency());
    Textra::omp::setNumThreads(num_threads);
// Set the number of threads to be used
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    Eigen::setNbThreads(num_threads);
    Textra::omp::setNumThreads(num_threads);
    tools::log->info("Using Eigen Tensor with {} threads", Textra::omp::tp->NumThreads());
    tools::log->info("Using Eigen  with {} threads", Eigen::nbThreads());
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

    long chiL  = 64;
    long chiR  = 64;
    long spin  = 32;
    long mpod  = 5;
    int  iters = 10;

    // 256 16 16 (8 = 4+4) --> A/B
    // 64 8 8 (6 = 3+3) --> A/B
    // 64 32 32 (6 < 5 + 5) --> A/B
    // 32 32 32 (5 < 5 + 5) --  A == B == C
    // 16 32 32 (4 < 5 + 5) --  C
    // 32 16 32 (5 < 4 + 5) --  A
    // 32 32 16 (5 < 4 + 5) --  B

    // Conclusion: Take A/B when spin > std::max(chiL,chiR)
    using cplx   = std::complex<double>;
    using real   = double;
    using Scalar = cplx;
    Eigen::Tensor<Scalar, 4> envL(chiL, chiL, mpod, mpod);
    Eigen::Tensor<Scalar, 4> envR(chiR, chiR, mpod, mpod);
    Eigen::Tensor<Scalar, 4> mpo(mpod, mpod, spin, spin);
    Eigen::Tensor<Scalar, 3> psi(spin, chiL, chiR);
    envL.setRandom();
    envR.setRandom();
    mpo.setRandom();
    psi.setRandom();
    tools::prof::t_total->tic();

    for(int iter = 0; iter < iters; iter++) {
        auto psi_out = contract::hamiltonian_squared_dot_psi_v1(psi, mpo, envL, envR);
        tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_v1->get_name(), psi_out.dimensions(), iter, iters,
                         tools::prof::t_ham_sq_psi_v1->get_last_time_interval());
    }
    tools::log->info("{} | psi dimensions {} | total time {:.4f} s", tools::prof::t_ham_sq_psi_v1->get_name(), psi.dimensions(),
                     tools::prof::t_ham_sq_psi_v1->get_measured_time());

    for(int iter = 0; iter < iters; iter++) {
        auto psi_out = contract::hamiltonian_squared_dot_psi_v2(psi, mpo, envL, envR);
        tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_v2->get_name(), psi_out.dimensions(), iter, iters,
                         tools::prof::t_ham_sq_psi_v2->get_last_time_interval());
    }
    tools::log->info("{} | psi dimensions {} | total time {:.4f} s", tools::prof::t_ham_sq_psi_v2->get_name(), psi.dimensions(),
                     tools::prof::t_ham_sq_psi_v2->get_measured_time());

    for(int iter = 0; iter < iters; iter++) {
        auto psi_out = contract::hamiltonian_squared_dot_psi_v3(psi, mpo, envL, envR);
        tools::log->info("{} | psi dimensions {} | iter {}/{} |  time {:.4f} s", tools::prof::t_ham_sq_psi_v3->get_name(), psi_out.dimensions(), iter, iters,
                         tools::prof::t_ham_sq_psi_v3->get_last_time_interval());
    }
    tools::log->info("{} | psi dimensions {} | total time {:.4f} s", tools::prof::t_ham_sq_psi_v3->get_name(), psi.dimensions(),
                     tools::prof::t_ham_sq_psi_v3->get_measured_time());

    tools::prof::t_total->toc();
    tools::log->info("total time {:.4f} s", tools::prof::t_total->get_measured_time());
}