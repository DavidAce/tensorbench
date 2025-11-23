
#include "benchmark/benchmark.h"
#include "config/config.h"
#include "debug/exceptions.h"
#include "debug/info.h"
#include "debug/stacktrace.h"
#include "general/enums.h"
#include "math/stat.h"
#include "math/tenx.h"
#include "mpi/mpi-tools.h"
#include "storage/results.h"
#include "tid/tid.h"
#include "tools/log.h"
#include "tools/prof.h"
#include <env/environment.h>
#include <omp.h>

#if defined(FLEXIBLAS_AVAILABLE)
    #include <flexiblas/flexiblas_api.h>
#endif

#if defined(TB_OPENBLAS)
    #include <openblas/cblas.h>
    #include <openblas/openblas_config.h>
#endif

#if defined(TB_MKL)
    #define MKL_Complex8  std::complex<float>
    #define MKL_Complex16 std::complex<double>
    #include <mkl.h>
    #include <mkl_service.h>
#endif

#if defined(TB_CUDA) && __has_include(<cuda.h>)
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif

void print_timers() {
    for(const auto &t : tid::get_tree("", tid::level::detailed)) tools::log->info("{}", t.str());
}

/*
 * Resources explaining MPI/OpenMP Hybrid parameters
 *
 * https://www.admin-magazine.com/HPC/Articles/Processor-Affinity-for-OpenMP-and-MPI
 *
 * These mpirun flags can be used:
 * -n 1 --report-bindings --map-by --map-by ppr:1:l3cache:pe=8 --bind-to socket -x OMP_PLACES=sockets -x OMP_PROC_BIND=close -x OMP_NUM_THREADS=4 -x
 * OMP_DISPLAY_AFFINITY=true ...
 *
 * From https://www.ibm.com/docs/en/smpi/10.2?topic=modifiers-map-by-pprnunit-map-by-pprnunitpen-options
 * The purpose of ppr:n:socket option is to launch n ranks on each socket.
 * The purpose of the ppr:n:socket:pe=m option is to launch n ranks per socket, with each rank using m cores.
 *
 *
 * Despite the fact that Eigen::Tensor uses C++11 threads, it will behave differently depending on the environment variable OMP_PLACES and OMP_BIND_PROCS.
 * OMP_PLACES: cores will cause all eigen threads to confined to a single core!
 * OMP_PLACES: sockets will allow the threads to spread out over the entire socket. This is the only option that really works.
 * OMP_PROC_BIND=close will put the threads onto the same socket.
 */

int main(int argc, char *argv[]) {
    auto t_main = tid::tic_scope("main");
    config::parse(argc, argv);
    tools::log = tools::Logger::setLogger("tensorbench", config::loglevel);

#if defined(_OPENMP) and defined(EIGEN_USE_THREADS)
    bool has_eigen1 = std::find(config::tb_modes.begin(), config::tb_modes.end(), tb_mode::eigen1) != config::tb_modes.end();
    bool has_eigen2 = std::find(config::tb_modes.begin(), config::tb_modes.end(), tb_mode::eigen2) != config::tb_modes.end();
    bool has_eigen3 = std::find(config::tb_modes.begin(), config::tb_modes.end(), tb_mode::eigen3) != config::tb_modes.end();

    if(has_eigen1 or has_eigen2 or has_eigen3) {
        auto get_omp_proc_bind = []() -> std::string {
            switch(omp_get_proc_bind()) {
                case 0: return "false";
                case 1: return "true";
                case 2: return "primary";
                case 3: return "close";
                case 4: return "spread";
                default: return "unknown";
            }
        };
        if(auto omp_proc_bind = get_omp_proc_bind(); omp_proc_bind != "false") {
            throw except::runtime_error("\n \t Detected OMP_PROC_BIND: {}.\n"
                                        "\t OpenMP core pinning interacts poorly with std::thread in Eigen::Tensor when EIGEN_USE_THREADS is defined.\n"
                                        "\t Please unset environment variables OMP_PROC_BIND and OMP_PLACES, or unset preprocessor variable EIGEN_USE_THREADS",
                                        omp_proc_bind);
        }
    }
#endif
    // Initialize MPI if this benchmark was run with mpirun
    mpi::init(argc, argv);

    // Register termination codes and what to do on exit
    debug::register_callbacks();
    if(mpi::world.id == 0) {
        tools::log->info("Hostname        : {}", debug::hostname());
        tools::log->info("Build hostname  : {}", env::build::hostname);
        tools::log->info("Git branch      : {}", env::git::branch);
        tools::log->info("    commit hash : {}", env::git::commit_hash);
        tools::log->info("    revision    : {}", env::git::revision);
        #if defined(FLEXIBLAS_AVAILABLE)
        char buffer[32] = {0};
        int  size       = flexiblas_current_backend(buffer, 32);
        if(size > 0) {
            tools::log->info("Flexiblas backend [{}] | num_threads {}", buffer, flexiblas_get_num_threads());
        } else {
            tools::log->info("Flexiblas backend read failed: size {}", size);
        }

        #endif
        config::showCpuName();

        bool has_cutensor = std::find(config::tb_modes.begin(), config::tb_modes.end(), tb_mode::cutensor) != config::tb_modes.end();
        bool has_matx     = std::find(config::tb_modes.begin(), config::tb_modes.end(), tb_mode::matx) != config::tb_modes.end();

        if(has_cutensor or has_matx) config::showGpuInfo();


        std::atexit(debug::print_mem_usage);
        std::atexit(print_timers);
        std::at_quick_exit(debug::print_mem_usage);
        std::at_quick_exit(print_timers);

    }

    for(int id = 0; id < mpi::world.size; ++id) {
        if(id == mpi::world.id) {
#pragma omp parallel
            {
                tools::log->debug("Hello from mpi rank {} of {} | omp thread {} of {} | on thread {}", mpi::world.id, mpi::world.size, omp_get_thread_num(),
                                  omp_get_max_threads(), sched_getcpu());
            }
        }
        tools::log->flush();
        mpi::barrier();
    }
    mpi::barrier();
    benchmark::iterate_benchmarks();
}