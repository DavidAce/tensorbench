
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
#include "tblis/util/configs.h"
#include "tblis/util/thread.h"
#include "tid/tid.h"
#include "tools/log.h"
#include "tools/prof.h"
#include <env/environment.h>
#include <omp.h>

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
    //    for(const auto &t : tid::get_tree("main", tid::level::normal)) tools::log->info("{}", t.str());
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
    auto t_main = tid::tic_scope("mains");
    config::parse(argc, argv);
    tools::log = tools::Logger::setLogger("tensorbench", config::loglevel);
    tools::log->info("Hostname        : {}", debug::hostname());
    tools::log->info("Build hostname  : {}", env::build::hostname);
    tools::log->info("Git branch      : {}", env::git::branch);
    tools::log->info("    commit hash : {}", env::git::commit_hash);
    tools::log->info("    revision    : {}", env::git::revision);

    // Initialize MPI if this benchmark was run with mpirun
    mpi::init(argc, argv);

    // Register termination codes and what to do on exit
    debug::register_callbacks();
    if(mpi::world.id == 0) {
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

#if defined(TB_CUDA) && __has_include(<cuda.h>)
    tools::log->info("Initializing CUDA");
    auto init_result = cudaSetDevice(0);
    //    auto init_result = cuInit(0);
    if(init_result != 0) throw std::runtime_error("cuInit returned " + std::to_string(init_result));
#endif

    benchmark::iterate_benchmarks<benchmark::fp32>();
    benchmark::iterate_benchmarks<benchmark::fp64>();
    benchmark::iterate_benchmarks<benchmark::cplx>();
}