#if defined(TB_CUDA)
    #include "config.h"
    #include "tools/log.h"
    #include <cuda_runtime_api.h>
    #include <string>
#endif

void config::initializeCuda(int gpun) {
#if defined(TB_CUDA)
    tools::log->info("Initializing CUDA device {}", gpun);
    auto init_result = cudaSetDevice(gpun);
    if(init_result != 0) throw std::runtime_error("cuInit returned " + std::to_string(init_result));
#endif
}

void config::showGpuInfo() {
#if defined(TB_CUDA)
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("GPU: Number of devices: %d\n", nDevices);

    for(int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop = {};
        cudaGetDeviceProperties(&prop, i);
        printf("     Device Number: %d\n", i);
        printf("       Device name: %s\n", prop.name);
        printf("       Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1024);
        printf("       Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("       Peak Memory Bandwidth (GB/s): %.1f\n", 2.0 * prop.memoryClockRate * (static_cast<double>(prop.memoryBusWidth) / 8) / 1.0e6);
        printf("       Total global memory (Gbytes) %.1f\n", (float) (prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
        printf("       Shared memory per block (Kbytes) %.1f\n", (float) (prop.sharedMemPerBlock) / 1024.0);
        printf("       minor-major: %d-%d\n", prop.minor, prop.major);
        printf("       Warp-size: %d\n", prop.warpSize);
        printf("       Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("       Concurrent computation/communication: %s\n\n", prop.deviceOverlap ? "yes" : "no");
    }
#endif
}