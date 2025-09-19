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
    if(init_result != 0) throw std::runtime_error("cudaSetDevice returned " + std::to_string(init_result) + ": " + cudaGetErrorString(init_result));
    #endif
}

std::string config::getGpuName(int deviceNumber) {
    #if defined(TB_CUDA)
    if(deviceNumber < 0) return {};
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if(deviceNumber + 1 > nDevices) tools::log->warn("requested cuda device number {} out of bounds | detected {} devices", deviceNumber, nDevices);
    if(nDevices >= 1) {
        deviceNumber        = std::clamp<int>(deviceNumber, 0, nDevices - 1);
        cudaDeviceProp prop = {};
        cudaGetDeviceProperties(&prop, deviceNumber);
        return prop.name;
    } else
        return {};
    #else
    return {};
    #endif
}

void config::showGpuInfo() {
    #if defined(TB_CUDA)
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("GPU: Number of devices: %d\n", nDevices);
    if(nDevices < 0) return;

    for(int i = 0; i < std::min(8, nDevices); i++) {
        cudaDeviceProp prop = {};
        cudaGetDeviceProperties(&prop, i);

        int memoryClockRate = -1;
        int deviceOverlap   = -1;
        cudaDeviceGetAttribute(&memoryClockRate, cudaDeviceAttr::cudaDevAttrMemoryClockRate, i);
        cudaDeviceGetAttribute(&deviceOverlap, cudaDeviceAttr::cudaDevAttrGpuOverlap, i);

        printf("     Device Number: %d\n", i);
        printf("       Device name: %s\n", prop.name);
        printf("       Memory Clock Rate (MHz): %d\n", memoryClockRate / 1024);
        printf("       Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("       Peak Memory Bandwidth (GB/s): %.1f\n", 2.0 * memoryClockRate * (static_cast<double>(prop.memoryBusWidth) / 8) / 1.0e6);
        printf("       Total global memory (Gbytes) %.1f\n", (float) (prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
        printf("       Shared memory per block (Kbytes) %.1f\n", (float) (prop.sharedMemPerBlock) / 1024.0);
        printf("       minor-major: %d-%d\n", prop.minor, prop.major);
        printf("       Warp-size: %d\n", prop.warpSize);
        printf("       Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("       Concurrent computation/communication: %s\n\n", deviceOverlap ? "yes" : "no");
    }
    #endif
}