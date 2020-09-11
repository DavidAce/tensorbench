//
// Created by david on 2019-10-06.
//

#pragma once

#ifdef _OPENMP
    #include <omp.h>
    #include <thread>
#endif
#define EIGEN_USE_GPU

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace Textra::omp {
#if defined(EIGEN_USE_GPU)
    // Shown here
    // https://svn.larosterna.com/oss/thirdparty/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu
    inline std::unique_ptr<Eigen::GpuDevice>        dev;
    inline std::unique_ptr<Eigen::CudaStreamDevice> stream;
    inline void                                     setNumThreads_(int num_threads) {
        if(not stream) stream  = std::make_unique<Eigen::CudaStreamDevice>();
        if(not dev and stream) dev = std::make_unique<Eigen::GpuDevice>(stream.get());
    }
#elif defined(_OPENMP) && defined(EIGEN_USE_THREADS)
    inline std::unique_ptr<Eigen::ThreadPool>       tp;
    inline std::unique_ptr<Eigen::ThreadPoolDevice> dev;
    inline void                                     setNumThreads(int num_threads) {
        if(not tp) tp = std::make_unique<Eigen::ThreadPool>(num_threads);
        if(not dev and tp) dev = std::make_unique<Eigen::ThreadPoolDevice>(tp.get(), num_threads);
    }



#else
    inline std::unique_ptr<Eigen::DefaultDevice> dev = std::make_unique<Eigen::DefaultDevice>() ;
    inline void setNumThreads([[maybe_unused]] int num_threads) {
        if(not dev) dev = std::make_unique<Eigen::DefaultDevice>();
    }
#endif

}
