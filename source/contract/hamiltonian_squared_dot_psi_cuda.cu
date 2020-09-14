

#if defined(EIGEN_USE_GPU)
//    #include <complex>
    #include <contract/contract.h>
    #include <tools/class_tic_toc.h>
    #include <tools/prof.h>
    #include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

template<typename Scalar, auto rank, auto rankL,auto rankR,auto rankC>
Eigen::Tensor<Scalar, rank> custom_contract (const Eigen::Tensor<Scalar,rankL> & lft, const Eigen::Tensor<Scalar,rankR> & rgt, const Textra::idxlistpair<Eigen::Index,rankC> & idx){
    static_assert(rank == rankL+rankR-2*rankC and "Resulting rank does not match input ranks" );
//    constexpr Eigen::Index res_rank = rankL+rankR-2*rankC;
    Eigen::DSizes<Eigen::Index,rank> res_dims;
    auto dimsL = lft.dimensions();
    auto dimsR = rgt.dimensions();
    size_t res_count = 0;
    for(size_t i = 0; i < dimsL.size(); i++) if(i == idx[i].first) dimsL[i] = -1;
    for(size_t i = 0; i < dimsR.size(); i++) if(i == idx[i].second) dimsR[i] = -1;
    for(size_t i = 0; i < dimsL.size(); i++) if(dimsL[i] != -1) res_dims[res_count++] = dimsL[i];
    for(size_t i = 0; i < dimsR.size(); i++) if(dimsR[i] != -1) res_dims[res_count++] = dimsR[i];
    Eigen::Tensor<Scalar,rank> res(res_dims);

    std::size_t res_bytes         = res.size()   * sizeof(Scalar);
    std::size_t lft_bytes         = lft.size() * sizeof(Scalar);
    std::size_t rgt_bytes         = rgt.size() * sizeof(Scalar);
    Scalar * d_res;
    Scalar * d_lft;
    Scalar * d_rgt;
    gpuMalloc((void **) (&d_res), res_bytes);
    gpuMalloc((void **) (&d_lft), lft_bytes);
    gpuMalloc((void **) (&d_rgt), rgt_bytes);
    gpuMemcpy(d_lft, lft.data(), lft_bytes,gpuMemcpyHostToDevice);
    gpuMemcpy(d_rgt, rgt.data(), rgt_bytes,gpuMemcpyHostToDevice);
    Eigen::TensorMap<Eigen::Tensor<Scalar, rank> > gpu_res(d_res, res.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, rankL> > gpu_lft(d_lft, lft.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, rankR> > gpu_rgt(d_rgt, rgt.dimensions());


    // Setup cuda
#if EIGEN_VERSION_AT_LEAST(3,3,90)
    Eigen::GpuStreamDevice stream;
#else
    Eigen::CudaStreamDevice stream;
#endif
    // Fix for eigen using too much shared data
    //    https://gitlab.com/libeigen/eigen/-/issues/1212
    Eigen::GpuDevice        gpudev(&stream);
    gpu_res.device(gpudev) = gpu_lft.contract(gpu_rgt, idx);
    gpuMemcpy(res.data(), d_res, res_bytes,gpuMemcpyDeviceToHost);
    gpuFree((void *) d_res);
    gpuFree((void *) d_lft);
    gpuFree((void *) d_rgt);

    return res;
}



template<typename Scalar>
Eigen::Tensor<Scalar, 3> contract::hamiltonian_squared_dot_psi_cuda(const Eigen::Tensor<Scalar, 3> &psi_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                    const Eigen::Tensor<Scalar, 4> &envL, const Eigen::Tensor<Scalar, 4> &envR) {
    // https://svn.larosterna.com/oss/thirdparty/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu
    tools::prof::t_ham_sq_psi_cuda->tic();
    Eigen::DSizes<long, 3>   dsizes = psi_in.dimensions();
    Eigen::Tensor<Scalar, 3> ham_sq_psi(dsizes);
    Eigen::Tensor<Scalar, 3> psi_shuffled = psi_in.shuffle(Textra::array3{1, 0, 2});

//    auto psi_envL = custom_contract<Scalar,5,3,4,1>(psi_shuffled,envL,Textra::idx(Textra::array1{0}, Textra::array1{0}));
//    auto psi_envL_mpo = custom_contract<Scalar,5,5,4,2>(psi_envL,mpo, Textra::idx(Textra::array2{0, 3}, Textra::array2{2, 0}));
//    auto psi_envL_mpo_mpo = custom_contract<Scalar,5,5,4,2>(psi_envL_mpo,mpo,Textra::idx(Textra::array2{4, 2}, Textra::array2{2, 0}));
//    ham_sq_psi = custom_contract<Scalar,3,5,4,3>(psi_envL_mpo_mpo,envR,Textra::idx(Textra::array3{0, 2, 3}, Textra::array3{0, 2, 3}));

    // Setup cuda
#if EIGEN_VERSION_AT_LEAST(3,3,90)
    Eigen::GpuStreamDevice stream;
#else
    Eigen::CudaStreamDevice stream;
#endif
    Eigen::GpuDevice        gpudev(&stream);
    Scalar *    d_ham_sq_psi;
    Scalar *    d_psi_shuffled;
    Scalar *    d_mpo;
    Scalar *    d_envL;
    Scalar *    d_envR;
    std::size_t d_ham_sq_psi_bytes   = ham_sq_psi.size() * sizeof(Scalar);
    std::size_t d_psi_shuffled_bytes = psi_shuffled.size() * sizeof(Scalar);
    std::size_t d_mpo_bytes          = mpo.size() * sizeof(Scalar);
    std::size_t d_envL_bytes         = envL.size() * sizeof(Scalar);
    std::size_t d_envR_bytes         = envR.size() * sizeof(Scalar);




    cudaMalloc((void **) (&d_ham_sq_psi), d_ham_sq_psi_bytes);
    cudaMalloc((void **) (&d_psi_shuffled), d_psi_shuffled_bytes);
    cudaMalloc((void **) (&d_mpo), d_mpo_bytes);
    cudaMalloc((void **) (&d_envL), d_envL_bytes);
    cudaMalloc((void **) (&d_envR), d_envR_bytes);
    cudaMemcpy(d_ham_sq_psi, ham_sq_psi.data(), d_ham_sq_psi_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi_shuffled, psi_shuffled.data(), d_psi_shuffled_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mpo, mpo.data(), d_mpo_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_envL, envL.data(), d_envL_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_envR, envR.data(), d_envR_bytes,cudaMemcpyHostToDevice);

    Eigen::TensorMap<Eigen::Tensor<Scalar, 3> > gpu_ham_sq_psi(d_ham_sq_psi, dsizes);
    Eigen::TensorMap<Eigen::Tensor<Scalar, 3> > gpu_psi_shuffled(d_psi_shuffled, psi_shuffled.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, 4> > gpu_mpo(d_mpo, mpo.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, 4> > gpu_envL(d_envL, envL.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, 4> > gpu_envR(d_envR, envR.dimensions());

    gpu_ham_sq_psi.device(gpudev) = gpu_psi_shuffled.contract(gpu_envL, Textra::idx(Textra::array1{0}, Textra::array1{0}))
                                        .contract(gpu_mpo, Textra::idx(Textra::array2{0, 3}, Textra::array2{2, 0}))
                                        .contract(gpu_mpo, Textra::idx(Textra::array2{4, 2}, Textra::array2{2, 0}))
                                        .contract(gpu_envR, Textra::idx(Textra::array3{0, 2, 3}, Textra::array3{0, 2, 3}))
                                        .shuffle(Textra::array3{1, 0, 2});

    cudaMemcpy(ham_sq_psi.data(), d_ham_sq_psi, d_ham_sq_psi_bytes,cudaMemcpyDeviceToHost);
    cudaFree((void *) d_ham_sq_psi);
    cudaFree((void *) d_psi_shuffled);
    cudaFree((void *) d_mpo);
    cudaFree((void *) d_envL);
    cudaFree((void *) d_envR);

    tools::prof::t_ham_sq_psi_cuda->toc();
    return ham_sq_psi;
}

using cplx = std::complex<double>;
using real = double;

//template Eigen::Tensor<cplx, 3> contract::hamiltonian_squared_dot_psi_cuda(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
//                                                                           const Eigen::Tensor<cplx, 4> &envL, const Eigen::Tensor<cplx, 4> &envR);
template Eigen::Tensor<real, 3> contract::hamiltonian_squared_dot_psi_cuda(const Eigen::Tensor<real, 3> &psi_in, const Eigen::Tensor<real, 4> &mpo,
                                                                           const Eigen::Tensor<real, 4> &envL, const Eigen::Tensor<real, 4> &envR);
#endif