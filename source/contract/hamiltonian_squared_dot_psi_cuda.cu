

#if defined(EIGEN_USE_GPU)
    #include <complex>
    #include <contract/contract.h>
    #include <tools/class_tic_toc.h>
    #include <tools/prof.h>

template<typename Scalar>
Eigen::Tensor<Scalar, 3> contract::hamiltonian_squared_dot_psi_cuda(const Eigen::Tensor<Scalar, 3> &psi_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                    const Eigen::Tensor<Scalar, 4> &envL, const Eigen::Tensor<Scalar, 4> &envR) {
    // https://svn.larosterna.com/oss/thirdparty/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu
    tools::prof::t_ham_sq_psi_cuda->tic();
    Eigen::DSizes<long, 3>   dsizes = psi_in.dimensions();
    Eigen::Tensor<Scalar, 3> ham_sq_psi(dsizes);
    Eigen::Tensor<Scalar, 3> psi_shuffled = psi_in.shuffle(Textra::array3{1, 0, 2});

    // Setup cuda
    Eigen::GpuStreamDevice stream;
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

    cudaMemcpy(d_ham_sq_psi, ham_sq_psi.data(), d_ham_sq_psi_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_psi_shuffled, psi_shuffled.data(), d_psi_shuffled_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mpo, mpo.data(), d_mpo_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_envL, envL.data(), d_envL_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_envR, envR.data(), d_envR_bytes, cudaMemcpyHostToDevice);

    Eigen::TensorMap<Eigen::Tensor<Scalar, 3> > gpu_ham_sq_psi(d_ham_sq_psi, dsizes);
    Eigen::TensorMap<Eigen::Tensor<Scalar, 3> > gpu_psi_shuffled(d_psi_shuffled, dsizes);
    Eigen::TensorMap<Eigen::Tensor<Scalar, 4> > gpu_mpo(d_mpo, mpo.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, 4> > gpu_envL(d_envL, envL.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, 4> > gpu_envR(d_envR, envR.dimensions());

    gpu_ham_sq_psi.device(gpudev) = gpu_psi_shuffled.contract(gpu_envL, Textra::idx(Textra::array1{0}, Textra::array1{0}))
                                        .contract(gpu_mpo, Textra::idx(Textra::array2{0, 3}, Textra::array2{2, 0}))
                                        .contract(gpu_mpo, Textra::idx(Textra::array2{4, 2}, Textra::array2{2, 0}))
                                        .contract(gpu_envR, Textra::idx(Textra::array3{0, 2, 3}, Textra::array3{0, 2, 3}))
                                        .shuffle(Textra::array3{1, 0, 2});

    cudaMemcpy(ham_sq_psi.data(), d_ham_sq_psi, d_ham_sq_psi_bytes, cudaMemcpyDeviceToHost);
    cudaFree((void *) d_psi_shuffled_bytes);
    cudaFree((void *) d_mpo_bytes);
    cudaFree((void *) d_envL_bytes);
    cudaFree((void *) d_envR_bytes);

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