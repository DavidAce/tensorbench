

#if defined(TB_CUTE)
//    #include <complex>
    #include <contract/contract.h>
    #include <tools/class_tic_toc.h>
    #include <tools/log.h>
    #include <tools/prof.h>
    #include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>


template<typename Scalar>
Eigen::Tensor<Scalar, 3> contract::hamiltonian_squared_dot_psi_cute(const Eigen::Tensor<Scalar, 3> &psi_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                    const Eigen::Tensor<Scalar, 4> &envL, const Eigen::Tensor<Scalar, 4> &envR) {
    tools::prof::t_ham_sq_psi_cute->tic();
    Eigen::DSizes<long, 3>   dsizes = psi_in.dimensions();
    Eigen::Tensor<Scalar, 3> ham_sq_psi(dsizes);
    Eigen::Tensor<Scalar, 3> psi_shuffled = psi_in.shuffle(Textra::array3{1, 0, 2});








    tools::prof::t_ham_sq_psi_cute->toc();
    return ham_sq_psi;
}

using cplx = std::complex<double>;
using fp32 = float;
using fp64 = double;

//template Eigen::Tensor<cplx, 3> contract::hamiltonian_squared_dot_psi_cute(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
//                                                                           const Eigen::Tensor<cplx, 4> &envL, const Eigen::Tensor<cplx, 4> &envR);
template Eigen::Tensor<fp32, 3> contract::hamiltonian_squared_dot_psi_cute(const Eigen::Tensor<fp32, 3> &psi_in, const Eigen::Tensor<fp32, 4> &mpo,
                                                                           const Eigen::Tensor<fp32, 4> &envL, const Eigen::Tensor<fp32, 4> &envR);
template Eigen::Tensor<fp64, 3> contract::hamiltonian_squared_dot_psi_cute(const Eigen::Tensor<fp64, 3> &psi_in, const Eigen::Tensor<fp64, 4> &mpo,
                                                                           const Eigen::Tensor<fp64, 4> &envL, const Eigen::Tensor<fp64, 4> &envR);
#endif