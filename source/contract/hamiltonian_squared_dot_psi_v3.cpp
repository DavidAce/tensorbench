#if defined(TB_CPU)

#include <contract/contract.h>
#include <complex>
#include <tools/prof.h>
#include <tools/class_tic_toc.h>

template<typename Scalar>
Eigen::Tensor<Scalar,3> contract::hamiltonian_squared_dot_psi_v3(const Eigen::Tensor<Scalar,3> & psi_in, const Eigen::Tensor<Scalar,4> & mpo, const Eigen::Tensor<Scalar,4> & envL, const Eigen::Tensor<Scalar,4> & envR){
    tools::prof::t_ham_sq_psi_v3->tic();
    Eigen::DSizes<long,3> dsizes = psi_in.dimensions();
    Eigen::Tensor<Scalar,3> ham_sq_psi(dsizes);
    Eigen::Tensor<Scalar,3> psi_shuffled = psi_in.shuffle(Textra::array3{1, 0, 2});
    ham_sq_psi.device(*Textra::omp::dev) =
        psi_shuffled
            .contract(envL , Textra::idx({0}, {0}))
            .contract(mpo  , Textra::idx({0, 3}, {2, 0}))
            .contract(mpo  , Textra::idx({4, 2}, {2, 0}))
            .contract(envR, Textra::idx({0, 2, 3}, {0, 2, 3}))
            .shuffle(Textra::array3{1, 0, 2});
    tools::prof::t_ham_sq_psi_v3->toc();
    return ham_sq_psi;

}

using cplx = std::complex<double>;
using fp32 = float;
using fp64 = double;

template Eigen::Tensor<cplx, 3> contract::hamiltonian_squared_dot_psi_v3(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
                                                                         const Eigen::Tensor<cplx, 4> &envL, const Eigen::Tensor<cplx, 4> &envR);
template Eigen::Tensor<fp32, 3> contract::hamiltonian_squared_dot_psi_v3(const Eigen::Tensor<fp32, 3> &psi_in, const Eigen::Tensor<fp32, 4> &mpo,
                                                                         const Eigen::Tensor<fp32, 4> &envL, const Eigen::Tensor<fp32, 4> &envR);
template Eigen::Tensor<fp64, 3> contract::hamiltonian_squared_dot_psi_v3(const Eigen::Tensor<fp64, 3> &psi_in, const Eigen::Tensor<fp64, 4> &mpo,
                                                                         const Eigen::Tensor<fp64, 4> &envL, const Eigen::Tensor<fp64, 4> &envR);
#endif