#if defined(TB_CPU1)

    #include <complex>
    #include <contract/contract.h>
    #include <tools/class_tic_toc.h>
    #include <tools/prof.h>

long get_ops_v1_m(long d, long chiL, long chiR, long m) {
    // m first
    long step1   = chiL * chiL * chiR * m * m * d;
    long step2_m = chiL * chiR * d * d * m * m * (d * m + 1);
    long step3   = chiL * chiR * d * m * m * m * (chiR * m + 1);
    long step4_m = chiL * chiR * d * d * (d * m * m * m + d * m + 1);
    return step1 + step2_m + step3 + step4_m;
}

long get_ops_v1_d(long d, long chiL, long chiR, long m) {
    // m first
    long step1   = chiL * chiL * chiR * m * m * d;
    long step2_d = chiL * chiR * d * m * m * m * (d * m + 1);
    long step3   = chiL * chiR * d * m * m * m * (chiR * m + 1);
    long step4_d = chiL * chiR * d * m * (d * m * m * m + m * m + 1);
    return step1 + step2_d + step3 + step4_d;
}

template<typename Scalar>
contract::ResultType<Scalar> contract::hamiltonian_squared_dot_psi_v1(const Eigen::Tensor<Scalar, 3> &psi_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                      const Eigen::Tensor<Scalar, 4> &envL, const Eigen::Tensor<Scalar, 4> &envR,
                                                                      std::string_view firstLeg) {
    tools::prof::t_ham_sq_psi_v1->tic();
    Eigen::DSizes<long, 3>   dsizes = psi_in.dimensions();
    Eigen::Tensor<Scalar, 3> ham_sq_psi(dsizes);

    if(firstLeg == "m") {
        Eigen::Tensor<Scalar, 3> psi_shuffled = psi_in.shuffle(Textra::array3{1, 0, 2});
        ham_sq_psi.device(*Textra::omp::dev)  = psi_shuffled.contract(envL, Textra::idx({0}, {0}))
                                                   .contract(mpo, Textra::idx({3, 0}, {0, 2}))
                                                   .contract(envR, Textra::idx({0, 3}, {0, 2}))
                                                   .contract(mpo, Textra::idx({1, 4, 2}, {0, 1, 2}))
                                                   .shuffle(Textra::array3{2, 0, 1});
        tools::prof::t_ham_sq_psi_v1->toc();
        return std::make_pair(ham_sq_psi, get_ops_v1_m(dsizes[0], dsizes[1], dsizes[2], mpo.dimension(0)));
    }
    if(firstLeg == "d") {
        Eigen::Tensor<Scalar, 3> psi_shuffled = psi_in.shuffle(Textra::array3{1, 0, 2});
        ham_sq_psi.device(*Textra::omp::dev)  = psi_shuffled.contract(envL, Textra::idx({0}, {0}))
                                                   .contract(mpo, Textra::idx({0, 3}, {2, 0}))
                                                   .contract(envR, Textra::idx({0, 3}, {0, 2}))
                                                   .contract(mpo, Textra::idx({2, 1, 4}, {2, 0, 1}))
                                                   .shuffle(Textra::array3{2, 0, 1});
        tools::prof::t_ham_sq_psi_v1->toc();
        return std::make_pair(ham_sq_psi, get_ops_v1_d(dsizes[0], dsizes[1], dsizes[2], mpo.dimension(0)));
    }
}

using cplx = std::complex<double>;
using fp32 = float;
using fp64 = double;

// template contract::ResultType<cplx> contract::hamiltonian_squared_dot_psi_v1(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
//                                                                           const Eigen::Tensor<cplx, 4> &envL, const Eigen::Tensor<cplx, 4> &envR,
//                                                                           std::string_view firstLeg);
// template contract::ResultType<fp32> contract::hamiltonian_squared_dot_psi_v1(const Eigen::Tensor<fp32, 3> &psi_in, const Eigen::Tensor<fp32, 4> &mpo,
//                                                                           const Eigen::Tensor<fp32, 4> &envL, const Eigen::Tensor<fp32, 4> &envR,
//                                                                           std::string_view firstLeg);
template contract::ResultType<fp64> contract::hamiltonian_squared_dot_psi_v1(const Eigen::Tensor<fp64, 3> &psi_in, const Eigen::Tensor<fp64, 4> &mpo,
                                                                             const Eigen::Tensor<fp64, 4> &envL, const Eigen::Tensor<fp64, 4> &envR,
                                                                             std::string_view firstLeg);

#endif