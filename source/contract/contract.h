#pragma once
#include <general/nmspc_tensor_extra.h>

namespace contract {
    template<typename Scalar>
    using ResultType = std::pair<Eigen::Tensor<Scalar, 3>, long>;
    
    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> hamiltonian_squared_dot_psi_v1(const Eigen::Tensor<Scalar, 3> &theta_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                          const Eigen::Tensor<Scalar, 4> &env2L, const Eigen::Tensor<Scalar, 4> &env2R, std::string_view leg = "m");

    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> hamiltonian_squared_dot_psi_v2(const Eigen::Tensor<Scalar, 3> &theta_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                          const Eigen::Tensor<Scalar, 4> &env2L, const Eigen::Tensor<Scalar, 4> &env2R, std::string_view leg = "m");

    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> hamiltonian_squared_dot_psi_v3(const Eigen::Tensor<Scalar, 3> &theta_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                          const Eigen::Tensor<Scalar, 4> &env2L, const Eigen::Tensor<Scalar, 4> &env2R, std::string_view leg = "m");

    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> hamiltonian_squared_dot_psi_cuda(const Eigen::Tensor<Scalar, 3> &theta_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                            const Eigen::Tensor<Scalar, 4> &env2L, const Eigen::Tensor<Scalar, 4> &env2R);

    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> hamiltonian_squared_dot_psi_acro(const Eigen::Tensor<Scalar, 3> &theta_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                            const Eigen::Tensor<Scalar, 4> &env2L, const Eigen::Tensor<Scalar, 4> &env2R);

    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> hamiltonian_squared_dot_psi_cute(const Eigen::Tensor<Scalar, 3> &theta_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                            const Eigen::Tensor<Scalar, 4> &env2L, const Eigen::Tensor<Scalar, 4> &env2R);

}