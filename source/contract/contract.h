#pragma once
#include <general/nmspc_tensor_extra.h>

namespace contract{

    template<typename Scalar>
    [[nodiscard]] Eigen::Tensor<Scalar,3> hamiltonian_squared_dot_psi_v1(const Eigen::Tensor<Scalar,3> & theta_in, const Eigen::Tensor<Scalar,4> & mpo, const Eigen::Tensor<Scalar,4> & env2L, const Eigen::Tensor<Scalar,4> & env2R);

    template<typename Scalar>
    [[nodiscard]] Eigen::Tensor<Scalar,3> hamiltonian_squared_dot_psi_v2(const Eigen::Tensor<Scalar,3> & theta_in, const Eigen::Tensor<Scalar,4> & mpo, const Eigen::Tensor<Scalar,4> & env2L, const Eigen::Tensor<Scalar,4> & env2R);

    template<typename Scalar>
    [[nodiscard]] Eigen::Tensor<Scalar,3> hamiltonian_squared_dot_psi_v3(const Eigen::Tensor<Scalar,3> & theta_in, const Eigen::Tensor<Scalar,4> & mpo, const Eigen::Tensor<Scalar,4> & env2L, const Eigen::Tensor<Scalar,4> & env2R);

    template<typename Scalar>
    [[nodiscard]] Eigen::Tensor<Scalar,3> hamiltonian_squared_dot_psi_cuda(const Eigen::Tensor<Scalar,3> & theta_in, const Eigen::Tensor<Scalar,4> & mpo, const Eigen::Tensor<Scalar,4> & env2L, const Eigen::Tensor<Scalar,4> & env2R);

}