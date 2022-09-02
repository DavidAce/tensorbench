#pragma once
#include "math/tenx.h"

namespace contract {
    template<typename Scalar>
    using ResultType = std::pair<Eigen::Tensor<Scalar, 3>, long>;
    
    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> tensor_product_eigen1(const Eigen::Tensor<Scalar, 3> &psi_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                          const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);

    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> tensor_product_eigen2(const Eigen::Tensor<Scalar, 3> &theta_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                          const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);

    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> tensor_product_eigen3(const Eigen::Tensor<Scalar, 3> &theta_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                          const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);

    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> tensor_product_cute(const Eigen::Tensor<Scalar, 3> &theta_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                            const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);
    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> tensor_product_xtensor(const Eigen::Tensor<Scalar, 3> &theta_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);

    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> tensor_product_tblis(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                 const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);

    template<typename Scalar>
    [[nodiscard]] ResultType<Scalar> tensor_product_cyclops(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 4> &mpo,
                                                          const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR);
}