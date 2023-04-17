#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
namespace h5pp {
    class File;
}
enum class tb_mode;
enum class tb_type;

template<typename T>
struct tb_setup {
    tb_mode             mode;
    tb_type             type;
    int                 nomp = 0;
    int                 nmpi = 0;
    Eigen::Tensor<T, 3> envL, envR, psi;
    Eigen::Tensor<T, 4> mpo;
    size_t              iters;
    std::string         dsetname;

    mutable std::vector<Eigen::Tensor<T, 3>> psi_check;
    tb_setup(tb_mode mode, tb_type type, int nomp, int nmpi, long spin, long chi, long chiL, long chiR, long mpoD, size_t iters);
    std::string string() const;
};

namespace benchmark {
    template<typename T>
    using ResultType = Eigen::Tensor<T, 3>;

    using cplx = std::complex<double>;
    using fp32 = float;
    using fp64 = double;

    template<typename T>
    extern void iterate_benchmarks();

    template<typename T>
    extern void run_benchmark(const tb_setup<T> &tbs);

    template<typename T>
    [[nodiscard]] ResultType<T> tensor_product_eigen1(const tb_setup<T> &tbs);

    template<typename T>
    [[nodiscard]] ResultType<T> tensor_product_eigen2(const tb_setup<T> &tbs);

    template<typename T>
    [[nodiscard]] ResultType<T> tensor_product_eigen3(const tb_setup<T> &tbs);

    template<typename T>
    [[nodiscard]] ResultType<T> tensor_product_cute(const tb_setup<T> &tbs);

    template<typename T>
    [[nodiscard]] ResultType<T> tensor_product_xtensor(const tb_setup<T> &tbs);

    template<typename T>
    [[nodiscard]] ResultType<T> tensor_product_tblis(const tb_setup<T> &tbs);

    template<typename T>
    [[nodiscard]] ResultType<T> tensor_product_cyclops(const tb_setup<T> &tbs);
}
