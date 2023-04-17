
#include "benchmark.h"

template<typename T>
benchmark::ResultType<T> benchmark::tensor_product_cute([[maybe_unused]] const tb_setup<T> &tbs) {
    return {};
}

using fp32 = benchmark::fp32;
using fp64 = benchmark::fp64;
using cplx = benchmark::cplx;

template benchmark::ResultType<fp32> benchmark::tensor_product_cute(const tb_setup<fp32> &tbs);
template benchmark::ResultType<fp64> benchmark::tensor_product_cute(const tb_setup<fp64> &tbs);
template benchmark::ResultType<cplx> benchmark::tensor_product_cute(const tb_setup<cplx> &tbs);