

#if !defined(EIGEN_USE_GPU)
#include <contract/contract.h>
#include <complex>
#include <tools/prof.h>
#include <tools/class_tic_toc.h>


using Eigen::Tensor;
typedef Tensor<float, 1>::DimensionPair DimPair;

template<int DataLayout>
void test_cuda_contraction(int m_size, int k_size, int n_size)
{
    std::cout << "Testing for (" << m_size << "," << k_size << "," << n_size << ")" << std::endl;
    // with these dimensions, the output has 300 * 140 elements, which is
    // more than 30 * 1024, which is the number of threads in blocks on
    // a 15 SM GK110 GPU
    Tensor<float, 2, DataLayout> t_left(m_size, k_size);
    Tensor<float, 2, DataLayout> t_right(k_size, n_size);
    Tensor<float, 2, DataLayout> t_result(m_size, n_size);
    Tensor<float, 2, DataLayout> t_result_gpu(m_size, n_size);
    Eigen::array<DimPair, 1> dims(DimPair(1, 0));

    t_left.setRandom();
    t_right.setRandom();

    std::size_t t_left_bytes = t_left.size()  * sizeof(float);
    std::size_t t_right_bytes = t_right.size() * sizeof(float);
    std::size_t t_result_bytes = t_result.size() * sizeof(float);

    float* d_t_left;
    float* d_t_right;
    float* d_t_result;

    cudaMalloc((void**)(&d_t_left), t_left_bytes);
    cudaMalloc((void**)(&d_t_right), t_right_bytes);
    cudaMalloc((void**)(&d_t_result), t_result_bytes);

    cudaMemcpy(d_t_left, t_left.data(), t_left_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_right, t_right.data(), t_right_bytes, cudaMemcpyHostToDevice);

    Eigen::CudaStreamDevice stream;
    Eigen::GpuDevice gpu_device(&stream);

    Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> >
        gpu_t_left(d_t_left, Eigen::array<int, 2>(m_size, k_size));
    Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> >
        gpu_t_right(d_t_right, Eigen::array<int, 2>(k_size, n_size));
    Eigen::TensorMap<Eigen::Tensor<float, 2, DataLayout> >
        gpu_t_result(d_t_result, Eigen::array<int, 2>(m_size, n_size));


    gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);
    t_result = t_left.contract(t_right, dims);

    cudaMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, cudaMemcpyDeviceToHost);
    for (DenseIndex i = 0; i < t_result.size(); i++) {
        if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
            continue;
        }
        if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
            continue;
        }
        std::cout << "mismatch detected at index " << i << ": " << t_result(i)
                  << " vs " <<  t_result_gpu(i) << std::endl;
        assert(false);
    }

    cudaFree((void*)d_t_left);
    cudaFree((void*)d_t_right);
    cudaFree((void*)d_t_result);
}




//template<typename Scalar>
//Eigen::Tensor<Scalar,3> contract::hamiltonian_squared_dot_psi_cuda(const Eigen::Tensor<Scalar,3> & psi_in, const Eigen::Tensor<Scalar,4> & mpo, const Eigen::Tensor<Scalar,4> & envL, const Eigen::Tensor<Scalar,4> & envR){
//    // https://svn.larosterna.com/oss/thirdparty/eigen/unsupported/test/cxx11_tensor_reduction_cuda.cu
//    Eigen::CudaStreamDevice stream;
//    Eigen::GpuDevice        gpudev(&stream);
//    tools::prof::t_ham_sq_psi_cuda->tic();
//    Eigen::DSizes<long,3> dsizes = psi_in.dimensions();
//    Eigen::Tensor<Scalar,3> ham_sq_psi(dsizes);
//    Eigen::Tensor<Scalar,3> psi_shuffled = psi_in.shuffle(Textra::array3{1, 0, 2});
//    ham_sq_psi.device(gpudev) =
//        psi_shuffled
//            .contract(envL , Textra::idx({0}, {0}))
//            .contract(mpo  , Textra::idx({0, 3}, {2, 0}))
//            .contract(mpo  , Textra::idx({4, 2}, {2, 0}))
//            .contract(envR, Textra::idx({0, 2, 3}, {0, 2, 3}))
//            .shuffle(Textra::array3{1, 0, 2});
//    tools::prof::t_ham_sq_psi_cuda->toc();
//    return ham_sq_psi;
//
//}
//
//using cplx = std::complex<double>;
//using real = double;
//
//template Eigen::Tensor<cplx,3> contract::hamiltonian_squared_dot_psi_cuda(const Eigen::Tensor<cplx,3> & psi_in, const Eigen::Tensor<cplx,4> & mpo, const Eigen::Tensor<cplx,4> & envL, const Eigen::Tensor<cplx,4> & envR);
//template Eigen::Tensor<real,3> contract::hamiltonian_squared_dot_psi_cuda(const Eigen::Tensor<real,3> & psi_in, const Eigen::Tensor<real,4> & mpo, const Eigen::Tensor<real,4> & envL, const Eigen::Tensor<real,4> & envR);
#endif