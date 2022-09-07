

#if defined(TB_CUTE)
    #include "contract.h"
    #include "tools/class_tic_toc.h"
    #include "tools/fmt.h"
    #include "tools/log.h"
    #include "tools/prof.h"
    #include <cstdio>
    #include <cstdlib>

    #include <cuda_runtime.h>
    #include <cutensor.h>

    #include <unordered_map>
    #include <vector>

    // Handle cuTENSOR errors
    #define HANDLE_ERROR(x)                                                                                                                                    \
        {                                                                                                                                                      \
            const auto err = x;                                                                                                                                \
            if(err != CUTENSOR_STATUS_SUCCESS) {                                                                                                               \
                tools::log->critical("{} in line {}", cutensorGetErrorString(err), __LINE__);                                                                  \
                exit(err);                                                                                                                                     \
            }                                                                                                                                                  \
        }

    #define HANDLE_CUDA_ERROR(x)                                                                                                                               \
        {                                                                                                                                                      \
            const auto err = x;                                                                                                                                \
            if(err != cudaSuccess) {                                                                                                                           \
                tools::log->critical("{} in line {}", cudaGetErrorString(err), __LINE__);                                                                      \
                exit(err);                                                                                                                                     \
            }                                                                                                                                                  \
        }

long get_ops_cute_L(long d, long chiL, long chiR, long m);
long get_ops_cute_R(long d, long chiL, long chiR, long m) {
    // Same as L, just swap chiL and chiR
    if(chiR > chiL) return get_ops_cute_L(d, chiR, chiL, m);
    else
        return get_ops_cute_L(d, chiL, chiR, m);
}

long get_ops_cute_L(long d, long chiL, long chiR, long m) {
    if(chiR > chiL) return get_ops_cute_R(d, chiL, chiR, m);
    if(d > m) {
        // d first
        long step1 = chiL * chiL * chiR * m * m * d;
        long step2 = chiL * chiR * d * m * m * m * (d * m + 1);
        long step3 = chiL * chiR * d * m * (chiR * m * m * m + m * m + 1);
        return step1 + step2 + step3;
    } else {
        // m first
        long step1 = chiL * chiL * chiR * m * d;
        long step2 = chiL * chiR * d * m * m * (d * m + 1);
        long step3 = chiL * chiR * chiR * d * m * (m + 1);
        return step1 + step2 + step3;
    }
}

template<typename Scalar>
class Meta {
    private:
    Scalar       *d_ptr = nullptr;
    const Scalar *h_ptr = nullptr;

    public:
    using value_type = Scalar;

    std::vector<int>     mode;
    std::vector<int64_t> extent;

    template<auto rank>
    Meta(const Eigen::Tensor<Scalar, rank> &tensor, const std::vector<int> &mode_) : h_ptr(tensor.data()), mode(mode_) {
        if(rank != mode.size()) throw std::runtime_error("Rank mismatch");
        for(size_t idx = 0; idx < rank; idx++) extent.push_back(tensor.dimension(idx));
    }

    Meta(const std::vector<int> &mode_, const std::vector<int64_t> &extent_) : mode(mode_), extent(extent_) {
        if(mode.size() != extent.size()) throw std::runtime_error("Mode and extent size mismatch");
    }

    ~Meta() {
        //        if(d_ptr) HANDLE_CUDA_ERROR(cudaFree(d_ptr));
    }
    auto size() {
        size_t size = 1;
        for(auto &ext : extent) size *= ext;
        return size;
    }

    size_t        byteSize() { return size() * sizeof(Scalar); }
    uint32_t      rank() { return mode.size(); }
    Scalar       *data_d() { return d_ptr; }
    const Scalar *data_h() { return h_ptr; }
    void          copyToDevice() {
//        size_t mf, ma;
//        cudaMemGetInfo(&mf, &ma);
//        tools::log->trace("CUDA: Free {} | Total {}", static_cast<double>(mf)/std::pow(1024,2), static_cast<double>(ma)/std::pow(1024,2));
        if(data_d() == nullptr) HANDLE_CUDA_ERROR(cudaMalloc((void **) &d_ptr, byteSize()));
        if(data_h() == nullptr) return; // Nothing to copy
        HANDLE_CUDA_ERROR(cudaMemcpy(data_d(), data_h(), byteSize(), cudaMemcpyHostToDevice));
    }

    void copyFromDevice(Scalar *data_h) {
        if(data_h == nullptr) throw std::runtime_error("Cannot copy from device: Host data is null");
        if(data_d() == nullptr) return; // Nothing to copy
        HANDLE_CUDA_ERROR(cudaMemcpy(data_h, data_d(), byteSize(), cudaMemcpyDeviceToHost));
    }

    void free() {
        if(d_ptr) { HANDLE_CUDA_ERROR(cudaFree(d_ptr)); }
    }
};

template<typename Scalar>
void cuTensorContract(Meta<Scalar> &tensor_R, Meta<Scalar> &tensor_A, Meta<Scalar> &tensor_B) {
    // CUDA types
    cudaDataType_t        typeCutensor;
    cutensorComputeType_t typeCompute;
    if constexpr(std::is_same_v<Scalar, double>) {
        typeCutensor = CUDA_R_64F;
        typeCompute  = CUTENSOR_COMPUTE_64F;
        tools::log->trace("Detected type fp64");
    } else if(std::is_same_v<Scalar, float>) {
        typeCutensor = CUDA_R_32F;
        typeCompute  = CUTENSOR_COMPUTE_32F;
        tools::log->trace("Detected type fp32");
    } else
        throw std::runtime_error("Wrong type selected");

    Scalar alpha = 1.0;
    Scalar beta  = 0.0;

    // Initialize cuTENSOR library
    cutensorHandle_t handle;
    cutensorInit(&handle);

    // Create Tensor Descriptors
    cutensorTensorDescriptor_t desc_A;
    cutensorTensorDescriptor_t desc_B;
    cutensorTensorDescriptor_t desc_R;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &desc_A, tensor_A.rank(), tensor_A.extent.data(), nullptr, typeCutensor, CUTENSOR_OP_IDENTITY));
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &desc_B, tensor_B.rank(), tensor_B.extent.data(), nullptr, typeCutensor, CUTENSOR_OP_IDENTITY));
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &desc_R, tensor_R.rank(), tensor_R.extent.data(), nullptr, typeCutensor, CUTENSOR_OP_IDENTITY));

    tools::log->trace("Initialize cuTENSOR and tensor descriptors");

    // Retrieve the memory alignment for each tensor
    uint32_t alignmentRequirement_A;
    uint32_t alignmentRequirement_B;
    uint32_t alignmentRequirement_R;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle, tensor_A.data_d(), &desc_A, &alignmentRequirement_A));
    HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle, tensor_B.data_d(), &desc_B, &alignmentRequirement_B));
    HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle, tensor_R.data_d(), &desc_R, &alignmentRequirement_R));

    tools::log->trace("Query best alignment requirement for our pointers");

    // Create the Contraction Descriptor
    cutensorContractionDescriptor_t desc;
    HANDLE_ERROR(cutensorInitContractionDescriptor(&handle, &desc, &desc_A, tensor_A.mode.data(), alignmentRequirement_A, &desc_B, tensor_B.mode.data(),
                                                   alignmentRequirement_B, &desc_R, tensor_R.mode.data(), alignmentRequirement_R, &desc_R, tensor_R.mode.data(),
                                                   alignmentRequirement_R, typeCompute));

    tools::log->trace("Initialize contraction descriptor");

    /* ***************************** */

    // Set the algorithm to use
    cutensorContractionFind_t find;
    HANDLE_ERROR(cutensorInitContractionFind(&handle, &find, CUTENSOR_ALGO_DEFAULT));

    tools::log->trace("Initialize settings to find algorithm");

    /* ***************************** */

    // Query workspace
    size_t worksize = 0;
    HANDLE_ERROR(cutensorContractionGetWorkspaceSize(&handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    // Allocate workspace
    void *work = nullptr;
    if(worksize > 0) {
        if(cudaSuccess != cudaMalloc(&work, worksize)) // This is optional!
        {
            work     = nullptr;
            worksize = 0;
        }
    }

    tools::log->trace("Query recommended workspace size and allocate it");

    /* ***************************** */

    // Create Contraction Plan
    cutensorContractionPlan_t plan;
    HANDLE_ERROR(cutensorInitContractionPlan(&handle, &plan, &desc, &find, worksize));

    tools::log->trace("Create plan for contraction");

    /* ***************************** */

    cutensorStatus_t err;
    cudaStream_t     stream = nullptr;
    // Execute the tensor contraction
    err = cutensorContraction(&handle, &plan, (void *) &alpha, tensor_A.data_d(), tensor_B.data_d(), (void *) &beta, tensor_R.data_d(), tensor_R.data_d(), work,
                              worksize, stream);
    cudaDeviceSynchronize();

    // Check for errors
    if(err != CUTENSOR_STATUS_SUCCESS) { tools::log->error("{}", cutensorGetErrorString(err)); }

    tools::log->trace("Execute contraction from plan");

    if(work) cudaFree(work);

    tools::log->trace("Successful completion");
}

template<typename Scalar>
contract::ResultType<Scalar> contract::tensor_product_cute(const Eigen::Tensor<Scalar, 3> &psi, const Eigen::Tensor<Scalar, 4> &mpo,
                                                           const Eigen::Tensor<Scalar, 3> &envL, const Eigen::Tensor<Scalar, 3> &envR) {
    tools::prof::t_cute->tic();
    Eigen::DSizes<long, 3>   dsizes = psi.dimensions();
    tools::log->info("cute: psi  dims {}", psi.dimensions());
    tools::log->info("cute: mpo  dims {}", mpo.dimensions());
    tools::log->info("cute: envL dims {}", envL.dimensions());
    tools::log->info("cute: envR dims {}", envR.dimensions());

    // Extents
    std::unordered_map<int, int64_t> ext;
    ext['i'] = psi.dimension(0);
    ext['j'] = psi.dimension(1);
    ext['k'] = psi.dimension(2);
    ext['l'] = mpo.dimension(0);
    ext['m'] = mpo.dimension(1);
    ext['n'] = mpo.dimension(3);
    ext['o'] = envL.dimension(1);
    ext['p'] = envR.dimension(1);

    tools::log->info("cute: ext {}", ext);
    tools::log->info("Define cuda tensor meta objects");
    Meta<Scalar> cu_psi(psi, {'i', 'j', 'k'});
    Meta<Scalar> cu_mpo(mpo, {'l', 'm', 'i', 'n'});
    Meta<Scalar> cu_envL(envL, {'j', 'o', 'l'});
    Meta<Scalar> cu_envR(envR, {'k', 'p', 'm'});
    Meta<Scalar> cu_psi_envL({'i', 'k', 'l', 'o'}, {ext['i'], ext['k'], ext['l'], ext['o']});
    Meta<Scalar> cu_psi_envL_mpo({'k', 'o', 'm', 'n'}, {ext['k'], ext['o'], ext['m'], ext['n']});
    Meta<Scalar> cu_ham_psi_sq({'n', 'o', 'p'}, {ext['n'], ext['o'], ext['p']});

    tools::log->info("Copy to device: step 1 of 3");
    cu_psi.copyToDevice();
    cu_envL.copyToDevice();
    cu_psi_envL.copyToDevice();

    tools::log->trace("Contract psi and envL");
    cuTensorContract(cu_psi_envL, cu_psi, cu_envL);

    cu_psi.free();
    cu_envL.free();

    tools::log->info("Copy to device: step 2 of 3");
    cu_mpo.copyToDevice();
    cu_psi_envL_mpo.copyToDevice();

    tools::log->info("Contract psi_envL and mpo");
    cuTensorContract(cu_psi_envL_mpo, cu_psi_envL, cu_mpo);

    cu_mpo.free();
    cu_psi_envL.free();

    tools::log->info("Copy to device: step 3 of 3");
    cu_envR.copyToDevice();
    cu_ham_psi_sq.copyToDevice();

    tools::log->info("Contract psi_envL_mpo and envR");
    cuTensorContract(cu_ham_psi_sq, cu_psi_envL_mpo, cu_envR);

    cu_envR.free();
    cu_psi_envL_mpo.free();

    Eigen::Tensor<Scalar, 3> ham_sq_psi(dsizes);
    cu_ham_psi_sq.copyFromDevice(ham_sq_psi.data());
    cu_ham_psi_sq.free();
    tools::prof::t_cute->toc();
    tools::log->info("Done");

    return std::make_pair(ham_sq_psi, get_ops_cute_L(dsizes[0], dsizes[1], dsizes[2], mpo.dimension(0)));
}

// using cx64 = std::complex<double>;
// using fp32 = float;
using fp64 = double;

// template contract::ResultType<cplx> contract::tensor_product_cute(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
//                                                                           const Eigen::Tensor<cplx, 4> &envL, const Eigen::Tensor<cplx, 4> &envR);
// template contract::ResultType<fp32> contract::tensor_product_cute(const Eigen::Tensor<fp32, 3> &psi_in, const Eigen::Tensor<fp32, 4> &mpo,
//                                                                           const Eigen::Tensor<fp32, 4> &envL, const Eigen::Tensor<fp32, 4> &envR);
template contract::ResultType<fp64> contract::tensor_product_cute(const Eigen::Tensor<fp64, 3> &psi, const Eigen::Tensor<fp64, 4> &mpo,
                                                                  const Eigen::Tensor<fp64, 3> &envL, const Eigen::Tensor<fp64, 3> &envR);
#endif