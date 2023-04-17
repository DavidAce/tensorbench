
#include "benchmark.h"
using fp32 = benchmark::fp32;
using fp64 = benchmark::fp64;
using cplx = benchmark::cplx;
#if defined(TB_CUTENSOR)
    #include "tid/tid.h"
    #include "tools/fmt.h"
    #include "tools/log.h"
    #include <cstdio>
    #include <cstdlib>
    #include <cuda_runtime.h>
    #include <cutensor.h>
    #include <stdexcept>
    #include <unordered_map>
    #include <vector>

    // Handle cuTENSOR errors
    #define HANDLE_ERROR(x)                                                                   \
        {                                                                                     \
            const auto err = x;                                                               \
            if(err != CUTENSOR_STATUS_SUCCESS) {                                              \
                tools::log->critical("{} in line {}", cutensorGetErrorString(err), __LINE__); \
                exit(err);                                                                    \
            }                                                                                 \
        }

    #define HANDLE_CUDA_ERROR(x)                                                          \
        {                                                                                 \
            const auto err = x;                                                           \
            if(err != cudaSuccess) {                                                      \
                tools::log->critical("{} in line {}", cudaGetErrorString(err), __LINE__); \
                exit(err);                                                                \
            }                                                                             \
        }

long get_ops_cute_L(long d, long chiL, long chiR, long m);
long get_ops_cute_R(long d, long chiL, long chiR, long m) {
    // Same as L, just swap chiL and chiR
    if(chiR > chiL)
        return get_ops_cute_L(d, chiR, chiL, m);
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

std::string getGpuName(int deviceNumber) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if(deviceNumber + 1 > nDevices) tools::log->warn("requested cuda device number {} out of bounds | detected {} devices", deviceNumber, nDevices);
    if(nDevices >= 1) {
        deviceNumber        = std::clamp<int>(deviceNumber, 0, nDevices - 1);
        cudaDeviceProp prop = {};
        cudaGetDeviceProperties(&prop, deviceNumber);
        return prop.name;
    } else
        return "";
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
        free();
        //                if(d_ptr) HANDLE_CUDA_ERROR(cudaFree(d_ptr));
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
        //        auto t_copy2gpu = tid::tic_token("copy2gpu");

        size_t mf, ma;
        cudaMemGetInfo(&mf, &ma);
        tools::log->trace("CUDA: Free {} | Total {}", static_cast<double>(mf) / std::pow(1024, 2), static_cast<double>(ma) / std::pow(1024, 2));
        if(data_d() == nullptr) {
            HANDLE_CUDA_ERROR(cudaMalloc((void **) &d_ptr, byteSize()));
            tools::log->trace("copyToDevice(): cudaMalloc {} bytes to ptr {}", byteSize(), fmt::ptr(d_ptr));
        }
        if(data_h() == nullptr) return; // Nothing to copy
        HANDLE_CUDA_ERROR(cudaMemcpy(data_d(), data_h(), byteSize(), cudaMemcpyHostToDevice));
    }

    void copyFromDevice(Scalar *data_h) {
        //        auto t_copy2cpu = tid::tic_token("copy2cpu");
        if(data_h == nullptr) throw std::runtime_error("Cannot copy from device: Host data is null");
        if(data_d() == nullptr) return; // Nothing to copy
        HANDLE_CUDA_ERROR(cudaMemcpy(data_h, data_d(), byteSize(), cudaMemcpyDeviceToHost));
    }

    void free() {
        if(d_ptr) {
            tools::log->trace("cudaFree {} bytes from ptr {}", byteSize(), fmt::ptr(d_ptr));
            HANDLE_CUDA_ERROR(cudaFree(d_ptr));
            d_ptr = nullptr;
        }
    }
};

template<typename Scalar>
void cuTensorContract(Meta<Scalar> &tensor_R, Meta<Scalar> &tensor_A, Meta<Scalar> &tensor_B) {
    auto t_con = tid::tic_token("contract");
    // CUDA types
    cudaDataType_t        typeCutensor;
    cutensorComputeType_t typeCompute;
    if(std::is_same_v<Scalar, fp32>) {
        typeCutensor = CUDA_R_32F;
        typeCompute  = CUTENSOR_COMPUTE_32F;
        tools::log->trace("Detected type fp32");
    } else if constexpr(std::is_same_v<Scalar, fp64>) {
        typeCutensor = CUDA_R_64F;
        typeCompute  = CUTENSOR_COMPUTE_64F;
        tools::log->trace("Detected type fp64");
    } else if(std::is_same_v<Scalar, cplx>) {
        typeCutensor = CUDA_C_64F;
        typeCompute  = CUTENSOR_COMPUTE_64F;
        tools::log->trace("Detected type cplx");

    } else
        throw std::runtime_error("Wrong type selected");

    Scalar alpha = 1.0;
    Scalar beta  = 0.0;

    // Initialize cuTENSOR library
    cutensorHandle_t *handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    // Create Tensor Descriptors
    cutensorTensorDescriptor_t desc_A;
    cutensorTensorDescriptor_t desc_B;
    cutensorTensorDescriptor_t desc_R;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &desc_A, tensor_A.rank(), tensor_A.extent.data(), nullptr, typeCutensor, CUTENSOR_OP_IDENTITY));
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &desc_B, tensor_B.rank(), tensor_B.extent.data(), nullptr, typeCutensor, CUTENSOR_OP_IDENTITY));
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &desc_R, tensor_R.rank(), tensor_R.extent.data(), nullptr, typeCutensor, CUTENSOR_OP_IDENTITY));

    tools::log->trace("Initialize cuTENSOR and tensor descriptors");

    // Retrieve the memory alignment for each tensor
    uint32_t alignmentRequirement_A;
    uint32_t alignmentRequirement_B;
    uint32_t alignmentRequirement_R;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, tensor_A.data_d(), &desc_A, &alignmentRequirement_A));
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, tensor_B.data_d(), &desc_B, &alignmentRequirement_B));
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, tensor_R.data_d(), &desc_R, &alignmentRequirement_R));

    tools::log->trace("Query best alignment requirement for our pointers");

    // Create the Contraction Descriptor
    cutensorContractionDescriptor_t desc;
    HANDLE_ERROR(cutensorInitContractionDescriptor(handle, &desc, &desc_A, tensor_A.mode.data(), alignmentRequirement_A, &desc_B, tensor_B.mode.data(),
                                                   alignmentRequirement_B, &desc_R, tensor_R.mode.data(), alignmentRequirement_R, &desc_R, tensor_R.mode.data(),
                                                   alignmentRequirement_R, typeCompute));

    tools::log->trace("Initialize contraction descriptor");

    /* ***************************** */

    // Set the algorithm to use
    cutensorContractionFind_t find;
    HANDLE_ERROR(cutensorInitContractionFind(handle, &find, CUTENSOR_ALGO_DEFAULT));

    tools::log->trace("Initialize settings to find algorithm");

    /* ***************************** */

    // Query workspace
    size_t worksize = 0;
    HANDLE_ERROR(cutensorContractionGetWorkspaceSize(handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    // Allocate workspace
    void *work = nullptr;
    if(worksize > 0) HANDLE_CUDA_ERROR(cudaMalloc(&work, worksize));

    tools::log->trace("Query recommended workspace size and allocate it");

    /* ***************************** */

    // Create Contraction Plan
    cutensorContractionPlan_t plan;
    HANDLE_ERROR(cutensorInitContractionPlan(handle, &plan, &desc, &find, worksize));

    tools::log->trace("Create plan for contraction");

    /* ***************************** */

    cudaStream_t stream = nullptr;
    // Execute the tensor contraction
    HANDLE_ERROR(cutensorContraction(handle, &plan, (void *) &alpha, tensor_A.data_d(), tensor_B.data_d(), (void *) &beta, tensor_R.data_d(), tensor_R.data_d(),
                                     work, worksize, stream));
    cudaDeviceSynchronize();

    tools::log->trace("Execute contraction from plan");

    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_CUDA_ERROR(cudaFree(work));
    tools::log->trace("Successful completion");
}
#endif

template<typename T>
benchmark::ResultType<T> benchmark::tensor_product_cute([[maybe_unused]] const tb_setup<T> &tbs) {
#if defined(TB_CUTE)
    auto                   t_cutensor = tid::tic_scope("cutensor");
    Eigen::DSizes<long, 3> dsizes = tbs.psi.dimensions();
    tbs.device                    = getGpuName(tbs.gpun);

    // Extents
    std::unordered_map<int, int64_t> ext;
    ext['i'] = tbs.psi.dimension(0);
    ext['j'] = tbs.psi.dimension(1);
    ext['k'] = tbs.psi.dimension(2);
    ext['l'] = tbs.mpo.dimension(0);
    ext['m'] = tbs.mpo.dimension(1);
    ext['n'] = tbs.mpo.dimension(3);
    ext['o'] = tbs.envL.dimension(1);
    ext['p'] = tbs.envR.dimension(1);

    tools::log->trace("cutensor: ext {}", ext);
    tools::log->trace("Define cuda tensor meta objects");
    Meta<T> cu_psi_envL({'i', 'k', 'l', 'o'}, {ext['i'], ext['k'], ext['l'], ext['o']});
    Meta<T> cu_psi_envL_mpo({'k', 'o', 'm', 'n'}, {ext['k'], ext['o'], ext['m'], ext['n']});
    Meta<T> cu_ham_psi_sq({'n', 'o', 'p'}, {ext['n'], ext['o'], ext['p']});

    tools::log->trace("Copy to device: step 0 of 3 (preamble)");

    auto t_contract = tid::tic_scope("contract");
    {
        Meta<T> cu_psi(tbs.psi, {'i', 'j', 'k'});
        Meta<T> cu_envL(tbs.envL, {'j', 'o', 'l'});
        tools::log->trace("Contract psi and envL");
        auto t_con1 = tid::tic_scope("psi_envL", tid::level::extra);
        cu_envL.copyToDevice();
        cu_psi.copyToDevice();
        cu_psi_envL.copyToDevice();
        cuTensorContract(cu_psi_envL, cu_psi, cu_envL);
    }

    {
        Meta<T> cu_mpo(tbs.mpo, {'l', 'm', 'i', 'n'});
        tools::log->trace("Contract psi_envL and mpo");
        auto t_con2 = tid::tic_scope("psi_envL_mpo", tid::level::extra);
        cu_mpo.copyToDevice();
        cu_psi_envL_mpo.copyToDevice();
        cuTensorContract(cu_psi_envL_mpo, cu_psi_envL, cu_mpo);
        cu_psi_envL.free();
    }

    {
        Meta<T> cu_envR(tbs.envR, {'k', 'p', 'm'});
        tools::log->trace("Contract psi_envL_mpo and envR");
        auto t_con3 = tid::tic_scope("psi_envL_mpo_envR", tid::level::extra);
        cu_ham_psi_sq.copyToDevice();
        cu_envR.copyToDevice();
        cuTensorContract(cu_ham_psi_sq, cu_psi_envL_mpo, cu_envR);
        cu_psi_envL_mpo.free();
    }

    Eigen::Tensor<T, 3> ham_sq_psi(dsizes);
    cu_ham_psi_sq.copyFromDevice(ham_sq_psi.data());
    return ham_sq_psi;
#else
    return {};
#endif
}

template benchmark::ResultType<fp32> benchmark::tensor_product_cute(const tb_setup<fp32> &tbs);
template benchmark::ResultType<fp64> benchmark::tensor_product_cute(const tb_setup<fp64> &tbs);
template benchmark::ResultType<cplx> benchmark::tensor_product_cute(const tb_setup<cplx> &tbs);
