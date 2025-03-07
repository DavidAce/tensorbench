
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
    cutensorDataType_t          typeCompute;
    cutensorComputeDescriptor_t descCompute;
    if constexpr (std::is_same_v<Scalar, fp32>) {
        typeCompute = CUTENSOR_R_32F;
        descCompute = CUTENSOR_COMPUTE_DESC_32F;
        tools::log->trace("Detected type fp32");
    } else if constexpr(std::is_same_v<Scalar, fp64>) {
        typeCompute = CUTENSOR_R_64F;
        descCompute = CUTENSOR_COMPUTE_DESC_64F;
        tools::log->trace("Detected type fp64");
    } else if constexpr (std::is_same_v<Scalar, cplx>) {
        typeCompute = CUTENSOR_C_64F;
        descCompute = CUTENSOR_COMPUTE_DESC_64F;
        tools::log->trace("Detected type cplx");

    } else
        throw std::runtime_error("Wrong type selected");

    Scalar alpha = 1.0;
    Scalar beta  = 0.0;

    const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
    // assert(uintptr_t(A_d) % kAlignment == 0);
    // assert(uintptr_t(B_d) % kAlignment == 0);
    // assert(uintptr_t(C_d) % kAlignment == 0);

    // Initialize cuTENSOR library
    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    // Create Tensor Descriptors
    cutensorTensorDescriptor_t desc_A;
    cutensorTensorDescriptor_t desc_B;
    cutensorTensorDescriptor_t desc_R;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &desc_A, tensor_A.rank(), tensor_A.extent.data(), nullptr, typeCompute, kAlignment));
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &desc_B, tensor_B.rank(), tensor_B.extent.data(), nullptr, typeCompute, kAlignment));
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &desc_R, tensor_R.rank(), tensor_R.extent.data(), nullptr, typeCompute, kAlignment));

    tools::log->trace("Initialize cuTENSOR and tensor descriptors");

    // Create the Contraction Descriptor
    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateContraction(handle,
                &desc,
                desc_A, tensor_A.mode.data(), /* unary operator A*/CUTENSOR_OP_IDENTITY,
                desc_B, tensor_B.mode.data(), /* unary operator B*/CUTENSOR_OP_IDENTITY,
                desc_R, tensor_R.mode.data(), /* unary operator C*/CUTENSOR_OP_IDENTITY,
                desc_R, tensor_R.mode.data(),
                descCompute));


    tools::log->trace("Initialize contraction descriptor");


    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
                desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType)));

    assert(scalarType == CUTENSOR_R_32F);
    /* ***************************** */

    /**************************
     * Set the algorithm to use
     ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(
                handle,
                &planPref,
                algo,
                CUTENSOR_JIT_MODE_NONE));

    tools::log->trace("Initialize settings to algorithm");

    /**********************
    * Query workspace estimate
    **********************/

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                desc,
                planPref,
                workspacePref,
                &workspaceSizeEstimate));


    tools::log->trace("Query recommended workspace size and allocate it");

    /**************************
         * Create Contraction Plan
         **************************/

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                &plan,
                desc,
                planPref,
                workspaceSizeEstimate));

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &actualWorkspaceSize,
                sizeof(actualWorkspaceSize)));

    // At this point the user knows exactly how much memory is need by the operation and
    // only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
    }

    tools::log->trace("Execute contraction from plan");
    /**********************
    * Execute
    **********************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    HANDLE_ERROR(cutensorContract(handle,
                plan,
                (void*) &alpha, tensor_A.data_d(), tensor_B.data_d(),
                (void*) &beta,  tensor_R.data_d(), tensor_R.data_d(),
                work, actualWorkspaceSize, stream));

    /**********************
     * Free allocated data
     **********************/
    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(desc_A));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(desc_B));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(desc_R));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
    HANDLE_CUDA_ERROR(cudaFree(work));
    tools::log->trace("Successful completion");
}
#endif

template<typename T>
benchmark::ResultType<T> benchmark::tensor_product_cutensor([[maybe_unused]] const tb_setup<T> &tbs) {
#if defined(TB_CUTENSOR)
    auto                   t_cutensor = tid::tic_scope("cutensor");
    Eigen::DSizes<long, 3> dsizes     = tbs.psi.dimensions();

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

template benchmark::ResultType<fp32> benchmark::tensor_product_cutensor(const tb_setup<fp32> &tbs);
template benchmark::ResultType<fp64> benchmark::tensor_product_cutensor(const tb_setup<fp64> &tbs);
template benchmark::ResultType<cplx> benchmark::tensor_product_cutensor(const tb_setup<cplx> &tbs);
