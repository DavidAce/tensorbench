#if defined(TB_ACRO)

#include <acrotensor/AcroTensor.hpp>
#include <complex>
#include <array>
#include <contract/contract.h>
#include <cuda.h>
#include <tools/class_tic_toc.h>
#include <tools/log.h>
#include <tools/prof.h>
template<typename Scalar>
Eigen::Tensor<Scalar, 3> contract::hamiltonian_squared_dot_psi_acro(const Eigen::Tensor<Scalar, 3> &psi_in, const Eigen::Tensor<Scalar, 4> &mpo,
                                                                    const Eigen::Tensor<Scalar, 4> &envL, const Eigen::Tensor<Scalar, 4> &envR) {
    tools::prof::t_ham_sq_psi_acro->tic();
    tools::log->set_level(spdlog::level::trace);
    tools::log->trace("Started acro contraction");
    Eigen::DSizes<long, 3>   dsizes = psi_in.dimensions();
    Eigen::Tensor<Scalar, 3> ham_sq_psi(dsizes);
    tools::log->trace("Shuffling psi");
    Eigen::Tensor<Scalar, 3> psi_shuffled = psi_in.shuffle(Textra::array3{1, 0, 2});

    //    ham_sq_psi.device(*Textra::omp::dev) = psi_shuffled.contract(envL, Textra::idx({0}, {0}))
    //                                               .contract(mpo, Textra::idx({0, 3}, {2, 0}))
    //                                               .contract(mpo, Textra::idx({4, 2}, {2, 0}))
    //                                               .contract(envR, Textra::idx({0, 2, 3}, {0, 2, 3}))
    //                                               .shuffle(Textra::array3{1, 0, 2});
    tools::log->trace("Setting up ranks");
    std::vector<int> ham_sq_psi_dims(3);
    std::vector<int> psi_shuffled_dims(3);
    std::vector<int> envL_dims(4);
    std::vector<int> envR_dims(4);
    std::vector<int> mpo_dims(4);
    tools::log->trace("Copying dimensions");
    for(long i = 0; i < 3; i++ ) ham_sq_psi_dims[i] = dsizes[i];
    for(long i = 0; i < 3; i++ ) psi_shuffled_dims[i] = psi_shuffled.dimension(i);
    for(long i = 0; i < 4; i++ ) envL_dims[i] = envL.dimension(i);
    for(long i = 0; i < 4; i++ ) envR_dims[i] = envR.dimension(i);
    for(long i = 0; i < 4; i++ ) mpo_dims[i] = mpo.dimension(i);

//
//
//    std::copy(std::begin(dsizes), std::end(dsizes), std::begin(ham_sq_psi_dims));
//    std::copy(std::begin(psi_shuffled.dimensions()), std::end(psi_shuffled.dimensions()), std::begin(psi_shuffled_dims));
//    std::copy(std::begin(envL.dimensions()), std::end(envL.dimensions()), std::begin(envL_dims));
//    std::copy(std::begin(envR.dimensions()), std::end(envR.dimensions()), std::begin(envR_dims));
//    std::copy(std::begin(mpo.dimensions()), std::end(mpo.dimensions()), std::begin(mpo_dims));
    tools::log->trace("Initializing CUDA");

    tools::log->trace("Allocating unified memory");

    size_t ham_sq_psi_bytes   = static_cast<size_t>(ham_sq_psi.size()) * sizeof(Scalar);
    size_t psi_shuffled_bytes = static_cast<size_t>(psi_shuffled.size()) * sizeof(Scalar);
    size_t envL_bytes         = static_cast<size_t>(envL.size()) * sizeof(Scalar);
    size_t envR_bytes         = static_cast<size_t>(envR.size()) * sizeof(Scalar);
    size_t mpo_bytes         = static_cast<size_t>(mpo.size()) * sizeof(Scalar);
    Scalar * u_ham_sq_psi;
    Scalar * u_psi_shuffled;
    Scalar * u_envL;
    Scalar * u_envR;
    Scalar * u_mpo1;
    Scalar * u_mpo2;

    cudaMallocManaged(&u_ham_sq_psi,ham_sq_psi_bytes );
    cudaMallocManaged(&u_psi_shuffled ,psi_shuffled_bytes  );
    cudaMallocManaged(&u_envL ,envL_bytes  );
    cudaMallocManaged(&u_envR ,envR_bytes  );
    cudaMallocManaged(&u_mpo1 ,mpo_bytes  );
    cudaMallocManaged(&u_mpo2 ,mpo_bytes  );



    //https://forums.developer.nvidia.com/t/unified-memory-oversubscription-and-page-faults/59470/4

    cudaMemPrefetchAsync(u_ham_sq_psi, ham_sq_psi_bytes, 0);
    cudaMemPrefetchAsync(u_psi_shuffled, psi_shuffled_bytes, 0);
    cudaMemPrefetchAsync(u_envL, envL_bytes, 0);
    cudaMemPrefetchAsync(u_envR, envR_bytes, 0);
    cudaMemPrefetchAsync(u_mpo1, mpo_bytes, 0);
    cudaMemPrefetchAsync(u_mpo2, mpo_bytes, 0);


    Eigen::TensorMap<Eigen::Tensor<Scalar, 3, Eigen::RowMajor>> ham_sq_psi_uni(u_ham_sq_psi, ham_sq_psi.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, 3, Eigen::RowMajor>> psi_shuffled_uni(u_psi_shuffled, psi_shuffled.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, 4, Eigen::RowMajor>> envL_uni(u_envL, envL.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, 4, Eigen::RowMajor>> envR_uni(u_envR, envR.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, 4, Eigen::RowMajor>> mpo1_uni(u_mpo1, mpo.dimensions());
    Eigen::TensorMap<Eigen::Tensor<Scalar, 4, Eigen::RowMajor>> mpo2_uni(u_mpo2, mpo.dimensions());

    tools::log->trace("Copying tensors from host to device and swapping layout on the fly");
    psi_shuffled_uni = Textra::to_RowMajor(psi_shuffled);
    envL_uni         = Textra::to_RowMajor(envL);
    envR_uni         = Textra::to_RowMajor(envR);
    mpo1_uni         = Textra::to_RowMajor(mpo);
    mpo2_uni         = Textra::to_RowMajor(mpo);


    tools::log->trace("Initializing acrotensors");
    acro::Tensor acro_ham_sq_psi(ham_sq_psi_dims, ham_sq_psi_uni.data(), ham_sq_psi_uni.data());
    acro::Tensor acro_psi_shuffled(psi_shuffled_dims, psi_shuffled_uni.data(), psi_shuffled_uni.data());
    acro::Tensor acro_envL(envL_dims, envL_uni.data(), envL_uni.data());
    acro::Tensor acro_envR(envR_dims, envR_uni.data(), envR_uni.data());
    acro::Tensor acro_mpo1(mpo_dims, mpo1_uni.data(), mpo1_uni.data());
    acro::Tensor acro_mpo2(mpo_dims, mpo2_uni.data(), mpo2_uni.data());

    //    tools::log->trace("Running CPU tensor contraction");

    //    acro::TensorEngine TE_cpu("CPUInterpreted");
    //    TE_cpu("H_q_r_s = P_i_j_k L_i_r_l_o M_l_m_j_n N_o_p_n_q R_k_s_m_p",acro_ham_sq_psi, acro_psi_shuffled, acro_envL,acro_mpo1,acro_mpo2,acro_envR);
    //    acro_ham_sq_psi.Print();    //Display the results of the contraction

    tools::log->trace("Moving tensors to GPU");
//    acro_ham_sq_psi.MapToGPU();
//    acro_psi_shuffled.MapToGPU();
//    acro_envL.MapToGPU();
//    acro_envR.MapToGPU();
//    acro_mpo1.MapToGPU();
//    acro_mpo2.MapToGPU();
//    acro_ham_sq_psi.SwitchToGPU();
//    acro_psi_shuffled.MoveToGPU();
//    acro_envL.MoveToGPU();
//    acro_envR.MoveToGPU();
//    acro_mpo1.MoveToGPU();
//    acro_mpo2.MoveToGPU();

    tools::log->trace("Running GPU tensor contraction");

    acro::TensorEngine TE_gpu("Cuda");
    TE_gpu("H_q_r_s = P_i_j_k L_i_r_l_o M_l_m_j_n N_o_p_n_q R_k_s_m_p", acro_ham_sq_psi, acro_psi_shuffled, acro_envL, acro_mpo1, acro_mpo2, acro_envR);
    // Wait for GPU to finish before accessing on host
    tools::log->trace("Syncronizing device");
    cudaDeviceSynchronize();
    tools::log->trace("Moving tensor from GPU");

//    acro_ham_sq_psi.MoveFromGPU();
    //    acro_ham_sq_psi.Print();    //Display the results of the contraction
    tools::log->trace("Copy data from device to host");
    ham_sq_psi = Textra::to_ColMajor(Eigen::TensorMap<Eigen::Tensor<Scalar, 3, Eigen::RowMajor>>(acro_ham_sq_psi.GetData(), dsizes));
//    Eigen::Tensor<Scalar,3,Eigen::RowMajor> ham_sq_psi_rowm(dsizes);
//    cudaMemcpy(ham_sq_psi_rowm.data(), acro_ham_sq_psi.GetData(), ham_sq_psi_bytes,cudaMemcpyDefault);
//

    tools::log->trace("Free CUDA memory");
    cudaFree(u_ham_sq_psi);
    cudaFree(u_psi_shuffled);
    cudaFree(u_envL);
    cudaFree(u_envR);
    cudaFree(u_mpo1);
    cudaFree(u_mpo2);


//    tools::log->trace("Swapping back the storage layout");
//    ham_sq_psi = Textra::to_ColMajor(Eigen::TensorMap<Eigen::Tensor<Scalar, 3, Eigen::RowMajor>>(ham_sq_psi_rowm.data(), dsizes));


    tools::log->trace("Finished acrotensor contractions");

    tools::prof::t_ham_sq_psi_acro->toc();

    Eigen::Tensor<Scalar, 3> ham_sq_psi_cpu(dsizes);
    ham_sq_psi_cpu.device(*Textra::omp::dev) =
        psi_shuffled
            .contract(envL , Textra::idx(Textra::array1{0}, Textra::array1{0}))
            .contract(mpo  , Textra::idx(Textra::array2{0, 3}, Textra::array2{2, 0}))
            .contract(mpo  , Textra::idx(Textra::array2{4, 2}, Textra::array2{2, 0}))
            .contract(envR,  Textra::idx(Textra::array3{0, 2, 3}, Textra::array3{0, 2, 3}))
            .shuffle(Textra::array3{1, 0, 2});

    for (long i = 0; i < ham_sq_psi_cpu.size(); i++) {
        if (std::abs(ham_sq_psi_cpu(i) - ham_sq_psi(i)) > 1e-2) {
            tools::log->error("Tensor mismatch > 1e-2 at index {:5}: cpu {:20.16f} != gpu {:20.16f}",i,ham_sq_psi_cpu(i) ,ham_sq_psi(i));
//            throw std::runtime_error("Tensor mismatch > 1e-4 at index "+ std::to_string(i)
//                                     + " " + std::to_string(ham_sq_psi_cpu(i)) + " " + std::to_string(ham_sq_psi(i)));
        }
//        if (not Eigen::internal::isApprox(ham_sq_psi_cpu(i),  ham_sq_psi(i), 1e-4)) {
//            tools::log->error("Tensor mismatch > 1e-4 at index {:5}: {:.8f} != {.8f}",i,ham_sq_psi_cpu(i) ,ham_sq_psi(i));
//            throw std::runtime_error("Tensor approx mismatch > 1e-4 at index "+ std::to_string(i)
//                                     + " " + std::to_string(ham_sq_psi_cpu(i)) + " " + std::to_string(ham_sq_psi(i)));
//        }

    }




    return ham_sq_psi;
}

using cplx = std::complex<double>;
using fp32 = float;
using fp64 = double;

// template Eigen::Tensor<cplx, 3> contract::hamiltonian_squared_dot_psi_acro(const Eigen::Tensor<cplx, 3> &psi_in, const Eigen::Tensor<cplx, 4> &mpo,
//                                                                           const Eigen::Tensor<cplx, 4> &envL, const Eigen::Tensor<cplx, 4> &envR);
// template Eigen::Tensor<fp32, 3> contract::hamiltonian_squared_dot_psi_acro(const Eigen::Tensor<fp32, 3> &psi_in, const Eigen::Tensor<fp32, 4> &mpo,
//                                                                           const Eigen::Tensor<fp32, 4> &envL, const Eigen::Tensor<fp32, 4> &envR);
template Eigen::Tensor<fp64, 3> contract::hamiltonian_squared_dot_psi_acro(const Eigen::Tensor<fp64, 3> &psi_in, const Eigen::Tensor<fp64, 4> &mpo,
                                                                           const Eigen::Tensor<fp64, 4> &envL, const Eigen::Tensor<fp64, 4> &envR);
#endif