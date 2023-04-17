#pragma once
#include "math/num.h"
#include "mpi-logger.h"
#include "tid/tid.h"
#include "tools/log.h"
#include <cassert>
#include <complex>
#include <general/sfinae.h>
#include <h5pp/details/h5ppFilesystem.h>
#include <h5pp/details/h5ppFormat.h>
#include <vector>

#if defined(TB_MPI)
    #include <mpi.h>
#endif

namespace mpi {

    inline bool on = false;

    struct comm {
        int id   = 0;
        int size = 1;

        template<typename T>
        T get_id() {
            return static_cast<T>(id);
        }
        template<typename T>
        T get_size() {
            return static_cast<T>(size);
        }
    };

    inline comm world;
    void        init(int argc, char *argv[]);
    void        finalize();
    void        barrier();

#if defined(TB_MPI)
    template<typename T>
    [[nodiscard]] constexpr MPI_Datatype get_dtype() noexcept {
        using D = typename std::decay<T>::type;
        /* clang-format off */
        if constexpr(std::is_same_v<T, char>)                           return MPI_CHAR;
        else if constexpr(std::is_same_v<D, signed char>)               return MPI_SIGNED_CHAR;
        else if constexpr(std::is_same_v<D, unsigned char>)             return MPI_UNSIGNED_CHAR;
        else if constexpr(std::is_same_v<D, wchar_t>)                   return MPI_WCHAR;
        else if constexpr(std::is_same_v<D, std::int8_t>)               return MPI_INT8_T;
        else if constexpr(std::is_same_v<D, std::int16_t>)              return MPI_INT16_T;
        else if constexpr(std::is_same_v<D, std::int32_t>)              return MPI_INT32_T;
        else if constexpr(std::is_same_v<D, std::int64_t>)              return MPI_INT64_T;
        else if constexpr(std::is_same_v<D, std::uint8_t>)              return MPI_UINT8_T;
        else if constexpr(std::is_same_v<D, std::uint16_t>)             return MPI_UINT16_T;
        else if constexpr(std::is_same_v<D, std::uint32_t>)             return MPI_UINT32_T;
        else if constexpr(std::is_same_v<D, std::uint64_t>)             return MPI_UINT64_T;
        else if constexpr(std::is_same_v<D, signed short>)              return MPI_SHORT;
        else if constexpr(std::is_same_v<D, signed int>)                return MPI_INT;
        else if constexpr(std::is_same_v<D, signed long int>)           return MPI_LONG;
        else if constexpr(std::is_same_v<D, signed long long int>)      return MPI_LONG_LONG;
        else if constexpr(std::is_same_v<D, unsigned short>)            return MPI_UNSIGNED_SHORT;
        else if constexpr(std::is_same_v<D, unsigned int>)              return MPI_UNSIGNED;
        else if constexpr(std::is_same_v<D, unsigned long int>)         return MPI_UNSIGNED_LONG;
        else if constexpr(std::is_same_v<D, unsigned long long int>)    return MPI_UNSIGNED_LONG_LONG;
        else if constexpr(std::is_same_v<D, float>)                     return MPI_FLOAT;
        else if constexpr(std::is_same_v<D, double>)                    return MPI_DOUBLE;
        else if constexpr(std::is_same_v<D, long double>)               return MPI_LONG_DOUBLE;
        else if constexpr(std::is_same_v<D, bool>)                      return MPI_C_BOOL;
        else if constexpr(std::is_same_v<D, std::complex<float>>)       return MPI_C_COMPLEX;
        else if constexpr(std::is_same_v<D, std::complex<double>>)      return MPI_C_DOUBLE_COMPLEX;
        else if constexpr(std::is_same_v<D, std::complex<long double>>) return MPI_C_LONG_DOUBLE_COMPLEX;
        else if constexpr(sfinae::has_Scalar_v<D>)                      return get_dtype<typename D::Scalar>();
        else if constexpr(sfinae::has_value_type_v<D>)                  return get_dtype<typename D::value_type>();
        else static_assert(sfinae::invalid_type_v<D>, "Could not match with MPI datatype");
        /* clang-format on */
    }
    template<typename T>
    [[nodiscard]] void *get_buffer(T &data) {
        static_assert(sfinae::has_data_v<T> or std::is_pointer_v<T> or std::is_array_v<T>, "Buffer cannot be casted to 'void *'");
        if constexpr(sfinae::has_data_v<T>) return static_cast<void *>(data.data());
        if constexpr(std::is_pointer_v<T> or std::is_array_v<T>) return static_cast<void *>(data);
    }

    template<typename T>
    [[nodiscard]] const void *get_cbuffer(const T &data) {
        static_assert(sfinae::has_data_v<T> or std::is_pointer_v<T> or std::is_array_v<T>, "Buffer cannot be casted to 'const void *'");
        if constexpr(sfinae::has_data_v<T>) return static_cast<const void *>(data.data());
        if constexpr(std::is_pointer_v<T> or std::is_array_v<T>) return static_cast<const void *>(data);
    }

    template<typename T>
    [[nodiscard]] int get_count(const T &data) {
        if constexpr(sfinae::has_size_v<T> or std::is_array_v<T>)
            return static_cast<int>(std::size(data));
        else
            return 1;
    }

    template<typename T>
    void send(const T &data, int dst, int tag) {
        auto t_mpi = tid::tic_token("mpi::send");
        if constexpr(sfinae::has_size_v<T>) {
            size_t count = data.size();
            MPI_Send(&count, 1, mpi::get_dtype<size_t>(), dst, 0, MPI_COMM_WORLD);
        }

        MPI_Send(mpi::get_cbuffer(data), mpi::get_count(data), mpi::get_dtype<T>(), dst, tag, MPI_COMM_WORLD);
    }

    template<typename T>
    void recv(T &data, int src, int tag) {
        auto t_mpi = tid::tic_token("mpi::recv");
        if constexpr(sfinae::has_size_v<T>) {
            size_t count;
            MPI_Recv(&count, 1, mpi::get_dtype<size_t>(), src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if constexpr(sfinae::has_resize_v<T>) data.resize(count);
            if(data.size() < count) throw std::runtime_error(fmt::format("mpi::recv: cointainer size {} < count {}", data.size(), count));
        }
        MPI_Recv(mpi::get_buffer(data), mpi::get_count(data), mpi::get_dtype<T>(), src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    template<typename T>
    void sendrecv(const T &send, const T &recv, int src, int dst, int tag) {
        auto t_mpi = tid::tic_token("mpi::sendrecv");
        // Start by sending the data size, so we can resize the receiving buffer accordingly
        int count = mpi::get_count(send);
        MPI_Sendrecv_replace(&count, 1, MPI_INT, dst, tag, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if constexpr(sfinae::has_resize_v<T>) {
            recv.resize(count); // Both containers are now ready to receive
        }
        MPI_Sendrecv(mpi::get_buffer(send), mpi::get_count(send), mpi::get_dtype<T>(), dst, tag, mpi::get_buffer(recv), mpi::get_count(recv),
                     mpi::get_dtype<T>(), src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    template<typename T>
    void sendrecv_replace(T &data, int src, int dst, int tag) {
        auto t_mpi = tid::tic_token("mpi::sendrecv_replace");
        if constexpr(sfinae::has_resize_v<T>) {
            // Start by sending the data size, so we can resize the receiving buffer accordingly
            int count = mpi::get_count(data);
            MPI_Sendrecv_replace(&count, 1, MPI_INT, dst, tag, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            data.resize(count); // Should not modify the src container
        }
        MPI_Sendrecv_replace(mpi::get_buffer(data), mpi::get_count(data), mpi::get_dtype<T>(), dst, tag, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    template<typename T>
    void bcast(T &data, int src) {
        auto t_mpi = tid::tic_token("mpi::bcast");
        int  count = mpi::get_count(data);
        MPI_Bcast(&count, 1, MPI_INT, src, MPI_COMM_WORLD);       // Send data size, so we can resize the receiving buffers accordingly
        if constexpr(sfinae::has_resize_v<T>) data.resize(count); // Resize receiving buffers. Should not modify the src container
        MPI_Bcast(mpi::get_buffer(data), mpi::get_count(data), mpi::get_dtype<T>(), src, MPI_COMM_WORLD);
    }

    template<typename S, typename R>
    void gatherv(const S &send, R &recv, int dst) {
        auto             t_mpi = tid::tic_token("mpi::gatherv");
        int              count = mpi::get_count(send);
        std::vector<int> counts(world.size);
        std::vector<int> displs;
        int              err = MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, dst, MPI_COMM_WORLD);
        if(err != 0) tools::log->error("mpi::gatherv: MPI_Gather exit code {}", err);
        if constexpr(sfinae::has_resize_v<R>) {
            if(world.id == dst) {
                displs         = num::disps(counts);
                auto sumcounts = num::sum(counts);
                recv.resize(sumcounts); // Resize receiving buffer.
            }
        }
        err = MPI_Gatherv(mpi::get_cbuffer(send), count, mpi::get_dtype<S>(), mpi::get_buffer(recv), counts.data(), displs.data(), mpi::get_dtype<R>(), dst,
                          MPI_COMM_WORLD);
        if(err != 0) tools::log->error("mpi::gatherv: MPI_Gatherv exit code {}", err);
    }

    template<typename S, typename R>
    void scatterv(const S &send, R &recv, int src, int recvcount = -1) {
        auto t_mpi = tid::tic_token("mpi::scatterv");
        if(recvcount == -1) recvcount = mpi::get_count(recv);
        if constexpr(sfinae::has_resize_v<R>) recv.resize(recvcount); // Resize receiving buffer.
        std::vector<int> sendcounts;
        std::vector<int> displs;
        if(mpi::world.id == src) sendcounts.resize(world.size);
        int err = MPI_Gather(&recvcount, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, src, MPI_COMM_WORLD);
        if(err != 0) tools::log->error("mpi::scatterv: MPI_Gather exit code {}", err);
        if(mpi::world.id == src) displs = num::disps(sendcounts);

        err = MPI_Scatterv(mpi::get_cbuffer(send), sendcounts.data(), displs.data(), mpi::get_dtype<S>(), mpi::get_buffer(recv), recvcount, mpi::get_dtype<R>(),
                           src, MPI_COMM_WORLD);
        if(err != 0) tools::log->error("mpi::scatterv: MPI_Scatterv exit code {}", err);
    }

    void scatter(std::vector<h5pp::fs::path> &data, int srcId);
    void scatter_r(std::vector<h5pp::fs::path> &data, int srcId); // Roundrobin
#endif
}