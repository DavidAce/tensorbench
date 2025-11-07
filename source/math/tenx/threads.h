#pragma once
#include <memory>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <thread>
namespace tenx {
    namespace threads {
        #if defined(EIGEN_USE_THREADS)
        namespace internal {


            class AffinityGuard {
                public:
                AffinityGuard() {
                    CPU_ZERO(&old_);
                    if (pthread_getaffinity_np(pthread_self(), sizeof(old_), &old_) == 0) have_old_ = true;

                    cpu_set_t set;
                    CPU_ZERO(&set);
                    unsigned n = std::thread::hardware_concurrency();
                    if (n == 0) n = CPU_SETSIZE;
                    if (n > CPU_SETSIZE) n = CPU_SETSIZE;
                    for (unsigned i = 0; i < n; ++i) CPU_SET(i, &set);

                    pthread_setaffinity_np(pthread_self(), sizeof(set), &set); // best effort
                }

                ~AffinityGuard() {
                    if (have_old_) pthread_setaffinity_np(pthread_self(), sizeof(old_), &old_);
                }

                private:
                cpu_set_t old_{};
                bool have_old_ = false;
            };
            struct ThreadPoolWrapper {
                private:
                std::unique_ptr<Eigen::ThreadPool> tp;

                public:
                std::unique_ptr<Eigen::ThreadPoolDevice> dev;

                ThreadPoolWrapper(int nt);
            };
            inline unsigned int                       num_threads = 1;
            extern std::unique_ptr<ThreadPoolWrapper> singleThreadWrapper;
            extern std::unique_ptr<ThreadPoolWrapper> multiThreadWrapper;
        }

        template<typename T>
        // requires std::is_integral_v<T>
        void       setNumThreads(T num) noexcept;
        extern int getNumThreads() noexcept;
        //        internal::ThreadPoolWrapper &get() noexcept;
        const std::unique_ptr<internal::ThreadPoolWrapper> &get() noexcept;
        #else
        namespace internal {
            struct DefaultDeviceWrapper {
                std::unique_ptr<Eigen::DefaultDevice> dev;
                DefaultDeviceWrapper();
            };
            inline unsigned int                          num_threads = 1;
            extern std::unique_ptr<DefaultDeviceWrapper> defaultDeviceWrapper;
        }

        void                                                   setNumThreads([[maybe_unused]] int num);
        const std::unique_ptr<internal::DefaultDeviceWrapper> &get() noexcept;
        #endif

    }
}