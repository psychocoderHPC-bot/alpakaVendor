/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <type_traits>

#include "alpaka/blas/internal/api/config.hpp"
#include "alpaka/blas/internal/api/iamaxKernel.hpp"

#if ALPAKAV_DEP_CUBLAS && ALPAKAV_HAS_CUBLAS
namespace alpaka::blas::internal
{
    template<typename T>
    struct CublasTraits;

    template<>
    struct CublasTraits<float>
    {
        static constexpr auto dataType = CUDA_R_32F;
        static constexpr auto exactComputeType = CUBLAS_COMPUTE_32F_PEDANTIC;
        static constexpr auto backendDefaultComputeType = CUBLAS_COMPUTE_32F;
    };

    template<>
    struct CublasTraits<double>
    {
        static constexpr auto dataType = CUDA_R_64F;
        static constexpr auto exactComputeType = CUBLAS_COMPUTE_64F_PEDANTIC;
        static constexpr auto backendDefaultComputeType = CUBLAS_COMPUTE_64F;
    };

    template<>
    struct CublasTraits<alpaka::math::Complex<float>>
    {
        static constexpr auto dataType = CUDA_C_32F;
        static constexpr auto exactComputeType = CUBLAS_COMPUTE_32F_PEDANTIC;
        static constexpr auto backendDefaultComputeType = CUBLAS_COMPUTE_32F;
    };

    template<>
    struct CublasTraits<alpaka::math::Complex<double>>
    {
        static constexpr auto dataType = CUDA_C_64F;
        static constexpr auto exactComputeType = CUBLAS_COMPUTE_64F_PEDANTIC;
        static constexpr auto backendDefaultComputeType = CUBLAS_COMPUTE_64F;
    };

    inline void check(cublasStatus_t status, char const* what)
    {
        if(status != CUBLAS_STATUS_SUCCESS)
            throw std::invalid_argument(
                std::string{what} + " failed with cuBLAS error code " + std::to_string(int(status)));
    }

    struct CublasHandle
    {
        cublasHandle_t handle{};

        explicit CublasHandle(auto nativeStream)
        {
            check(cublasCreate(&handle), "cublasCreate");
            check(cublasSetStream(handle, nativeStream), "cublasSetStream");
        }

        ~CublasHandle()
        {
            if(handle != nullptr)
                static_cast<void>(cublasDestroy(handle));
        }
    };

    inline auto toCublasOp(Transpose transpose)
    {
        switch(transpose)
        {
        case Transpose::none:
            return CUBLAS_OP_N;
        case Transpose::transposed:
            return CUBLAS_OP_T;
        case Transpose::conjugateTransposed:
            return CUBLAS_OP_C;
        }
        return CUBLAS_OP_N;
    }

    /**
     * RAII guard that switches a cuBLAS handle to device pointer mode for the duration of a scope and restores the
     * previously active mode afterwards.
     *
     * Reduction routines write their scalar result to a device-accessible pointer, so the handle must be in device
     * pointer mode while the call runs. Restoring the previous mode keeps the handle consistent even if the backend
     * call throws, and avoids leaking the device mode into any later use of the same handle.
     */
    struct CublasPointerModeGuard
    {
        cublasHandle_t handle;
        cublasPointerMode_t previous = CUBLAS_POINTER_MODE_HOST;

        explicit CublasPointerModeGuard(cublasHandle_t handleIn) : handle(handleIn)
        {
            check(cublasGetPointerMode(handle, &previous), "cublasGetPointerMode");
            check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
        }

        ~CublasPointerModeGuard()
        {
            // Do not throw from a destructor; the previous mode is the best-effort fallback.
            static_cast<void>(cublasSetPointerMode(handle, previous));
        }

        CublasPointerModeGuard(CublasPointerModeGuard const&) = delete;
        CublasPointerModeGuard& operator=(CublasPointerModeGuard const&) = delete;
    };

    template<typename T>
    inline auto toCublasGemvOp(Transpose transpose)
    {
        switch(transpose)
        {
        case Transpose::none:
            return CUBLAS_OP_T;
        case Transpose::transposed:
            return CUBLAS_OP_N;
        case Transpose::conjugateTransposed:
            if constexpr(RealScalar<T>)
                return CUBLAS_OP_N;
            else
                throw std::invalid_argument(
                    "CUDA GEMV does not support row-major conjugate-transposed complex operands yet.");
        }
        return CUBLAS_OP_T;
    }

    inline auto toCublasFill(Triangle triangle)
    {
        return triangle == Triangle::upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    }

    inline auto toCublasDiag(Diagonal diagonal)
    {
        return diagonal == Diagonal::unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
    }

    inline auto toCublasSide(Side side)
    {
        return side == Side::left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    }

    inline auto swappedTriangle(Triangle triangle)
    {
        return triangle == Triangle::upper ? Triangle::lower : Triangle::upper;
    }

    template<typename T>
    inline void setMathMode(cublasHandle_t handle, Options const& options)
    {
        if constexpr(std::same_as<T, float> || std::same_as<T, alpaka::math::Complex<float>>)
        {
            if(options.precision == Precision::exact)
                check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
            else
                check(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
        }
    }

    inline void setAtomicsMode(cublasHandle_t handle, Options const& options)
    {
        if(options.algorithm == Algorithm::deterministic)
            check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode");
        else if(options.algorithm == Algorithm::fastest)
            check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED), "cublasSetAtomicsMode");
    }

    template<typename T>
    [[nodiscard]] inline auto computeTypeFor(Options const& options)
    {
        return options.precision == Precision::exact ? CublasTraits<T>::exactComputeType
                                                     : CublasTraits<T>::backendDefaultComputeType;
    }

    void alpakaFnDispatch(
        CopyFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto const& x,
        auto& y,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasScopy(
                            handle,
                            int(xd.n),
                            static_cast<float const*>(xd.constPtr),
                            int(xd.inc),
                            static_cast<float*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasScopy");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDcopy(
                            handle,
                            int(xd.n),
                            static_cast<double const*>(xd.constPtr),
                            int(xd.inc),
                            static_cast<double*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasDcopy");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCcopy(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasCcopy");
                else
                    check(
                        cublasZcopy(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasZcopy");
            });
    }

    void alpakaFnDispatch(
        SwapFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto& x,
        auto& y,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSswap(
                            handle,
                            int(xd.n),
                            static_cast<float*>(xd.mutPtr),
                            int(xd.inc),
                            static_cast<float*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasSswap");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDswap(
                            handle,
                            int(xd.n),
                            static_cast<double*>(xd.mutPtr),
                            int(xd.inc),
                            static_cast<double*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasDswap");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCswap(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuComplex*>(xd.mutPtr),
                            int(xd.inc),
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasCswap");
                else
                    check(
                        cublasZswap(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuDoubleComplex*>(xd.mutPtr),
                            int(xd.inc),
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasZswap");
            });
    }

    void alpakaFnDispatch(
        ScalFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto alpha,
        auto& x,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                T alphaT = static_cast<T>(alpha);
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSscal(handle, int(xd.n), &alphaT, static_cast<float*>(xd.mutPtr), int(xd.inc)),
                        "cublasSscal");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDscal(handle, int(xd.n), &alphaT, static_cast<double*>(xd.mutPtr), int(xd.inc)),
                        "cublasDscal");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCscal(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex*>(xd.mutPtr),
                            int(xd.inc)),
                        "cublasCscal");
                else
                    check(
                        cublasZscal(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex*>(xd.mutPtr),
                            int(xd.inc)),
                        "cublasZscal");
            });
    }

    void alpakaFnDispatch(
        AxpyFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto alpha,
        auto const& x,
        auto& y,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                T alphaT = static_cast<T>(alpha);
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSaxpy(
                            handle,
                            int(xd.n),
                            &alphaT,
                            static_cast<float const*>(xd.constPtr),
                            int(xd.inc),
                            static_cast<float*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasSaxpy");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDaxpy(
                            handle,
                            int(xd.n),
                            &alphaT,
                            static_cast<double const*>(xd.constPtr),
                            int(xd.inc),
                            static_cast<double*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasDaxpy");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCaxpy(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasCaxpy");
                else
                    check(
                        cublasZaxpy(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasZaxpy");
            });
    }

    void alpakaFnDispatch(
        DotFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto const& x,
        auto const& y,
        auto& result,
        Options options)
    {
        using T = std::remove_cv_t<Value_t<ALPAKA_TYPEOF(x)>>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                CublasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSdot(
                            handle,
                            int(xd.n),
                            static_cast<float const*>(xd.constPtr),
                            int(xd.inc),
                            static_cast<float const*>(yd.constPtr),
                            int(yd.inc),
                            resultPtr),
                        "cublasSdot");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDdot(
                            handle,
                            int(xd.n),
                            static_cast<double const*>(xd.constPtr),
                            int(xd.inc),
                            static_cast<double const*>(yd.constPtr),
                            int(yd.inc),
                            resultPtr),
                        "cublasDdot");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCdotu(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<cuComplex const*>(yd.constPtr),
                            int(yd.inc),
                            reinterpret_cast<cuComplex*>(resultPtr)),
                        "cublasCdotu");
                else
                    check(
                        cublasZdotu(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<cuDoubleComplex const*>(yd.constPtr),
                            int(yd.inc),
                            reinterpret_cast<cuDoubleComplex*>(resultPtr)),
                        "cublasZdotu");
            });
    }

    void alpakaFnDispatch(
        DotcFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto const& x,
        auto const& y,
        auto& result,
        Options options)
    {
        using Scalar = std::remove_cv_t<Value_t<ALPAKA_TYPEOF(x)>>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto const nInt = checkedCast<int>(xd.n, "dotc n");
        auto const incxInt = checkedCast<int>(xd.inc, "dotc incx");
        auto const incyInt = checkedCast<int>(yd.inc, "dotc incy");
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<Scalar>(handle, options);
                CublasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<Scalar, float>)
                    check(
                        cublasSdot(
                            handle,
                            nInt,
                            static_cast<float const*>(xd.constPtr),
                            incxInt,
                            static_cast<float const*>(yd.constPtr),
                            incyInt,
                            resultPtr),
                        "cublasSdot");
                else if constexpr(std::same_as<Scalar, double>)
                    check(
                        cublasDdot(
                            handle,
                            nInt,
                            static_cast<double const*>(xd.constPtr),
                            incxInt,
                            static_cast<double const*>(yd.constPtr),
                            incyInt,
                            resultPtr),
                        "cublasDdot");
                else if constexpr(std::same_as<Scalar, alpaka::math::Complex<float>>)
                    check(
                        cublasCdotc(
                            handle,
                            nInt,
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            incxInt,
                            reinterpret_cast<cuComplex const*>(yd.constPtr),
                            incyInt,
                            reinterpret_cast<cuComplex*>(resultPtr)),
                        "cublasCdotc");
                else
                    check(
                        cublasZdotc(
                            handle,
                            nInt,
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            incxInt,
                            reinterpret_cast<cuDoubleComplex const*>(yd.constPtr),
                            incyInt,
                            reinterpret_cast<cuDoubleComplex*>(resultPtr)),
                        "cublasZdotc");
            });
    }

    void alpakaFnDispatch(
        Nrm2Fn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto const& x,
        auto& result,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                CublasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSnrm2(handle, int(xd.n), static_cast<float const*>(xd.constPtr), int(xd.inc), resultPtr),
                        "cublasSnrm2");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDnrm2(
                            handle,
                            int(xd.n),
                            static_cast<double const*>(xd.constPtr),
                            int(xd.inc),
                            resultPtr),
                        "cublasDnrm2");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasScnrm2(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            int(xd.inc),
                            resultPtr),
                        "cublasScnrm2");
                else
                    check(
                        cublasDznrm2(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            int(xd.inc),
                            resultPtr),
                        "cublasDznrm2");
            });
    }

    void alpakaFnDispatch(
        AsumFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto const& x,
        auto& result,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                CublasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSasum(handle, int(xd.n), static_cast<float const*>(xd.constPtr), int(xd.inc), resultPtr),
                        "cublasSasum");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDasum(
                            handle,
                            int(xd.n),
                            static_cast<double const*>(xd.constPtr),
                            int(xd.inc),
                            resultPtr),
                        "cublasDasum");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasScasum(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            int(xd.inc),
                            resultPtr),
                        "cublasScasum");
                else
                    check(
                        cublasDzasum(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            int(xd.inc),
                            resultPtr),
                        "cublasDzasum");
            });
    }

    void alpakaFnDispatch(
        IamaxFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto const& x,
        auto& result,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                CublasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasIsamax(
                            handle,
                            int(xd.n),
                            static_cast<float const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIsamax");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasIdamax(
                            handle,
                            int(xd.n),
                            static_cast<double const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIdamax");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasIcamax(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIcamax");
                else
                    check(
                        cublasIzamax(
                            handle,
                            int(xd.n),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIzamax");
            });
        // cuBLAS already returns a 1-based index for n > 0. Enforce 0 for n <= 0 independently of the vendor in a
        // regular alpaka kernel on the same queue, preserving sequencing and queue-kind semantics (e.g. blocking).
        queue.enqueue(
            alpaka::onHost::ThreadSpec{1u, 1u},
            IamaxZeroForEmptyKernel{},
            reinterpret_cast<int*>(resultPtr),
            static_cast<int>(xd.n));
    }

    void alpakaFnDispatch(
        GemmFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto const& B,
        auto beta,
        auto& C,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(A)>;
        auto const ad = makeMatrixDescriptor(A);
        auto const bd = makeMatrixDescriptor(B);
        auto const cd = makeMatrixDescriptor(C);
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                setAtomicsMode(handle, options);
                T alphaT = static_cast<T>(alpha);
                T betaT = static_cast<T>(beta);
                check(
                    cublasGemmEx(
                        handle,
                        toCublasOp(bd.transpose),
                        toCublasOp(ad.transpose),
                        int(cd.cols),
                        int(cd.rows),
                        int(ad.transpose == Transpose::none ? ad.cols : ad.rows),
                        &alphaT,
                        bd.constPtr,
                        CublasTraits<T>::dataType,
                        int(bd.ld),
                        ad.constPtr,
                        CublasTraits<T>::dataType,
                        int(ad.ld),
                        &betaT,
                        cd.mutPtr,
                        CublasTraits<T>::dataType,
                        int(cd.ld),
                        computeTypeFor<T>(options),
                        CUBLAS_GEMM_DEFAULT),
                    "cublasGemmEx");
            });
    }

    void alpakaFnDispatch(
        StridedBatchedGemmFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto const& B,
        auto beta,
        auto& C,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(A)>;
        auto const ad = makeBatchedMatrixDescriptor(A);
        auto const bd = makeBatchedMatrixDescriptor(B);
        auto const cd = makeBatchedMatrixDescriptor(C);
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                setAtomicsMode(handle, options);
                T alphaT = static_cast<T>(alpha);
                T betaT = static_cast<T>(beta);
                check(
                    cublasGemmStridedBatchedEx(
                        handle,
                        toCublasOp(bd.transpose),
                        toCublasOp(ad.transpose),
                        int(cd.cols),
                        int(cd.rows),
                        int(ad.transpose == Transpose::none ? ad.cols : ad.rows),
                        &alphaT,
                        bd.constPtr,
                        CublasTraits<T>::dataType,
                        int(bd.ld),
                        static_cast<long long>(bd.batchStride),
                        ad.constPtr,
                        CublasTraits<T>::dataType,
                        int(ad.ld),
                        static_cast<long long>(ad.batchStride),
                        &betaT,
                        cd.mutPtr,
                        CublasTraits<T>::dataType,
                        int(cd.ld),
                        static_cast<long long>(cd.batchStride),
                        int(cd.batchCount),
                        computeTypeFor<T>(options),
                        CUBLAS_GEMM_DEFAULT),
                    "cublasGemmStridedBatchedEx");
            });
    }

    void alpakaFnDispatch(
        GemvFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto const& x,
        auto beta,
        auto& y,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(A)>;
        auto const ad = makeMatrixDescriptor(A);
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                setAtomicsMode(handle, options);
                T alphaT = static_cast<T>(alpha);
                T betaT = static_cast<T>(beta);
                auto const op = toCublasGemvOp<T>(ad.transpose);
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSgemv(
                            handle,
                            op,
                            int(ad.cols),
                            int(ad.rows),
                            &alphaT,
                            static_cast<float const*>(ad.constPtr),
                            int(ad.ld),
                            static_cast<float const*>(xd.constPtr),
                            int(xd.inc),
                            &betaT,
                            static_cast<float*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasSgemv");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDgemv(
                            handle,
                            op,
                            int(ad.cols),
                            int(ad.rows),
                            &alphaT,
                            static_cast<double const*>(ad.constPtr),
                            int(ad.ld),
                            static_cast<double const*>(xd.constPtr),
                            int(xd.inc),
                            &betaT,
                            static_cast<double*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasDgemv");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCgemv(
                            handle,
                            op,
                            int(ad.cols),
                            int(ad.rows),
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex const*>(ad.constPtr),
                            int(ad.ld),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<cuComplex*>(&betaT),
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasCgemv");
                else
                    check(
                        cublasZgemv(
                            handle,
                            op,
                            int(ad.cols),
                            int(ad.rows),
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex const*>(ad.constPtr),
                            int(ad.ld),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            int(xd.inc),
                            reinterpret_cast<cuDoubleComplex*>(&betaT),
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            int(yd.inc)),
                        "cublasZgemv");
            });
    }

    void alpakaFnDispatch(
        TrsmFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        Side side,
        auto alpha,
        auto const& A,
        auto& B,
        Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(A)>;
        auto const ad = makeMatrixDescriptor(A);
        auto const bd = makeMatrixDescriptor(B);
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                setAtomicsMode(handle, options);
                T alphaT = static_cast<T>(alpha);
                auto const colSide = side == Side::left ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
                auto const colTriangle = swappedTriangle(ad.triangle);
                auto const colOp = toCublasOp(ad.transpose);
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasStrsm(
                            handle,
                            colSide,
                            toCublasFill(colTriangle),
                            colOp,
                            toCublasDiag(ad.diagonal),
                            int(bd.cols),
                            int(bd.rows),
                            &alphaT,
                            static_cast<float const*>(ad.constPtr),
                            int(ad.ld),
                            static_cast<float*>(bd.mutPtr),
                            int(bd.ld)),
                        "cublasStrsm");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDtrsm(
                            handle,
                            colSide,
                            toCublasFill(colTriangle),
                            colOp,
                            toCublasDiag(ad.diagonal),
                            int(bd.cols),
                            int(bd.rows),
                            &alphaT,
                            static_cast<double const*>(ad.constPtr),
                            int(ad.ld),
                            static_cast<double*>(bd.mutPtr),
                            int(bd.ld)),
                        "cublasDtrsm");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCtrsm(
                            handle,
                            colSide,
                            toCublasFill(colTriangle),
                            colOp,
                            toCublasDiag(ad.diagonal),
                            int(bd.cols),
                            int(bd.rows),
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex const*>(ad.constPtr),
                            int(ad.ld),
                            reinterpret_cast<cuComplex*>(bd.mutPtr),
                            int(bd.ld)),
                        "cublasCtrsm");
                else
                    check(
                        cublasZtrsm(
                            handle,
                            colSide,
                            toCublasFill(colTriangle),
                            colOp,
                            toCublasDiag(ad.diagonal),
                            int(bd.cols),
                            int(bd.rows),
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex const*>(ad.constPtr),
                            int(ad.ld),
                            reinterpret_cast<cuDoubleComplex*>(bd.mutPtr),
                            int(bd.ld)),
                        "cublasZtrsm");
            });
    }

    void alpakaFnDispatch(
        HerkFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto beta,
        auto& C,
        Options options)
    {
        // Value_t keeps cv-qualifiers; dispatch on the unqualified scalar so a const-element A (read-only input)
        // selects the same vendor branch as a writable A.
        using T = std::remove_cv_t<Value_t<ALPAKA_TYPEOF(A)>>;
        static_assert(ComplexScalar<T>, "herk supports only complex scalar types.");
        auto const ad = makeMatrixDescriptor(A);
        auto const cd = makeMatrixDescriptor(C);
        // Logical (post-op) extents: op(A) is n x k. The public wrapper intercepts the degenerate n == 0 / k == 0
        // cases before dispatch, so this routine is only called for a well-defined update (n, k > 0).
        auto const n = ad.transpose == Transpose::none ? ad.rows : ad.cols;
        auto const k = ad.transpose == Transpose::none ? ad.cols : ad.rows;
        auto const nInt = checkedCast<int>(n, "herk n");
        auto const kInt = checkedCast<int>(k, "herk k");
        auto const adLd = checkedCast<int>(ad.ld, "herk A ld");
        auto const cdLd = checkedCast<int>(cd.ld, "herk C ld");
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                setAtomicsMode(handle, options);
                using Real = Real_t<T>;
                Real alphaT = static_cast<Real>(alpha);
                Real betaT = static_cast<Real>(beta);
                // Row-major C = alpha*M*adjoint(M) + beta*C is, seen column-major, D = C^T. cuBLAS herk computes
                // D = op(B)*op(B)^H, and real alpha/beta avoid conjugating the coefficients. Reinterpreting
                // row-major A as B = A^T gives: public none (M = A) -> D = A*A^H = B^H*B, so op(B) = conjugate
                // transpose; public conjTransposed (M = A^H) -> D = A^T*conj(A) = B*B^H, so op(B) = none.
                auto const colOp = ad.transpose == Transpose::none ? CUBLAS_OP_C : CUBLAS_OP_N;
                auto const colTriangle = swappedTriangle(cd.triangle);
                if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCherk(
                            handle,
                            toCublasFill(colTriangle),
                            colOp,
                            nInt,
                            kInt,
                            &alphaT,
                            reinterpret_cast<cuComplex const*>(ad.constPtr),
                            adLd,
                            &betaT,
                            reinterpret_cast<cuComplex*>(cd.mutPtr),
                            cdLd),
                        "cublasCherk");
                else
                    check(
                        cublasZherk(
                            handle,
                            toCublasFill(colTriangle),
                            colOp,
                            nInt,
                            kInt,
                            &alphaT,
                            reinterpret_cast<cuDoubleComplex const*>(ad.constPtr),
                            adLd,
                            &betaT,
                            reinterpret_cast<cuDoubleComplex*>(cd.mutPtr),
                            cdLd),
                        "cublasZherk");
            });
    }

    void alpakaFnDispatch(
        SyrkFn::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto beta,
        auto& C,
        Options options)
    {
        using T = std::remove_cv_t<Value_t<ALPAKA_TYPEOF(A)>>;
        static_assert(RealScalar<T>, "syrk supports only real scalar types.");
        // The public syrk entry converts alpha/beta once; the dispatch receives canonical T scalars already and must
        // not re-cast them (convert-once semantics).
        static_assert(std::same_as<decltype(alpha), T>, "syrk alpha must arrive as the canonical scalar.");
        static_assert(std::same_as<decltype(beta), T>, "syrk beta must arrive as the canonical scalar.");
        auto const ad = makeMatrixDescriptor(A);
        auto const cd = makeMatrixDescriptor(C);
        auto const n = ad.transpose == Transpose::none ? ad.rows : ad.cols;
        auto const k = ad.transpose == Transpose::none ? ad.cols : ad.rows;
        auto const nInt = checkedCast<int>(n, "syrk n");
        auto const kInt = checkedCast<int>(k, "syrk k");
        auto const adLd = checkedCast<int>(ad.ld, "syrk A ld");
        auto const cdLd = checkedCast<int>(cd.ld, "syrk C ld");
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                setAtomicsMode(handle, options);
                // Row-major C = alpha*M*M^T + beta*C with C row-major n x n is, seen column-major,
                // D = C^T = alpha*M^T*M + beta*D. cuBLAS syrk computes D = op(B)*op(B)^T, so we need
                // op(B) = M^T. M = op_public(A), so M^T = op_flipped(A) where flipped(none)=T, flipped(T)=N.
                auto const colOp = ad.transpose == Transpose::none ? CUBLAS_OP_T : CUBLAS_OP_N;
                auto const colTriangle = swappedTriangle(cd.triangle);
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSsyrk(
                            handle,
                            toCublasFill(colTriangle),
                            colOp,
                            nInt,
                            kInt,
                            &alpha,
                            static_cast<float const*>(ad.constPtr),
                            adLd,
                            &beta,
                            static_cast<float*>(cd.mutPtr),
                            cdLd),
                        "cublasSsyrk");
                else
                    check(
                        cublasDsyrk(
                            handle,
                            toCublasFill(colTriangle),
                            colOp,
                            nInt,
                            kInt,
                            &alpha,
                            static_cast<double const*>(ad.constPtr),
                            adLd,
                            &beta,
                            static_cast<double*>(cd.mutPtr),
                            cdLd),
                        "cublasDsyrk");
            });
    }
} // namespace alpaka::blas::internal
#endif
