/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
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
                // int(status) only formats the vendor error enum; it is not a descriptor cast (descriptor narrowing
                // goes through checkedVendorInt above).
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
        switch(triangle)
        {
        case Triangle::upper:
            return CUBLAS_FILL_MODE_UPPER;
        case Triangle::lower:
            return CUBLAS_FILL_MODE_LOWER;
        case Triangle::full:
            break;
        }
        throw std::invalid_argument("cuBLAS triangle mapping requires an explicit upper(A) or lower(A) annotation.");
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
        switch(triangle)
        {
        case Triangle::upper:
            return Triangle::lower;
        case Triangle::lower:
            return Triangle::upper;
        case Triangle::full:
            break;
        }
        throw std::invalid_argument("Triangular annotation must not be Triangle::full.");
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Cuda>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc");
        auto const ydIncInt = checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc");
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
                            xdNInt,
                            static_cast<float const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<float*>(yd.mutPtr),
                            ydIncInt),
                        "cublasScopy");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDcopy(
                            handle,
                            xdNInt,
                            static_cast<double const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<double*>(yd.mutPtr),
                            ydIncInt),
                        "cublasDcopy");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCcopy(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            ydIncInt),
                        "cublasCcopy");
                else
                    check(
                        cublasZcopy(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            ydIncInt),
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Cuda>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc");
        auto const ydIncInt = checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc");
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
                            xdNInt,
                            static_cast<float*>(xd.mutPtr),
                            xdIncInt,
                            static_cast<float*>(yd.mutPtr),
                            ydIncInt),
                        "cublasSswap");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDswap(
                            handle,
                            xdNInt,
                            static_cast<double*>(xd.mutPtr),
                            xdIncInt,
                            static_cast<double*>(yd.mutPtr),
                            ydIncInt),
                        "cublasDswap");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCswap(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuComplex*>(xd.mutPtr),
                            xdIncInt,
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            ydIncInt),
                        "cublasCswap");
                else
                    check(
                        cublasZswap(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuDoubleComplex*>(xd.mutPtr),
                            xdIncInt,
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            ydIncInt),
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Cuda>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc");
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                T alphaT = static_cast<T>(alpha);
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSscal(handle, xdNInt, &alphaT, static_cast<float*>(xd.mutPtr), xdIncInt),
                        "cublasSscal");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDscal(handle, xdNInt, &alphaT, static_cast<double*>(xd.mutPtr), xdIncInt),
                        "cublasDscal");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCscal(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex*>(xd.mutPtr),
                            xdIncInt),
                        "cublasCscal");
                else
                    check(
                        cublasZscal(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex*>(xd.mutPtr),
                            xdIncInt),
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Cuda>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc");
        auto const ydIncInt = checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc");
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
                            xdNInt,
                            &alphaT,
                            static_cast<float const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<float*>(yd.mutPtr),
                            ydIncInt),
                        "cublasSaxpy");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDaxpy(
                            handle,
                            xdNInt,
                            &alphaT,
                            static_cast<double const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<double*>(yd.mutPtr),
                            ydIncInt),
                        "cublasDaxpy");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCaxpy(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            ydIncInt),
                        "cublasCaxpy");
                else
                    check(
                        cublasZaxpy(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            ydIncInt),
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Cuda>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc");
        auto const ydIncInt = checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc");
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
                            xdNInt,
                            static_cast<float const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<float const*>(yd.constPtr),
                            ydIncInt,
                            resultPtr),
                        "cublasSdot");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDdot(
                            handle,
                            xdNInt,
                            static_cast<double const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<double const*>(yd.constPtr),
                            ydIncInt,
                            resultPtr),
                        "cublasDdot");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCdotu(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<cuComplex const*>(yd.constPtr),
                            ydIncInt,
                            reinterpret_cast<cuComplex*>(resultPtr)),
                        "cublasCdotu");
                else
                    check(
                        cublasZdotu(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<cuDoubleComplex const*>(yd.constPtr),
                            ydIncInt,
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Cuda>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc");
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                CublasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSnrm2(handle, xdNInt, static_cast<float const*>(xd.constPtr), xdIncInt, resultPtr),
                        "cublasSnrm2");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDnrm2(handle, xdNInt, static_cast<double const*>(xd.constPtr), xdIncInt, resultPtr),
                        "cublasDnrm2");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasScnrm2(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            xdIncInt,
                            resultPtr),
                        "cublasScnrm2");
                else
                    check(
                        cublasDznrm2(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            xdIncInt,
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Cuda>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc");
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                CublasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSasum(handle, xdNInt, static_cast<float const*>(xd.constPtr), xdIncInt, resultPtr),
                        "cublasSasum");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDasum(handle, xdNInt, static_cast<double const*>(xd.constPtr), xdIncInt, resultPtr),
                        "cublasDasum");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasScasum(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            xdIncInt,
                            resultPtr),
                        "cublasScasum");
                else
                    check(
                        cublasDzasum(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            xdIncInt,
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Cuda>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc");
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
                            xdNInt,
                            static_cast<float const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIsamax");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasIdamax(
                            handle,
                            xdNInt,
                            static_cast<double const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIdamax");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasIcamax(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIcamax");
                else
                    check(
                        cublasIzamax(
                            handle,
                            xdNInt,
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIzamax");
            });
        // cuBLAS already returns a 1-based index for n > 0. Enforce 0 for n <= 0 independently of the vendor in a
        // regular alpaka kernel on the same queue, preserving sequencing and queue-kind semantics (e.g. blocking).
        queue.enqueue(
            alpaka::onHost::ThreadSpec{1u, 1u},
            IamaxZeroForEmptyKernel{},
            reinterpret_cast<int*>(resultPtr),
            xdNInt);
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
        auto const cdColsInt = checkedVendorInt<alpaka::api::Cuda>(cd.cols, "cols");
        auto const cdRowsInt = checkedVendorInt<alpaka::api::Cuda>(cd.rows, "rows");
        auto const adGemmKInt
            = checkedVendorInt<alpaka::api::Cuda>(ad.transpose == Transpose::none ? ad.cols : ad.rows, "gemm k");
        auto const bdLdInt = checkedVendorInt<alpaka::api::Cuda>(bd.ld, "ld");
        auto const adLdInt = checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld");
        auto const cdLdInt = checkedVendorInt<alpaka::api::Cuda>(cd.ld, "ld");
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
                        cdColsInt,
                        cdRowsInt,
                        adGemmKInt,
                        &alphaT,
                        bd.constPtr,
                        CublasTraits<T>::dataType,
                        bdLdInt,
                        ad.constPtr,
                        CublasTraits<T>::dataType,
                        adLdInt,
                        &betaT,
                        cd.mutPtr,
                        CublasTraits<T>::dataType,
                        cdLdInt,
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
        auto const cdColsInt = checkedVendorInt<alpaka::api::Cuda>(cd.cols, "cols");
        auto const cdRowsInt = checkedVendorInt<alpaka::api::Cuda>(cd.rows, "rows");
        auto const adGemmKInt
            = checkedVendorInt<alpaka::api::Cuda>(ad.transpose == Transpose::none ? ad.cols : ad.rows, "gemm k");
        auto const bdLdInt = checkedVendorInt<alpaka::api::Cuda>(bd.ld, "ld");
        auto const adLdInt = checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld");
        auto const cdLdInt = checkedVendorInt<alpaka::api::Cuda>(cd.ld, "ld");
        auto const cdBatchcountInt = checkedVendorInt<alpaka::api::Cuda>(cd.batchCount, "batchCount");
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
                        cdColsInt,
                        cdRowsInt,
                        adGemmKInt,
                        &alphaT,
                        bd.constPtr,
                        CublasTraits<T>::dataType,
                        bdLdInt,
                        static_cast<long long>(bd.batchStride),
                        ad.constPtr,
                        CublasTraits<T>::dataType,
                        adLdInt,
                        static_cast<long long>(ad.batchStride),
                        &betaT,
                        cd.mutPtr,
                        CublasTraits<T>::dataType,
                        cdLdInt,
                        static_cast<long long>(cd.batchStride),
                        cdBatchcountInt,
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
        auto const adColsInt = checkedVendorInt<alpaka::api::Cuda>(ad.cols, "cols");
        auto const adRowsInt = checkedVendorInt<alpaka::api::Cuda>(ad.rows, "rows");
        auto const adLdInt = checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc");
        auto const ydIncInt = checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc");
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
                            adColsInt,
                            adRowsInt,
                            &alphaT,
                            static_cast<float const*>(ad.constPtr),
                            adLdInt,
                            static_cast<float const*>(xd.constPtr),
                            xdIncInt,
                            &betaT,
                            static_cast<float*>(yd.mutPtr),
                            ydIncInt),
                        "cublasSgemv");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDgemv(
                            handle,
                            op,
                            adColsInt,
                            adRowsInt,
                            &alphaT,
                            static_cast<double const*>(ad.constPtr),
                            adLdInt,
                            static_cast<double const*>(xd.constPtr),
                            xdIncInt,
                            &betaT,
                            static_cast<double*>(yd.mutPtr),
                            ydIncInt),
                        "cublasDgemv");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCgemv(
                            handle,
                            op,
                            adColsInt,
                            adRowsInt,
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex const*>(ad.constPtr),
                            adLdInt,
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<cuComplex*>(&betaT),
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            ydIncInt),
                        "cublasCgemv");
                else
                    check(
                        cublasZgemv(
                            handle,
                            op,
                            adColsInt,
                            adRowsInt,
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex const*>(ad.constPtr),
                            adLdInt,
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<cuDoubleComplex*>(&betaT),
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            ydIncInt),
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
        validateTriangularAnnotation(A);
        auto const ad = makeMatrixDescriptor(A);
        auto const bd = makeMatrixDescriptor(B);
        auto const bdColsInt = checkedVendorInt<alpaka::api::Cuda>(bd.cols, "cols");
        auto const bdRowsInt = checkedVendorInt<alpaka::api::Cuda>(bd.rows, "rows");
        auto const adLdInt = checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld");
        auto const bdLdInt = checkedVendorInt<alpaka::api::Cuda>(bd.ld, "ld");
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
                            bdColsInt,
                            bdRowsInt,
                            &alphaT,
                            static_cast<float const*>(ad.constPtr),
                            adLdInt,
                            static_cast<float*>(bd.mutPtr),
                            bdLdInt),
                        "cublasStrsm");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDtrsm(
                            handle,
                            colSide,
                            toCublasFill(colTriangle),
                            colOp,
                            toCublasDiag(ad.diagonal),
                            bdColsInt,
                            bdRowsInt,
                            &alphaT,
                            static_cast<double const*>(ad.constPtr),
                            adLdInt,
                            static_cast<double*>(bd.mutPtr),
                            bdLdInt),
                        "cublasDtrsm");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCtrsm(
                            handle,
                            colSide,
                            toCublasFill(colTriangle),
                            colOp,
                            toCublasDiag(ad.diagonal),
                            bdColsInt,
                            bdRowsInt,
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex const*>(ad.constPtr),
                            adLdInt,
                            reinterpret_cast<cuComplex*>(bd.mutPtr),
                            bdLdInt),
                        "cublasCtrsm");
                else
                    check(
                        cublasZtrsm(
                            handle,
                            colSide,
                            toCublasFill(colTriangle),
                            colOp,
                            toCublasDiag(ad.diagonal),
                            bdColsInt,
                            bdRowsInt,
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex const*>(ad.constPtr),
                            adLdInt,
                            reinterpret_cast<cuDoubleComplex*>(bd.mutPtr),
                            bdLdInt),
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
