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
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<float const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            static_cast<float*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasScopy");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDcopy(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<double const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            static_cast<double*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasDcopy");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCcopy(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasCcopy");
                else
                    check(
                        cublasZcopy(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
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
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<float*>(xd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            static_cast<float*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasSswap");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDswap(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<double*>(xd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            static_cast<double*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasDswap");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCswap(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuComplex*>(xd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasCswap");
                else
                    check(
                        cublasZswap(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuDoubleComplex*>(xd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
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
                        cublasSscal(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            &alphaT,
                            static_cast<float*>(xd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc")),
                        "cublasSscal");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDscal(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            &alphaT,
                            static_cast<double*>(xd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc")),
                        "cublasDscal");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCscal(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex*>(xd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc")),
                        "cublasCscal");
                else
                    check(
                        cublasZscal(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex*>(xd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc")),
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
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            &alphaT,
                            static_cast<float const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            static_cast<float*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasSaxpy");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDaxpy(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            &alphaT,
                            static_cast<double const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            static_cast<double*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasDaxpy");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCaxpy(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasCaxpy");
                else
                    check(
                        cublasZaxpy(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
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
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](cudaStream_t nativeStream)
            {
                CublasHandle cublas{nativeStream};
                auto handle = cublas.handle;
                setMathMode<T>(handle, options);
                check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSdot(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<float const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            static_cast<float const*>(yd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc"),
                            resultPtr),
                        "cublasSdot");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDdot(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<double const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            static_cast<double const*>(yd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc"),
                            resultPtr),
                        "cublasDdot");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCdotu(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<cuComplex const*>(yd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc"),
                            reinterpret_cast<cuComplex*>(resultPtr)),
                        "cublasCdotu");
                else
                    check(
                        cublasZdotu(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<cuDoubleComplex const*>(yd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc"),
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
                check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
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
                check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSnrm2(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<float const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            resultPtr),
                        "cublasSnrm2");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDnrm2(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<double const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            resultPtr),
                        "cublasDnrm2");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasScnrm2(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            resultPtr),
                        "cublasScnrm2");
                else
                    check(
                        cublasDznrm2(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
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
                check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasSasum(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<float const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            resultPtr),
                        "cublasSasum");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDasum(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<double const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            resultPtr),
                        "cublasDasum");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasScasum(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            resultPtr),
                        "cublasScasum");
                else
                    check(
                        cublasDzasum(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
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
                check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
                if constexpr(std::same_as<T, float>)
                    check(
                        cublasIsamax(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<float const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIsamax");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasIdamax(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            static_cast<double const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIdamax");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasIcamax(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIcamax");
                else
                    check(
                        cublasIzamax(
                            handle,
                            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<int*>(resultPtr)),
                        "cublasIzamax");
            });
        // cuBLAS already returns a 1-based index for n > 0. Enforce 0 for n <= 0 independently of the vendor in a
        // regular alpaka kernel on the same queue, preserving sequencing and queue-kind semantics (e.g. blocking).
        queue.enqueue(
            alpaka::onHost::ThreadSpec{1u, 1u},
            IamaxZeroForEmptyKernel{},
            reinterpret_cast<int*>(resultPtr),
            checkedVendorInt<alpaka::api::Cuda>(xd.n, "n"));
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
                        checkedVendorInt<alpaka::api::Cuda>(cd.cols, "cols"),
                        checkedVendorInt<alpaka::api::Cuda>(cd.rows, "rows"),
                        checkedVendorInt<alpaka::api::Cuda>(
                            ad.transpose == Transpose::none ? ad.cols : ad.rows,
                            "gemm k"),
                        &alphaT,
                        bd.constPtr,
                        CublasTraits<T>::dataType,
                        checkedVendorInt<alpaka::api::Cuda>(bd.ld, "ld"),
                        ad.constPtr,
                        CublasTraits<T>::dataType,
                        checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld"),
                        &betaT,
                        cd.mutPtr,
                        CublasTraits<T>::dataType,
                        checkedVendorInt<alpaka::api::Cuda>(cd.ld, "ld"),
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
                        checkedVendorInt<alpaka::api::Cuda>(cd.cols, "cols"),
                        checkedVendorInt<alpaka::api::Cuda>(cd.rows, "rows"),
                        checkedVendorInt<alpaka::api::Cuda>(
                            ad.transpose == Transpose::none ? ad.cols : ad.rows,
                            "gemm k"),
                        &alphaT,
                        bd.constPtr,
                        CublasTraits<T>::dataType,
                        checkedVendorInt<alpaka::api::Cuda>(bd.ld, "ld"),
                        static_cast<long long>(bd.batchStride),
                        ad.constPtr,
                        CublasTraits<T>::dataType,
                        checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld"),
                        static_cast<long long>(ad.batchStride),
                        &betaT,
                        cd.mutPtr,
                        CublasTraits<T>::dataType,
                        checkedVendorInt<alpaka::api::Cuda>(cd.ld, "ld"),
                        static_cast<long long>(cd.batchStride),
                        checkedVendorInt<alpaka::api::Cuda>(cd.batchCount, "batchCount"),
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
                            checkedVendorInt<alpaka::api::Cuda>(ad.cols, "cols"),
                            checkedVendorInt<alpaka::api::Cuda>(ad.rows, "rows"),
                            &alphaT,
                            static_cast<float const*>(ad.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld"),
                            static_cast<float const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            &betaT,
                            static_cast<float*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasSgemv");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDgemv(
                            handle,
                            op,
                            checkedVendorInt<alpaka::api::Cuda>(ad.cols, "cols"),
                            checkedVendorInt<alpaka::api::Cuda>(ad.rows, "rows"),
                            &alphaT,
                            static_cast<double const*>(ad.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld"),
                            static_cast<double const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            &betaT,
                            static_cast<double*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasDgemv");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCgemv(
                            handle,
                            op,
                            checkedVendorInt<alpaka::api::Cuda>(ad.cols, "cols"),
                            checkedVendorInt<alpaka::api::Cuda>(ad.rows, "rows"),
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex const*>(ad.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld"),
                            reinterpret_cast<cuComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<cuComplex*>(&betaT),
                            reinterpret_cast<cuComplex*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
                        "cublasCgemv");
                else
                    check(
                        cublasZgemv(
                            handle,
                            op,
                            checkedVendorInt<alpaka::api::Cuda>(ad.cols, "cols"),
                            checkedVendorInt<alpaka::api::Cuda>(ad.rows, "rows"),
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex const*>(ad.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld"),
                            reinterpret_cast<cuDoubleComplex const*>(xd.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(xd.inc, "inc"),
                            reinterpret_cast<cuDoubleComplex*>(&betaT),
                            reinterpret_cast<cuDoubleComplex*>(yd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(yd.inc, "inc")),
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
                            checkedVendorInt<alpaka::api::Cuda>(bd.cols, "cols"),
                            checkedVendorInt<alpaka::api::Cuda>(bd.rows, "rows"),
                            &alphaT,
                            static_cast<float const*>(ad.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld"),
                            static_cast<float*>(bd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(bd.ld, "ld")),
                        "cublasStrsm");
                else if constexpr(std::same_as<T, double>)
                    check(
                        cublasDtrsm(
                            handle,
                            colSide,
                            toCublasFill(colTriangle),
                            colOp,
                            toCublasDiag(ad.diagonal),
                            checkedVendorInt<alpaka::api::Cuda>(bd.cols, "cols"),
                            checkedVendorInt<alpaka::api::Cuda>(bd.rows, "rows"),
                            &alphaT,
                            static_cast<double const*>(ad.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld"),
                            static_cast<double*>(bd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(bd.ld, "ld")),
                        "cublasDtrsm");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        cublasCtrsm(
                            handle,
                            colSide,
                            toCublasFill(colTriangle),
                            colOp,
                            toCublasDiag(ad.diagonal),
                            checkedVendorInt<alpaka::api::Cuda>(bd.cols, "cols"),
                            checkedVendorInt<alpaka::api::Cuda>(bd.rows, "rows"),
                            reinterpret_cast<cuComplex*>(&alphaT),
                            reinterpret_cast<cuComplex const*>(ad.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld"),
                            reinterpret_cast<cuComplex*>(bd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(bd.ld, "ld")),
                        "cublasCtrsm");
                else
                    check(
                        cublasZtrsm(
                            handle,
                            colSide,
                            toCublasFill(colTriangle),
                            colOp,
                            toCublasDiag(ad.diagonal),
                            checkedVendorInt<alpaka::api::Cuda>(bd.cols, "cols"),
                            checkedVendorInt<alpaka::api::Cuda>(bd.rows, "rows"),
                            reinterpret_cast<cuDoubleComplex*>(&alphaT),
                            reinterpret_cast<cuDoubleComplex const*>(ad.constPtr),
                            checkedVendorInt<alpaka::api::Cuda>(ad.ld, "ld"),
                            reinterpret_cast<cuDoubleComplex*>(bd.mutPtr),
                            checkedVendorInt<alpaka::api::Cuda>(bd.ld, "ld")),
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
