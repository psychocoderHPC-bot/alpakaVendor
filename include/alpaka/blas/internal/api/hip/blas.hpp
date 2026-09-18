/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <type_traits>

#include "alpaka/blas/internal/api/config.hpp"

#if ALPAKAV_DEP_ROCBLAS && ALPAKAV_HAS_ROCBLAS
namespace alpaka::blas::internal
{
    template<typename T>
    struct RocblasHandleTraits;

    inline void check(rocblas_status status, char const* what)
    {
        if(status != rocblas_status_success)
            throw std::invalid_argument(
                std::string{what} + " failed with rocBLAS error code " + std::to_string(int(status)));
    }

    struct RocblasHandle
    {
        rocblas_handle handle{};

        explicit RocblasHandle(auto nativeStream)
        {
            check(rocblas_create_handle(&handle), "rocblas_create_handle");
            check(rocblas_set_stream(handle, nativeStream), "rocblas_set_stream");
        }

        ~RocblasHandle()
        {
            if(handle != nullptr)
                static_cast<void>(rocblas_destroy_handle(handle));
        }
    };

    inline auto toRocblasOp(Transpose transpose)
    {
        switch(transpose)
        {
        case Transpose::none:
            return rocblas_operation_none;
        case Transpose::transposed:
            return rocblas_operation_transpose;
        case Transpose::conjugateTransposed:
            return rocblas_operation_conjugate_transpose;
        }
        return rocblas_operation_none;
    }

    template<typename T>
    inline auto toRocblasGemvOp(Transpose transpose)
    {
        switch(transpose)
        {
        case Transpose::none:
            return rocblas_operation_transpose;
        case Transpose::transposed:
            return rocblas_operation_none;
        case Transpose::conjugateTransposed:
            if constexpr(RealScalar<T>)
                return rocblas_operation_none;
            else
                throw std::invalid_argument(
                    "HIP GEMV does not support row-major conjugate-transposed complex operands yet.");
        }
        return rocblas_operation_transpose;
    }

    inline auto toRocblasFill(Triangle triangle)
    {
        return triangle == Triangle::upper ? rocblas_fill_upper : rocblas_fill_lower;
    }

    inline auto toRocblasDiag(Diagonal diagonal)
    {
        return diagonal == Diagonal::unit ? rocblas_diagonal_unit : rocblas_diagonal_non_unit;
    }

    inline auto toRocblasSide(Side side)
    {
        return side == Side::left ? rocblas_side_left : rocblas_side_right;
    }

    inline auto swappedTriangle(Triangle triangle)
    {
        return triangle == Triangle::upper ? Triangle::lower : Triangle::upper;
    }

    template<typename T>
    inline void setPointerMode(rocblas_handle handle)
    {
        check(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device), "rocblas_set_pointer_mode");
    }

    inline void setAtomicsMode(rocblas_handle handle, Options const& options)
    {
        if(options.algorithm == Algorithm::deterministic)
        {
#    if defined(__GNUC__) || defined(__clang__)
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#    endif
            check(rocblas_set_atomics_mode(handle, rocblas_atomics_not_allowed), "rocblas_set_atomics_mode");
#    if defined(__GNUC__) || defined(__clang__)
#        pragma GCC diagnostic pop
#    endif
        }
        else if(options.algorithm == Algorithm::fastest)
        {
#    if defined(__GNUC__) || defined(__clang__)
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#    endif
            check(rocblas_set_atomics_mode(handle, rocblas_atomics_allowed), "rocblas_set_atomics_mode");
#    if defined(__GNUC__) || defined(__clang__)
#        pragma GCC diagnostic pop
#    endif
        }
    }

    void alpakaFnDispatch(
        CopyFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto const& x,
        auto& y,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_scopy(
                            handle,
                            xd.n,
                            static_cast<float const*>(xd.constPtr),
                            xd.inc,
                            static_cast<float*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_scopy");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dcopy(
                            handle,
                            xd.n,
                            static_cast<double const*>(xd.constPtr),
                            xd.inc,
                            static_cast<double*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_dcopy");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_ccopy(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_float_complex*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_ccopy");
                else
                    check(
                        rocblas_zcopy(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_double_complex*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_zcopy");
            });
    }

    void alpakaFnDispatch(
        SwapFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto& x,
        auto& y,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sswap(
                            handle,
                            xd.n,
                            static_cast<float*>(xd.mutPtr),
                            xd.inc,
                            static_cast<float*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_sswap");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dswap(
                            handle,
                            xd.n,
                            static_cast<double*>(xd.mutPtr),
                            xd.inc,
                            static_cast<double*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_dswap");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cswap(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_float_complex*>(xd.mutPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_float_complex*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_cswap");
                else
                    check(
                        rocblas_zswap(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_double_complex*>(xd.mutPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_double_complex*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_zswap");
            });
    }

    void alpakaFnDispatch(
        ScalFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto alpha,
        auto& x,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                T alphaT = static_cast<T>(alpha);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sscal(handle, xd.n, &alphaT, static_cast<float*>(xd.mutPtr), xd.inc),
                        "rocblas_sscal");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dscal(handle, xd.n, &alphaT, static_cast<double*>(xd.mutPtr), xd.inc),
                        "rocblas_dscal");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cscal(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex*>(xd.mutPtr),
                            xd.inc),
                        "rocblas_cscal");
                else
                    check(
                        rocblas_zscal(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex*>(xd.mutPtr),
                            xd.inc),
                        "rocblas_zscal");
            });
    }

    void alpakaFnDispatch(
        AxpyFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto alpha,
        auto const& x,
        auto& y,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                T alphaT = static_cast<T>(alpha);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_saxpy(
                            handle,
                            xd.n,
                            &alphaT,
                            static_cast<float const*>(xd.constPtr),
                            xd.inc,
                            static_cast<float*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_saxpy");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_daxpy(
                            handle,
                            xd.n,
                            &alphaT,
                            static_cast<double const*>(xd.constPtr),
                            xd.inc,
                            static_cast<double*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_daxpy");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_caxpy(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_float_complex*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_caxpy");
                else
                    check(
                        rocblas_zaxpy(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_double_complex*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_zaxpy");
            });
    }

    void alpakaFnDispatch(
        DotFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto const& x,
        auto const& y,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setPointerMode<T>(handle);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sdot(
                            handle,
                            xd.n,
                            static_cast<float const*>(xd.constPtr),
                            xd.inc,
                            static_cast<float const*>(yd.constPtr),
                            yd.inc,
                            resultPtr),
                        "rocblas_sdot");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_ddot(
                            handle,
                            xd.n,
                            static_cast<double const*>(xd.constPtr),
                            xd.inc,
                            static_cast<double const*>(yd.constPtr),
                            yd.inc,
                            resultPtr),
                        "rocblas_ddot");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cdotu(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_float_complex const*>(yd.constPtr),
                            yd.inc,
                            reinterpret_cast<rocblas_float_complex*>(resultPtr)),
                        "rocblas_cdotu");
                else
                    check(
                        rocblas_zdotu(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_double_complex const*>(yd.constPtr),
                            yd.inc,
                            reinterpret_cast<rocblas_double_complex*>(resultPtr)),
                        "rocblas_zdotu");
            });
    }

    void alpakaFnDispatch(
        Nrm2Fn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto const& x,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setPointerMode<T>(handle);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_snrm2(handle, xd.n, static_cast<float const*>(xd.constPtr), xd.inc, resultPtr),
                        "rocblas_snrm2");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dnrm2(handle, xd.n, static_cast<double const*>(xd.constPtr), xd.inc, resultPtr),
                        "rocblas_dnrm2");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_scnrm2(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xd.inc,
                            resultPtr),
                        "rocblas_scnrm2");
                else
                    check(
                        rocblas_dznrm2(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xd.inc,
                            resultPtr),
                        "rocblas_dznrm2");
            });
    }

    void alpakaFnDispatch(
        AsumFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto const& x,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setPointerMode<T>(handle);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sasum(handle, xd.n, static_cast<float const*>(xd.constPtr), xd.inc, resultPtr),
                        "rocblas_sasum");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dasum(handle, xd.n, static_cast<double const*>(xd.constPtr), xd.inc, resultPtr),
                        "rocblas_dasum");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_scasum(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xd.inc,
                            resultPtr),
                        "rocblas_scasum");
                else
                    check(
                        rocblas_dzasum(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xd.inc,
                            resultPtr),
                        "rocblas_dzasum");
            });
    }

    void alpakaFnDispatch(
        IamaxFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto const& x,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setPointerMode<T>(handle);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_isamax(
                            handle,
                            xd.n,
                            static_cast<float const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_int*>(resultPtr)),
                        "rocblas_isamax");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_idamax(
                            handle,
                            xd.n,
                            static_cast<double const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_int*>(resultPtr)),
                        "rocblas_idamax");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_icamax(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_int*>(resultPtr)),
                        "rocblas_icamax");
                else
                    check(
                        rocblas_izamax(
                            handle,
                            xd.n,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_int*>(resultPtr)),
                        "rocblas_izamax");
            });
    }

    void alpakaFnDispatch(
        GemmFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
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
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setAtomicsMode(handle, options);
                T alphaT = static_cast<T>(alpha);
                T betaT = static_cast<T>(beta);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sgemm(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cd.cols,
                            cd.rows,
                            ad.transpose == Transpose::none ? ad.cols : ad.rows,
                            &alphaT,
                            static_cast<float const*>(bd.constPtr),
                            bd.ld,
                            static_cast<float const*>(ad.constPtr),
                            ad.ld,
                            &betaT,
                            static_cast<float*>(cd.mutPtr),
                            cd.ld),
                        "rocblas_sgemm");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dgemm(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cd.cols,
                            cd.rows,
                            ad.transpose == Transpose::none ? ad.cols : ad.rows,
                            &alphaT,
                            static_cast<double const*>(bd.constPtr),
                            bd.ld,
                            static_cast<double const*>(ad.constPtr),
                            ad.ld,
                            &betaT,
                            static_cast<double*>(cd.mutPtr),
                            cd.ld),
                        "rocblas_dgemm");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cgemm(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cd.cols,
                            cd.rows,
                            ad.transpose == Transpose::none ? ad.cols : ad.rows,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex const*>(bd.constPtr),
                            bd.ld,
                            reinterpret_cast<rocblas_float_complex const*>(ad.constPtr),
                            ad.ld,
                            reinterpret_cast<rocblas_float_complex*>(&betaT),
                            reinterpret_cast<rocblas_float_complex*>(cd.mutPtr),
                            cd.ld),
                        "rocblas_cgemm");
                else
                    check(
                        rocblas_zgemm(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cd.cols,
                            cd.rows,
                            ad.transpose == Transpose::none ? ad.cols : ad.rows,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex const*>(bd.constPtr),
                            bd.ld,
                            reinterpret_cast<rocblas_double_complex const*>(ad.constPtr),
                            ad.ld,
                            reinterpret_cast<rocblas_double_complex*>(&betaT),
                            reinterpret_cast<rocblas_double_complex*>(cd.mutPtr),
                            cd.ld),
                        "rocblas_zgemm");
            });
    }

    void alpakaFnDispatch(
        StridedBatchedGemmFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
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
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setAtomicsMode(handle, options);
                T alphaT = static_cast<T>(alpha);
                T betaT = static_cast<T>(beta);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sgemm_strided_batched(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cd.cols,
                            cd.rows,
                            ad.transpose == Transpose::none ? ad.cols : ad.rows,
                            &alphaT,
                            static_cast<float const*>(bd.constPtr),
                            bd.ld,
                            bd.batchStride,
                            static_cast<float const*>(ad.constPtr),
                            ad.ld,
                            ad.batchStride,
                            &betaT,
                            static_cast<float*>(cd.mutPtr),
                            cd.ld,
                            cd.batchStride,
                            cd.batchCount),
                        "rocblas_sgemm_strided_batched");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dgemm_strided_batched(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cd.cols,
                            cd.rows,
                            ad.transpose == Transpose::none ? ad.cols : ad.rows,
                            &alphaT,
                            static_cast<double const*>(bd.constPtr),
                            bd.ld,
                            bd.batchStride,
                            static_cast<double const*>(ad.constPtr),
                            ad.ld,
                            ad.batchStride,
                            &betaT,
                            static_cast<double*>(cd.mutPtr),
                            cd.ld,
                            cd.batchStride,
                            cd.batchCount),
                        "rocblas_dgemm_strided_batched");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cgemm_strided_batched(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cd.cols,
                            cd.rows,
                            ad.transpose == Transpose::none ? ad.cols : ad.rows,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex const*>(bd.constPtr),
                            bd.ld,
                            bd.batchStride,
                            reinterpret_cast<rocblas_float_complex const*>(ad.constPtr),
                            ad.ld,
                            ad.batchStride,
                            reinterpret_cast<rocblas_float_complex*>(&betaT),
                            reinterpret_cast<rocblas_float_complex*>(cd.mutPtr),
                            cd.ld,
                            cd.batchStride,
                            cd.batchCount),
                        "rocblas_cgemm_strided_batched");
                else
                    check(
                        rocblas_zgemm_strided_batched(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cd.cols,
                            cd.rows,
                            ad.transpose == Transpose::none ? ad.cols : ad.rows,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex const*>(bd.constPtr),
                            bd.ld,
                            bd.batchStride,
                            reinterpret_cast<rocblas_double_complex const*>(ad.constPtr),
                            ad.ld,
                            ad.batchStride,
                            reinterpret_cast<rocblas_double_complex*>(&betaT),
                            reinterpret_cast<rocblas_double_complex*>(cd.mutPtr),
                            cd.ld,
                            cd.batchStride,
                            cd.batchCount),
                        "rocblas_zgemm_strided_batched");
            });
    }

    void alpakaFnDispatch(
        GemvFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
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
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setAtomicsMode(handle, options);
                T alphaT = static_cast<T>(alpha);
                T betaT = static_cast<T>(beta);
                auto const op = toRocblasGemvOp<T>(ad.transpose);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sgemv(
                            handle,
                            op,
                            ad.cols,
                            ad.rows,
                            &alphaT,
                            static_cast<float const*>(ad.constPtr),
                            ad.ld,
                            static_cast<float const*>(xd.constPtr),
                            xd.inc,
                            &betaT,
                            static_cast<float*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_sgemv");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dgemv(
                            handle,
                            op,
                            ad.cols,
                            ad.rows,
                            &alphaT,
                            static_cast<double const*>(ad.constPtr),
                            ad.ld,
                            static_cast<double const*>(xd.constPtr),
                            xd.inc,
                            &betaT,
                            static_cast<double*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_dgemv");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cgemv(
                            handle,
                            op,
                            ad.cols,
                            ad.rows,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex const*>(ad.constPtr),
                            ad.ld,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_float_complex*>(&betaT),
                            reinterpret_cast<rocblas_float_complex*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_cgemv");
                else
                    check(
                        rocblas_zgemv(
                            handle,
                            op,
                            ad.cols,
                            ad.rows,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex const*>(ad.constPtr),
                            ad.ld,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xd.inc,
                            reinterpret_cast<rocblas_double_complex*>(&betaT),
                            reinterpret_cast<rocblas_double_complex*>(yd.mutPtr),
                            yd.inc),
                        "rocblas_zgemv");
            });
    }

    void alpakaFnDispatch(
        TrsmFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
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
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setAtomicsMode(handle, options);
                T alphaT = static_cast<T>(alpha);
                auto const colSide = side == Side::left ? rocblas_side_right : rocblas_side_left;
                auto const colTriangle = swappedTriangle(ad.triangle);
                auto const colOp = toRocblasOp(ad.transpose);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_strsm(
                            handle,
                            colSide,
                            toRocblasFill(colTriangle),
                            colOp,
                            toRocblasDiag(ad.diagonal),
                            bd.cols,
                            bd.rows,
                            &alphaT,
                            static_cast<float const*>(ad.constPtr),
                            ad.ld,
                            static_cast<float*>(bd.mutPtr),
                            bd.ld),
                        "rocblas_strsm");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dtrsm(
                            handle,
                            colSide,
                            toRocblasFill(colTriangle),
                            colOp,
                            toRocblasDiag(ad.diagonal),
                            bd.cols,
                            bd.rows,
                            &alphaT,
                            static_cast<double const*>(ad.constPtr),
                            ad.ld,
                            static_cast<double*>(bd.mutPtr),
                            bd.ld),
                        "rocblas_dtrsm");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_ctrsm(
                            handle,
                            colSide,
                            toRocblasFill(colTriangle),
                            colOp,
                            toRocblasDiag(ad.diagonal),
                            bd.cols,
                            bd.rows,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex const*>(ad.constPtr),
                            ad.ld,
                            reinterpret_cast<rocblas_float_complex*>(bd.mutPtr),
                            bd.ld),
                        "rocblas_ctrsm");
                else
                    check(
                        rocblas_ztrsm(
                            handle,
                            colSide,
                            toRocblasFill(colTriangle),
                            colOp,
                            toRocblasDiag(ad.diagonal),
                            bd.cols,
                            bd.rows,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex const*>(ad.constPtr),
                            ad.ld,
                            reinterpret_cast<rocblas_double_complex*>(bd.mutPtr),
                            bd.ld),
                        "rocblas_ztrsm");
            });
    }

    void alpakaFnDispatch(
        SyrkFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto beta,
        auto& C,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(A)>;
        static_assert(RealScalar<T>, "syrk supports only real scalar types.");
        auto const ad = makeMatrixDescriptor(A);
        auto const cd = makeMatrixDescriptor(C);
        auto const n = ad.transpose == Transpose::none ? ad.rows : ad.cols;
        auto const k = ad.transpose == Transpose::none ? ad.cols : ad.rows;
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setPointerMode<T>(handle);
                T alphaT = static_cast<T>(alpha);
                T betaT = static_cast<T>(beta);
                // Row-major C = alpha*M*M^T + beta*C is, seen column-major, D = C^T.
                // rocBLAS syrk computes D = op(B)*op(B)^T, so pass op(B)=M^T when A is as-stored.
                auto const colOp = ad.transpose == Transpose::none ? rocblas_operation_transpose
                                                                   : rocblas_operation_none;
                auto const colTriangle = swappedTriangle(cd.triangle);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_ssyrk(
                            handle,
                            toRocblasFill(colTriangle),
                            colOp,
                            n,
                            k,
                            &alphaT,
                            static_cast<float const*>(ad.constPtr),
                            ad.ld,
                            &betaT,
                            static_cast<float*>(cd.mutPtr),
                            cd.ld),
                        "rocblas_ssyrk");
                else
                    check(
                        rocblas_dsyrk(
                            handle,
                            toRocblasFill(colTriangle),
                            colOp,
                            n,
                            k,
                            &alphaT,
                            static_cast<double const*>(ad.constPtr),
                            ad.ld,
                            &betaT,
                            static_cast<double*>(cd.mutPtr),
                            cd.ld),
                        "rocblas_dsyrk");
            });
    }
} // namespace alpaka::blas::internal
#endif
