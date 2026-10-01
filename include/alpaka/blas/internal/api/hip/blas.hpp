/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

#include "alpaka/blas/internal/api/config.hpp"
#include "alpaka/blas/internal/api/iamaxKernel.hpp"

#if ALPAKAV_DEP_ROCBLAS && ALPAKAV_HAS_ROCBLAS
namespace alpaka::blas::internal
{
    template<typename T>
    struct RocblasHandleTraits;

    inline void check(rocblas_status status, char const* what)
    {
        if(status != rocblas_status_success)
            throw std::invalid_argument(
                // int(status) only formats the vendor error enum; it is not a descriptor cast (descriptor narrowing
                // goes through checkedVendorInt above).
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
        switch(triangle)
        {
        case Triangle::upper:
            return rocblas_fill_upper;
        case Triangle::lower:
            return rocblas_fill_lower;
        case Triangle::full:
            break;
        }
        throw std::invalid_argument("rocBLAS triangle mapping requires an explicit upper(A) or lower(A) annotation.");
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

    /**
     * RAII guard that switches a rocBLAS handle to device pointer mode for the duration of a scope and restores the
     * previously active mode afterwards.
     *
     * Reduction routines write their scalar result to a device-accessible pointer, so the handle must be in device
     * pointer mode while the call runs. Restoring the previous mode keeps the handle consistent even if the backend
     * call throws, and avoids leaking the device mode into any later use of the same handle.
     */
    struct RocblasPointerModeGuard
    {
        rocblas_handle handle;
        rocblas_pointer_mode previous = rocblas_pointer_mode_host;

        explicit RocblasPointerModeGuard(rocblas_handle handleIn) : handle(handleIn)
        {
            check(rocblas_get_pointer_mode(handle, &previous), "rocblas_get_pointer_mode");
            check(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device), "rocblas_set_pointer_mode");
        }

        ~RocblasPointerModeGuard()
        {
            // Do not throw from a destructor; the previous mode is the best-effort fallback.
            static_cast<void>(rocblas_set_pointer_mode(handle, previous));
        }

        RocblasPointerModeGuard(RocblasPointerModeGuard const&) = delete;
        RocblasPointerModeGuard& operator=(RocblasPointerModeGuard const&) = delete;
    };

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
        auto const xdNInt = checkedVendorInt<alpaka::api::Hip>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Hip>(xd.inc, "inc");
        auto const ydIncInt = checkedVendorInt<alpaka::api::Hip>(yd.inc, "inc");
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_scopy(
                            handle,
                            xdNInt,
                            static_cast<float const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<float*>(yd.mutPtr),
                            ydIncInt),
                        "rocblas_scopy");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dcopy(
                            handle,
                            xdNInt,
                            static_cast<double const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<double*>(yd.mutPtr),
                            ydIncInt),
                        "rocblas_dcopy");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_ccopy(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_float_complex*>(yd.mutPtr),
                            ydIncInt),
                        "rocblas_ccopy");
                else
                    check(
                        rocblas_zcopy(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_double_complex*>(yd.mutPtr),
                            ydIncInt),
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Hip>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Hip>(xd.inc, "inc");
        auto const ydIncInt = checkedVendorInt<alpaka::api::Hip>(yd.inc, "inc");
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sswap(
                            handle,
                            xdNInt,
                            static_cast<float*>(xd.mutPtr),
                            xdIncInt,
                            static_cast<float*>(yd.mutPtr),
                            ydIncInt),
                        "rocblas_sswap");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dswap(
                            handle,
                            xdNInt,
                            static_cast<double*>(xd.mutPtr),
                            xdIncInt,
                            static_cast<double*>(yd.mutPtr),
                            ydIncInt),
                        "rocblas_dswap");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cswap(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_float_complex*>(xd.mutPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_float_complex*>(yd.mutPtr),
                            ydIncInt),
                        "rocblas_cswap");
                else
                    check(
                        rocblas_zswap(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_double_complex*>(xd.mutPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_double_complex*>(yd.mutPtr),
                            ydIncInt),
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Hip>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Hip>(xd.inc, "inc");
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                T alphaT = static_cast<T>(alpha);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sscal(handle, xdNInt, &alphaT, static_cast<float*>(xd.mutPtr), xdIncInt),
                        "rocblas_sscal");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dscal(handle, xdNInt, &alphaT, static_cast<double*>(xd.mutPtr), xdIncInt),
                        "rocblas_dscal");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cscal(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex*>(xd.mutPtr),
                            xdIncInt),
                        "rocblas_cscal");
                else
                    check(
                        rocblas_zscal(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex*>(xd.mutPtr),
                            xdIncInt),
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Hip>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Hip>(xd.inc, "inc");
        auto const ydIncInt = checkedVendorInt<alpaka::api::Hip>(yd.inc, "inc");
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
                            xdNInt,
                            &alphaT,
                            static_cast<float const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<float*>(yd.mutPtr),
                            ydIncInt),
                        "rocblas_saxpy");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_daxpy(
                            handle,
                            xdNInt,
                            &alphaT,
                            static_cast<double const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<double*>(yd.mutPtr),
                            ydIncInt),
                        "rocblas_daxpy");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_caxpy(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_float_complex*>(yd.mutPtr),
                            ydIncInt),
                        "rocblas_caxpy");
                else
                    check(
                        rocblas_zaxpy(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_double_complex*>(yd.mutPtr),
                            ydIncInt),
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
        using T = std::remove_cv_t<Value_t<ALPAKA_TYPEOF(x)>>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        auto const xdNInt = checkedVendorInt<alpaka::api::Hip>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Hip>(xd.inc, "inc");
        auto const ydIncInt = checkedVendorInt<alpaka::api::Hip>(yd.inc, "inc");
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                RocblasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sdot(
                            handle,
                            xdNInt,
                            static_cast<float const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<float const*>(yd.constPtr),
                            ydIncInt,
                            resultPtr),
                        "rocblas_sdot");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_ddot(
                            handle,
                            xdNInt,
                            static_cast<double const*>(xd.constPtr),
                            xdIncInt,
                            static_cast<double const*>(yd.constPtr),
                            ydIncInt,
                            resultPtr),
                        "rocblas_ddot");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cdotu(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_float_complex const*>(yd.constPtr),
                            ydIncInt,
                            reinterpret_cast<rocblas_float_complex*>(resultPtr)),
                        "rocblas_cdotu");
                else
                    check(
                        rocblas_zdotu(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_double_complex const*>(yd.constPtr),
                            ydIncInt,
                            reinterpret_cast<rocblas_double_complex*>(resultPtr)),
                        "rocblas_zdotu");
            });
    }

    void alpakaFnDispatch(
        DotcFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto const& x,
        auto const& y,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using Scalar = std::remove_cv_t<Value_t<ALPAKA_TYPEOF(x)>>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto const nInt = checkedVendorInt<alpaka::api::Hip>(xd.n, "dotc n");
        auto const incxInt = checkedVendorInt<alpaka::api::Hip>(xd.inc, "dotc incx");
        auto const incyInt = checkedVendorInt<alpaka::api::Hip>(yd.inc, "dotc incy");
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                RocblasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<Scalar, float>)
                    check(
                        rocblas_sdot(
                            handle,
                            nInt,
                            static_cast<float const*>(xd.constPtr),
                            incxInt,
                            static_cast<float const*>(yd.constPtr),
                            incyInt,
                            resultPtr),
                        "rocblas_sdot");
                else if constexpr(std::same_as<Scalar, double>)
                    check(
                        rocblas_ddot(
                            handle,
                            nInt,
                            static_cast<double const*>(xd.constPtr),
                            incxInt,
                            static_cast<double const*>(yd.constPtr),
                            incyInt,
                            resultPtr),
                        "rocblas_ddot");
                else if constexpr(std::same_as<Scalar, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cdotc(
                            handle,
                            nInt,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            incxInt,
                            reinterpret_cast<rocblas_float_complex const*>(yd.constPtr),
                            incyInt,
                            reinterpret_cast<rocblas_float_complex*>(resultPtr)),
                        "rocblas_cdotc");
                else
                    check(
                        rocblas_zdotc(
                            handle,
                            nInt,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            incxInt,
                            reinterpret_cast<rocblas_double_complex const*>(yd.constPtr),
                            incyInt,
                            reinterpret_cast<rocblas_double_complex*>(resultPtr)),
                        "rocblas_zdotc");
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Hip>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Hip>(xd.inc, "inc");
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                RocblasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_snrm2(handle, xdNInt, static_cast<float const*>(xd.constPtr), xdIncInt, resultPtr),
                        "rocblas_snrm2");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dnrm2(handle, xdNInt, static_cast<double const*>(xd.constPtr), xdIncInt, resultPtr),
                        "rocblas_dnrm2");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_scnrm2(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xdIncInt,
                            resultPtr),
                        "rocblas_scnrm2");
                else
                    check(
                        rocblas_dznrm2(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xdIncInt,
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Hip>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Hip>(xd.inc, "inc");
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                RocblasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_sasum(handle, xdNInt, static_cast<float const*>(xd.constPtr), xdIncInt, resultPtr),
                        "rocblas_sasum");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dasum(handle, xdNInt, static_cast<double const*>(xd.constPtr), xdIncInt, resultPtr),
                        "rocblas_dasum");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_scasum(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xdIncInt,
                            resultPtr),
                        "rocblas_scasum");
                else
                    check(
                        rocblas_dzasum(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xdIncInt,
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
        auto const xdNInt = checkedVendorInt<alpaka::api::Hip>(xd.n, "n");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Hip>(xd.inc, "inc");
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                RocblasPointerModeGuard pointerModeGuard{handle};
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_isamax(
                            handle,
                            xdNInt,
                            static_cast<float const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_int*>(resultPtr)),
                        "rocblas_isamax");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_idamax(
                            handle,
                            xdNInt,
                            static_cast<double const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_int*>(resultPtr)),
                        "rocblas_idamax");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_icamax(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_int*>(resultPtr)),
                        "rocblas_icamax");
                else
                    check(
                        rocblas_izamax(
                            handle,
                            xdNInt,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_int*>(resultPtr)),
                        "rocblas_izamax");
            });
        // rocBLAS already returns a 1-based index for n > 0. Enforce 0 for n <= 0 independently of the vendor in a
        // regular alpaka kernel on the same queue, preserving sequencing and queue-kind semantics (e.g. blocking).
        queue.enqueue(
            alpaka::onHost::ThreadSpec{1u, 1u},
            IamaxZeroForEmptyKernel{},
            reinterpret_cast<rocblas_int*>(resultPtr),
            xdNInt);
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
        auto const cdColsInt = checkedVendorInt<alpaka::api::Hip>(cd.cols, "cols");
        auto const cdRowsInt = checkedVendorInt<alpaka::api::Hip>(cd.rows, "rows");
        auto const adGemmKInt
            = checkedVendorInt<alpaka::api::Hip>(ad.transpose == Transpose::none ? ad.cols : ad.rows, "gemm k");
        auto const bdLdInt = checkedVendorInt<alpaka::api::Hip>(bd.ld, "ld");
        auto const adLdInt = checkedVendorInt<alpaka::api::Hip>(ad.ld, "ld");
        auto const cdLdInt = checkedVendorInt<alpaka::api::Hip>(cd.ld, "ld");
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
                            cdColsInt,
                            cdRowsInt,
                            adGemmKInt,
                            &alphaT,
                            static_cast<float const*>(bd.constPtr),
                            bdLdInt,
                            static_cast<float const*>(ad.constPtr),
                            adLdInt,
                            &betaT,
                            static_cast<float*>(cd.mutPtr),
                            cdLdInt),
                        "rocblas_sgemm");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dgemm(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cdColsInt,
                            cdRowsInt,
                            adGemmKInt,
                            &alphaT,
                            static_cast<double const*>(bd.constPtr),
                            bdLdInt,
                            static_cast<double const*>(ad.constPtr),
                            adLdInt,
                            &betaT,
                            static_cast<double*>(cd.mutPtr),
                            cdLdInt),
                        "rocblas_dgemm");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cgemm(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cdColsInt,
                            cdRowsInt,
                            adGemmKInt,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex const*>(bd.constPtr),
                            bdLdInt,
                            reinterpret_cast<rocblas_float_complex const*>(ad.constPtr),
                            adLdInt,
                            reinterpret_cast<rocblas_float_complex*>(&betaT),
                            reinterpret_cast<rocblas_float_complex*>(cd.mutPtr),
                            cdLdInt),
                        "rocblas_cgemm");
                else
                    check(
                        rocblas_zgemm(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cdColsInt,
                            cdRowsInt,
                            adGemmKInt,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex const*>(bd.constPtr),
                            bdLdInt,
                            reinterpret_cast<rocblas_double_complex const*>(ad.constPtr),
                            adLdInt,
                            reinterpret_cast<rocblas_double_complex*>(&betaT),
                            reinterpret_cast<rocblas_double_complex*>(cd.mutPtr),
                            cdLdInt),
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
        auto const cdColsInt = checkedVendorInt<alpaka::api::Hip>(cd.cols, "cols");
        auto const cdRowsInt = checkedVendorInt<alpaka::api::Hip>(cd.rows, "rows");
        auto const adGemmKInt
            = checkedVendorInt<alpaka::api::Hip>(ad.transpose == Transpose::none ? ad.cols : ad.rows, "gemm k");
        auto const bdLdInt = checkedVendorInt<alpaka::api::Hip>(bd.ld, "ld");
        auto const adLdInt = checkedVendorInt<alpaka::api::Hip>(ad.ld, "ld");
        auto const cdLdInt = checkedVendorInt<alpaka::api::Hip>(cd.ld, "ld");
        auto const cdBatchcountInt = checkedVendorInt<alpaka::api::Hip>(cd.batchCount, "batchCount");
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
                            cdColsInt,
                            cdRowsInt,
                            adGemmKInt,
                            &alphaT,
                            static_cast<float const*>(bd.constPtr),
                            bdLdInt,
                            bd.batchStride,
                            static_cast<float const*>(ad.constPtr),
                            adLdInt,
                            ad.batchStride,
                            &betaT,
                            static_cast<float*>(cd.mutPtr),
                            cdLdInt,
                            cd.batchStride,
                            cdBatchcountInt),
                        "rocblas_sgemm_strided_batched");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dgemm_strided_batched(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cdColsInt,
                            cdRowsInt,
                            adGemmKInt,
                            &alphaT,
                            static_cast<double const*>(bd.constPtr),
                            bdLdInt,
                            bd.batchStride,
                            static_cast<double const*>(ad.constPtr),
                            adLdInt,
                            ad.batchStride,
                            &betaT,
                            static_cast<double*>(cd.mutPtr),
                            cdLdInt,
                            cd.batchStride,
                            cdBatchcountInt),
                        "rocblas_dgemm_strided_batched");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cgemm_strided_batched(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cdColsInt,
                            cdRowsInt,
                            adGemmKInt,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex const*>(bd.constPtr),
                            bdLdInt,
                            bd.batchStride,
                            reinterpret_cast<rocblas_float_complex const*>(ad.constPtr),
                            adLdInt,
                            ad.batchStride,
                            reinterpret_cast<rocblas_float_complex*>(&betaT),
                            reinterpret_cast<rocblas_float_complex*>(cd.mutPtr),
                            cdLdInt,
                            cd.batchStride,
                            cdBatchcountInt),
                        "rocblas_cgemm_strided_batched");
                else
                    check(
                        rocblas_zgemm_strided_batched(
                            handle,
                            toRocblasOp(bd.transpose),
                            toRocblasOp(ad.transpose),
                            cdColsInt,
                            cdRowsInt,
                            adGemmKInt,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex const*>(bd.constPtr),
                            bdLdInt,
                            bd.batchStride,
                            reinterpret_cast<rocblas_double_complex const*>(ad.constPtr),
                            adLdInt,
                            ad.batchStride,
                            reinterpret_cast<rocblas_double_complex*>(&betaT),
                            reinterpret_cast<rocblas_double_complex*>(cd.mutPtr),
                            cdLdInt,
                            cd.batchStride,
                            cdBatchcountInt),
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
        auto const adColsInt = checkedVendorInt<alpaka::api::Hip>(ad.cols, "cols");
        auto const adRowsInt = checkedVendorInt<alpaka::api::Hip>(ad.rows, "rows");
        auto const adLdInt = checkedVendorInt<alpaka::api::Hip>(ad.ld, "ld");
        auto const xdIncInt = checkedVendorInt<alpaka::api::Hip>(xd.inc, "inc");
        auto const ydIncInt = checkedVendorInt<alpaka::api::Hip>(yd.inc, "inc");
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
                        "rocblas_sgemv");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dgemv(
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
                        "rocblas_dgemv");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cgemv(
                            handle,
                            op,
                            adColsInt,
                            adRowsInt,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex const*>(ad.constPtr),
                            adLdInt,
                            reinterpret_cast<rocblas_float_complex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_float_complex*>(&betaT),
                            reinterpret_cast<rocblas_float_complex*>(yd.mutPtr),
                            ydIncInt),
                        "rocblas_cgemv");
                else
                    check(
                        rocblas_zgemv(
                            handle,
                            op,
                            adColsInt,
                            adRowsInt,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex const*>(ad.constPtr),
                            adLdInt,
                            reinterpret_cast<rocblas_double_complex const*>(xd.constPtr),
                            xdIncInt,
                            reinterpret_cast<rocblas_double_complex*>(&betaT),
                            reinterpret_cast<rocblas_double_complex*>(yd.mutPtr),
                            ydIncInt),
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
        validateTriangularAnnotation(A);
        auto const ad = makeMatrixDescriptor(A);
        auto const bd = makeMatrixDescriptor(B);
        auto const bdColsInt = checkedVendorInt<alpaka::api::Hip>(bd.cols, "cols");
        auto const bdRowsInt = checkedVendorInt<alpaka::api::Hip>(bd.rows, "rows");
        auto const adLdInt = checkedVendorInt<alpaka::api::Hip>(ad.ld, "ld");
        auto const bdLdInt = checkedVendorInt<alpaka::api::Hip>(bd.ld, "ld");
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
                            bdColsInt,
                            bdRowsInt,
                            &alphaT,
                            static_cast<float const*>(ad.constPtr),
                            adLdInt,
                            static_cast<float*>(bd.mutPtr),
                            bdLdInt),
                        "rocblas_strsm");
                else if constexpr(std::same_as<T, double>)
                    check(
                        rocblas_dtrsm(
                            handle,
                            colSide,
                            toRocblasFill(colTriangle),
                            colOp,
                            toRocblasDiag(ad.diagonal),
                            bdColsInt,
                            bdRowsInt,
                            &alphaT,
                            static_cast<double const*>(ad.constPtr),
                            adLdInt,
                            static_cast<double*>(bd.mutPtr),
                            bdLdInt),
                        "rocblas_dtrsm");
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_ctrsm(
                            handle,
                            colSide,
                            toRocblasFill(colTriangle),
                            colOp,
                            toRocblasDiag(ad.diagonal),
                            bdColsInt,
                            bdRowsInt,
                            reinterpret_cast<rocblas_float_complex*>(&alphaT),
                            reinterpret_cast<rocblas_float_complex const*>(ad.constPtr),
                            adLdInt,
                            reinterpret_cast<rocblas_float_complex*>(bd.mutPtr),
                            bdLdInt),
                        "rocblas_ctrsm");
                else
                    check(
                        rocblas_ztrsm(
                            handle,
                            colSide,
                            toRocblasFill(colTriangle),
                            colOp,
                            toRocblasDiag(ad.diagonal),
                            bdColsInt,
                            bdRowsInt,
                            reinterpret_cast<rocblas_double_complex*>(&alphaT),
                            reinterpret_cast<rocblas_double_complex const*>(ad.constPtr),
                            adLdInt,
                            reinterpret_cast<rocblas_double_complex*>(bd.mutPtr),
                            bdLdInt),
                        "rocblas_ztrsm");
            });
    }

    void alpakaFnDispatch(
        HerkFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto beta,
        auto& C,
        [[maybe_unused]] Options options)
    {
        // Value_t keeps cv-qualifiers; dispatch on the unqualified scalar so a const-element A (read-only input)
        // selects the same vendor branch as a writable A.
        using T = std::remove_cv_t<Value_t<ALPAKA_TYPEOF(A)>>;
        static_assert(ComplexScalar<T>, "herk supports only complex scalar types.");
        auto const ad = makeMatrixDescriptor(A);
        auto const cd = makeMatrixDescriptor(C);
        // Logical (post-op) extents: op(A) is n x k. The public wrapper intercepts the degenerate n == 0 / k == 0
        // cases before dispatch, so this routine is only called for a well-defined update (n, k > 0).
        auto const nInt
            = checkedVendorInt<alpaka::api::Hip>(ad.transpose == Transpose::none ? ad.rows : ad.cols, "herk n");
        auto const kInt
            = checkedVendorInt<alpaka::api::Hip>(ad.transpose == Transpose::none ? ad.cols : ad.rows, "herk k");
        auto const adLd = checkedVendorInt<alpaka::api::Hip>(ad.ld, "herk A ld");
        auto const cdLd = checkedVendorInt<alpaka::api::Hip>(cd.ld, "herk C ld");
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setAtomicsMode(handle, options);
                using Real = Real_t<T>;
                Real alphaT = static_cast<Real>(alpha);
                Real betaT = static_cast<Real>(beta);
                // Row-major C = alpha*M*adjoint(M) + beta*C is, seen column-major, D = C^T. rocBLAS herk computes
                // D = op(B)*op(B)^H, and real alpha/beta avoid conjugating the coefficients. Reinterpreting
                // row-major A as B = A^T gives: public none (M = A) -> op(B) = conjugate transpose;
                // public conjTransposed (M = A^H) -> op(B) = none.
                auto const colOp
                    = ad.transpose == Transpose::none ? rocblas_operation_conjugate_transpose : rocblas_operation_none;
                auto const colTriangle = swappedTriangle(cd.triangle);
                if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    check(
                        rocblas_cherk(
                            handle,
                            toRocblasFill(colTriangle),
                            colOp,
                            nInt,
                            kInt,
                            &alphaT,
                            reinterpret_cast<rocblas_float_complex const*>(ad.constPtr),
                            adLd,
                            &betaT,
                            reinterpret_cast<rocblas_float_complex*>(cd.mutPtr),
                            cdLd),
                        "rocblas_cherk");
                else
                    check(
                        rocblas_zherk(
                            handle,
                            toRocblasFill(colTriangle),
                            colOp,
                            nInt,
                            kInt,
                            &alphaT,
                            reinterpret_cast<rocblas_double_complex const*>(ad.constPtr),
                            adLd,
                            &betaT,
                            reinterpret_cast<rocblas_double_complex*>(cd.mutPtr),
                            cdLd),
                        "rocblas_zherk");
            });
    }

    void alpakaFnDispatch(
        SyrkFn::Spec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>,
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
        auto const nInt
            = checkedVendorInt<alpaka::api::Hip>(ad.transpose == Transpose::none ? ad.rows : ad.cols, "syrk n");
        auto const kInt
            = checkedVendorInt<alpaka::api::Hip>(ad.transpose == Transpose::none ? ad.cols : ad.rows, "syrk k");
        auto const adLd = checkedVendorInt<alpaka::api::Hip>(ad.ld, "syrk A ld");
        auto const cdLd = checkedVendorInt<alpaka::api::Hip>(cd.ld, "syrk C ld");
        queue.enqueueNativeFn(
            [=](hipStream_t nativeStream)
            {
                RocblasHandle rocblas{nativeStream};
                auto handle = rocblas.handle;
                setAtomicsMode(handle, options);
                // Row-major C = alpha*M*M^T + beta*C is, seen column-major, D = C^T.
                // rocBLAS syrk computes D = op(B)*op(B)^T, so pass op(B)=M^T when A is as-stored.
                auto const colOp
                    = ad.transpose == Transpose::none ? rocblas_operation_transpose : rocblas_operation_none;
                auto const colTriangle = swappedTriangle(cd.triangle);
                if constexpr(std::same_as<T, float>)
                    check(
                        rocblas_ssyrk(
                            handle,
                            toRocblasFill(colTriangle),
                            colOp,
                            nInt,
                            kInt,
                            &alpha,
                            static_cast<float const*>(ad.constPtr),
                            adLd,
                            &beta,
                            static_cast<float*>(cd.mutPtr),
                            cdLd),
                        "rocblas_ssyrk");
                else
                    check(
                        rocblas_dsyrk(
                            handle,
                            toRocblasFill(colTriangle),
                            colOp,
                            nInt,
                            kInt,
                            &alpha,
                            static_cast<double const*>(ad.constPtr),
                            adLd,
                            &beta,
                            static_cast<double*>(cd.mutPtr),
                            cdLd),
                        "rocblas_dsyrk");
            });
    }
} // namespace alpaka::blas::internal
#endif
