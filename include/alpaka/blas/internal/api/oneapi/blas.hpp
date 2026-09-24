/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <complex>
#include <optional>
#include <type_traits>
#include <vector>

#include "alpaka/blas/internal/api/config.hpp"

#if ALPAKAV_HAS_ONEMKL_BLAS
namespace alpaka::blas::internal
{
    inline auto toOneMklTranspose(Transpose transpose)
    {
        switch(transpose)
        {
        case Transpose::none:
            return oneapi::mkl::transpose::nontrans;
        case Transpose::transposed:
            return oneapi::mkl::transpose::trans;
        case Transpose::conjugateTransposed:
            return oneapi::mkl::transpose::conjtrans;
        }
        return oneapi::mkl::transpose::nontrans;
    }

    inline auto toOneMklUplo(Triangle triangle)
    {
        return triangle == Triangle::upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
    }

    inline auto toOneMklDiag(Diagonal diagonal)
    {
        return diagonal == Diagonal::unit ? oneapi::mkl::diag::unit : oneapi::mkl::diag::nonunit;
    }

    inline auto toOneMklSide(Side side)
    {
        return side == Side::left ? oneapi::mkl::side::left : oneapi::mkl::side::right;
    }

    template<typename T>
    struct OneMklType
    {
        using type = std::remove_cv_t<T>;
    };

    template<RealScalar T>
    struct OneMklType<alpaka::math::Complex<T>>
    {
        using type = std::complex<T>;
    };

    template<typename T>
    using OneMklType_t = typename OneMklType<std::remove_cv_t<T>>::type;

    template<typename T>
    void validateOneMklInterop()
    {
        if constexpr(ComplexScalar<T>)
        {
            static_assert(
                sizeof(std::remove_cv_t<T>) == sizeof(OneMklType_t<T>),
                "alpaka::math::Complex must match std::complex storage for oneMKL interop.");
            static_assert(
                alignof(std::remove_cv_t<T>) == alignof(OneMklType_t<T>),
                "alpaka::math::Complex must match std::complex alignment for oneMKL interop.");
        }
    }

    template<typename T>
    auto oneMklPtr(void const* ptr) -> OneMklType_t<T> const*
    {
        validateOneMklInterop<T>();
        return static_cast<OneMklType_t<T> const*>(ptr);
    }

    template<typename T>
    auto oneMklPtr(void* ptr) -> OneMklType_t<T>*
    {
        validateOneMklInterop<T>();
        return static_cast<OneMklType_t<T>*>(ptr);
    }

    template<typename T>
    auto oneMklValuePtr(T* ptr) -> OneMklType_t<T>*
    {
        validateOneMklInterop<T>();
        return reinterpret_cast<OneMklType_t<T>*>(ptr);
    }

    template<typename T, typename T_Value>
    auto toOneMklScalar(T_Value const& value) -> OneMklType_t<T>
    {
        validateOneMklInterop<T>();
        return static_cast<OneMklType_t<T>>(static_cast<std::remove_cv_t<T>>(value));
    }

    template<typename T>
    [[nodiscard]] auto oneMklComputeModeFor(Options const& options) -> std::optional<oneapi::mkl::blas::compute_mode>
    {
        using oneapi::mkl::blas::compute_mode;

        if(options.precision == Precision::exact || options.algorithm == Algorithm::deterministic)
            return compute_mode::standard;

        if(options.algorithm == Algorithm::fastest)
        {
            // oneMKL exposes alternate compute modes for more than single-precision real and complex GEMM: they are
            // available for SYRK (and other routines) as well. Both the compute mode and the underlying library
            // support are routine- and device-dependent, so a request for an alternate mode is best-effort: if a
            // routine or device does not support it, oneMKL falls back to the routine default. Double-precision
            // requests intentionally stay with the routine default here.
            if constexpr(
                std::same_as<std::remove_cv_t<T>, float>
                || std::same_as<std::remove_cv_t<T>, alpaka::math::Complex<float>>)
                return compute_mode::any | compute_mode::prefer_alternate;
        }

        return std::nullopt;
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        CopyFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&& queue,
        auto const& x,
        auto& y,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                return oneapi::mkl::blas::copy(
                    q,
                    xd.n,
                    oneMklPtr<T>(xd.constPtr),
                    xd.inc,
                    oneMklPtr<T>(yd.mutPtr),
                    yd.inc,
                    deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        SwapFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&& queue,
        auto& x,
        auto& y,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                return oneapi::mkl::blas::swap(
                    q,
                    xd.n,
                    oneMklPtr<T>(xd.mutPtr),
                    xd.inc,
                    oneMklPtr<T>(yd.mutPtr),
                    yd.inc,
                    deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        ScalFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&& queue,
        auto alpha,
        auto& x,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto alphaT = toOneMklScalar<T>(alpha);
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                return oneapi::mkl::blas::scal(q, xd.n, alphaT, oneMklPtr<T>(xd.mutPtr), xd.inc, deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        AxpyFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&& queue,
        auto alpha,
        auto const& x,
        auto& y,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto alphaT = toOneMklScalar<T>(alpha);
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                return oneapi::mkl::blas::axpy(
                    q,
                    xd.n,
                    alphaT,
                    oneMklPtr<T>(xd.constPtr),
                    xd.inc,
                    oneMklPtr<T>(yd.mutPtr),
                    yd.inc,
                    deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        DotFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
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
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                if constexpr(ComplexScalar<T>)
                    return oneapi::mkl::blas::dotu(
                        q,
                        xd.n,
                        oneMklPtr<T>(xd.constPtr),
                        xd.inc,
                        oneMklPtr<T>(yd.constPtr),
                        yd.inc,
                        oneMklValuePtr(resultPtr),
                        deps);
                else
                    return oneapi::mkl::blas::dot(
                        q,
                        xd.n,
                        oneMklPtr<T>(xd.constPtr),
                        xd.inc,
                        oneMklPtr<T>(yd.constPtr),
                        yd.inc,
                        oneMklValuePtr(resultPtr),
                        deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        DotcFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&& queue,
        auto const& x,
        auto const& y,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using Scalar = std::remove_cv_t<Value_t<ALPAKA_TYPEOF(x)>>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                if constexpr(ComplexScalar<Scalar>)
                    return oneapi::mkl::blas::dotc(
                        q,
                        xd.n,
                        oneMklPtr<Scalar>(xd.constPtr),
                        xd.inc,
                        oneMklPtr<Scalar>(yd.constPtr),
                        yd.inc,
                        oneMklValuePtr(resultPtr),
                        deps);
                else
                    return oneapi::mkl::blas::dot(
                        q,
                        xd.n,
                        oneMklPtr<Scalar>(xd.constPtr),
                        xd.inc,
                        oneMklPtr<Scalar>(yd.constPtr),
                        yd.inc,
                        oneMklValuePtr(resultPtr),
                        deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        Nrm2Fn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&& queue,
        auto const& x,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                return oneapi::mkl::blas::nrm2(
                    q,
                    xd.n,
                    oneMklPtr<T>(xd.constPtr),
                    xd.inc,
                    oneMklValuePtr(resultPtr),
                    deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        AsumFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&& queue,
        auto const& x,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                return oneapi::mkl::blas::asum(
                    q,
                    xd.n,
                    oneMklPtr<T>(xd.constPtr),
                    xd.inc,
                    oneMklValuePtr(resultPtr),
                    deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        IamaxFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&& queue,
        auto const& x,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                auto event = oneapi::mkl::blas::iamax(
                    q,
                    xd.n,
                    oneMklPtr<T>(xd.constPtr),
                    xd.inc,
                    oneMklValuePtr(resultPtr),
                    deps);
                return q.submit(
                    [&](sycl::handler& handler)
                    {
                        handler.depends_on(event);
                        // oneMKL Iamax returns a 0-based index. Convert to the documented 1-based index and
                        // write 0 for an empty vector (n == 0), matching netlib BLAS.
                        handler.host_task(
                            [=]()
                            {
                                if(xd.n > 0)
                                    ++resultPtr[0];
                                else
                                    resultPtr[0] = 0;
                            });
                    });
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        GemvFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto const& x,
        auto beta,
        auto& y,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(A)>;
        auto const ad = makeMatrixDescriptor(A);
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto alphaT = toOneMklScalar<T>(alpha);
        auto betaT = toOneMklScalar<T>(beta);
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                return oneapi::mkl::blas::row_major::gemv(
                    q,
                    toOneMklTranspose(ad.transpose),
                    ad.rows,
                    ad.cols,
                    alphaT,
                    oneMklPtr<T>(ad.constPtr),
                    ad.ld,
                    oneMklPtr<T>(xd.constPtr),
                    xd.inc,
                    betaT,
                    oneMklPtr<T>(yd.mutPtr),
                    yd.inc,
                    deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        GemmFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
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
        auto alphaT = toOneMklScalar<T>(alpha);
        auto betaT = toOneMklScalar<T>(beta);
        auto const computeMode = oneMklComputeModeFor<T>(options);
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                if(computeMode.has_value())
                    return oneapi::mkl::blas::row_major::gemm(
                        q,
                        toOneMklTranspose(ad.transpose),
                        toOneMklTranspose(bd.transpose),
                        cd.rows,
                        cd.cols,
                        ad.transpose == Transpose::none ? ad.cols : ad.rows,
                        alphaT,
                        oneMklPtr<T>(ad.constPtr),
                        ad.ld,
                        oneMklPtr<T>(bd.constPtr),
                        bd.ld,
                        betaT,
                        oneMklPtr<T>(cd.mutPtr),
                        cd.ld,
                        *computeMode,
                        deps);

                return oneapi::mkl::blas::row_major::gemm(
                    q,
                    toOneMklTranspose(ad.transpose),
                    toOneMklTranspose(bd.transpose),
                    cd.rows,
                    cd.cols,
                    ad.transpose == Transpose::none ? ad.cols : ad.rows,
                    alphaT,
                    oneMklPtr<T>(ad.constPtr),
                    ad.ld,
                    oneMklPtr<T>(bd.constPtr),
                    bd.ld,
                    betaT,
                    oneMklPtr<T>(cd.mutPtr),
                    cd.ld,
                    deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        StridedBatchedGemmFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
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
        auto alphaT = toOneMklScalar<T>(alpha);
        auto betaT = toOneMklScalar<T>(beta);
        auto const computeMode = oneMklComputeModeFor<T>(options);
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                if(computeMode.has_value())
                    return oneapi::mkl::blas::row_major::gemm_batch(
                        q,
                        toOneMklTranspose(ad.transpose),
                        toOneMklTranspose(bd.transpose),
                        cd.rows,
                        cd.cols,
                        ad.transpose == Transpose::none ? ad.cols : ad.rows,
                        alphaT,
                        oneMklPtr<T>(ad.constPtr),
                        ad.ld,
                        ad.batchStride,
                        oneMklPtr<T>(bd.constPtr),
                        bd.ld,
                        bd.batchStride,
                        betaT,
                        oneMklPtr<T>(cd.mutPtr),
                        cd.ld,
                        cd.batchStride,
                        cd.batchCount,
                        *computeMode,
                        deps);

                return oneapi::mkl::blas::row_major::gemm_batch(
                    q,
                    toOneMklTranspose(ad.transpose),
                    toOneMklTranspose(bd.transpose),
                    cd.rows,
                    cd.cols,
                    ad.transpose == Transpose::none ? ad.cols : ad.rows,
                    alphaT,
                    oneMklPtr<T>(ad.constPtr),
                    ad.ld,
                    ad.batchStride,
                    oneMklPtr<T>(bd.constPtr),
                    bd.ld,
                    bd.batchStride,
                    betaT,
                    oneMklPtr<T>(cd.mutPtr),
                    cd.ld,
                    cd.batchStride,
                    cd.batchCount,
                    deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        TrsmFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
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
        auto alphaT = toOneMklScalar<T>(alpha);
        auto const computeMode = oneMklComputeModeFor<T>(options);
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                if(computeMode.has_value())
                    return oneapi::mkl::blas::row_major::trsm(
                        q,
                        toOneMklSide(side),
                        toOneMklUplo(ad.triangle),
                        toOneMklTranspose(ad.transpose),
                        toOneMklDiag(ad.diagonal),
                        bd.rows,
                        bd.cols,
                        alphaT,
                        oneMklPtr<T>(ad.constPtr),
                        ad.ld,
                        oneMklPtr<T>(bd.mutPtr),
                        bd.ld,
                        *computeMode,
                        deps);

                return oneapi::mkl::blas::row_major::trsm(
                    q,
                    toOneMklSide(side),
                    toOneMklUplo(ad.triangle),
                    toOneMklTranspose(ad.transpose),
                    toOneMklDiag(ad.diagonal),
                    bd.rows,
                    bd.cols,
                    alphaT,
                    oneMklPtr<T>(ad.constPtr),
                    ad.ld,
                    oneMklPtr<T>(bd.mutPtr),
                    bd.ld,
                    deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        HerkFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
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
        // cases (including the beta scaling semantics) before dispatch, so this routine is only called for a
        // well-defined update (n, k > 0).
        auto const n = ad.transpose == Transpose::none ? ad.rows : ad.cols;
        auto const k = ad.transpose == Transpose::none ? ad.cols : ad.rows;
        // oneMKL herk expects real scalars (value_or_pointer<Treal>), so use REAL coefficients, not the complex type.
        auto const alphaT = static_cast<Real_t<T>>(alpha);
        auto const betaT = static_cast<Real_t<T>>(beta);
        // oneMKL alternate compute modes (prefer_alternate) are GEMM-only and must NOT be requested for HERK, so the
        // options are translated conservatively: a standard/deterministic request maps to compute_mode::standard,
        // any other request leaves the oneMKL routine default (compute_mode::unset). The mode is always passed
        // explicitly so options are never silently discarded; the mapping mirrors the GEMM/trsm dispatches' helper
        // for the supported modes while never requesting the GEMM-only alternate mode.
        auto const computeMode = options.precision == Precision::exact || options.algorithm == Algorithm::deterministic
                                     ? oneapi::mkl::blas::compute_mode::standard
                                     : oneapi::mkl::blas::compute_mode::unset;
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                // oneMKL is row-major native, so the public triangle/operation are forwarded unchanged.
                return oneapi::mkl::blas::row_major::herk(
                    q,
                    toOneMklUplo(cd.triangle),
                    toOneMklTranspose(ad.transpose),
                    n,
                    k,
                    alphaT,
                    oneMklPtr<T>(ad.constPtr),
                    ad.ld,
                    betaT,
                    oneMklPtr<T>(cd.mutPtr),
                    cd.ld,
                    computeMode,
                    deps);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        SyrkFn::Spec<alpaka::api::OneApi, T_DeviceKind>,
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
        // For real operands conjugateTransposed is the identity-conjugated transpose: normalize to transposed.
        auto const op
            = ad.transpose == Transpose::none ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
        // oneMKL constructs its value_or_pointer from the scalar by value; no conversion is needed because alpha/beta
        // already have the canonical element type T.
        auto const alphaT = alpha;
        auto const betaT = beta;
        auto const computeMode = oneMklComputeModeFor<T>(options);
        queue.enqueueNativeFn(
            [=](sycl::queue q) -> sycl::event
            {
                auto deps = std::vector<sycl::event>{q.ext_oneapi_submit_barrier()};
                if(computeMode.has_value())
                    return oneapi::mkl::blas::row_major::syrk(
                        q,
                        toOneMklUplo(cd.triangle),
                        op,
                        n,
                        k,
                        alphaT,
                        oneMklPtr<T>(ad.constPtr),
                        ad.ld,
                        betaT,
                        oneMklPtr<T>(cd.mutPtr),
                        cd.ld,
                        *computeMode,
                        deps);

                return oneapi::mkl::blas::row_major::syrk(
                    q,
                    toOneMklUplo(cd.triangle),
                    op,
                    n,
                    k,
                    alphaT,
                    oneMklPtr<T>(ad.constPtr),
                    ad.ld,
                    betaT,
                    oneMklPtr<T>(cd.mutPtr),
                    cd.ld,
                    deps);
            });
    }

} // namespace alpaka::blas::internal
#endif
