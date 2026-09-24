/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <algorithm>
#include <cmath>

#include "alpaka/blas/internal/api/config.hpp"

#if ALPAKAV_DEP_OPENBLAS && ALPAKAV_HAS_OPENBLAS
namespace alpaka::blas::internal
{
    template<typename T>
    struct OpenBlas;

    template<>
    struct OpenBlas<float>
    {
        static void copy(int n, float const* x, int incx, float* y, int incy)
        {
            cblas_scopy(n, x, incx, y, incy);
        }

        static void swap(int n, float* x, int incx, float* y, int incy)
        {
            cblas_sswap(n, x, incx, y, incy);
        }

        static void scal(int n, float alpha, float* x, int incx)
        {
            cblas_sscal(n, alpha, x, incx);
        }

        static void axpy(int n, float alpha, float const* x, int incx, float* y, int incy)
        {
            cblas_saxpy(n, alpha, x, incx, y, incy);
        }

        static float dot(int n, float const* x, int incx, float const* y, int incy)
        {
            return cblas_sdot(n, x, incx, y, incy);
        }

        static float dotc(int n, float const* x, int incx, float const* y, int incy)
        {
            return cblas_sdot(n, x, incx, y, incy);
        }

        static float nrm2(int n, float const* x, int incx)
        {
            return cblas_snrm2(n, x, incx);
        }

        static float asum(int n, float const* x, int incx)
        {
            return cblas_sasum(n, x, incx);
        }

        static int iamax(int n, float const* x, int incx)
        {
            // netlib BLAS: n <= 0 yields 0, otherwise a 1-based index. cblas_isamax returns 0 for n <= 0,
            // so the +1 is skipped to keep the documented contract "1-based index, 0 if n == 0".
            auto const result = cblas_isamax(n, x, incx);
            return n > 0 ? result + 1 : result;
        }
    };

    template<>
    struct OpenBlas<double>
    {
        static void copy(int n, double const* x, int incx, double* y, int incy)
        {
            cblas_dcopy(n, x, incx, y, incy);
        }

        static void swap(int n, double* x, int incx, double* y, int incy)
        {
            cblas_dswap(n, x, incx, y, incy);
        }

        static void scal(int n, double alpha, double* x, int incx)
        {
            cblas_dscal(n, alpha, x, incx);
        }

        static void axpy(int n, double alpha, double const* x, int incx, double* y, int incy)
        {
            cblas_daxpy(n, alpha, x, incx, y, incy);
        }

        static double dot(int n, double const* x, int incx, double const* y, int incy)
        {
            return cblas_ddot(n, x, incx, y, incy);
        }

        static double dotc(int n, double const* x, int incx, double const* y, int incy)
        {
            return cblas_ddot(n, x, incx, y, incy);
        }

        static double nrm2(int n, double const* x, int incx)
        {
            return cblas_dnrm2(n, x, incx);
        }

        static double asum(int n, double const* x, int incx)
        {
            return cblas_dasum(n, x, incx);
        }

        static int iamax(int n, double const* x, int incx)
        {
            // netlib BLAS: n <= 0 yields 0, otherwise a 1-based index. cblas_idamax returns 0 for n <= 0,
            // so the +1 is skipped to keep the documented contract "1-based index, 0 if n == 0".
            auto const result = cblas_idamax(n, x, incx);
            return n > 0 ? result + 1 : result;
        }
    };

    template<>
    struct OpenBlas<alpaka::math::Complex<float>>
    {
        using T = alpaka::math::Complex<float>;

        static void copy(int n, T const* x, int incx, T* y, int incy)
        {
            cblas_ccopy(n, x, incx, y, incy);
        }

        static void swap(int n, T* x, int incx, T* y, int incy)
        {
            cblas_cswap(n, x, incx, y, incy);
        }

        static void scal(int n, T alpha, T* x, int incx)
        {
            cblas_cscal(n, &alpha, x, incx);
        }

        static void axpy(int n, T alpha, T const* x, int incx, T* y, int incy)
        {
            cblas_caxpy(n, &alpha, x, incx, y, incy);
        }

        static T dot(int n, T const* x, int incx, T const* y, int incy)
        {
            T result{};
            cblas_cdotu_sub(n, x, incx, y, incy, &result);
            return result;
        }

        static T dotc(int n, T const* x, int incx, T const* y, int incy)
        {
            T result{};
            cblas_cdotc_sub(n, x, incx, y, incy, &result);
            return result;
        }

        static float nrm2(int n, T const* x, int incx)
        {
            return cblas_scnrm2(n, x, incx);
        }

        static float asum(int n, T const* x, int incx)
        {
            return cblas_scasum(n, x, incx);
        }

        static int iamax(int n, T const* x, int incx)
        {
            // netlib BLAS: n <= 0 yields 0, otherwise a 1-based index. cblas_icamax returns 0 for n <= 0,
            // so the +1 is skipped to keep the documented contract "1-based index, 0 if n == 0".
            auto const result = cblas_icamax(n, x, incx);
            return n > 0 ? result + 1 : result;
        }
    };

    template<>
    struct OpenBlas<alpaka::math::Complex<double>>
    {
        using T = alpaka::math::Complex<double>;

        static void copy(int n, T const* x, int incx, T* y, int incy)
        {
            cblas_zcopy(n, x, incx, y, incy);
        }

        static void swap(int n, T* x, int incx, T* y, int incy)
        {
            cblas_zswap(n, x, incx, y, incy);
        }

        static void scal(int n, T alpha, T* x, int incx)
        {
            cblas_zscal(n, &alpha, x, incx);
        }

        static void axpy(int n, T alpha, T const* x, int incx, T* y, int incy)
        {
            cblas_zaxpy(n, &alpha, x, incx, y, incy);
        }

        static T dot(int n, T const* x, int incx, T const* y, int incy)
        {
            T result{};
            cblas_zdotu_sub(n, x, incx, y, incy, &result);
            return result;
        }

        static T dotc(int n, T const* x, int incx, T const* y, int incy)
        {
            T result{};
            cblas_zdotc_sub(n, x, incx, y, incy, &result);
            return result;
        }

        static double nrm2(int n, T const* x, int incx)
        {
            return cblas_dznrm2(n, x, incx);
        }

        static double asum(int n, T const* x, int incx)
        {
            return cblas_dzasum(n, x, incx);
        }

        static int iamax(int n, T const* x, int incx)
        {
            // netlib BLAS: n <= 0 yields 0, otherwise a 1-based index. cblas_izamax returns 0 for n <= 0,
            // so the +1 is skipped to keep the documented contract "1-based index, 0 if n == 0".
            auto const result = cblas_izamax(n, x, incx);
            return n > 0 ? result + 1 : result;
        }
    };

    inline CBLAS_TRANSPOSE toCblasTranspose(Transpose transpose)
    {
        switch(transpose)
        {
        case Transpose::none:
            return CblasNoTrans;
        case Transpose::transposed:
            return CblasTrans;
        case Transpose::conjugateTransposed:
            return CblasConjTrans;
        }
        return CblasNoTrans;
    }

    inline CBLAS_UPLO toCblasUplo(Triangle triangle)
    {
        return triangle == Triangle::upper ? CblasUpper : CblasLower;
    }

    inline CBLAS_DIAG toCblasDiag(Diagonal diagonal)
    {
        return diagonal == Diagonal::unit ? CblasUnit : CblasNonUnit;
    }

    inline CBLAS_SIDE toCblasSide(Side side)
    {
        return side == Side::left ? CblasLeft : CblasRight;
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        CopyFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        auto const& x,
        auto& y,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](auto)
            {
                OpenBlas<T>::copy(
                    int(xd.n),
                    static_cast<T const*>(xd.constPtr),
                    int(xd.inc),
                    static_cast<T*>(yd.mutPtr),
                    int(yd.inc));
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        SwapFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        auto& x,
        auto& y,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        queue.enqueueNativeFn(
            [=](auto)
            {
                OpenBlas<T>::swap(
                    int(xd.n),
                    static_cast<T*>(xd.mutPtr),
                    int(xd.inc),
                    static_cast<T*>(yd.mutPtr),
                    int(yd.inc));
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        ScalFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        auto alpha,
        auto& x,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        queue.enqueueNativeFn(
            [=](auto)
            { OpenBlas<T>::scal(int(xd.n), static_cast<T>(alpha), static_cast<T*>(xd.mutPtr), int(xd.inc)); });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        AxpyFn::Spec<alpaka::api::Host, T_DeviceKind>,
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
            [=](auto)
            {
                OpenBlas<T>::axpy(
                    int(xd.n),
                    static_cast<T>(alpha),
                    static_cast<T const*>(xd.constPtr),
                    int(xd.inc),
                    static_cast<T*>(yd.mutPtr),
                    int(yd.inc));
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        DotFn::Spec<alpaka::api::Host, T_DeviceKind>,
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
            [=](auto)
            {
                resultPtr[0] = OpenBlas<T>::dot(
                    int(xd.n),
                    static_cast<T const*>(xd.constPtr),
                    int(xd.inc),
                    static_cast<T const*>(yd.constPtr),
                    int(yd.inc));
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        DotcFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        auto const& x,
        auto const& y,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using Scalar = std::remove_cv_t<Value_t<ALPAKA_TYPEOF(x)>>;
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto const nInt = checkedCast<int>(xd.n, "dotc n");
        auto const incxInt = checkedCast<int>(xd.inc, "dotc incx");
        auto const incyInt = checkedCast<int>(yd.inc, "dotc incy");
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](auto)
            {
                resultPtr[0] = OpenBlas<Scalar>::dotc(
                    nInt,
                    static_cast<Scalar const*>(xd.constPtr),
                    incxInt,
                    static_cast<Scalar const*>(yd.constPtr),
                    incyInt);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        Nrm2Fn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        auto const& x,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](auto)
            { resultPtr[0] = OpenBlas<T>::nrm2(int(xd.n), static_cast<T const*>(xd.constPtr), int(xd.inc)); });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        AsumFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        auto const& x,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](auto)
            { resultPtr[0] = OpenBlas<T>::asum(int(xd.n), static_cast<T const*>(xd.constPtr), int(xd.inc)); });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        IamaxFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        auto const& x,
        auto& result,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(x)>;
        auto const xd = makeVectorDescriptor(x);
        auto* resultPtr = alpaka::onHost::data(getView(result));
        queue.enqueueNativeFn(
            [=](auto)
            { resultPtr[0] = OpenBlas<T>::iamax(int(xd.n), static_cast<T const*>(xd.constPtr), int(xd.inc)); });
    }

    template<typename T>
    static void hostGemmImpl(
        MatrixDescriptor const& A,
        MatrixDescriptor const& B,
        MatrixDescriptor const& C,
        T alpha,
        T beta)
    {
        if constexpr(std::same_as<T, float>)
            cblas_sgemm(
                CblasRowMajor,
                toCblasTranspose(A.transpose),
                toCblasTranspose(B.transpose),
                int(C.rows),
                int(C.cols),
                int(A.transpose == Transpose::none ? A.cols : A.rows),
                alpha,
                static_cast<float const*>(A.constPtr),
                int(A.ld),
                static_cast<float const*>(B.constPtr),
                int(B.ld),
                beta,
                static_cast<float*>(C.mutPtr),
                int(C.ld));
        else if constexpr(std::same_as<T, double>)
            cblas_dgemm(
                CblasRowMajor,
                toCblasTranspose(A.transpose),
                toCblasTranspose(B.transpose),
                int(C.rows),
                int(C.cols),
                int(A.transpose == Transpose::none ? A.cols : A.rows),
                alpha,
                static_cast<double const*>(A.constPtr),
                int(A.ld),
                static_cast<double const*>(B.constPtr),
                int(B.ld),
                beta,
                static_cast<double*>(C.mutPtr),
                int(C.ld));
        else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
            cblas_cgemm(
                CblasRowMajor,
                toCblasTranspose(A.transpose),
                toCblasTranspose(B.transpose),
                int(C.rows),
                int(C.cols),
                int(A.transpose == Transpose::none ? A.cols : A.rows),
                &alpha,
                static_cast<T const*>(A.constPtr),
                int(A.ld),
                static_cast<T const*>(B.constPtr),
                int(B.ld),
                &beta,
                static_cast<T*>(C.mutPtr),
                int(C.ld));
        else
            cblas_zgemm(
                CblasRowMajor,
                toCblasTranspose(A.transpose),
                toCblasTranspose(B.transpose),
                int(C.rows),
                int(C.cols),
                int(A.transpose == Transpose::none ? A.cols : A.rows),
                &alpha,
                static_cast<T const*>(A.constPtr),
                int(A.ld),
                static_cast<T const*>(B.constPtr),
                int(B.ld),
                &beta,
                static_cast<T*>(C.mutPtr),
                int(C.ld));
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        GemmFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto const& B,
        auto beta,
        auto& C,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(A)>;
        auto const ad = makeMatrixDescriptor(A);
        auto const bd = makeMatrixDescriptor(B);
        auto const cd = makeMatrixDescriptor(C);
        queue.enqueueNativeFn([=](auto) { hostGemmImpl<T>(ad, bd, cd, static_cast<T>(alpha), static_cast<T>(beta)); });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        StridedBatchedGemmFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto const& B,
        auto beta,
        auto& C,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(A)>;
        auto const ad = makeBatchedMatrixDescriptor(A);
        auto const bd = makeBatchedMatrixDescriptor(B);
        auto const cd = makeBatchedMatrixDescriptor(C);
        queue.enqueueNativeFn(
            [=](auto)
            {
                for(std::int64_t i = 0; i < cd.batchCount; ++i)
                {
                    MatrixDescriptor batchA{};
                    batchA.constPtr = static_cast<T const*>(ad.constPtr) + i * ad.batchStride;
                    batchA.rows = ad.rows;
                    batchA.cols = ad.cols;
                    batchA.ld = ad.ld;
                    batchA.transpose = ad.transpose;
                    batchA.triangle = ad.triangle;
                    batchA.diagonal = ad.diagonal;
                    MatrixDescriptor batchB{};
                    batchB.constPtr = static_cast<T const*>(bd.constPtr) + i * bd.batchStride;
                    batchB.rows = bd.rows;
                    batchB.cols = bd.cols;
                    batchB.ld = bd.ld;
                    batchB.transpose = bd.transpose;
                    batchB.triangle = bd.triangle;
                    batchB.diagonal = bd.diagonal;
                    MatrixDescriptor batchC{};
                    batchC.mutPtr = static_cast<T*>(cd.mutPtr) + i * cd.batchStride;
                    batchC.rows = cd.rows;
                    batchC.cols = cd.cols;
                    batchC.ld = cd.ld;
                    batchC.transpose = cd.transpose;
                    batchC.triangle = cd.triangle;
                    batchC.diagonal = cd.diagonal;
                    hostGemmImpl<T>(batchA, batchB, batchC, static_cast<T>(alpha), static_cast<T>(beta));
                }
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        GemvFn::Spec<alpaka::api::Host, T_DeviceKind>,
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
        queue.enqueueNativeFn(
            [=](auto)
            {
                if constexpr(std::same_as<T, float>)
                    cblas_sgemv(
                        CblasRowMajor,
                        toCblasTranspose(ad.transpose),
                        int(ad.rows),
                        int(ad.cols),
                        static_cast<float>(alpha),
                        static_cast<float const*>(ad.constPtr),
                        int(ad.ld),
                        static_cast<float const*>(xd.constPtr),
                        int(xd.inc),
                        static_cast<float>(beta),
                        static_cast<float*>(yd.mutPtr),
                        int(yd.inc));
                else if constexpr(std::same_as<T, double>)
                    cblas_dgemv(
                        CblasRowMajor,
                        toCblasTranspose(ad.transpose),
                        int(ad.rows),
                        int(ad.cols),
                        static_cast<double>(alpha),
                        static_cast<double const*>(ad.constPtr),
                        int(ad.ld),
                        static_cast<double const*>(xd.constPtr),
                        int(xd.inc),
                        static_cast<double>(beta),
                        static_cast<double*>(yd.mutPtr),
                        int(yd.inc));
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    cblas_cgemv(
                        CblasRowMajor,
                        toCblasTranspose(ad.transpose),
                        int(ad.rows),
                        int(ad.cols),
                        &alpha,
                        static_cast<T const*>(ad.constPtr),
                        int(ad.ld),
                        static_cast<T const*>(xd.constPtr),
                        int(xd.inc),
                        &beta,
                        static_cast<T*>(yd.mutPtr),
                        int(yd.inc));
                else
                    cblas_zgemv(
                        CblasRowMajor,
                        toCblasTranspose(ad.transpose),
                        int(ad.rows),
                        int(ad.cols),
                        &alpha,
                        static_cast<T const*>(ad.constPtr),
                        int(ad.ld),
                        static_cast<T const*>(xd.constPtr),
                        int(xd.inc),
                        &beta,
                        static_cast<T*>(yd.mutPtr),
                        int(yd.inc));
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        TrsmFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        Side side,
        auto alpha,
        auto const& A,
        auto& B,
        [[maybe_unused]] Options options)
    {
        using T = Value_t<ALPAKA_TYPEOF(A)>;
        auto const ad = makeMatrixDescriptor(A);
        auto const bd = makeMatrixDescriptor(B);
        queue.enqueueNativeFn(
            [=](auto)
            {
                if constexpr(std::same_as<T, float>)
                    cblas_strsm(
                        CblasRowMajor,
                        toCblasSide(side),
                        toCblasUplo(ad.triangle),
                        toCblasTranspose(ad.transpose),
                        toCblasDiag(ad.diagonal),
                        int(bd.rows),
                        int(bd.cols),
                        static_cast<float>(alpha),
                        static_cast<float const*>(ad.constPtr),
                        int(ad.ld),
                        static_cast<float*>(bd.mutPtr),
                        int(bd.ld));
                else if constexpr(std::same_as<T, double>)
                    cblas_dtrsm(
                        CblasRowMajor,
                        toCblasSide(side),
                        toCblasUplo(ad.triangle),
                        toCblasTranspose(ad.transpose),
                        toCblasDiag(ad.diagonal),
                        int(bd.rows),
                        int(bd.cols),
                        static_cast<double>(alpha),
                        static_cast<double const*>(ad.constPtr),
                        int(ad.ld),
                        static_cast<double*>(bd.mutPtr),
                        int(bd.ld));
                else if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    cblas_ctrsm(
                        CblasRowMajor,
                        toCblasSide(side),
                        toCblasUplo(ad.triangle),
                        toCblasTranspose(ad.transpose),
                        toCblasDiag(ad.diagonal),
                        int(bd.rows),
                        int(bd.cols),
                        &alpha,
                        static_cast<T const*>(ad.constPtr),
                        int(ad.ld),
                        static_cast<T*>(bd.mutPtr),
                        int(bd.ld));
                else
                    cblas_ztrsm(
                        CblasRowMajor,
                        toCblasSide(side),
                        toCblasUplo(ad.triangle),
                        toCblasTranspose(ad.transpose),
                        toCblasDiag(ad.diagonal),
                        int(bd.rows),
                        int(bd.cols),
                        &alpha,
                        static_cast<T const*>(ad.constPtr),
                        int(ad.ld),
                        static_cast<T*>(bd.mutPtr),
                        int(bd.ld));
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        HerkFn::Spec<alpaka::api::Host, T_DeviceKind>,
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
        auto const n = ad.transpose == Transpose::none ? ad.rows : ad.cols;
        auto const k = ad.transpose == Transpose::none ? ad.cols : ad.rows;
        auto const nInt = checkedCast<int>(n, "herk n");
        auto const kInt = checkedCast<int>(k, "herk k");
        auto const adLd = checkedCast<int>(ad.ld, "herk A ld");
        auto const cdLd = checkedCast<int>(cd.ld, "herk C ld");
        queue.enqueueNativeFn(
            [=](auto)
            {
                // cblas_*herk computes C = alpha*op(A)*op(A)^H + beta*C with real alpha/beta, so the public
                // triangle/operation are forwarded unchanged in row-major order.
                if constexpr(std::same_as<T, alpaka::math::Complex<float>>)
                    cblas_cherk(
                        CblasRowMajor,
                        toCblasUplo(cd.triangle),
                        toCblasTranspose(ad.transpose),
                        nInt,
                        kInt,
                        static_cast<float>(alpha),
                        static_cast<T const*>(ad.constPtr),
                        adLd,
                        static_cast<float>(beta),
                        static_cast<T*>(cd.mutPtr),
                        cdLd);
                else
                    cblas_zherk(
                        CblasRowMajor,
                        toCblasUplo(cd.triangle),
                        toCblasTranspose(ad.transpose),
                        nInt,
                        kInt,
                        static_cast<double>(alpha),
                        static_cast<T const*>(ad.constPtr),
                        adLd,
                        static_cast<double>(beta),
                        static_cast<T*>(cd.mutPtr),
                        cdLd);
            });
    }

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    void alpakaFnDispatch(
        SyrkFn::Spec<alpaka::api::Host, T_DeviceKind>,
        auto&& queue,
        auto alpha,
        auto const& A,
        auto beta,
        auto& C,
        [[maybe_unused]] Options options)
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
        // For real operands conjugateTransposed is the identity-conjugated transpose: normalize to transposed.
        auto const op = ad.transpose == Transpose::none ? CblasNoTrans : CblasTrans;
        queue.enqueueNativeFn(
            [=](auto)
            {
                if constexpr(std::same_as<T, float>)
                    cblas_ssyrk(
                        CblasRowMajor,
                        toCblasUplo(cd.triangle),
                        op,
                        nInt,
                        kInt,
                        alpha,
                        static_cast<float const*>(ad.constPtr),
                        adLd,
                        beta,
                        static_cast<float*>(cd.mutPtr),
                        cdLd);
                else
                    cblas_dsyrk(
                        CblasRowMajor,
                        toCblasUplo(cd.triangle),
                        op,
                        nInt,
                        kInt,
                        alpha,
                        static_cast<double const*>(ad.constPtr),
                        adLd,
                        beta,
                        static_cast<double*>(cd.mutPtr),
                        cdLd);
            });
    }
} // namespace alpaka::blas::internal
#endif
