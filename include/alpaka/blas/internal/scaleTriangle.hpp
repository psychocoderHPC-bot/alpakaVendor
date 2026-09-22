/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/blas/internal/utility.hpp"

namespace alpaka::blas::internal
{
    /** Device-generic kernel which scales (or zeroes) the selected triangle of a square matrix in place.
     *
     * Only the selected triangle of ``C`` is touched; the opposite triangle and padding are left unchanged.
     * When ``beta == 0`` the old content of the triangle is not read, the entries are only overwritten
     * with zeros. This implements the degenerate ``k == 0`` (empty rank-k product) herk semantics without
     * touching ``A`` at all: the selected triangle becomes ``beta * C``.
     *
     * For complex ``C`` the diagonal stays real: ``beta`` is a real scalar, so the real (or zero) diagonal is
     * scaled in its real part only and its imaginary part remains zero.
     */
    struct ScaleTriangleKernel
    {
        template<alpaka::concepts::IMdSpan T_CView, typename T_Scalar>
        ALPAKA_FN_ACC void operator()(
            onAcc::concepts::Acc auto const& acc,
            T_CView C,
            std::int64_t n,
            Triangle triangle,
            T_Scalar beta) const
        {
            using T = std::remove_cv_t<alpaka::GetValueType_t<T_CView>>;
            auto const nIdx = static_cast<std::uint32_t>(n);
            // The index domain covers only the row range. Iterating the full n*n square index domain would overflow
            // the uint32_t indices for n just above 65535; splitting the work into rows keeps every index below n.
            for(auto const [row] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{nIdx}))
            {
                // Visit the selected triangle only: upper is (row, col) with row <= col, lower is col <= row.
                uint32_t const colBegin = triangle == Triangle::upper ? row : 0u;
                uint32_t const colEnd = triangle == Triangle::upper ? nIdx : row + 1u;
                for(uint32_t col = colBegin; col < colEnd; ++col)
                {
                    auto const idx = alpaka::Vec{static_cast<std::size_t>(row), static_cast<std::size_t>(col)};
                    if(beta == T{0})
                        C[idx] = T{0};
                    else if constexpr(ComplexScalar<T>)
                    {
                        // herk keeps the diagonal real whenever it writes it: beta is real, so scale the real part
                        // and keep the diagonal imaginary part at zero; off-diagonal elements scale in both parts.
                        auto const betaR = static_cast<Real_t<T>>(beta);
                        if(row == col)
                            C[idx] = T{betaR * C[idx].real(), Real_t<T>{0}};
                        else
                            C[idx] = T{betaR * static_cast<T>(C[idx]).real(), betaR * static_cast<T>(C[idx]).imag()};
                    }
                    else
                        C[idx] = static_cast<T>(beta * static_cast<T>(C[idx]));
                }
            }
        }
    };

    /** Leading-dimension width of the vendor BLAS herk of the backend that owns ``queue``.
     *
     * The herk dispatch headers narrow every leading dimension through this width before calling the vendor tool:
     * host/cuda/hip use the 32-bit CBLAS/cuBLAS/rocBLAS ``int``, while the oneMKL C++ API takes ``int64_t``. The
     * degenerate triangle-scale branch applies the same fence, so its metadata checks never diverge from the
     * ``k > 0`` vendored path of the same backend.
     */
    template<typename T_Queue>
    using HerkLdInt_t = std::conditional_t<
        std::is_same_v<std::remove_cvref_t<decltype(alpaka::getApi(std::declval<T_Queue&>()))>, alpaka::api::OneApi>,
        std::int64_t,
        int>;

    /** Scale (or zero) the selected triangle of ``C`` in place on the device/queue of ``queue``.
     *
     * The operation is enqueued and asynchronous like any other BLAS call. ``A`` is never accessed.
     *
     * ``beta == 1`` is a true no-op: the selected triangle, including a complex diagonal imaginary part, stays
     * byte-identical and nothing is enqueued.
     * ``beta == 0`` zeroes the selected triangle without reading its previous values.
     *
     * The kernel traverses one row index per work item, so any row count up to ``UINT32_MAX`` is representable; the
     * only remaining guard rejects metadata beyond the 64-bit descriptor envelope, which is shared with the vendor
     * herk paths. The leading dimension is narrowed through the same vendor-int gate (see ``HerkLdInt_t``) as the
     * non-degenerate herk dispatch on the same backend: ``int64`` on oneMKL (which uses 64-bit dimensions),
     * ``checkedCast<int>`` on host/cuda/hip (whose vendor herk rejects ``ld > 2^31 - 1``), so a degenerate
     * ``k == 0`` / ``alpha == 0`` call never diverges from the ``k > 0`` path of its own backend.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param C input/output result matrix, annotated ``upper(C)`` or ``lower(C)``.
     * @param beta real scalar multiplier (beta == 0 zeroes the triangle without reading the old values).
     */
    inline void enqueueScaleTriangle(auto& queue, concepts::MatrixView auto& C, auto beta)
    {
        using T = Value_t<ALPAKA_TYPEOF(C)>;
        auto const cd = makeMatrixDescriptor(C);
        std::int64_t const n = cd.rows;
        if(n == 0)
            return;
        using Real = Real_t<T>;
        auto const betaR = static_cast<Real>(beta);
        if(betaR == Real{1})
            return; // true no-op: the selected triangle must stay byte-identical (incl. a complex diagonal imag).
        // The kernel's index domain is uint32_t: the row range must fit. With a 1D row index there is no quadratic
        // n*n overflow like the old full-square domain; only n > UINT32_MAX needs rejection. The leading dimension
        // is narrowed through the same vendor-int gate as the k>0 dispatch of this backend (see HerkLdInt_t): no
        // 64-bit fence is imposed on the degenerate branch beyond what the backend's own herk accepts.
        if(n > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("herk scale: number of rows is too large.");
        auto const nU = static_cast<std::uint32_t>(n);
        // The degenerate branch promises the same metadata fence as the backend's non-degenerate herk dispatch:
        // oneMKL takes int64 leading dimensions while host/cuda/hip take 32-bit vendor ints, so an enormously
        // pitched C must be rejected here exactly when the k>0 path of the same backend would reject it.
        auto const ldChecked
            = static_cast<std::int64_t>(checkedCast<HerkLdInt_t<ALPAKA_TYPEOF(queue)>>(cd.ld, "herk scale C ld"));
        auto const extent = alpaka::Vec<std::uint32_t, 1u>{nU};
        auto const cv = alpaka::makeMdSpan(
            static_cast<T*>(cd.mutPtr),
            alpaka::Vec<std::uint32_t, 2u>{nU, nU},
            alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(ldChecked) * sizeof(T), sizeof(T)});
        auto const frameSpec = alpaka::onHost::getFrameSpec(queue.getDevice(), alpaka::exec::anyExecutor, extent);
        queue.enqueue(frameSpec, alpaka::KernelBundle{ScaleTriangleKernel{}, cv, n, getTriangle(C), betaR});
    }
} // namespace alpaka::blas::internal
