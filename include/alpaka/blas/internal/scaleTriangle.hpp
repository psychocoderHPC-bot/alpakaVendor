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
            for(auto [row, col] : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInGrid,
                    IdxRange{alpaka::Vec{std::uint32_t{0u}, std::uint32_t{0u}}, alpaka::Vec{nIdx, nIdx}}))
            {
                bool const inTriangle = triangle == Triangle::upper ? col >= row : col <= row;
                if(!inTriangle)
                    continue;
                auto const idx = alpaka::Vec{static_cast<std::size_t>(row), static_cast<std::size_t>(col)};
                if(beta == T{0})
                    C[idx] = T{0};
                else if constexpr(ComplexScalar<T>)
                {
                    // herk keeps the diagonal real whenever it writes it: beta is real, so scale the real part and
                    // keep the diagonal imaginary part at zero; off-diagonal elements scale in both parts.
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
    };

    /** Scale (or zero) the selected triangle of ``C`` in place on the device/queue of ``queue``.
     *
     * The operation is enqueued and asynchronous like any other BLAS call. ``A`` is never accessed.
     *
     * ``beta == 1`` is a true no-op: the selected triangle, including a complex diagonal imaginary part, stays
     * byte-identical and nothing is enqueued.
     * ``beta == 0`` zeroes the selected triangle without reading its previous values.
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
        // Reject dimensions/leading dimensions that cannot be represented safely: the grid index domain of the
        // kernel is uint32_t (n*n must fit), and the row pitch must fit the vendor integer width when the same
        // problem reaches the vendor herk paths.
        if(n > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("herk scale: number of rows is too large.");
        if(n > 65535)
            throw std::invalid_argument("herk scale: n*n exceeds the kernel index domain.");
        auto const nU = static_cast<std::uint32_t>(n);
        // Keep the degenerate branch's metadata contract aligned with the vendor herk dispatches: host/cuda/hip take
        // a 32-bit leading dimension, so an oversized C ld must be rejected here too (the vendor path is bypassed).
        auto const ldChecked = checkedCast<int>(cd.ld, "herk scale C ld");
        auto const extent = alpaka::Vec<std::uint32_t, 2u>{nU, nU};
        auto const cv = alpaka::makeMdSpan(
            static_cast<T*>(cd.mutPtr),
            extent,
            alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(ldChecked) * sizeof(T), sizeof(T)});
        auto const frameSpec = alpaka::onHost::getFrameSpec(queue.getDevice(), alpaka::exec::anyExecutor, extent);
        queue.enqueue(frameSpec, alpaka::KernelBundle{ScaleTriangleKernel{}, cv, n, getTriangle(C), betaR});
    }
} // namespace alpaka::blas::internal
