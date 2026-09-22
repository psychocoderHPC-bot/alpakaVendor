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
     * with zeros. This implements the degenerate ``k == 0`` / ``alpha == 0`` syrk semantics without
     * touching ``A`` at all.
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
                    else
                        C[idx] = static_cast<T>(beta * static_cast<T>(C[idx]));
                }
            }
        }
    };

    /** Scale (or zero) the selected triangle of ``C`` in place on the device/queue of ``queue``.
     *
     * The operation is enqueued and asynchronous like any other BLAS call. ``A`` is never accessed.
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
        // Reject dimensions/leading dimensions that cannot be represented safely. The kernel traverses one row index
        // per work item, so the row count must fit the (uint32_t) index domain of the alpaka frame specification;
        // unlike the old n*n square domain this bound holds for any n, no quadratic overflow exists. The row pitch
        // must also fit the vendor integer width so the degenerate branch stays aligned with the vendor syrk paths.
        if(n > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("syrk scale: number of rows is too large.");
        auto const ldChecked = checkedCast<int>(cd.ld, "syrk scale C ld");
        auto const nU = static_cast<std::uint32_t>(n);
        auto const extent = alpaka::Vec<std::uint32_t, 1u>{nU};
        auto const cv = alpaka::makeMdSpan(
            static_cast<T*>(cd.mutPtr),
            alpaka::Vec<std::uint32_t, 2u>{nU, nU},
            alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(ldChecked) * sizeof(T), sizeof(T)});
        auto const frameSpec = alpaka::onHost::getFrameSpec(queue.getDevice(), alpaka::exec::anyExecutor, extent);
        queue.enqueue(
            frameSpec,
            alpaka::KernelBundle{ScaleTriangleKernel{}, cv, n, getTriangle(C), static_cast<T>(beta)});
    }
} // namespace alpaka::blas::internal
