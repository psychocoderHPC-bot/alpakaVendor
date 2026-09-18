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
                else
                    C[idx] = static_cast<T>(beta * static_cast<T>(C[idx]));
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
        auto const nU = static_cast<std::uint32_t>(n);
        auto const extent = alpaka::Vec<std::uint32_t, 2u>{nU, nU};
        auto const cv = alpaka::makeMdSpan(
            static_cast<T*>(cd.mutPtr),
            extent,
            alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(cd.ld) * sizeof(T), sizeof(T)});
        auto const frameSpec = alpaka::onHost::getFrameSpec(queue.getDevice(), alpaka::exec::anyExecutor, extent);
        queue.enqueue(
            frameSpec,
            alpaka::KernelBundle{ScaleTriangleKernel{}, cv, n, getTriangle(C), static_cast<T>(beta)});
    }
} // namespace alpaka::blas::internal
