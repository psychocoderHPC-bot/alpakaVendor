/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/blas/internal/api/blas.hpp"

namespace alpaka::blas::onHost
{
    /**
     * Copy one vector into another.
     *
     * @param queue alpaka queue that defines when the operation executes.
     * @param x source vector.
     * @param y destination vector with the same logical extent as ``x``.
     * @param options optional backend hints. Current host backends ignore them, but keeping the parameter makes call
     *        sites uniform with other BLAS routines.
     */
    void copy(auto& queue, concepts::VectorView auto const& x, concepts::VectorView auto& y, Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(x)>>();
        internal::validateSameVectorExtent(x, y, "copy");
        internal::CopyFn::call(queue, x, y, options);
    }

    /**
     * Swap two vectors element by element.
     *
     * @param queue alpaka queue that defines when the swap executes.
     * @param x first vector, overwritten with the original contents of ``y``.
     * @param y second vector, overwritten with the original contents of ``x``.
     * @param options optional backend hints.
     */
    void swap(auto& queue, concepts::VectorView auto& x, concepts::VectorView auto& y, Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(x)>>();
        internal::validateSameVectorExtent(x, y, "swap");
        internal::SwapFn::call(queue, x, y, options);
    }

    /**
     * Scale a vector in place.
     *
     * Computes ``x = alpha * x``.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param alpha scalar multiplier.
     * @param x vector updated in place.
     * @param options optional backend hints.
     */
    void scal(auto& queue, auto alpha, concepts::VectorView auto& x, Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(x)>>();
        internal::ScalFn::call(queue, alpha, x, options);
    }

    /**
     * Perform the classic AXPY update.
     *
     * Computes ``y = alpha * x + y``.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param alpha scalar multiplier applied to ``x``.
     * @param x input vector.
     * @param y input/output vector updated in place.
     * @param options optional backend hints.
     */
    void axpy(
        auto& queue,
        auto alpha,
        concepts::VectorView auto const& x,
        concepts::VectorView auto& y,
        Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(x)>>();
        internal::validateSameVectorExtent(x, y, "axpy");
        internal::AxpyFn::call(queue, alpha, x, y, options);
    }

    /**
     * Compute a vector dot product.
     *
     * Computes ``result[0] = sum_i x[i] * y[i]``.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param x first input vector.
     * @param y second input vector.
     * @param result single-element output view that receives the scalar result.
     * @param options optional backend hints.
     */
    void dot(
        auto& queue,
        concepts::VectorView auto const& x,
        concepts::VectorView auto const& y,
        concepts::VectorView auto& result,
        Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(x)>>();
        internal::validateSameVectorExtent(x, y, "dot");
        internal::validateScalarResult(x, result, "dot");
        internal::DotFn::call(queue, x, y, result, options);
    }

    /**
     * Compute the Euclidean norm of a vector.
     *
     * Computes ``result[0] = ||x||_2``.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param x input vector.
     * @param result single-element output view that receives the real-valued norm.
     * @param options optional backend hints.
     */
    void nrm2(auto& queue, concepts::VectorView auto const& x, concepts::VectorView auto& result, Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(x)>>();
        internal::validateScalarResult(x, result, "nrm2");
        internal::Nrm2Fn::call(queue, x, result, options);
    }

    /**
     * Compute the BLAS ASUM reduction.
     *
     * For real values this is ``sum_i abs(x[i])``. For complex values vendor BLAS uses the standard 1-norm style
     * reduction ``sum_i (abs(real(x[i])) + abs(imag(x[i])))``.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param x input vector.
     * @param result single-element output view that receives the real-valued reduction result.
     * @param options optional backend hints.
     */
    void asum(auto& queue, concepts::VectorView auto const& x, concepts::VectorView auto& result, Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(x)>>();
        internal::validateScalarResult(x, result, "asum");
        internal::AsumFn::call(queue, x, result, options);
    }

    /**
     * Return the 1-based index of the entry with largest absolute value.
     *
     * This follows the BLAS convention, so the first element has index ``1`` rather than ``0``.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param x input vector.
     * @param result single-element integer output view that receives the BLAS index.
     * @param options optional backend hints.
     */
    void iamax(
        auto& queue,
        concepts::VectorView auto const& x,
        concepts::VectorView auto& result,
        Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(x)>>();
        internal::validateScalarResult(x, result, "iamax");
        internal::IamaxFn::call(queue, x, result, options);
    }

    /**
     * Matrix-vector multiplication.
     *
     * Computes ``y = alpha * op(A) * x + beta * y`` where ``op(A)`` is one of:
     *
     * - ``A``
     * - ``transposed(A)``
     * - ``conjTransposed(A)``
     *
     * The wrapper derives the logical matrix shape from the view and its annotations, then checks that the extents of
     * ``x`` and ``y`` match the selected operation.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param alpha scalar multiplier for ``op(A) * x``.
     * @param A matrix operand. Pass ``transposed(A)`` or ``conjTransposed(A)`` if the stored layout should be read
     *        differently without moving data.
     * @param x input vector whose extent must match ``op(A).cols``.
     * @param beta scalar multiplier applied to the existing contents of ``y``.
     * @param y input/output vector whose extent must match ``op(A).rows``.
     * @param options optional backend hints such as preferred math mode.
     */
    void gemv(
        auto& queue,
        auto alpha,
        concepts::MatrixView auto const& A,
        concepts::VectorView auto const& x,
        auto beta,
        concepts::VectorView auto& y,
        Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(A)>>();
        internal::validateGemv(A, x, y);
        internal::GemvFn::call(queue, alpha, A, x, beta, y, options);
    }

    /**
     * Matrix-matrix multiplication.
     *
     * Computes ``C = alpha * op(A) * op(B) + beta * C``. Both matrix operands may be annotated with
     * ``transposed()`` or ``conjTransposed()``.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param alpha scalar multiplier for the matrix product.
     * @param A left matrix operand. Its logical extent becomes ``op(A).rows x op(A).cols`` after annotations.
     * @param B right matrix operand. Its logical extent becomes ``op(B).rows x op(B).cols`` after annotations.
     * @param beta scalar multiplier applied to the existing contents of ``C``.
     * @param C input/output result matrix. Its extent must match ``op(A).rows x op(B).cols``.
     * @param options optional backend hints. For example, some CUDA paths use them to select a math mode.
     */
    void gemm(
        auto& queue,
        auto alpha,
        concepts::MatrixView auto const& A,
        concepts::MatrixView auto const& B,
        auto beta,
        concepts::MatrixView auto& C,
        Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(A)>>();
        internal::validateGemm(A, B, C);
        internal::GemmFn::call(queue, alpha, A, B, beta, C, options);
    }

    /**
     * Strided batched matrix-matrix multiplication.
     *
     * Computes the GEMM update independently for every batch:
     * ``C[b] = alpha * op(A[b]) * op(B[b]) + beta * C[b]``.
     *
     * A batched matrix view is a three-dimensional view interpreted as ``[batch, row, column]``. Consecutive batches
     * are separated by the natural stride of the view.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param alpha scalar multiplier for each batch product.
     * @param A left batched matrix operand.
     * @param B right batched matrix operand.
     * @param beta scalar multiplier applied to the existing contents of every batch in ``C``.
     * @param C input/output batched result matrices.
     * @param options optional backend hints.
     */
    void stridedBatchedGemm(
        auto& queue,
        auto alpha,
        concepts::BatchedMatrixView auto const& A,
        concepts::BatchedMatrixView auto const& B,
        auto beta,
        concepts::BatchedMatrixView auto& C,
        Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(A)>>();
        auto const ad = internal::makeBatchedMatrixDescriptor(A);
        auto const bd = internal::makeBatchedMatrixDescriptor(B);
        auto const cd = internal::makeBatchedMatrixDescriptor(C);
        if(ad.batchCount != bd.batchCount || ad.batchCount != cd.batchCount)
            throw std::invalid_argument("stridedBatchedGemm requires matching batch counts.");
        if((detail::getTranspose(A) == Transpose::none ? ad.cols : ad.rows)
           != (detail::getTranspose(B) == Transpose::none ? bd.rows : bd.cols))
            throw std::invalid_argument("stridedBatchedGemm requires op(A).cols == op(B).rows.");
        if(cd.rows != (detail::getTranspose(A) == Transpose::none ? ad.rows : ad.cols)
           || cd.cols != (detail::getTranspose(B) == Transpose::none ? bd.cols : bd.rows))
            throw std::invalid_argument("stridedBatchedGemm output extent mismatch.");
        internal::StridedBatchedGemmFn::call(queue, alpha, A, B, beta, C, options);
    }

    /**
     * Solve a triangular linear system with multiple right-hand sides.
     *
     * For ``side == Side::left`` this solves ``op(A) * X = alpha * B`` and overwrites ``B`` with ``X``.
     * For ``side == Side::right`` it solves ``X * op(A) = alpha * B``.
     *
     * ``A`` should usually be wrapped in one or more annotations:
     *
     * - ``upper(A)`` or ``lower(A)`` to select the stored triangular half
     * - ``unitDiag(A)`` or ``nonUnitDiag(A)`` to describe the diagonal
     * - optionally ``transposed(A)`` or ``conjTransposed(A)``
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param side which side the triangular operand acts from.
     * @param alpha scalar multiplier applied to the right-hand side(s).
     * @param A triangular coefficient matrix, optionally annotated as described above.
     * @param B input/output matrix of right-hand sides, overwritten with the solution.
     * @param options optional backend hints.
     */
    void trsm(
        auto& queue,
        Side side,
        auto alpha,
        concepts::MatrixView auto const& A,
        concepts::MatrixView auto& B,
        Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(A)>>();
        internal::validateTrsm(side, A, B);
        internal::TrsmFn::call(queue, side, alpha, A, B, options);
    }

    /**
     * Hermitian rank-k update.
     *
     * Computes the selected triangle of ``C = alpha * M * conjTranspose(M) + beta * C`` where ``M = op(A)`` has shape
     * ``n x k`` and ``C`` is ``n x n``.
     *
     * Only complex scalar types (``alpaka::math::Complex<float>`` and ``alpaka::math::Complex<double>``) are
     * supported.
     * ``A`` and ``C`` must share the same complex element type.
     *
     * ``alpha`` and ``beta`` must be real values convertible to ``Real_t<T>``; complex coefficients are rejected
     * (including ones with a zero imaginary part). ``A`` is a general dense matrix and may be annotated
     * ``conjTransposed(A)`` (or left plain); the plain ``transposed(A)`` annotation is rejected because it is not a
     * standard HERK operation. ``C`` must carry an explicit ``upper(C)`` or ``lower(C)`` selection; the opposite
     * triangle and any padding are left unchanged. Transpose and unit-diagonal annotations on ``C`` are rejected.
     *
     * On an actual update the written diagonal is real (its imaginary part is ignored). A true no-op (``n == 0``, or a
     * zero product contribution combined with ``beta == 1``) may leave ``C`` unchanged, including its diagonal; no
     * unconditional diagonal canonicalization is promised. No special zero-scalar fast path is guaranteed, so ``k ==
     * 0`` or ``alpha == 0`` are not guaranteed to leave the old selected triangle unread, and ``beta == 0`` is not
     * guaranteed to skip reading the old selected ``C`` values.
     *
     * ``A`` and ``C`` must not overlap.
     *
     * The real-valued counterpart is ``syrk``.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param alpha real scalar multiplier for the rank-k product.
     * @param A input matrix, optionally ``conjTransposed(A)``.
     * @param beta real scalar multiplier applied to the selected triangle of the existing ``C``.
     * @param C input/output result matrix, annotated ``upper(C)`` or ``lower(C)``.
     * @param options optional backend hints.
     */
    void herk(
        auto& queue,
        auto alpha,
        concepts::MatrixView auto const& A,
        auto beta,
        concepts::MatrixView auto& C,
        Options options = {})
    {
        using T = internal::Value_t<ALPAKA_TYPEOF(A)>;
        static_assert(ComplexScalar<T>, "herk supports only complex scalar types.");
        static_assert(
            std::same_as<T, internal::Value_t<ALPAKA_TYPEOF(C)>>,
            "herk requires A and C to have the same element type.");
        static_assert(
            RealScalar<std::remove_cv_t<decltype(alpha)>>,
            "herk requires a real alpha, not a complex coefficient.");
        static_assert(
            RealScalar<std::remove_cv_t<decltype(beta)>>,
            "herk requires a real beta, not a complex coefficient.");
        internal::validateHerk(A, C);
        internal::HerkFn::call(queue, alpha, A, beta, C, options);
    }
} // namespace alpaka::blas::onHost
