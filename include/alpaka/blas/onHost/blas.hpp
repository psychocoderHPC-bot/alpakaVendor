/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <cstdint>

#include "alpaka/blas/internal/api/blas.hpp"
#include "alpaka/blas/internal/scaleTriangle.hpp"

namespace alpaka::blas::onHost
{
    /**
     * @note All routines in this namespace are enqueued on the supplied alpaka queue. On a default (non-blocking)
     * queue they complete asynchronously; a blocking queue returns only after the enqueued work has completed. Wait
     *       for the queue with ``alpaka::onHost::wait(queue)`` before reading any output. Operand views and the result
     *       buffer must stay alive and be mutable until the enqueued work has completed; the wrappers capture the
     * views and data pointers when the routine is called. On accelerator backends the single-element result buffer
     * must additionally be accessible to the backend that writes it: device-accessible for CUDA/HIP, and also
     *       host-accessible (shared or host USM) for oneAPI ``iamax``, where oneMKL writes the index on the device and
     * a host task then dereferences it.
     */

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
        internal::validateWritable<ALPAKA_TYPEOF(y)>();
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
        internal::validateWritable<ALPAKA_TYPEOF(x)>();
        internal::validateWritable<ALPAKA_TYPEOF(y)>();
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
        internal::validateWritable<ALPAKA_TYPEOF(x)>();
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
        internal::validateWritable<ALPAKA_TYPEOF(y)>();
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
        internal::validateWritable<ALPAKA_TYPEOF(result)>();
        internal::validateSameVectorExtent(x, y, "dot");
        internal::validateScalarResult(x, result, "dot");
        // Guard the dispatch with the result element type: a mismatch must throw before any backend call is
        // instantiated, because vendor reduction routines are typed on their scalar result pointer. ``Value_t`` is
        // cv-preserving, so both sides are compared with cv removed to accept read-only (const-element) inputs.
        if constexpr(std::same_as<
                         std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(result)>>,
                         std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(x)>>>)
            internal::DotFn::call(queue, x, y, result, options);
        else
            throw std::invalid_argument("dot requires a result buffer with the routine's result type.");
    }

    /**
     * Compute the conjugated dot product.
     *
     * Computes ``result[0] = sum_i conj(x[i]) * y[i]``. The first operand is conjugated; the second is not. For
     * real-valued vectors this is identical to ``dot``. For complex-valued vectors it differs from ``dot``, which
     * leaves both operands unconjugated.
     *
     * ``dotc`` is available on the same backends as ``dot`` (OpenBLAS/cuBLAS/rocBLAS/oneMKL host paths); it is not
     * provided for OpenMP or the generic native alpaka CPU queues.
     *
     * alpaka 1D vector views are always contiguous (the reported element pitch equals the element size), so the BLAS
     * increments passed to the backend are always 1; non-unit 1D strides are not expressible through alpaka 1D views.
     * A view shifted by ``getSubView`` (or an MdSpan created directly from an offset pointer) is honored via its base
     * pointer.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param x first input vector, conjugated before multiplication.
     * @param y second input vector, used as-is.
     * @param result single-element output view that receives the scalar result.
     * @param options optional backend hints.
     */
    void dotc(
        auto& queue,
        concepts::VectorView auto const& x,
        concepts::VectorView auto const& y,
        concepts::VectorView auto& result,
        Options options = {})
        requires(
            std::same_as<
                std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(x)>>,
                std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(y)>>>
            && std::same_as<
                std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(x)>>,
                std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(result)>>>
            && !std::is_const_v<alpaka::GetValueType_t<alpaka::blas::detail::unannotated_t<ALPAKA_TYPEOF(result)>>>)
    {
        using XValue = internal::Value_t<ALPAKA_TYPEOF(x)>;
        using YValue = internal::Value_t<ALPAKA_TYPEOF(y)>;
        using ResultValue = internal::Value_t<ALPAKA_TYPEOF(result)>;
        // Routine-local unqualified scalar. ``Value_t`` is cv-preserving so read-only input views (``MdSpan<T
        // const>``) compare equal with writable ones, while mismatched element types (``x<float>, y<double>``) are
        // still rejected below.
        using Scalar = std::remove_cv_t<XValue>;
        static_assert(
            std::same_as<Scalar, std::remove_cv_t<YValue>>,
            "dotc requires x and y to have the same element type.");
        static_assert(
            std::same_as<Scalar, std::remove_cv_t<ResultValue>>,
            "dotc requires result to have the same element type as x and y.");
        internal::validateScalarSupport<Scalar>();
        internal::validateWritable<ALPAKA_TYPEOF(result)>();
        internal::validateSameVectorExtent(x, y, "dotc");
        internal::validateScalarResult(x, result, "dotc");
        internal::DotcFn::call(queue, x, y, result, options);
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
        internal::validateWritable<ALPAKA_TYPEOF(result)>();
        internal::validateScalarResult(x, result, "nrm2");
        if constexpr(std::same_as<
                         internal::Value_t<ALPAKA_TYPEOF(result)>,
                         Real_t<internal::Value_t<ALPAKA_TYPEOF(x)>>>)
            internal::Nrm2Fn::call(queue, x, result, options);
        else
            throw std::invalid_argument("nrm2 requires a result buffer with the routine's result type.");
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
        internal::validateWritable<ALPAKA_TYPEOF(result)>();
        internal::validateScalarResult(x, result, "asum");
        if constexpr(std::same_as<
                         internal::Value_t<ALPAKA_TYPEOF(result)>,
                         Real_t<internal::Value_t<ALPAKA_TYPEOF(x)>>>)
            internal::AsumFn::call(queue, x, result, options);
        else
            throw std::invalid_argument("asum requires a result buffer with the routine's result type.");
    }

    /**
     * Return the 1-based index of the entry with largest absolute value.
     *
     * Returns 0 if the vector is empty (n == 0).
     *
     * This follows the BLAS convention, so the first element has index ``1`` rather than ``0``.
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param x input vector.
     * @param result single-element integer output view that receives the BLAS index.
     * @param options optional backend hints.
     *
     * @note The result buffer is written by the enqueued work, so it must outlive the queue wait and be accessible to
     *       the backend that writes it. CUDA/HIP compute the index on the device (device-accessible required); oneAPI
     *       ``iamax`` writes the index on the device and then dereferences it from a host task, so a shared or
     *       host-accessible buffer is required as well.
     */
    void iamax(
        auto& queue,
        concepts::VectorView auto const& x,
        concepts::VectorView auto& result,
        Options options = {})
    {
        internal::validateScalarSupport<internal::Value_t<ALPAKA_TYPEOF(x)>>();
        internal::validateWritable<ALPAKA_TYPEOF(result)>();
        internal::validateScalarResult(x, result, "iamax");
        // The result element type must be exactly a 32-bit signed integer: the host backend writes through
        // ``int``, CUDA/HIP reinterpret_cast the result pointer to ``int*`` and write 4 bytes, and oneMKL has an
        // int32 overload for an int result. Accepting a narrower or wider integral type would overflow the buffer or
        // leave high bytes indeterminate. The wider oneMKL result width is tracked in issue #19.
        using R = internal::Value_t<ALPAKA_TYPEOF(result)>;
        if constexpr(std::same_as<R, std::int32_t>)
            internal::IamaxFn::call(queue, x, result, options);
        else
            throw std::invalid_argument("iamax requires a 32-bit signed integer result buffer.");
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
        internal::validateWritable<ALPAKA_TYPEOF(y)>();
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
        internal::validateWritable<ALPAKA_TYPEOF(C)>();
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
        internal::validateWritable<ALPAKA_TYPEOF(C)>();
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
        internal::validateWritable<ALPAKA_TYPEOF(B)>();
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
     * On an actual update the written diagonal is real (its imaginary part is ignored); a true no-op (``n == 0``, or a
     * zero product contribution combined with ``beta == 1``) may leave ``C`` unchanged, including its diagonal; no
     * unconditional diagonal canonicalization is promised.
     *
     * Degenerate product contributions (``k == 0``, i.e. an empty rank-k product, or ``alpha == 0``) have
     * well-defined ``beta`` scaling semantics:
     *
     * - ``n == 0`` is a true no-op: no element of ``C`` is read or written.
     * - the selected triangle is ``beta * C``; with ``beta == 1`` this is a true no-op and with ``beta == 0`` the
     *   selected triangle is zeroed without reading its previous values.
     * - for complex ``C`` the diagonal stays real: the real scalar ``beta`` scales the diagonal's real part and its
     *   imaginary part remains zero, and off-diagonal elements scale in both real and imaginary part.
     *
     * The degenerate ``k == 0`` / ``alpha == 0`` path never reads ``A`` and never calls the backend BLAS routine; it
     * runs a queued triangle-scale kernel on the same queue, so ordering against other queued work is preserved. Its
     * metadata checks mirror the backend's own herk dispatch (the leading dimension is narrowed through the same
     * vendor-int width, 32-bit on host/cuda/hip and 64-bit on oneMKL), so an enormously pitched ``C`` is rejected
     * exactly when the ``k > 0`` path of the same backend would reject it.
     *
     * ``A`` and ``C`` must not overlap.
     *
     * The real-valued counterpart is the standard BLAS ``syrk`` (real symmetric rank-k).
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
        requires(
            ComplexScalar<std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(A)>>>
            && std::same_as<
                std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(A)>>,
                std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(C)>>>
            && RealScalar<std::remove_cv_t<decltype(alpha)>> && RealScalar<std::remove_cv_t<decltype(beta)>>
            && !std::is_const_v<alpaka::GetValueType_t<alpaka::blas::detail::unannotated_t<ALPAKA_TYPEOF(C)>>>)
    {
        // Value_t keeps cv-qualifiers; herk dispatches with the unqualified scalar type so a const-element A selects
        // the same vendor branch as a writable A while the validated writable C stays checked below.
        using T = std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(A)>>;
        internal::validateWritable<ALPAKA_TYPEOF(C)>();
        internal::validateHerk(A, C);
        auto const ad = internal::makeMatrixDescriptor(A);
        auto const n = internal::getTranspose(A) == Transpose::none ? ad.rows : ad.cols;
        auto const k = internal::getTranspose(A) == Transpose::none ? ad.cols : ad.rows;
        if(n == 0)
            return; // nothing to do, no data access.
        if(k == 0 || static_cast<Real_t<T>>(alpha) == Real_t<T>{0})
        {
            // The empty/zero rank-k product leaves the selected triangle as beta * C; A must not be read.
            internal::enqueueScaleTriangle(queue, C, beta);
            return;
        }
        internal::HerkFn::call(queue, alpha, A, beta, C, options);
    }

    /**
     * Symmetric rank-k update.
     *
     * Computes the selected triangle of ``C = alpha * op(A) * transpose(op(A)) + beta * C`` where ``M = op(A)`` has
     * shape ``n x k`` and ``C`` is ``n x n``.
     *
     * Only real scalar types (``float``, ``double``) are supported. Complex symmetric rank-k is intentionally not
     * exposed here; the complex Hermitian rank-k counterpart is the standard BLAS ``herk`` routine (``C =
     * alpha * op(A) * op(A)^H + beta * C`` with ``op(A)^H`` the conjugate transpose), provided by this library.
     *
     * ``A`` is a general dense matrix and may be annotated ``transposed(A)`` or ``conjTransposed(A)``. For real
     * operands ``conjTransposed(A)`` is equivalent to ``transposed(A)`` (conjugation is the identity on real types)
     * and is normalized to the transposed operation. ``C`` must carry an explicit ``upper(C)`` or ``lower(C)``
     * selection; the opposite triangle and any padding are left unchanged. Transpose and unit-diagonal annotations on
     * ``C`` are rejected.
     *
     * The coefficients are converted exactly once at this public entry into the canonical scalar type of the
     * operands; the backends receive the already-converted values and never re-cast them.
     *
     * Degenerate cases are handled without touching the operands that must not be read:
     * - ``n == 0`` is a no-op and no data is accessed at all.
     * - ``k == 0`` or ``alpha == 0`` produce ``beta * C`` on the selected triangle; ``A`` is never read.
     * - ``beta == 0`` writes ``alpha * op(A) * transpose(op(A))`` to the selected triangle; the old content of the
     *   triangle is not read.
     *
     * ``A`` and ``C`` must not alias (no overlapping storage).
     *
     * @param queue alpaka queue that defines when the work runs.
     * @param alpha real scalar multiplier for the rank-k product.
     * @param A input matrix, optionally ``transposed(A)`` or ``conjTransposed(A)``.
     * @param beta real scalar multiplier applied to the selected triangle of the existing ``C``.
     * @param C input/output result matrix, annotated ``upper(C)`` or ``lower(C)``.
     * @param options optional backend hints.
     */
    void syrk(
        auto& queue,
        auto alpha,
        concepts::MatrixView auto const& A,
        auto beta,
        concepts::MatrixView auto& C,
        Options options = {})
        requires(
            RealScalar<std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(A)>>>
            && std::same_as<
                std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(A)>>,
                std::remove_cv_t<internal::Value_t<ALPAKA_TYPEOF(C)>>>
            && RealScalar<std::remove_cv_t<decltype(alpha)>> && RealScalar<std::remove_cv_t<decltype(beta)>>
            && !std::is_const_v<alpaka::GetValueType_t<alpaka::blas::detail::unannotated_t<ALPAKA_TYPEOF(C)>>>)
    {
        using T = internal::Value_t<ALPAKA_TYPEOF(A)>;
        using Scalar = std::remove_cv_t<T>;
        static_assert(RealScalar<Scalar>, "syrk supports only real scalar types.");
        internal::validateWritable<ALPAKA_TYPEOF(C)>();
        internal::validateSyrk(A, C);
        auto const ad = internal::makeMatrixDescriptor(A);
        auto const n = internal::getTranspose(A) == Transpose::none ? ad.rows : ad.cols;
        auto const k = internal::getTranspose(A) == Transpose::none ? ad.cols : ad.rows;
        if(n == 0)
            return; // nothing to do, no data access.
        // Convert the scalar coefficients exactly once at the public entry; the backends receive already-converted
        // canonical Scalar values and never re-cast them.
        Scalar const alphaScalar = static_cast<Scalar>(alpha);
        Scalar const betaScalar = static_cast<Scalar>(beta);
        if(k == 0 || alphaScalar == Scalar{0})
        {
            // The result is beta * C on the selected triangle and A must not be read.
            internal::enqueueScaleTriangle(queue, C, betaScalar);
            return;
        }
        internal::SyrkFn::call(queue, alphaScalar, A, betaScalar, C, options);
    }
} // namespace alpaka::blas::onHost
