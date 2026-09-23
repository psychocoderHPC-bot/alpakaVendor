/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "alpaka/blas/common.hpp"

namespace alpaka::blas::internal
{
    using alpaka::blas::detail::getDiagonal;
    using alpaka::blas::detail::getTranspose;
    using alpaka::blas::detail::getTriangle;
    using alpaka::blas::detail::getView;

    /** Scalar type of a BLAS operand view.
     *
     * ``Value_t`` preserves cv-qualifiers: a const-element input view keeps its const element type so callers can
     * distinguish read-only operands from writable ones. Routines that dispatch on the scalar type derive a
     * routine-local unqualified scalar with ``std::remove_cv_t<Value_t<T>>``; the writable-operand guard
     * ``validateWritable()`` rejects a const-element view on an operand that is written in place.
     */
    template<typename T>
    using Value_t = alpaka::GetValueType_t<detail::unannotated_t<T>>;

    template<typename T>
    constexpr bool isSupportedScalar_v = Scalar<Value_t<T>>;

    /** Compile-time guard: a BLAS operand that is written in place must be a view with non-const element type.
     *
     * ``Value_t`` preserves cv-qualifiers, so backend dispatch derives its unqualified scalar locally with
     * ``std::remove_cv_t<Value_t<T>>``. Without this guard, a const-element view passed to a writable operand would
     * select the same backend branch and the backend would write through a pointer the caller declared read-only
     * (UB).
     */
    template<typename T>
    constexpr void validateWritable()
    {
        static_assert(
            !std::is_const_v<alpaka::GetValueType_t<detail::unannotated_t<T>>>,
            "The BLAS operand must be a writable view (element type must not be const).");
    }

    template<typename T>
    constexpr auto asRealMagnitude(T value)
    {
        using Real = Real_t<T>;
        if constexpr(ComplexScalar<T>)
            return std::abs(value.real()) + std::abs(value.imag());
        else
            return std::abs(static_cast<Real>(value));
    }

    template<typename T>
    constexpr auto asSquaredMagnitude(T value)
    {
        using Real = Real_t<T>;
        if constexpr(ComplexScalar<T>)
            return value.real() * value.real() + value.imag() * value.imag();
        else
        {
            auto const realValue = static_cast<Real>(value);
            return realValue * realValue;
        }
    }

    template<typename T>
    constexpr auto conjugateIfNeeded(T value)
    {
        return value;
    }

    template<typename T>
    constexpr auto conjugate(T value)
    {
        if constexpr(ComplexScalar<T>)
            return T{value.real(), -value.imag()};
        else
            return value;
    }

    template<typename T_View>
    [[nodiscard]] inline auto extents(T_View const& view)
    {
        return alpaka::onHost::getExtents(getView(view));
    }

    template<typename T_View>
    [[nodiscard]] inline auto pitches(T_View const& view)
    {
        return alpaka::onHost::getPitches(getView(view));
    }

    template<typename T_View>
    [[nodiscard]] inline auto data(T_View&& view)
    {
        return alpaka::onHost::data(getView(std::forward<T_View>(view)));
    }

    template<std::integral T_Int, std::integral T_Value>
    [[nodiscard]] inline T_Int checkedCast(T_Value value, char const* what)
    {
        if((std::is_signed_v<T_Value> && value < 0)
           || static_cast<unsigned long long>(value)
                  > static_cast<unsigned long long>(std::numeric_limits<T_Int>::max()))
            throw std::invalid_argument(std::string{what} + " is out of range.");
        return static_cast<T_Int>(value);
    }

    struct VectorDescriptor
    {
        void const* constPtr = nullptr;
        void* mutPtr = nullptr;
        std::int64_t n = 0;
        std::int64_t inc = 1;
    };

    struct MatrixDescriptor
    {
        void const* constPtr = nullptr;
        void* mutPtr = nullptr;
        std::int64_t rows = 0;
        std::int64_t cols = 0;
        std::int64_t ld = 0;
        Transpose transpose = Transpose::none;
        Triangle triangle = Triangle::full;
        Diagonal diagonal = Diagonal::nonUnit;
    };

    struct BatchedMatrixDescriptor : MatrixDescriptor
    {
        std::int64_t batchCount = 0;
        std::int64_t batchStride = 0;
    };

    template<typename T_View>
    [[nodiscard]] inline VectorDescriptor makeVectorDescriptor(T_View&& view)
    {
        using View = std::remove_reference_t<decltype(getView(view))>;
        static_assert(View::dim() == 1u);
        auto const& base = getView(view);
        auto const ex = alpaka::onHost::getExtents(base);
        auto const pt = alpaka::onHost::getPitches(base);
        if(pt.x() % sizeof(Value_t<View>) != 0u)
            throw std::invalid_argument("Vector pitch must be a multiple of the element size.");
        auto const stride = pt.x() / sizeof(Value_t<View>);
        return VectorDescriptor{
            .constPtr = static_cast<void const*>(alpaka::onHost::data(base)),
            .mutPtr = const_cast<void*>(static_cast<void const*>(alpaka::onHost::data(base))),
            .n = checkedCast<std::int64_t>(ex.x(), "vector extent"),
            .inc = checkedCast<std::int64_t>(stride, "vector stride")};
    }

    template<typename T_View>
    [[nodiscard]] inline MatrixDescriptor makeMatrixDescriptor(T_View&& view)
    {
        using View = std::remove_reference_t<decltype(getView(view))>;
        static_assert(View::dim() == 2u);
        auto const& base = getView(view);
        auto const ex = alpaka::onHost::getExtents(base);
        auto const pt = alpaka::onHost::getPitches(base);
        auto const strideCol = pt.x() / sizeof(Value_t<View>);
        if(strideCol != 1u)
            throw std::invalid_argument("Only row-major dense 2D views are supported.");
        if(pt.y() % sizeof(Value_t<View>) != 0u)
            throw std::invalid_argument("Row pitch must be a multiple of the element size.");
        auto const strideRow = pt.y() / sizeof(Value_t<View>);
        if(strideRow < ex.x())
            throw std::invalid_argument("Invalid row-major leading dimension.");
        return MatrixDescriptor{
            .constPtr = static_cast<void const*>(alpaka::onHost::data(base)),
            .mutPtr = const_cast<void*>(static_cast<void const*>(alpaka::onHost::data(base))),
            .rows = checkedCast<std::int64_t>(ex.y(), "matrix rows"),
            .cols = checkedCast<std::int64_t>(ex.x(), "matrix cols"),
            .ld = checkedCast<std::int64_t>(strideRow, "matrix ld"),
            .transpose = getTranspose(view),
            .triangle = getTriangle(view),
            .diagonal = getDiagonal(view)};
    }

    template<typename T_View>
    [[nodiscard]] inline BatchedMatrixDescriptor makeBatchedMatrixDescriptor(T_View&& view)
    {
        using View = std::remove_reference_t<decltype(getView(view))>;
        static_assert(View::dim() == 3u);
        auto const& base = getView(view);
        auto const ex = alpaka::onHost::getExtents(base);
        auto const pt = alpaka::onHost::getPitches(base);
        auto const strideCol = pt.x() / sizeof(Value_t<View>);
        if(strideCol != 1u)
            throw std::invalid_argument("Only row-major dense 3D batched views are supported.");
        if(pt.y() % sizeof(Value_t<View>) != 0u)
            throw std::invalid_argument("Row pitch must be a multiple of the element size.");
        if(pt.z() % sizeof(Value_t<View>) != 0u)
            throw std::invalid_argument("Batch pitch must be a multiple of the element size.");
        auto const strideRow = pt.y() / sizeof(Value_t<View>);
        auto const strideBatch = pt.z() / sizeof(Value_t<View>);
        if(strideRow < ex.x())
            throw std::invalid_argument("Invalid batched matrix leading dimension.");
        BatchedMatrixDescriptor desc{};
        desc.constPtr = static_cast<void const*>(alpaka::onHost::data(base));
        desc.mutPtr = const_cast<void*>(static_cast<void const*>(alpaka::onHost::data(base)));
        desc.rows = checkedCast<std::int64_t>(ex.y(), "matrix rows");
        desc.cols = checkedCast<std::int64_t>(ex.x(), "matrix cols");
        desc.ld = checkedCast<std::int64_t>(strideRow, "matrix ld");
        desc.transpose = getTranspose(view);
        desc.triangle = getTriangle(view);
        desc.diagonal = getDiagonal(view);
        desc.batchCount = checkedCast<std::int64_t>(ex.z(), "batch count");
        desc.batchStride = checkedCast<std::int64_t>(strideBatch, "batch stride");
        return desc;
    }

    template<typename T_Matrix>
    [[nodiscard]] inline auto opRows(T_Matrix const& A)
    {
        return getTranspose(A) == Transpose::none ? makeMatrixDescriptor(A).rows : makeMatrixDescriptor(A).cols;
    }

    template<typename T_Matrix>
    [[nodiscard]] inline auto opCols(T_Matrix const& A)
    {
        return getTranspose(A) == Transpose::none ? makeMatrixDescriptor(A).cols : makeMatrixDescriptor(A).rows;
    }

    template<typename T>
    inline void validateScalarSupport()
    {
        static_assert(Scalar<T>, "Unsupported BLAS scalar type.");
    }

    template<typename T_Matrix>
    inline void validateTriangularAnnotation(T_Matrix const& A)
    {
        if(getTriangle(A) == Triangle::full)
            throw std::invalid_argument("Triangular BLAS operations require upper(A) or lower(A).");
    }

    template<typename T_A, typename T_B, typename T_C>
    inline void validateGemm(T_A const& A, T_B const& B, T_C const& C)
    {
        auto const ad = makeMatrixDescriptor(A);
        auto const bd = makeMatrixDescriptor(B);
        auto const cd = makeMatrixDescriptor(C);
        auto const m = getTranspose(A) == Transpose::none ? ad.rows : ad.cols;
        auto const kA = getTranspose(A) == Transpose::none ? ad.cols : ad.rows;
        auto const kB = getTranspose(B) == Transpose::none ? bd.rows : bd.cols;
        auto const n = getTranspose(B) == Transpose::none ? bd.cols : bd.rows;
        if(kA != kB)
            throw std::invalid_argument("gemm requires op(A).cols == op(B).rows.");
        if(cd.rows != m || cd.cols != n)
            throw std::invalid_argument("gemm requires C to match op(A).rows x op(B).cols.");
    }

    template<typename T_A, typename T_X, typename T_Y>
    inline void validateGemv(T_A const& A, T_X const& x, T_Y const& y)
    {
        auto const ad = makeMatrixDescriptor(A);
        auto const xd = makeVectorDescriptor(x);
        auto const yd = makeVectorDescriptor(y);
        auto const m = getTranspose(A) == Transpose::none ? ad.rows : ad.cols;
        auto const n = getTranspose(A) == Transpose::none ? ad.cols : ad.rows;
        if(xd.n != n)
            throw std::invalid_argument("gemv requires x extent to match op(A).cols.");
        if(yd.n != m)
            throw std::invalid_argument("gemv requires y extent to match op(A).rows.");
    }

    template<typename T_X, typename T_Y>
    inline void validateSameVectorExtent(T_X const& x, T_Y const& y, char const* what)
    {
        if(makeVectorDescriptor(x).n != makeVectorDescriptor(y).n)
            throw std::invalid_argument(std::string{what} + " requires matching vector extents.");
    }

    template<typename T_X, typename T_Result>
    inline void validateScalarResult(T_X const&, T_Result const& result, char const* what)
    {
        if(extents(result).x() != 1u)
            throw std::invalid_argument(std::string{what} + " requires a one-element result buffer.");
    }

    template<typename T_A, typename T_B>
    inline void validateTrsm(Side side, T_A const& A, T_B const& B)
    {
        validateTriangularAnnotation(A);
        auto const ad = makeMatrixDescriptor(A);
        auto const bd = makeMatrixDescriptor(B);
        auto const aOrder = getTranspose(A) == Transpose::none ? ad.rows : ad.cols;
        auto const otherOrder = getTranspose(A) == Transpose::none ? ad.cols : ad.rows;
        if(aOrder != otherOrder)
            throw std::invalid_argument("trsm requires a square triangular matrix.");
        if(side == Side::left)
        {
            if(ad.rows != bd.rows)
                throw std::invalid_argument("trsm left requires A.rows == B.rows.");
        }
        else if(ad.rows != bd.cols)
            throw std::invalid_argument("trsm right requires A.rows == B.cols.");
    }

    template<typename T_A, typename T_C>
    inline void validateHerk(T_A const& A, T_C const& C)
    {
        // Value_t keeps cv-qualifiers, so both A (possibly const-element) and C (writable) are compared after
        // removing const: herk requires a complex element type shared by A and C, and rejects mismatched complex
        // scalars (finding: Complex<float> A with Complex<double> C must not silently select a backend overload).
        using AValue = std::remove_cv_t<Value_t<T_A>>;
        using CValue = std::remove_cv_t<Value_t<T_C>>;
        static_assert(ComplexScalar<AValue>, "herk requires a complex A.");
        static_assert(std::same_as<AValue, CValue>, "herk requires A and C to have the same element type.");
        auto const ad = makeMatrixDescriptor(A);
        auto const cd = makeMatrixDescriptor(C);
        // A is a general dense matrix: reject triangle/unit-diagonal annotations.
        if(ad.triangle != Triangle::full)
            throw std::invalid_argument("herk requires a general dense A without a triangle annotation.");
        if(ad.diagonal != Diagonal::nonUnit)
            throw std::invalid_argument("herk requires a general dense A without a unit-diagonal annotation.");
        // HERK accepts the as-stored matrix or its conjugate transpose only.
        if(ad.transpose != Transpose::none && ad.transpose != Transpose::conjugateTransposed)
            throw std::invalid_argument("herk accepts A or conjTransposed(A), not a plain transposed(A).");
        // C must carry an explicit upper/lower selection.
        if(cd.triangle == Triangle::full)
            throw std::invalid_argument("herk requires upper(C) or lower(C).");
        // C must not be transposed or unit-diagonal.
        if(cd.transpose != Transpose::none)
            throw std::invalid_argument("herk rejects a transposed C.");
        if(cd.diagonal != Diagonal::nonUnit)
            throw std::invalid_argument("herk rejects a unit-diagonal C.");
        // op(A) is n x k; C must be n x n.
        auto const n = getTranspose(A) == Transpose::none ? ad.rows : ad.cols;
        if(cd.rows != n || cd.cols != n)
            throw std::invalid_argument("herk requires C to be n x n where n is op(A) rows.");
    }

    template<typename T_A, typename T_C>
    inline void validateSyrk(T_A const& A, T_C const& C)
    {
        static_assert(
            std::same_as<std::remove_cv_t<Value_t<T_A>>, std::remove_cv_t<Value_t<T_C>>>,
            "syrk requires A and C to have the same element type.");
        static_assert(
            !std::is_const_v<alpaka::GetValueType_t<detail::unannotated_t<T_C>>>,
            "syrk requires a writable C view (element type must not be const).");
        auto const ad = makeMatrixDescriptor(A);
        auto const cd = makeMatrixDescriptor(C);
        // A is a general dense matrix: reject triangle/unit-diagonal annotations.
        if(ad.triangle != Triangle::full)
            throw std::invalid_argument("syrk requires a general dense A without a triangle annotation.");
        if(ad.diagonal != Diagonal::nonUnit)
            throw std::invalid_argument("syrk requires a general dense A without a unit-diagonal annotation.");
        // C must carry an explicit upper/lower selection.
        if(cd.triangle == Triangle::full)
            throw std::invalid_argument("syrk requires upper(C) or lower(C).");
        // C must not be transposed or unit-diagonal.
        if(cd.transpose != Transpose::none)
            throw std::invalid_argument("syrk rejects a transposed C.");
        if(cd.diagonal != Diagonal::nonUnit)
            throw std::invalid_argument("syrk rejects a unit-diagonal C.");
        // op(A) is n x k; C must be n x n.
        auto const n = getTranspose(A) == Transpose::none ? ad.rows : ad.cols;
        if(cd.rows != n || cd.cols != n)
            throw std::invalid_argument("syrk requires C to be n x n where n is op(A) rows.");
    }

    template<typename T>
    constexpr auto zeroValue()
    {
        return T{};
    }

    template<typename T>
    constexpr auto oneValue()
    {
        if constexpr(ComplexScalar<T>)
            return T{1, 0};
        else
            return T{1};
    }

    template<typename T>
    inline void storeOneElement(auto& result, T value)
    {
        auto& base = getView(result);
        base.data()[0] = value;
    }
} // namespace alpaka::blas::internal
