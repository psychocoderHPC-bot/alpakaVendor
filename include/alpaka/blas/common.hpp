/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/math/Complex.hpp>

#include <concepts>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace alpaka::blas
{
    /**
     * How a matrix operand is interpreted before it is passed to the backend BLAS call.
     */
    enum class Transpose
    {
        none, ///< Use the matrix as stored.
        transposed, ///< Use the plain transpose, so rows become columns.
        conjugateTransposed ///< Use the conjugate transpose, mainly for complex-valued matrices.
    };

    /**
     * Which triangular part of a matrix is considered meaningful.
     *
     * This matters for routines such as ``trsm`` where BLAS only reads one half of the matrix.
     */
    enum class Triangle
    {
        full, ///< The full matrix participates.
        upper, ///< Only the upper triangular part is used.
        lower ///< Only the lower triangular part is used.
    };

    /**
     * Whether a triangular matrix should be treated as having an explicit or implicit unit diagonal.
     */
    enum class Diagonal
    {
        nonUnit, ///< Read the stored diagonal values.
        unit ///< Pretend every diagonal entry is ``1`` and ignore the stored diagonal values.
    };

    /**
     * Which side a special matrix operand is applied from.
     *
     * For ``trsm`` this decides whether the triangular matrix multiplies from the left
     * (``op(A) * X = B``) or from the right (``X * op(A) = B``).
     */
    enum class Side
    {
        left,
        right
    };

    /**
     * Hint for backends that offer multiple math modes.
     *
     * ``exact`` asks for the precise scalar type requested by the user. ``backendDefault`` lets the backend pick
     * its default mode. Backends that do not expose a matching knob simply ignore the hint.
     */
    enum class Precision
    {
        exact,
        backendDefault
    };

    /**
     * Hint for backends that expose multiple algorithm choices.
     *
     * Some vendor libraries can trade reproducibility for speed. Backends without such a distinction ignore the hint.
     */
    enum class Algorithm
    {
        backendDefault,
        deterministic,
        fastest
    };

    /**
     * Optional execution hints passed to BLAS entry points.
     */
    struct Options
    {
        Precision precision = Precision::exact; ///< Preferred math mode when the backend supports one.
        Algorithm algorithm = Algorithm::backendDefault; ///< Preferred backend algorithm, if selectable.
    };

    template<typename T>
    concept RealScalar = std::same_as<std::remove_cv_t<T>, float> || std::same_as<std::remove_cv_t<T>, double>;

    template<typename T>
    struct is_alpaka_complex : std::false_type
    {
    };

    template<typename T>
    struct is_alpaka_complex<alpaka::math::Complex<T>> : std::bool_constant<RealScalar<T>>
    {
    };

    template<typename T>
    concept ComplexScalar = is_alpaka_complex<std::remove_cv_t<T>>::value;

    /** Scalar types supported by the public BLAS wrappers. */
    template<typename T>
    concept Scalar = RealScalar<T> || ComplexScalar<T>;

    template<typename T>
    struct Real;

    template<RealScalar T>
    struct Real<T>
    {
        using type = std::remove_cv_t<T>;
    };

    template<ComplexScalar T>
    struct Real<T>
    {
        using type = typename std::remove_cv_t<T>::value_type;
    };

    template<typename T>
    using Real_t = typename Real<std::remove_cvref_t<T>>::type;

    /**
     * Lightweight wrapper that keeps a view together with BLAS-specific interpretation flags.
     *
     * Users normally create this through helpers such as ``transposed()``, ``lower()``, or ``unitDiag()`` instead of
     * constructing it directly.
     */
    template<typename T_View>
    struct AnnotatedView
    {
        T_View view; ///< The original alpaka mdspan-like view.
        Transpose transpose = Transpose::none; ///< Whether BLAS should apply ``op(view)`` first.
        Triangle triangle = Triangle::full; ///< Which triangular half is considered valid.
        Diagonal diagonal = Diagonal::nonUnit; ///< Whether the diagonal entries are stored or implied to be one.
    };

    namespace detail
    {
        template<typename T>
        struct Unannotated
        {
            using type = std::remove_cvref_t<T>;
        };

        template<typename T_View>
        struct Unannotated<AnnotatedView<T_View>>
        {
            using type = std::remove_cvref_t<T_View>;
        };

        template<typename T>
        using unannotated_t = typename Unannotated<std::remove_cvref_t<T>>::type;

        template<typename T>
        struct is_annotated_view : std::false_type
        {
        };

        template<typename T_View>
        struct is_annotated_view<AnnotatedView<T_View>> : std::true_type
        {
        };

        template<typename T>
        constexpr decltype(auto) getView(T&& x)
        {
            return std::forward<T>(x);
        }

        template<typename T_View>
        constexpr decltype(auto) getView(AnnotatedView<T_View>& x)
        {
            return (x.view);
        }

        template<typename T_View>
        constexpr decltype(auto) getView(AnnotatedView<T_View> const& x)
        {
            return (x.view);
        }

        template<typename T_View>
        constexpr decltype(auto) getView(AnnotatedView<T_View>&& x)
        {
            return std::move(x.view);
        }

        template<typename T>
        constexpr Transpose getTranspose([[maybe_unused]] T const& x)
        {
            return Transpose::none;
        }

        template<typename T_View>
        constexpr Transpose getTranspose(AnnotatedView<T_View> const& x)
        {
            return x.transpose;
        }

        template<typename T>
        constexpr Triangle getTriangle([[maybe_unused]] T const& x)
        {
            return Triangle::full;
        }

        template<typename T_View>
        constexpr Triangle getTriangle(AnnotatedView<T_View> const& x)
        {
            return x.triangle;
        }

        template<typename T>
        constexpr Diagonal getDiagonal([[maybe_unused]] T const& x)
        {
            return Diagonal::nonUnit;
        }

        template<typename T_View>
        constexpr Diagonal getDiagonal(AnnotatedView<T_View> const& x)
        {
            return x.diagonal;
        }

        template<typename T_View>
        [[nodiscard]] constexpr auto annotate(T_View&& view)
        {
            return AnnotatedView<std::remove_cvref_t<T_View>>{std::forward<T_View>(view)};
        }

        template<typename T_View>
        [[nodiscard]] constexpr auto asAnnotated(T_View&& view)
        {
            if constexpr(is_annotated_view<std::remove_cvref_t<T_View>>::value)
                return std::forward<T_View>(view);
            else
                return annotate(std::forward<T_View>(view));
        }
    } // namespace detail

    /**
     * Mark a matrix view as transposed for routines such as ``gemv`` or ``gemm``.
     *
     * This does not move data. It only changes how the BLAS wrapper interprets the view.
     */
    template<typename T_View>
    [[nodiscard]] constexpr auto transposed(T_View&& view)
    {
        auto annotated = detail::asAnnotated(std::forward<T_View>(view));
        annotated.transpose = Transpose::transposed;
        return annotated;
    }

    /**
     * Mark a matrix view as conjugate-transposed.
     *
     * This is the complex-valued counterpart to ``transposed()`` and is useful for Hermitian-style algebra.
     */
    template<typename T_View>
    [[nodiscard]] constexpr auto conjTransposed(T_View&& view)
    {
        auto annotated = detail::asAnnotated(std::forward<T_View>(view));
        annotated.transpose = Transpose::conjugateTransposed;
        return annotated;
    }

    /**
     * Mark a matrix as upper triangular.
     *
     * The annotation can be stacked with ``transposed()``, ``conjTransposed()``, ``unitDiag()``, or
     * ``nonUnitDiag()``. Mixing ``upper()`` and ``lower()`` on the same view is rejected.
     */
    template<typename T_View>
    [[nodiscard]] constexpr auto upper(T_View&& view)
    {
        auto annotated = detail::asAnnotated(std::forward<T_View>(view));
        if(annotated.triangle == Triangle::lower)
            throw std::invalid_argument("Conflicting triangular annotation: lower then upper.");
        annotated.triangle = Triangle::upper;
        return annotated;
    }

    /**
     * Mark a matrix as lower triangular.
     *
     * The annotation can be stacked with ``transposed()``, ``conjTransposed()``, ``unitDiag()``, or
     * ``nonUnitDiag()``. Mixing ``lower()`` and ``upper()`` on the same view is rejected.
     */
    template<typename T_View>
    [[nodiscard]] constexpr auto lower(T_View&& view)
    {
        auto annotated = detail::asAnnotated(std::forward<T_View>(view));
        if(annotated.triangle == Triangle::upper)
            throw std::invalid_argument("Conflicting triangular annotation: upper then lower.");
        annotated.triangle = Triangle::lower;
        return annotated;
    }

    /**
     * Mark a triangular matrix as having an implicit unit diagonal.
     *
     * BLAS routines then behave as if the diagonal entries were all one and do not need to read the stored values.
     */
    template<typename T_View>
    [[nodiscard]] constexpr auto unitDiag(T_View&& view)
    {
        auto annotated = detail::asAnnotated(std::forward<T_View>(view));
        annotated.diagonal = Diagonal::unit;
        return annotated;
    }

    /**
     * Mark a triangular matrix as using the stored diagonal values.
     */
    template<typename T_View>
    [[nodiscard]] constexpr auto nonUnitDiag(T_View&& view)
    {
        auto annotated = detail::asAnnotated(std::forward<T_View>(view));
        annotated.diagonal = Diagonal::nonUnit;
        return annotated;
    }

    namespace concepts
    {
        /** Any alpaka mdspan-like view, annotated or plain. */
        template<typename T>
        concept View = alpaka::concepts::IMdSpan<detail::unannotated_t<T>>;

        /** A one-dimensional BLAS view, used by Level-1 routines. */
        template<typename T>
        concept VectorView = View<T> && detail::unannotated_t<T>::dim() == 1u;

        /** A two-dimensional BLAS view, used by matrix-vector and matrix-matrix routines. */
        template<typename T>
        concept MatrixView = View<T> && detail::unannotated_t<T>::dim() == 2u;

        /** A three-dimensional view interpreted as ``[batch, row, column]`` for strided batched GEMM. */
        template<typename T>
        concept BatchedMatrixView = View<T> && detail::unannotated_t<T>::dim() == 3u;
    } // namespace concepts
} // namespace alpaka::blas
