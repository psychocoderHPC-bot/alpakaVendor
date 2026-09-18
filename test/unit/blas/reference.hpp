/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <vector>

#include "alpaka/blas.hpp"

namespace alpakaVendor::test::blas
{
    template<typename T>
    inline auto zero()
    {
        return T{};
    }

    template<typename T>
    inline auto conj(T value)
    {
        if constexpr(alpaka::blas::ComplexScalar<T>)
            return T{value.real(), -value.imag()};
        else
            return value;
    }

    template<typename T>
    inline T applyTranspose(
        T const* ptr,
        std::size_t ld,
        std::size_t row,
        std::size_t col,
        alpaka::blas::Transpose transpose)
    {
        switch(transpose)
        {
        case alpaka::blas::Transpose::none:
            return ptr[row * ld + col];
        case alpaka::blas::Transpose::transposed:
            return ptr[col * ld + row];
        case alpaka::blas::Transpose::conjugateTransposed:
            return conj(ptr[col * ld + row]);
        }
        return {};
    }

    template<typename T>
    inline void gemmRef(
        T alpha,
        T const* a,
        std::size_t lda,
        std::size_t aRows,
        std::size_t aCols,
        alpaka::blas::Transpose transA,
        T const* b,
        std::size_t ldb,
        std::size_t bRows,
        std::size_t bCols,
        alpaka::blas::Transpose transB,
        T beta,
        T* c,
        std::size_t ldc,
        std::size_t cRows,
        std::size_t cCols)
    {
        auto const m = transA == alpaka::blas::Transpose::none ? aRows : aCols;
        auto const k = transA == alpaka::blas::Transpose::none ? aCols : aRows;
        auto const n = transB == alpaka::blas::Transpose::none ? bCols : bRows;
        for(std::size_t row = 0; row < m; ++row)
            for(std::size_t col = 0; col < n; ++col)
            {
                T sum{};
                for(std::size_t kk = 0; kk < k; ++kk)
                    sum += applyTranspose(a, lda, row, kk, transA) * applyTranspose(b, ldb, kk, col, transB);
                c[row * ldc + col] = alpha * sum + beta * c[row * ldc + col];
            }
        alpaka::unused(cRows, cCols);
    }

    template<typename T>
    inline void gemvRef(
        T alpha,
        T const* a,
        std::size_t lda,
        std::size_t rows,
        std::size_t cols,
        alpaka::blas::Transpose trans,
        T const* x,
        T beta,
        T* y)
    {
        auto const outRows = trans == alpaka::blas::Transpose::none ? rows : cols;
        auto const inner = trans == alpaka::blas::Transpose::none ? cols : rows;
        for(std::size_t row = 0; row < outRows; ++row)
        {
            T sum{};
            for(std::size_t col = 0; col < inner; ++col)
                sum += applyTranspose(a, lda, row, col, trans) * x[col];
            y[row] = alpha * sum + beta * y[row];
        }
    }

    template<typename T>
    inline T dotRef(T const* x, T const* y, std::size_t n)
    {
        T sum{};
        for(std::size_t i = 0; i < n; ++i)
            sum += x[i] * y[i];
        return sum;
    }

    // Real symmetric rank-k update: writes the selected triangle of
    // C = alpha * M * M^T + beta * C where M = op(A) is n x k.
    // The opposite triangle and padding are left unchanged.
    template<typename T>
    inline void syrkRef(
        T alpha,
        T const* a,
        std::size_t lda,
        std::size_t aRows,
        std::size_t aCols,
        alpaka::blas::Transpose trans,
        T beta,
        T* c,
        std::size_t ldc,
        std::size_t n,
        alpaka::blas::Triangle triangle)
    {
        auto const k = trans == alpaka::blas::Transpose::none ? aCols : aRows;
        for(std::size_t i = 0; i < n; ++i)
        {
            auto const inTri = [triangle](std::size_t r, std::size_t col)
            { return triangle == alpaka::blas::Triangle::upper ? col >= r : col <= r; };
            for(std::size_t j = 0; j < n; ++j)
            {
                if(!inTri(i, j))
                    continue;
                T sum{};
                for(std::size_t kk = 0; kk < k; ++kk)
                    sum += applyTranspose(a, lda, i, kk, trans) * applyTranspose(a, lda, j, kk, trans);
                c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
            }
        }
    }

    template<typename T>
    inline auto nrm2Ref(T const* x, std::size_t n)
    {
        using Real = alpaka::blas::Real_t<T>;
        Real sum{};
        for(std::size_t i = 0; i < n; ++i)
        {
            if constexpr(alpaka::blas::ComplexScalar<T>)
                sum += x[i].real() * x[i].real() + x[i].imag() * x[i].imag();
            else
                sum += x[i] * x[i];
        }
        return std::sqrt(sum);
    }

    template<typename T>
    inline auto asumRef(T const* x, std::size_t n)
    {
        using Real = alpaka::blas::Real_t<T>;
        Real sum{};
        for(std::size_t i = 0; i < n; ++i)
        {
            if constexpr(alpaka::blas::ComplexScalar<T>)
                sum += std::abs(x[i].real()) + std::abs(x[i].imag());
            else
                sum += std::abs(x[i]);
        }
        return sum;
    }

    template<typename T>
    inline int iamaxRef(T const* x, std::size_t n)
    {
        using Real = alpaka::blas::Real_t<T>;
        Real best = -1;
        int idx = 1;
        for(std::size_t i = 0; i < n; ++i)
        {
            Real value = 0;
            if constexpr(alpaka::blas::ComplexScalar<T>)
                value = std::abs(x[i].real()) + std::abs(x[i].imag());
            else
                value = std::abs(x[i]);
            if(value > best)
            {
                best = value;
                idx = static_cast<int>(i) + 1;
            }
        }
        return idx;
    }
} // namespace alpakaVendor::test::blas
