/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <algorithm>
#include <alpakaTest/deviceHelper.hpp>
#include <cmath>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

#include "../unit/blas/reference.hpp"
#include "../unit/test.hpp"
#include "alpaka/blas.hpp"

using namespace alpakaVendor::test;

// Row-major leading dimension in elements for a 2D alpaka view.
template<typename T_View>
auto ldOf(T_View const& view)
{
    return view.getPitches().y() / sizeof(alpaka::GetValueType_t<T_View>);
}

template<typename T>
using RealOf = alpaka::blas::Real_t<T>;

// Fill a matrix with nonzero imaginary parts, so the conjugation/operation mapping is exercised.
template<typename T_View>
void fillMatrixComplex(T_View& view, std::size_t rows, std::size_t cols)
{
    using T = alpaka::GetValueType_t<std::remove_cvref_t<T_View>>;
    using Real = RealOf<T>;
    for(std::size_t i = 0; i < rows; ++i)
        for(std::size_t j = 0; j < cols; ++j)
            view[alpaka::Vec<uint32_t, 2u>{i, j}]
                = T{static_cast<Real>(10 * i + j + 1), static_cast<Real>(7 * i + 3 * j + 2)};
}

template<typename T_View>
void fillMatrixSentinel(T_View& view, std::size_t rows, std::size_t cols, auto sentinel)
{
    using T = alpaka::GetValueType_t<std::remove_cvref_t<T_View>>;
    for(std::size_t i = 0; i < rows; ++i)
        for(std::size_t j = 0; j < cols; ++j)
            view[alpaka::Vec<uint32_t, 2u>{i, j}] = T(sentinel);
}

// Copy a 2D view's raw row-major buffer (row pitch ld) into a std::vector.
template<typename T>
auto copyRaw(T const* ptr, std::size_t rows, std::size_t cols, std::size_t ld) -> std::vector<T>
{
    std::vector<T> out(rows * ld);
    for(std::size_t i = 0; i < rows; ++i)
        std::copy_n(ptr + i * ld, cols, out.data() + i * ld);
    return out;
}

template<typename T>
bool isFinite(T value)
{
    using Real = RealOf<T>;
    return std::isfinite(static_cast<Real>(value.real())) && std::isfinite(static_cast<Real>(value.imag()));
}

// Whether the public herk entry forms for the given argument types. Expressing the call through a variable-template
// requires-expression makes the negative cases (unsatisfied constraints) produce `false` instead of a hard error.
template<typename TQueue, typename TAlpha, typename TViewA, typename TBeta, typename TViewC>
inline constexpr bool herkCallable = requires(TQueue& queue, TAlpha alpha, TViewA& A, TBeta beta, TViewC& C) {
    alpaka::blas::onHost::herk(queue, alpha, A, beta, C);
};

TEMPLATE_LIST_TEST_CASE("BLAS herk complex Hermitian rank-k update", "[integr][blas][rankk][herk]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<float>;
        using Real = RealOf<Scalar>;
        constexpr uint32_t n = 5u;
        constexpr uint32_t k = 3u;

        auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
        auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        auto Ccopy = alpaka::onHost::allocHostLike(C);
        fillMatrixComplex(A, n, k);
        fillMatrixComplex(C, n, n);
        for(uint32_t i = 0; i < n; ++i)
            for(uint32_t j = 0; j < n; ++j)
                Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];

        auto upperC = alpaka::blas::upper(C);
        alpaka::blas::onHost::herk(queue, Real{1.5f}, A, Real{-0.5f}, upperC, options);
        alpaka::onHost::wait(queue);

        auto const ldA = ldOf(A);
        auto const ldC = ldOf(C);
        auto const Aref = copyRaw(A.data(), n, k, ldA);
        auto Cref = copyRaw(Ccopy.data(), n, n, ldC);
        blas::herkRef(
            Real{1.5f},
            Aref.data(),
            ldA,
            n,
            k,
            alpaka::blas::Transpose::none,
            Real{-0.5f},
            Cref.data(),
            ldC,
            n,
            alpaka::blas::Triangle::upper);
        for(uint32_t i = 0; i < n; ++i)
            for(uint32_t j = 0; j < n; ++j)
            {
                if(j >= i)
                {
                    CHECK(
                        C[alpaka::Vec<uint32_t, 2u>{i, j}].real()
                        == Catch::Approx(Cref[i * ldC + j].real()).epsilon(1e-4f).margin(1e-4f));
                    CHECK(
                        C[alpaka::Vec<uint32_t, 2u>{i, j}].imag()
                        == Catch::Approx(Cref[i * ldC + j].imag()).epsilon(1e-4f).margin(1e-4f));
                }
                else
                {
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}]);
                }
            }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk double precision conjTransposed input and lower triangle",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<double>;
        using Real = RealOf<Scalar>;
        constexpr uint32_t n = 4u;
        constexpr uint32_t k = 6u;

        // A stored as k x n; conjTransposed(A) is n x k.
        auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{k, n});
        auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        auto Ccopy = alpaka::onHost::allocHostLike(C);
        fillMatrixComplex(A, k, n);
        fillMatrixComplex(C, n, n);
        for(uint32_t i = 0; i < n; ++i)
            for(uint32_t j = 0; j < n; ++j)
                Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];

        auto conjTransposedA = alpaka::blas::conjTransposed(A);
        auto lowerC = alpaka::blas::lower(C);
        alpaka::blas::onHost::herk(queue, Real{0.75}, conjTransposedA, Real{2.0}, lowerC, options);
        alpaka::onHost::wait(queue);

        auto const ldA = ldOf(A);
        auto const ldC = ldOf(C);
        auto const Aref = copyRaw(A.data(), k, n, ldA);
        auto Cref = copyRaw(Ccopy.data(), n, n, ldC);
        blas::herkRef(
            Real{0.75},
            Aref.data(),
            ldA,
            k,
            n,
            alpaka::blas::Transpose::conjugateTransposed,
            Real{2.0},
            Cref.data(),
            ldC,
            n,
            alpaka::blas::Triangle::lower);
        for(uint32_t i = 0; i < n; ++i)
            for(uint32_t j = 0; j < n; ++j)
            {
                if(j <= i)
                {
                    CHECK(
                        C[alpaka::Vec<uint32_t, 2u>{i, j}].real()
                        == Catch::Approx(Cref[i * ldC + j].real()).epsilon(1e-12).margin(1e-12));
                    CHECK(
                        C[alpaka::Vec<uint32_t, 2u>{i, j}].imag()
                        == Catch::Approx(Cref[i * ldC + j].imag()).epsilon(1e-12).margin(1e-12));
                }
                else
                {
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}]);
                }
            }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk n=1 and padded layout preserve padding and opposite triangle",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<float>;
        using Real = RealOf<Scalar>;
        // n=1 case: A is 1 x 2, C is 1 x 1.
        {
            constexpr uint32_t n = 1u;
            constexpr uint32_t k = 2u;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            auto Ccopy = alpaka::onHost::allocHostLike(C);
            fillMatrixComplex(A, n, k);
            Ccopy[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = C[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = Scalar{2.0f, 3.0f};
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{1.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(A.data(), n, k, ldA);
            auto Cref = copyRaw(Ccopy.data(), n, n, ldC);
            blas::herkRef(
                Real{1.0f},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Real{1.0f},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::upper);
            CHECK(
                C[alpaka::Vec<uint32_t, 2u>{0u, 0u}].real()
                == Catch::Approx(Cref[0].real()).epsilon(1e-4f).margin(1e-4f));
            CHECK(C[alpaka::Vec<uint32_t, 2u>{0u, 0u}].imag() == Catch::Approx(0.0f).margin(1e-5f));
        }
        // Padded C (ld > n) with padding sentinels and distinct opposite-triangle sentinels.
        {
            constexpr uint32_t n = 4u;
            constexpr uint32_t k = 3u;
            constexpr uint32_t pad = 7u;
            auto storageA = alpaka::onHost::allocUnified<Scalar>(device, n * k);
            auto A = alpaka::makeMdSpan(
                storageA.data(),
                alpaka::Vec<uint32_t, 2u>{n, k},
                alpaka::Vec<std::size_t, 2u>{k * sizeof(Scalar), sizeof(Scalar)});
            auto storageC = alpaka::onHost::allocUnified<Scalar>(device, n * pad);
            auto C = alpaka::makeMdSpan(
                storageC.data(),
                alpaka::Vec<uint32_t, 2u>{n, n},
                alpaka::Vec<std::size_t, 2u>{pad * sizeof(Scalar), sizeof(Scalar)});
            auto storageCcopy = alpaka::onHost::allocUnified<Scalar>(device, n * pad);
            auto Ccopy = alpaka::makeMdSpan(
                storageCcopy.data(),
                alpaka::Vec<uint32_t, 2u>{n, n},
                alpaka::Vec<std::size_t, 2u>{pad * sizeof(Scalar), sizeof(Scalar)});
            fillMatrixComplex(A, n, k);
            auto const padSentinel = Scalar{Real(0x5'a5a5'a5af), Real(0x5'a5a5'a5af)};
            auto const upperSentinel = Scalar{Real(0x1111'1111), Real(0x1111'1111)};
            auto const lowerSentinel = Scalar{Real(0x2222'2222), Real(0x2222'2222)};
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < pad; ++j)
                {
                    auto v = (j < n) ? (j >= i ? upperSentinel : lowerSentinel) : padSentinel;
                    Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = v;
                    C[alpaka::Vec<uint32_t, 2u>{i, j}] = v;
                }
            auto lowerC = alpaka::blas::lower(C);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{0.5f}, lowerC, options);
            alpaka::onHost::wait(queue);
            auto const ldA = k;
            auto const ldC = ldOf(C);
            REQUIRE(ldC == pad);
            auto const Aref = copyRaw(storageA.data(), n, k, ldA);
            auto Cref = copyRaw(storageCcopy.data(), n, n, ldC);
            // Padding is left untouched, so seed the reference with the same padding sentinels.
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = n; j < pad; ++j)
                    Cref[i * ldC + j] = padSentinel;
            blas::herkRef(
                Real{1.0f},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Real{0.5f},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::lower);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < pad; ++j)
                {
                    if(j < n && j <= i)
                    {
                        CHECK(
                            C[alpaka::Vec<uint32_t, 2u>{i, j}].real()
                            == Catch::Approx(Cref[i * ldC + j].real()).epsilon(1e-4f).margin(1e-4f));
                        CHECK(
                            C[alpaka::Vec<uint32_t, 2u>{i, j}].imag()
                            == Catch::Approx(Cref[i * ldC + j].imag()).epsilon(1e-4f).margin(1e-4f));
                    }
                    else
                    {
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}]);
                    }
                }
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk alpha=0 / beta=0 no-contamination semantics",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<float>;
        using Real = RealOf<Scalar>;
        constexpr uint32_t n = 3u;
        constexpr uint32_t k = 3u;
        auto const nanV = Scalar{static_cast<Real>(std::nan("")), static_cast<Real>(std::nan(""))};

        // alpha=0: the product term is zero, so the selected triangle becomes beta*C (the beta scaling still
        // applies; the diagonal is real since a zero product was added). Seed A with NaN; the selected
        // triangle must be finite and equal to beta*C, proving the NaN in A did not contaminate the result.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixSentinel(A, n, k, nanV);
            fillMatrixComplex(C, n, n);
            auto const before = copyRaw(C.data(), n, n, ldOf(C));
            auto upperC = alpaka::blas::upper(C);
            auto const ldC = ldOf(C);
            alpaka::blas::onHost::herk(queue, Real{0.0f}, A, Real{2.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                {
                    CHECK(isFinite(C[alpaka::Vec<uint32_t, 2u>{i, j}]));
                    // alpha=0, beta=2: selected triangle = beta*C, with the (scaled) diagonal real.
                    Scalar expected{
                        2.0f * before[i * ldC + j].real(),
                        (i == j) ? Real{0} : 2.0f * before[i * ldC + j].imag()};
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == expected);
                }
        }
        // beta=0: C = alpha*M*adjoint(M). Seed C with NaN; the selected triangle must be finite and equal to the
        // product.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixComplex(A, n, k);
            fillMatrixSentinel(C, n, n, nanV);
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{0.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(A.data(), n, k, ldA);
            std::vector<Scalar> Cref(n * ldC, Scalar{});
            blas::herkRef(
                Real{1.0f},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Real{0.0f},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::upper);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                {
                    CHECK(isFinite(C[alpaka::Vec<uint32_t, 2u>{i, j}]));
                    CHECK(
                        C[alpaka::Vec<uint32_t, 2u>{i, j}].real()
                        == Catch::Approx(Cref[i * ldC + j].real()).epsilon(1e-4f).margin(1e-4f));
                    CHECK(
                        C[alpaka::Vec<uint32_t, 2u>{i, j}].imag()
                        == Catch::Approx(Cref[i * ldC + j].imag()).epsilon(1e-4f).margin(1e-4f));
                }
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk empty dimensions and coefficient matrix",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<float>;
        using Real = RealOf<Scalar>;
        auto const nanV = Scalar{static_cast<Real>(std::nan("")), static_cast<Real>(std::nan(""))};
        // n=0: assert C's storage is untouched (no-op contract). Use a backing buffer large enough to hold the
        // would-be result so any spurious write is observable, then build 0-row views over it.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{0u, 2u});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{1u, 1u});
            C[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = Scalar{7.0f, -3.0f};
            auto const before = C[alpaka::Vec<uint32_t, 2u>{0u, 0u}];
            auto const upperC = alpaka::blas::upper(C.getSubSharedBuffer(alpaka::Vec<uint32_t, 2u>{0u, 0u}));
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{1.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            CHECK(C[alpaka::Vec<uint32_t, 2u>{0u, 0u}] == before);
        }
        // k=0: the rank-k product is empty, so the selected triangle becomes beta * C. beta == 1 is a true no-op
        // (the triangle, including a complex diagonal imaginary part, stays byte-identical); beta == 0 zeroes the
        // selected triangle without reading it; any other beta scales the triangle in place. For complex C the
        // diagonal stays real (imag == 0) whenever it is scaled.
        for(auto const betaR : {0.0f, 1.0f, 2.5f})
        {
            constexpr uint32_t n = 3u;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, 0u});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixComplex(A, n, 0u);
            fillMatrixComplex(C, n, n);
            // Seed a nonzero imaginary part on the diagonal so a preserved/zeroed diagonal is observable.
            for(uint32_t i = 0; i < n; ++i)
                C[alpaka::Vec<uint32_t, 2u>{i, i}].imag(Real{7.0f});
            // The opposite (lower) triangle is outside the operation: seed it with a distinct known value and require
            // it to stay byte-identical, proving the scale kernel writes only the selected half.
            auto const lowerSentinel = Scalar{Real{1.25f}, Real{-3.75f}};
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < i; ++j)
                    C[alpaka::Vec<uint32_t, 2u>{i, j}] = lowerSentinel;
            auto const before = copyRaw(C.data(), n, n, ldOf(C));
            auto const ldC = ldOf(C);
            auto const upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, betaR, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                {
                    if(betaR == 0.0f)
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Scalar{0.0f, 0.0f});
                    else if(betaR == 1.0f)
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == before[i * ldC + j]);
                    else if(i == j)
                        // Diagonal stays real: beta scales the real part, the imaginary part is zero.
                        CHECK(
                            C[alpaka::Vec<uint32_t, 2u>{i, j}] == Scalar{betaR * before[i * ldC + j].real(), Real{0}});
                    else
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == betaR * before[i * ldC + j]);
                }
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < i; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == lowerSentinel);
        }
        // Lower selection variant: the opposite (upper) triangle must be preserved byte-identically as well.
        {
            constexpr uint32_t n = 3u;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, 0u});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixComplex(A, n, 0u);
            fillMatrixComplex(C, n, n);
            auto const upperSentinel = Scalar{Real{4.5f}, Real{-2.25f}};
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i + 1u; j < n; ++j)
                    C[alpaka::Vec<uint32_t, 2u>{i, j}] = upperSentinel;
            auto const before = copyRaw(C.data(), n, n, ldOf(C));
            auto const ldC = ldOf(C);
            auto const lowerC = alpaka::blas::lower(C);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{2.5f}, lowerC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j <= i; ++j)
                {
                    if(i == j)
                        CHECK(
                            C[alpaka::Vec<uint32_t, 2u>{i, j}] == Scalar{2.5f * before[i * ldC + j].real(), Real{0}});
                    else
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == 2.5f * before[i * ldC + j]);
                }
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i + 1u; j < n; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == upperSentinel);
        }
        // Degenerate kernel must not read C when beta == 0: seed the selected triangle with NaN; a read would
        // propagate NaN into the zeroed result. This covers both degenerate legs: k == 0 (with alpha != 0) and
        // alpha == 0 (with k != 0); both route through the same enqueueScaleTriangle kernel.
        {
            constexpr uint32_t n = 3u;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, 0u});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixComplex(A, n, 0u);
            fillMatrixSentinel(C, n, n, nanV);
            auto const upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{0.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Scalar{0.0f, 0.0f});
        }
        {
            constexpr uint32_t n = 3u;
            constexpr uint32_t k = 2u;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixComplex(A, n, k);
            fillMatrixSentinel(C, n, n, nanV);
            auto const upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::herk(queue, Real{0.0f}, A, Real{0.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Scalar{0.0f, 0.0f});
        }
        // Coefficient matrix: alpha in {0, 1, negative} and beta in {0, 1, nontrivial}. Fresh A and C per
        // combination so each case starts from the same initial state.
        for(auto const alphaR : {0.0f, 1.0f, -1.5f})
            for(auto const betaR : {0.0f, 1.0f, -2.5f})
            {
                constexpr uint32_t n = 3u;
                constexpr uint32_t k = 2u;
                auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
                auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
                fillMatrixComplex(A, n, k);
                fillMatrixComplex(C, n, n);
                auto const before = copyRaw(C.data(), n, n, ldOf(C));
                auto lowerC = alpaka::blas::lower(C);
                alpaka::blas::onHost::herk(queue, alphaR, A, betaR, lowerC, options);
                alpaka::onHost::wait(queue);
                auto const ldA = ldOf(A);
                auto const ldC = ldOf(C);
                auto const Aref = copyRaw(A.data(), n, k, ldA);
                auto Cref = copyRaw(before.data(), n, n, ldC);
                blas::herkRef(
                    alphaR,
                    Aref.data(),
                    ldA,
                    n,
                    k,
                    alpaka::blas::Transpose::none,
                    betaR,
                    Cref.data(),
                    ldC,
                    n,
                    alpaka::blas::Triangle::lower);
                for(uint32_t i = 0; i < n; ++i)
                    for(uint32_t j = 0; j < n; ++j)
                    {
                        if(j <= i)
                        {
                            CHECK(
                                C[alpaka::Vec<uint32_t, 2u>{i, j}].real()
                                == Catch::Approx(Cref[i * ldC + j].real()).epsilon(1e-4f).margin(1e-4f));
                            CHECK(
                                C[alpaka::Vec<uint32_t, 2u>{i, j}].imag()
                                == Catch::Approx(Cref[i * ldC + j].imag()).epsilon(1e-4f).margin(1e-4f));
                        }
                        else
                        {
                            CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == before[i * ldC + j]);
                        }
                    }
            }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk diagonal is real on update, preserved on no-op",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<float>;
        using Real = RealOf<Scalar>;
        constexpr uint32_t n = 3u;
        constexpr uint32_t k = 2u;

        // Update with a nonzero product contribution: the diagonal imaginary part must be discarded.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixComplex(A, n, k);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    C[alpaka::Vec<uint32_t, 2u>{i, j}] = Scalar{2.0f + 0.5f * i, 9.0f - 2.0f * i};
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{1.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
            {
                CHECK(C[alpaka::Vec<uint32_t, 2u>{i, i}].imag() == Catch::Approx(0.0f).margin(1e-5f));
                CHECK(C[alpaka::Vec<uint32_t, 2u>{i, i}].real() > 0.0f);
            }
        }
        // No-op (alpha=0, beta=1): C is left unchanged, including the diagonal imaginary part. Do not assume the
        // vendor canonicalizes the diagonal in this case.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixComplex(A, n, k);
            fillMatrixSentinel(C, n, n, Scalar{4.0f, -7.0f});
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::herk(queue, Real{0.0f}, A, Real{1.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Scalar{4.0f, -7.0f});
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk degenerate k=0 scale stays queue-ordered without intermediate waits",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<float>;
        using Real = RealOf<Scalar>;
        constexpr uint32_t n = 3u;
        // Producer/consumer chain with no intermediate wait: a queued fill overwrites C, then the degenerate k==0
        // herk must scale that produced value, then a queued memcpy reads C back to host. If the scale kernel were not
        // enqueued/in-order, the consumer would observe either the unscaled producer value (missing enqueue or wrong
        // stream) or a stale value. The expected result is deterministic: beta * (producer value).
        auto A0 = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, 0u});
        auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        auto hostC = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        fillMatrixComplex(A0, n, 0u);
        auto const producerValue = Scalar{Real{3.0f}, Real{-1.5f}};
        auto zeroC = [&](auto& view)
        {
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    view[alpaka::Vec<uint32_t, 2u>{i, j}] = producerValue;
        };
        // Two betas: 2 (positive scaling), and -3 (negative scaling). beta==1 would be a true no-op that enqueues
        // nothing, so it is not usable to prove ordering.
        for(auto const betaR : {Real{2.0f}, Real{-3.0f}})
        {
            zeroC(C);
            auto upperC = alpaka::blas::upper(C);
            // Producer (queued fill), degenerate k==0 herk, consumer (queued memcpy): no wait in between.
            alpaka::onHost::fill(queue, C, producerValue);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A0, betaR, upperC, options);
            alpaka::onHost::memcpy(queue, hostC, C);
            alpaka::onHost::wait(queue);
            auto expectedProd = producerValue;
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                {
                    Scalar expected{betaR * expectedProd.real(), Real{0}};
                    if(i != j)
                        expected = betaR * expectedProd;
                    CHECK(hostC[alpaka::Vec<uint32_t, 2u>{i, j}] == expected);
                }
        }
        // Ordering half: the producer must be observed before the scale. Use two halves with different betas chained
        // into one queue without waits; the first herk writes a nonzero pattern, the second scales it.
        {
            zeroC(C);
            auto upperC = alpaka::blas::upper(C);
            // Producer: k=2 non-degenerate herk overwrites C fully (beta=0).
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, 2u});
            fillMatrixComplex(A, n, 2u);
            // Consumer copies back through the queue.
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{0.0f}, upperC, options);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A0, Real{2.0f}, upperC, options);
            alpaka::onHost::memcpy(queue, hostC, C);
            alpaka::onHost::wait(queue);
            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(A.data(), n, 2u, ldA);
            // Reference: the producer herk (alpha=1,beta=0) writes into the pre-state, then beta=2 scales it.
            auto prodRef = std::vector<Scalar>(n * ldC, Scalar{producerValue});
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    prodRef[i * ldC + j] = producerValue;
            blas::herkRef(
                Real{1.0f},
                Aref.data(),
                ldA,
                n,
                2u,
                alpaka::blas::Transpose::none,
                Real{0.0f},
                prodRef.data(),
                ldC,
                n,
                alpaka::blas::Triangle::upper);
            // apply beta=2 scaling on the upper triangle
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                {
                    if(i == j)
                        prodRef[i * ldC + j] = Scalar{2.0f * prodRef[i * ldC + j].real(), Real{0}};
                    else
                        prodRef[i * ldC + j] = 2.0f * prodRef[i * ldC + j];
                }
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                {
                    CHECK(
                        hostC[alpaka::Vec<uint32_t, 2u>{i, j}].real()
                        == Catch::Approx(prodRef[i * ldC + j].real()).epsilon(1e-4f).margin(1e-4f));
                    CHECK(
                        hostC[alpaka::Vec<uint32_t, 2u>{i, j}].imag()
                        == Catch::Approx(prodRef[i * ldC + j].imag()).epsilon(1e-4f).margin(1e-4f));
                }
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk degenerate scale with padded ld treats triangle and padding as disjoint",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<float>;
        using Real = RealOf<Scalar>;
        auto const nanV = Scalar{static_cast<Real>(std::nan("")), static_cast<Real>(std::nan(""))};
        constexpr uint32_t n = 4u;
        constexpr uint32_t pad = 7u; // ld = pad > n: padding columns exist to the right of every row.
        // k == 0 degenerate branch: the selected (upper) triangle becomes beta * C.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, 0u});
            auto storageC = alpaka::onHost::allocUnified<Scalar>(device, n * pad);
            auto C = alpaka::makeMdSpan(
                storageC.data(),
                alpaka::Vec<uint32_t, 2u>{n, n},
                alpaka::Vec<std::size_t, 2u>{pad * sizeof(Scalar), sizeof(Scalar)});
            // Selected triangle: NaN so a read would propagate NaN. Opposite triangle and padding: distinct
            // sentinels that must stay byte-identical.
            auto const lowerSentinel = Scalar{Real(0x1'1111'1111), Real(0x1'1111'1111)};
            auto const padSentinel = Scalar{Real(0x5'a5a5'a5af), Real(0x5'a5a5'a5af)};
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < pad; ++j)
                {
                    if(j >= n)
                        C[alpaka::Vec<uint32_t, 2u>{i, j}] = padSentinel;
                    else if(j >= i)
                        C[alpaka::Vec<uint32_t, 2u>{i, j}] = nanV; // selected upper triangle: must be zeroed
                    else
                        C[alpaka::Vec<uint32_t, 2u>{i, j}] = lowerSentinel; // opposite triangle: untouched
                }
            auto upperC = alpaka::blas::upper(C);
            // beta == 0: the selected triangle is exactly zeroed and the old NaN values must not be read.
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{0.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < pad; ++j)
                {
                    if(j >= n)
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == padSentinel);
                    else if(j >= i)
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Scalar{0.0f, 0.0f});
                    else
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == lowerSentinel);
                }
        }
        // beta != 0 and beta != 1 on a padded C: the selected triangle scales in place (real diagonal), the
        // opposite triangle and the padding stay byte-identical.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, 0u});
            auto storageC = alpaka::onHost::allocUnified<Scalar>(device, n * pad);
            auto C = alpaka::makeMdSpan(
                storageC.data(),
                alpaka::Vec<uint32_t, 2u>{n, n},
                alpaka::Vec<std::size_t, 2u>{pad * sizeof(Scalar), sizeof(Scalar)});
            auto const lowerSentinel = Scalar{Real(0x2'2222'2222), Real(0x2'2222'2222)};
            auto const padSentinel = Scalar{Real(0x6'a6a6'a6af), Real(0x6'a6a6'a6af)};
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < pad; ++j)
                {
                    if(j >= n)
                        C[alpaka::Vec<uint32_t, 2u>{i, j}] = padSentinel;
                    else if(j >= i)
                        C[alpaka::Vec<uint32_t, 2u>{i, j}] = Scalar{Real{2.0f * i + j + 1}, Real{9.0f - 3.0f * i + j}};
                    else
                        C[alpaka::Vec<uint32_t, 2u>{i, j}] = lowerSentinel;
                }
            auto const before = copyRaw(storageC.data(), n, n, pad);
            auto const ldC = static_cast<std::size_t>(pad);
            auto lowerC = alpaka::blas::lower(C);
            auto const betaR = Real{2.5f};
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, betaR, lowerC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < pad; ++j)
                {
                    if(j >= n)
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == padSentinel);
                    else if(j <= i)
                    {
                        if(i == j)
                            CHECK(
                                C[alpaka::Vec<uint32_t, 2u>{i, j}]
                                == Scalar{betaR * before[i * ldC + j].real(), Real{0}});
                        else
                            CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == betaR * before[i * ldC + j]);
                    }
                    else
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == before[i * ldC + j]);
                }
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk validation rejects bad shapes, annotations, and complex coefficients",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        using Scalar = alpaka::math::Complex<float>;
        constexpr uint32_t n = 3u;
        constexpr uint32_t k = 3u;
        auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
        auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        fillMatrixComplex(A, n, k);
        fillMatrixComplex(C, n, n);

        auto upperC = alpaka::blas::upper(C);
        auto upperA = alpaka::blas::upper(A);
        auto lowerA = alpaka::blas::lower(A);
        auto unitUpperC = alpaka::blas::unitDiag(alpaka::blas::upper(C));
        auto transposedC = alpaka::blas::transposed(alpaka::blas::upper(C));
        // Missing triangle on C.
        CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, A, 1.0f, C), std::invalid_argument);
        // Triangle annotation on A.
        CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, upperA, 1.0f, upperC), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, lowerA, 1.0f, upperC), std::invalid_argument);
        // Unit-diagonal on C.
        CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, A, 1.0f, unitUpperC), std::invalid_argument);
        // Transposed C.
        CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, A, 1.0f, transposedC), std::invalid_argument);
        // Plain transposed(A) is not a valid HERK operation.
        CHECK_THROWS_AS(
            alpaka::blas::onHost::herk(queue, 1.0f, alpaka::blas::transposed(A), 1.0f, upperC),
            std::invalid_argument);
        // Wrong-size C.
        auto Cwrong = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n + 1u});
        fillMatrixComplex(Cwrong, n, n + 1);
        auto upperCwrong = alpaka::blas::upper(Cwrong);
        CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, A, 1.0f, upperCwrong), std::invalid_argument);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk rejects complex coefficients, mismatched types and const C (static)",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        using Scalar = alpaka::math::Complex<float>;
        using Real = RealOf<Scalar>;

        constexpr auto extA = alpaka::Vec<uint32_t, 2u>{3u, 2u};
        constexpr auto extC = alpaka::Vec<uint32_t, 2u>{3u, 3u};
        auto storageA = alpaka::onHost::allocUnified<Scalar>(device, 6u);
        auto storageC = alpaka::onHost::allocUnified<Scalar>(device, 9u);
        auto A = alpaka::makeMdSpan(
            storageA.data(),
            extA,
            alpaka::Vec<std::size_t, 2u>{2u * sizeof(Scalar), sizeof(Scalar)});
        auto C = alpaka::makeMdSpan(
            storageC.data(),
            extC,
            alpaka::Vec<std::size_t, 2u>{3u * sizeof(Scalar), sizeof(Scalar)});
        fillMatrixComplex(A, 3u, 2u);
        fillMatrixComplex(C, 3u, 3u);
        auto const upperC = alpaka::blas::upper(C);
        auto const Aconst = alpaka::makeMdSpan(
            static_cast<Scalar const*>(storageA.data()),
            extA,
            alpaka::Vec<std::size_t, 2u>{2u * sizeof(Scalar), sizeof(Scalar)});
        auto const Cconst = alpaka::makeMdSpan(
            static_cast<Scalar const*>(storageC.data()),
            extC,
            alpaka::Vec<std::size_t, 2u>{3u * sizeof(Scalar), sizeof(Scalar)});
        auto const upperCconst = alpaka::blas::upper(Cconst);
        // A float-element (real) matrix cannot be used for herk; the empty view type is enough to prove rejection.
        auto Afloat = alpaka::makeMdSpan(
            static_cast<float*>(nullptr),
            extA,
            alpaka::Vec<std::size_t, 2u>{2u * sizeof(float), sizeof(float)});
        // A complex-double A cannot be combined with a complex-float C (mismatched complex element types must be
        // rejected, not silently dispatched on one of the two types). Empty views are enough: the requires clause
        // and validateHerk() fire before any data access.
        auto Adouble = alpaka::makeMdSpan(
            static_cast<alpaka::math::Complex<double>*>(nullptr),
            extA,
            alpaka::Vec<std::size_t, 2u>{
                2u * sizeof(alpaka::math::Complex<double>),
                sizeof(alpaka::math::Complex<double>)});
        using TViewA = decltype(A);
        using TViewC = decltype(upperC);
        using TViewAconst = decltype(Aconst);
        using TViewCconst = decltype(upperCconst);
        using TViewAfloat = decltype(Afloat);
        using TViewAdouble = decltype(Adouble);

        // Positive control: writable upper(C), complex A, real coefficients.
        static_assert(herkCallable<int, Real, TViewA, Real, TViewC>);
        // Read-only A works: a const-element A is a valid input view (herk dispatches with the unqualified scalar
        // derived locally from the cv-preserving Value_t).
        static_assert(herkCallable<int, Real, TViewAconst, Real, TViewC>);
        // Rejections: const-element C, real-valued A, complex alpha, complex beta, mismatched A/C element types, and
        // a Complex<double> A combined with a Complex<float> C.
        static_assert(!herkCallable<int, Real, TViewA, Real, TViewCconst>);
        static_assert(!herkCallable<int, Real, TViewAfloat, Real, TViewC>);
        static_assert(!herkCallable<int, Scalar, TViewA, Real, TViewC>);
        static_assert(!herkCallable<int, Real, TViewA, Scalar, TViewC>);
        static_assert(!herkCallable<int, double, TViewAdouble, double, TViewC>);
        SUCCEED();
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk read-only A (const element) accepted for all herk scalar types",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};
        auto runCase = [&]<typename Scalar>()
        {
            using Real = RealOf<Scalar>;
            constexpr uint32_t n = 3u;
            constexpr uint32_t k = 2u;
            auto Abuf = alpaka::onHost::allocUnified<Scalar>(device, n * k);
            auto Cbuf = alpaka::onHost::allocUnified<Scalar>(device, n * n);
            auto A = alpaka::makeMdSpan(
                Abuf.data(),
                alpaka::Vec<uint32_t, 2u>{n, k},
                alpaka::Vec<std::size_t, 2u>{k * sizeof(Scalar), sizeof(Scalar)});
            auto C = alpaka::makeMdSpan(
                Cbuf.data(),
                alpaka::Vec<uint32_t, 2u>{n, n},
                alpaka::Vec<std::size_t, 2u>{n * sizeof(Scalar), sizeof(Scalar)});
            fillMatrixComplex(A, n, k);
            fillMatrixComplex(C, n, n);
            // Copy the pre-state so the reference can apply the beta scaling.
            auto before = copyRaw(Cbuf.data(), n, n, ldOf(C));
            // Build a read-only view over A: MdSpan<const Scalar>.
            auto Aconst = alpaka::makeMdSpan(
                static_cast<Scalar const*>(Abuf.data()),
                alpaka::Vec<uint32_t, 2u>{n, k},
                alpaka::Vec<std::size_t, 2u>{k * sizeof(Scalar), sizeof(Scalar)});
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, Aconst, Real{0.5f}, upperC, options);
            alpaka::onHost::wait(queue);
            // Compare against the reference for C = 1.0*A*adjoint(A) + 0.5*C over the copied pre-state.
            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(Abuf.data(), n, k, ldA);
            auto Cref = copyRaw(before.data(), n, n, ldC);
            blas::herkRef(
                Real{1.0f},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Real{0.5f},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::upper);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                {
                    auto const eps = std::same_as<Real, double> ? 1e-12 : 1e-4;
                    CHECK(
                        C[alpaka::Vec<uint32_t, 2u>{i, j}].real()
                        == Catch::Approx(Cref[i * ldC + j].real()).epsilon(eps));
                    CHECK(
                        C[alpaka::Vec<uint32_t, 2u>{i, j}].imag()
                        == Catch::Approx(Cref[i * ldC + j].imag()).epsilon(eps));
                }
        };
        runCase.template operator()<alpaka::math::Complex<float>>();
        runCase.template operator()<alpaka::math::Complex<double>>();
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS herk oversized dimensions rejected by checkedCast",
    "[integr][blas][rankk][herk]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        using Scalar = alpaka::math::Complex<float>;
        constexpr uint32_t n = 3u;
        constexpr uint32_t k = 3u;
        // An enormous leading dimension: the pitch overflows the 32-bit vendor int parameters used by the
        // host/cuda/hip herk dispatches (and by the degenerate scale-kernel branch).
        constexpr std::size_t hugeLd = static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) + 2u;
        auto Astorage = alpaka::onHost::allocUnified<Scalar>(device, 16u);
        auto Cstorage = alpaka::onHost::allocUnified<Scalar>(device, 16u);
        auto A = alpaka::makeMdSpan(
            Astorage.data(),
            alpaka::Vec<uint32_t, 2u>{n, k},
            alpaka::Vec<std::size_t, 2u>{hugeLd * sizeof(Scalar), sizeof(Scalar)});
        auto C = alpaka::makeMdSpan(
            Cstorage.data(),
            alpaka::Vec<uint32_t, 2u>{n, n},
            alpaka::Vec<std::size_t, 2u>{hugeLd * sizeof(Scalar), sizeof(Scalar)});
        auto Cst = alpaka::onHost::allocUnified<Scalar>(device, 16u);
        auto Cbig = alpaka::makeMdSpan(
            Cst.data(),
            alpaka::Vec<uint32_t, 2u>{n, n},
            alpaka::Vec<std::size_t, 2u>{hugeLd * sizeof(Scalar), sizeof(Scalar)});
        auto Cnormal = alpaka::makeMdSpan(
            Cst.data(),
            alpaka::Vec<uint32_t, 2u>{n, n},
            alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(n) * sizeof(Scalar), sizeof(Scalar)});
        auto Anormal = alpaka::makeMdSpan(
            Astorage.data(),
            alpaka::Vec<uint32_t, 2u>{n, k},
            alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(n) * sizeof(Scalar), sizeof(Scalar)});
        // Deliberately fill none of the oversized views: the enormous pitch means any element write would overflow
        // the tiny backing storage. The rejection must come from metadata alone, before any data access.
        auto upperC = alpaka::blas::upper(C);
        auto upperCbig = alpaka::blas::upper(Cbig);
        auto upperCnormal = alpaka::blas::upper(Cnormal);
        // Backend-generic guards of the degenerate triangle-scale path (enqueueScaleTriangle): they run for every
        // backend including oneAPI, so they are tested outside the !OneApi exclusion below.
        //
        // Scale-path metadata envelope: the degenerate branch is a generic kernel. With 64-bit index math there is
        // no n*n index-domain limit (the kernel traverses one row per work item), so n in (65535, UINT32_MAX] is
        // accepted and only n > UINT32_MAX is rejected (the uint32 index domain of the frame); the C leading
        // dimension is narrowed through the same vendor-int gate as the backend's own herk (int64 on oneMKL,
        // 32-bit vendor int elsewhere), so an enormously-pitched C is accepted on the no-op (beta == 1) path
        // without touching any metadata -- the no-op never reads or writes, on any backend.
        alpaka::blas::onHost::herk(queue, 0.0f, Anormal, 1.0f, upperCbig);
        alpaka::onHost::wait(queue);
        // Scale-path n>uint32-max guard: a degenerate (k==0) herk whose C claims more than uint32 rows must be
        // rejected before any kernel is enqueued. Metadata only -- a tiny backing storage is used. C's pitch must
        // itself encode the huge row count so the shape validation passes and only the scale path's n>uint32-max
        // guard rejects the call.
        {
            auto AstorageN = alpaka::onHost::allocUnified<Scalar>(device, 16u);
            auto CstorageN = alpaka::onHost::allocUnified<Scalar>(device, 16u);
            constexpr auto hugeN = static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) + 2u;
            // k == 0 forces the degenerate scale path; alpha and beta are irrelevant to the n>uint32-max guard. A is
            // never read, so its row pitch is arbitrary.
            auto AhugeN = alpaka::makeMdSpan(
                AstorageN.data(),
                alpaka::Vec<std::size_t, 2u>{hugeN, std::size_t{0u}},
                alpaka::Vec<std::size_t, 2u>{std::size_t{0u}, sizeof(Scalar)});
            auto ChugeN = alpaka::makeMdSpan(
                CstorageN.data(),
                alpaka::Vec<std::size_t, 2u>{hugeN, hugeN},
                alpaka::Vec<std::size_t, 2u>{hugeN * sizeof(Scalar), sizeof(Scalar)});
            // k==0 degenerates to the scale kernel; with beta != 1 the n>uint32-max guard must throw.
            auto upperChugeN = alpaka::blas::upper(ChugeN);
            CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, AhugeN, 2.0f, upperChugeN), std::invalid_argument);
        }
        // An intermediate n in (65535, UINT32_MAX] is accepted by the scale path (no n*n index-domain limit exists
        // anymore); it is only exercised here on the beta==1 no-op path because a real kernel run over that range
        // needs an n*n element allocation (infeasible as metadata-only). The no-op proves the 65536 fence is gone:
        // no metadata guard triggers for such an n.
        {
            constexpr auto midN = static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 2u;
            static_assert(midN > 65535u && midN <= std::numeric_limits<uint32_t>::max());
            auto AstorageM = alpaka::onHost::allocUnified<Scalar>(device, 16u);
            auto CstorageM = alpaka::onHost::allocUnified<Scalar>(device, 16u);
            auto AmidN = alpaka::makeMdSpan(
                AstorageM.data(),
                alpaka::Vec<std::size_t, 2u>{midN, std::size_t{0u}},
                alpaka::Vec<std::size_t, 2u>{std::size_t{0u}, sizeof(Scalar)});
            auto CmidN = alpaka::makeMdSpan(
                CstorageM.data(),
                alpaka::Vec<std::size_t, 2u>{midN, midN},
                alpaka::Vec<std::size_t, 2u>{midN * sizeof(Scalar), sizeof(Scalar)});
            auto upperCmidN = alpaka::blas::upper(CmidN);
            // beta == 1: the degenerate branch is a true no-op, accepted for the (65535, UINT32_MAX] range without
            // touching the metadata of C.
            alpaka::blas::onHost::herk(queue, 1.0f, AmidN, 1.0f, upperCmidN);
            alpaka::onHost::wait(queue);
            SUCCEED();
        }
        if constexpr(!std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::OneApi>)
        {
            CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, A, 1.0f, upperC), std::invalid_argument);
            // C-ld only oversized: the A descriptor is well-formed, C's cdLd must still be rejected.
            CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, A, 1.0f, upperCbig), std::invalid_argument);
            // Degenerate branch (k==0/alpha==0) bypasses the vendor dispatch, so an oversized C ld must be rejected
            // by the scale path itself with the same vendor-int fence as the k>0 dispatch of this backend. An
            // oversized A ld is harmless there (A is never read) and must be accepted while C is well-formed --
            // verified by the succeeding calls below.
            // Degenerate huge-C-ld parity: the k==0 branch must throw on the 32-bit-vendor-int backends exactly like
            // the k>0 path above (upperCbig), proving the degenerate branch never enqueues a kernel whose row
            // offsets exceed the tiny backing storage on those backends. A is k==0, so it is never read (its pitch
            // stays arbitrary); only the C ld is oversized. oneMKL takes int64 dimensions, so its degenerate branch
            // may accept the same metadata (matching its k>0 dispatch, which also accepts it).
            {
                auto Ak0 = alpaka::makeMdSpan(
                    Astorage.data(),
                    alpaka::Vec<uint32_t, 2u>{n, 0u},
                    alpaka::Vec<std::size_t, 2u>{hugeLd * sizeof(Scalar), sizeof(Scalar)});
                CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, Ak0, 2.0f, upperCbig), std::invalid_argument);
            }
            // Oversized A ld but well-formed C: accepted in the degenerate branch (A untouched), and a non-degenerate
            // run proceeds far enough to reject via the A dispatch checkedCast.
            CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, A, 1.0f, upperCnormal), std::invalid_argument);
            // Metadata-only oversized n and k: the vendor int parameters overflow, so the checkedCast<int>
            // in the real dispatch rejects before any data access. Claimed extents must pass the int64 descriptor
            // and the shape validation, then fail checkedCast<int>.
            {
                // n > INT_MAX (2^31-1), k == 3. The claimed extents and row pitch must be internally consistent so
                // the shape validation passes; only the dispatch's checkedCast<int>(n) rejects the call.
                constexpr auto hugeN = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1u;
                auto AstorageN = alpaka::onHost::allocUnified<Scalar>(device, 16u);
                auto CstorageN = alpaka::onHost::allocUnified<Scalar>(device, 16u);
                auto AhugeN = alpaka::makeMdSpan(
                    AstorageN.data(),
                    alpaka::Vec<std::size_t, 2u>{hugeN, static_cast<std::size_t>(k)},
                    alpaka::Vec<std::size_t, 2u>{hugeN * sizeof(Scalar), sizeof(Scalar)});
                auto ChugeN = alpaka::makeMdSpan(
                    CstorageN.data(),
                    alpaka::Vec<std::size_t, 2u>{hugeN, hugeN},
                    alpaka::Vec<std::size_t, 2u>{hugeN * sizeof(Scalar), sizeof(Scalar)});
                auto upperChugeN = alpaka::blas::upper(ChugeN);
                CHECK_THROWS_AS(
                    alpaka::blas::onHost::herk(queue, 1.0f, AhugeN, 1.0f, upperChugeN),
                    std::invalid_argument);
                // k > INT_MAX, n == 3. op(A) is n x k so C stays 3x3. A's row pitch encodes the hugek columns.
                constexpr auto hugeK = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1u;
                auto AstorageK = alpaka::onHost::allocUnified<Scalar>(device, 16u);
                auto AhugeK = alpaka::makeMdSpan(
                    AstorageK.data(),
                    alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(n), hugeK},
                    alpaka::Vec<std::size_t, 2u>{hugeK * sizeof(Scalar), sizeof(Scalar)});
                auto Ch = alpaka::onHost::allocUnified<Scalar>(device, 16u);
                auto CnormalK = alpaka::makeMdSpan(
                    Ch.data(),
                    alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(n), static_cast<std::size_t>(n)},
                    alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(n) * sizeof(Scalar), sizeof(Scalar)});
                auto upperCnormalK = alpaka::blas::upper(CnormalK);
                CHECK_THROWS_AS(
                    alpaka::blas::onHost::herk(queue, 1.0f, AhugeK, 1.0f, upperCnormalK),
                    std::invalid_argument);
            }
            // Queue sanity after the metadata rejections: nothing may have been enqueued, the queue must still
            // accept and run a perfectly valid herk.
            auto sanityA = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto sanityC = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixComplex(sanityA, n, k);
            fillMatrixComplex(sanityC, n, n);
            auto upperSanityC = alpaka::blas::upper(sanityC);
            alpaka::blas::onHost::herk(queue, 1.0f, sanityA, 0.5f, upperSanityC);
            alpaka::onHost::wait(queue);
            // The valid update ran: the diagonal is real for the first element.
            CHECK(sanityC[alpaka::Vec<uint32_t, 2u>{0u, 0u}].imag() == Catch::Approx(0.0f).margin(1e-5f));
        }
    }
}
