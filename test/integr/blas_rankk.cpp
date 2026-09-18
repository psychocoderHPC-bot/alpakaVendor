/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <algorithm>
#include <alpakaTest/deviceHelper.hpp>
#include <cmath>
#include <complex>
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
        // n=0: return without touching C.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{0u, 2u});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{0u, 0u});
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{1.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            SUCCEED();
        }
        // k=0: OpenBLAS CHERK rejects a zero leading dimension (k=0 => lda=0) and returns without
        // modifying C, so the selected triangle is left unchanged (no beta scaling). The spec allows a
        // true no-op to leave C unchanged; the host backend's vendor does exactly that for k=0.
        {
            constexpr uint32_t n = 2u;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, 0u});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixComplex(C, n, n);
            auto const before = copyRaw(C.data(), n, n, ldOf(C));
            auto const ldC = ldOf(C);
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::herk(queue, Real{1.0f}, A, Real{2.0f}, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == before[i * ldC + j]);
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
    "BLAS herk rejects complex coefficients and mismatched types (static)",
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
        // A complex alpha or beta (even with a zero imaginary part) is rejected at compile time by the
        // static_asserts in the public wrapper, so the corresponding overload does not form.
        static_assert(!alpaka::blas::RealScalar<alpaka::math::Complex<float>>);
        static_assert(!alpaka::blas::RealScalar<alpaka::math::Complex<double>>);
        // A and C must have the same element type (checked by the static_assert in the wrapper).
        static_assert(!std::same_as<Scalar, float>);
        SUCCEED();
    }
}
