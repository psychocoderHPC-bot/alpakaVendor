/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <algorithm>
#include <alpakaTest/deviceHelper.hpp>
#include <cmath>
#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

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

template<typename T_Device>
consteval bool supportsDoubleSyrk(T_Device const&)
{
    using Api = std::remove_cvref_t<decltype(alpaka::getApi(std::declval<T_Device>()))>;
    using DeviceKind = std::remove_cvref_t<decltype(alpaka::getDeviceKind(std::declval<T_Device>()))>;
    return !std::same_as<Api, alpaka::api::OneApi> || std::same_as<DeviceKind, alpaka::deviceKind::Cpu>;
}

void fillMatrix(auto& view, std::size_t rows, std::size_t cols)
{
    using T = alpaka::GetValueType_t<std::remove_cvref_t<decltype(view)>>;
    for(std::size_t i = 0; i < rows; ++i)
        for(std::size_t j = 0; j < cols; ++j)
            view[alpaka::Vec<uint32_t, 2u>{i, j}] = T(static_cast<typename alpaka::blas::Real_t<T>>(i * 10 + j + 1));
}

void fillMatrixSentinel(auto& view, std::size_t rows, std::size_t cols, auto sentinel)
{
    using T = alpaka::GetValueType_t<std::remove_cvref_t<decltype(view)>>;
    for(std::size_t i = 0; i < rows; ++i)
        for(std::size_t j = 0; j < cols; ++j)
            view[alpaka::Vec<uint32_t, 2u>{i, j}] = T(sentinel);
}

// Whether the public syrk entry forms for the given argument types. Expressing the call through a variable-template
// requires-expression makes the negative cases (unsatisfied constraints) produce `false` instead of a hard error.
template<typename TQueue, typename TAlpha, typename TViewA, typename TBeta, typename TViewC>
inline constexpr bool syrkCallable = requires(TQueue& queue, TAlpha alpha, TViewA& A, TBeta beta, TViewC& C) {
    alpaka::blas::onHost::syrk(queue, alpha, A, beta, C);
};

// Copy a 2D view's raw row-major buffer (pitches) into a std::vector.
auto copyRaw(auto const& view, std::size_t rows, std::size_t cols, std::size_t ld)
    -> std::vector<alpaka::GetValueType_t<std::remove_cvref_t<decltype(view)>>>
{
    using T = alpaka::GetValueType_t<std::remove_cvref_t<decltype(view)>>;
    auto const* ptr = view.data();
    std::vector<T> out(rows * ld);
    for(std::size_t i = 0; i < rows; ++i)
        std::copy_n(ptr + i * ld, cols, out.data() + i * ld);
    return out;
}

TEMPLATE_LIST_TEST_CASE("BLAS syrk real symmetric rank-k update", "[integr][blas][rankk][syrk]", TestBackends)
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

        constexpr uint32_t n = 5u;
        constexpr uint32_t k = 3u;

        // As-stored A, upper(C): single precision.
        {
            using Scalar = float;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            auto Ccopy = alpaka::onHost::allocHostLike(C);
            fillMatrix(A, n, k);
            fillMatrix(C, n, n);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];

            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::syrk(queue, Scalar{1.5}, A, Scalar{-0.5}, upperC, options);
            alpaka::onHost::wait(queue);

            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(A, n, k, ldA);
            auto Cref = copyRaw(Ccopy, n, n, ldC);
            blas::syrkRef(
                Scalar{1.5},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Scalar{-0.5},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::upper);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                {
                    if(j >= i)
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(1e-4f));
                    else
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}]);
                }
        }

        // As-stored A, upper(C): double precision.
        if constexpr(supportsDoubleSyrk(device))
        {
            using Scalar = double;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            auto Ccopy = alpaka::onHost::allocHostLike(C);
            fillMatrix(A, n, k);
            fillMatrix(C, n, n);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];

            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::syrk(queue, Scalar{1.5}, A, Scalar{-0.5}, upperC, options);
            alpaka::onHost::wait(queue);

            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(A, n, k, ldA);
            auto Cref = copyRaw(Ccopy, n, n, ldC);
            blas::syrkRef(
                Scalar{1.5},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Scalar{-0.5},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::upper);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                {
                    if(j >= i)
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(1e-12));
                    else
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}]);
                }
        }

        // As-stored A, upper(C): square n == k.
        {
            constexpr uint32_t m = 4u;
            using Scalar = float;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{m, m});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{m, m});
            auto Ccopy = alpaka::onHost::allocHostLike(C);
            fillMatrix(A, m, m);
            fillMatrix(C, m, m);
            for(uint32_t i = 0; i < m; ++i)
                for(uint32_t j = 0; j < m; ++j)
                    Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];

            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::syrk(queue, Scalar{1.25}, A, Scalar{-1.0}, upperC, options);
            alpaka::onHost::wait(queue);

            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(A, m, m, ldA);
            auto Cref = copyRaw(Ccopy, m, m, ldC);
            blas::syrkRef(
                Scalar{1.25},
                Aref.data(),
                ldA,
                m,
                m,
                alpaka::blas::Transpose::none,
                Scalar{-1.0},
                Cref.data(),
                ldC,
                m,
                alpaka::blas::Triangle::upper);
            for(uint32_t i = 0; i < m; ++i)
                for(uint32_t j = 0; j < m; ++j)
                {
                    if(j >= i)
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(1e-4f));
                    else
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}]);
                }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("BLAS syrk transposed input and lower triangle", "[integr][blas][rankk][syrk]", TestBackends)
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

        constexpr uint32_t n = 4u;
        constexpr uint32_t k = 6u;

        // A stored as k x n; transposed(A) is n x k.
        auto runCase = [&queue, &options, n, k](auto& A, auto& C, auto& Ccopy)
        {
            using T = alpaka::GetValueType_t<std::remove_cvref_t<decltype(A)>>;
            auto transposedA = alpaka::blas::transposed(A);
            auto lowerC = alpaka::blas::lower(C);
            alpaka::blas::onHost::syrk(queue, T{0.75}, transposedA, T{2.0}, lowerC, options);
            alpaka::onHost::wait(queue);

            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(A, k, n, ldA);
            auto Cref = copyRaw(Ccopy, n, n, ldC);
            blas::syrkRef(
                T{0.75},
                Aref.data(),
                ldA,
                k,
                n,
                alpaka::blas::Transpose::transposed,
                T{2.0},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::lower);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                {
                    if(j <= i)
                    {
                        double const epsilon = std::same_as<T, float> ? 1e-4 : 1e-12;
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(epsilon));
                    }
                    else
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}]);
                }
        };

        // Double precision.
        if constexpr(supportsDoubleSyrk(device))
        {
            auto A = alpaka::onHost::allocUnified<double>(device, alpaka::Vec<uint32_t, 2u>{k, n});
            auto C = alpaka::onHost::allocUnified<double>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            auto Ccopy = alpaka::onHost::allocHostLike(C);
            fillMatrix(A, k, n);
            fillMatrix(C, n, n);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];
            runCase(A, C, Ccopy);
        }

        // Single precision.
        {
            auto A = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{k, n});
            auto C = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            auto Ccopy = alpaka::onHost::allocHostLike(C);
            fillMatrix(A, k, n);
            fillMatrix(C, n, n);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];
            runCase(A, C, Ccopy);
        }

        // transposed(A), lower(C): square n == k.
        {
            constexpr uint32_t m = 4u;
            auto A = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{m, m});
            auto C = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{m, m});
            auto Ccopy = alpaka::onHost::allocHostLike(C);
            fillMatrix(A, m, m);
            fillMatrix(C, m, m);
            for(uint32_t i = 0; i < m; ++i)
                for(uint32_t j = 0; j < m; ++j)
                    Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];

            auto lowerC = alpaka::blas::lower(C);
            alpaka::blas::onHost::syrk(queue, float{1.25}, alpaka::blas::transposed(A), float{-1.0}, lowerC, options);
            alpaka::onHost::wait(queue);

            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(A, m, m, ldA);
            auto Cref = copyRaw(Ccopy, m, m, ldC);
            blas::syrkRef(
                float{1.25},
                Aref.data(),
                ldA,
                m,
                m,
                alpaka::blas::Transpose::transposed,
                float{-1.0},
                Cref.data(),
                ldC,
                m,
                alpaka::blas::Triangle::lower);
            for(uint32_t i = 0; i < m; ++i)
                for(uint32_t j = 0; j < m; ++j)
                {
                    if(j <= i)
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(1e-4f));
                    else
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}]);
                }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("BLAS syrk n==1 and n==k boundary cases", "[integr][blas][rankk][syrk]", TestBackends)
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

        using Scalar = float;
        auto runCase = [&](uint32_t n, uint32_t k)
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            auto Ccopy = alpaka::onHost::allocHostLike(C);
            fillMatrix(A, n, k);
            fillMatrix(C, n, n);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::syrk(queue, Scalar{1.25}, A, Scalar{-0.5}, upperC, options);
            alpaka::onHost::wait(queue);
            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(A, n, k, ldA);
            auto Cref = copyRaw(Ccopy, n, n, ldC);
            blas::syrkRef(
                Scalar{1.25},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Scalar{-0.5},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::upper);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                {
                    if(j >= i)
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(1e-4f));
                    else
                        CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}]);
                }
        };
        // n == 1 (the smallest non-degenerate case): A is 1 x k, C is 1 x 1.
        runCase(1u, 2u);
        // n == k (square A, most common syrk shape).
        runCase(4u, 4u);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS syrk alpha=0 and beta=0 semantics (no contamination)",
    "[integr][blas][rankk][syrk]",
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

        using Scalar = float;
        constexpr uint32_t n = 3u;
        constexpr uint32_t k = 3u;

        // alpha=0: the selected triangle must stay exactly beta*C. A seeded with NaN must not be read.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixSentinel(A, n, k, std::nan(""));
            fillMatrixSentinel(C, n, n, Scalar{3.0});
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::syrk(queue, Scalar{0.0}, A, Scalar{2.0}, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Scalar{6.0}).epsilon(1e-5f));
        }

        // k=0: result is beta*C with A unread (A is NaN).
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, 0u});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixSentinel(C, n, n, Scalar{2.0});
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{3.0}, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Scalar{6.0}).epsilon(1e-5f));
        }

        // alpha=0 and beta=0: the selected triangle becomes zero, old C NaN must not be read.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixSentinel(A, n, k, std::nan(""));
            fillMatrixSentinel(C, n, n, std::nan(""));
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::syrk(queue, Scalar{0.0}, A, Scalar{0.0}, upperC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Scalar{0.0}).epsilon(1e-5f));
        }

        // Lower triangle: alpha=0 scales the selected (lower) triangle by beta; the upper triangle stays unchanged.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrixSentinel(A, n, k, std::nan(""));
            fillMatrixSentinel(C, n, n, Scalar{7.0});
            auto lowerC = alpaka::blas::lower(C);
            alpaka::blas::onHost::syrk(queue, Scalar{0.0}, A, Scalar{2.0}, lowerC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j <= i; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Scalar{14.0}).epsilon(1e-5f));
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i + 1; j < n; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Scalar{7.0}).epsilon(1e-5f));
        }

        // Lower triangle beta=0: lower triangle becomes A*A^T with old C NaN unread; upper unchanged NaN-free.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrix(A, n, k);
            fillMatrixSentinel(C, n, n, std::nan(""));
            auto lowerC = alpaka::blas::lower(C);
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{0.0}, lowerC, options);
            alpaka::onHost::wait(queue);
            auto const ldA = ldOf(A);
            auto const Aref = copyRaw(A, n, k, ldA);
            auto const ldC = ldOf(C);
            auto Cref = std::vector<Scalar>(n * ldC, Scalar{0.0});
            blas::syrkRef(
                Scalar{1.0},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Scalar{0.0},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::lower);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j <= i; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(1e-4f));
        }

        // beta=0: the selected triangle must equal the reference computed from a zero C; old C NaN is not read.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C2 = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrix(A, n, k);
            fillMatrixSentinel(C2, n, n, std::nan(""));
            auto upperC2 = alpaka::blas::upper(C2);
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{0.0}, upperC2, options);
            alpaka::onHost::wait(queue);
            auto const ldA = ldOf(A);
            auto const Aref = copyRaw(A, n, k, ldA);
            auto const ldC2 = ldOf(C2);
            auto Cref2 = std::vector<Scalar>(n * ldC2, Scalar{0.0});
            blas::syrkRef(
                Scalar{1.0},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Scalar{0.0},
                Cref2.data(),
                ldC2,
                n,
                alpaka::blas::Triangle::upper);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                    CHECK(C2[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref2[i * ldC2 + j]).epsilon(1e-4f));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("BLAS syrk validation rejects bad annotations", "[integr][blas][rankk][syrk]", TestBackends)
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
        using Scalar = float;
        constexpr uint32_t n = 3u;
        constexpr uint32_t k = 3u;
        auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
        auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        fillMatrix(A, n, k);
        fillMatrix(C, n, n);

        auto upperC = alpaka::blas::upper(C);
        auto upperA = alpaka::blas::upper(A);
        auto unitUpperC = alpaka::blas::unitDiag(alpaka::blas::upper(C));
        // Missing triangle on C.
        CHECK_THROWS_AS(alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{1.0}, C), std::invalid_argument);
        // Triangle annotation on A.
        CHECK_THROWS_AS(
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, upperA, Scalar{1.0}, upperC),
            std::invalid_argument);
        // Unit-diagonal on C.
        CHECK_THROWS_AS(
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{1.0}, unitUpperC),
            std::invalid_argument);
        // Wrong-size C (n x n required, give n x (n+1)).
        auto Cwrong = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n + 1u});
        fillMatrix(Cwrong, n, n + 1);
        auto upperCwrong = alpaka::blas::upper(Cwrong);
        CHECK_THROWS_AS(
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{1.0}, upperCwrong),
            std::invalid_argument);
        // Unit-diagonal on A.
        auto unitDiagA = alpaka::blas::unitDiag(A);
        CHECK_THROWS_AS(
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, unitDiagA, Scalar{1.0}, upperC),
            std::invalid_argument);
        // Transposed C.
        auto transposedUpperC = alpaka::blas::transposed(upperC);
        CHECK_THROWS_AS(
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{1.0}, transposedUpperC),
            std::invalid_argument);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS syrk conjTransposed equals transposed and read-only A",
    "[integr][blas][rankk][syrk]",
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

        constexpr uint32_t n = 3u;
        constexpr uint32_t k = 4u;
        // As-stored A is k x n; transposed/conjTransposed give an n x k operand.
        auto runCase = [&](auto const& A, auto const& C)
        {
            using Scalar = std::remove_cv_t<alpaka::GetValueType_t<std::remove_cvref_t<decltype(A)>>>;
            auto Ctrans = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            auto Cconj = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                {
                    Ctrans[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];
                    Cconj[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];
                }
            auto const transA = alpaka::blas::transposed(A);
            auto const conjA = alpaka::blas::conjTransposed(A);
            auto const upperCtrans = alpaka::blas::upper(Ctrans);
            auto const upperCconj = alpaka::blas::upper(Cconj);
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, transA, Scalar{0.5}, upperCtrans, options);
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, conjA, Scalar{0.5}, upperCconj, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    CHECK(
                        Ctrans[alpaka::Vec<uint32_t, 2u>{i, j}]
                        == Catch::Approx(Cconj[alpaka::Vec<uint32_t, 2u>{i, j}]));
        };

        // float.
        auto Af = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{k, n});
        auto Cf = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        fillMatrix(Af, k, n);
        fillMatrix(Cf, n, n);
        runCase(Af, Cf);

        // double.
        if constexpr(supportsDoubleSyrk(device))
        {
            auto Ad = alpaka::onHost::allocUnified<double>(device, alpaka::Vec<uint32_t, 2u>{k, n});
            auto Cd = alpaka::onHost::allocUnified<double>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrix(Ad, k, n);
            fillMatrix(Cd, n, n);
            runCase(Ad, Cd);
        }

        // Read-only A (MdSpan<float const>) dispatches through the cv-stripped Value_t path.
        auto Aro = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{k, n});
        auto Cro = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        fillMatrix(Aro, k, n);
        fillMatrix(Cro, n, n);
        alpaka::onHost::wait(queue);
        auto const Areadonly = alpaka::makeMdSpan(
            static_cast<float const*>(Aro.data()),
            alpaka::Vec<uint32_t, 2u>{k, n},
            alpaka::Vec<std::size_t, 2u>{n * sizeof(float), sizeof(float)});
        runCase(Areadonly, Cro);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS syrk const-double A numerical update and negative type contracts",
    "[integr][blas][rankk][syrk]",
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

        using Scalar = double;
        // Numerical const-double A: a read-only MdSpan<double const> A must produce the identical result to the
        // writable A reference computation (convert-once scalar semantics must hold for real double coefficients).
        if constexpr(supportsDoubleSyrk(device))
        {
            constexpr uint32_t n = 4u;
            constexpr uint32_t k = 3u;
            auto Abuf = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto Cbuf = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            auto Cref = alpaka::onHost::allocHostLike(Cbuf);
            fillMatrix(Abuf, n, k);
            fillMatrix(Cbuf, n, n);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                    Cref[alpaka::Vec<uint32_t, 2u>{i, j}] = Cbuf[alpaka::Vec<uint32_t, 2u>{i, j}];
            // Run once with a read-only (const-element) A and once with the writable A view; both must produce the
            // identical result. A disagreement would indicate a cv-dispatch regression in the scalar conversion.
            auto Crun = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            for(uint32_t run = 0; run < 2; ++run)
            {
                for(uint32_t i = 0; i < n; ++i)
                    for(uint32_t j = 0; j < n; ++j)
                        Crun[alpaka::Vec<uint32_t, 2u>{i, j}] = Cbuf[alpaka::Vec<uint32_t, 2u>{i, j}];
                auto upperC = alpaka::blas::upper(Crun);
                if(run == 0)
                {
                    auto const Aconst = alpaka::makeMdSpan(
                        static_cast<Scalar const*>(Abuf.data()),
                        alpaka::Vec<uint32_t, 2u>{n, k},
                        alpaka::onHost::getPitches(Abuf));
                    alpaka::blas::onHost::syrk(queue, Scalar{1.25}, Aconst, Scalar{-0.5}, upperC, options);
                }
                else
                    alpaka::blas::onHost::syrk(queue, Scalar{1.25}, Abuf, Scalar{-0.5}, upperC, options);
                alpaka::onHost::wait(queue);
                auto const ldA = ldOf(Abuf);
                auto const ldC = ldOf(Crun);
                auto const ArefData = copyRaw(Abuf, n, k, ldA);
                auto CrefData = copyRaw(Cref, n, n, ldC);
                blas::syrkRef(
                    Scalar{1.25},
                    ArefData.data(),
                    ldA,
                    n,
                    k,
                    alpaka::blas::Transpose::none,
                    Scalar{-0.5},
                    CrefData.data(),
                    ldC,
                    n,
                    alpaka::blas::Triangle::upper);
                for(uint32_t i = 0; i < n; ++i)
                    for(uint32_t j = 0; j < n; ++j)
                    {
                        if(j >= i)
                            CHECK(
                                Crun[alpaka::Vec<uint32_t, 2u>{i, j}]
                                == Catch::Approx(CrefData[i * ldC + j]).epsilon(1e-12));
                        else
                        {
                            UNSCOPED_INFO("run=" << run << " (0=const A, 1=writable A)");
                            CHECK(Crun[alpaka::Vec<uint32_t, 2u>{i, j}] == Cref[alpaka::Vec<uint32_t, 2u>{i, j}]);
                        }
                    }
            }
        }

        // Negative compile-time contracts, expressed through the requires-clause of the public syrk entry:
        // (i) A/C element-type mismatch is rejected (the overload does not form); (ii) a const-element C is rejected;
        // (iii) complex coefficients are rejected; (iv) real A with complex C is rejected. The empty-view types are
        // sufficient to prove the constraints, so no device operation runs here.
        {
            constexpr auto extA = alpaka::Vec<uint32_t, 2u>{3u, 2u};
            constexpr auto extC = alpaka::Vec<uint32_t, 2u>{3u, 3u};
            auto storageA = alpaka::onHost::allocUnified<float>(device, 6u);
            auto storageC = alpaka::onHost::allocUnified<float>(device, 9u);
            auto Afloat = alpaka::makeMdSpan(
                storageA.data(),
                extA,
                alpaka::Vec<std::size_t, 2u>{2u * sizeof(float), sizeof(float)});
            auto Cfloat = alpaka::makeMdSpan(
                storageC.data(),
                extC,
                alpaka::Vec<std::size_t, 2u>{3u * sizeof(float), sizeof(float)});
            auto const upperC = alpaka::blas::upper(Cfloat);
            auto const Cdouble = alpaka::makeMdSpan(
                static_cast<double*>(nullptr),
                extC,
                alpaka::Vec<std::size_t, 2u>{3u * sizeof(double), sizeof(double)});
            auto const upperCdouble = alpaka::blas::upper(Cdouble);
            auto const Cconst = alpaka::makeMdSpan(
                static_cast<float const*>(storageC.data()),
                extC,
                alpaka::Vec<std::size_t, 2u>{3u * sizeof(float), sizeof(float)});
            auto const upperCconst = alpaka::blas::upper(Cconst);
            using TViewA = decltype(Afloat);
            using TViewC = decltype(upperC);
            using TViewCconst = decltype(upperCconst);
            using TViewCdouble = decltype(upperCdouble);
            // Positive control: float A + float writable upper(C) + float coefficients.
            static_assert(syrkCallable<int, float, TViewA, float, TViewC>);
            // Rejections: const-element C, A/C element-type mismatch, and complex coefficients.
            static_assert(!syrkCallable<int, float, TViewA, float, TViewCconst>); // const-element C rejected
            static_assert(!syrkCallable<int, float, TViewA, float, TViewCdouble>); // A/C element-type mismatch
            static_assert(!syrkCallable<int, alpaka::math::Complex<float>, TViewA, float, TViewC>); // complex alpha
            static_assert(!syrkCallable<int, float, TViewA, alpaka::math::Complex<float>, TViewC>); // complex beta
            SUCCEED();
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS syrk preserves padding and opposite triangle for padded C and A",
    "[integr][blas][rankk][syrk]",
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

        using Scalar = float;
        constexpr uint32_t n = 4u;
        constexpr uint32_t k = 3u;
        constexpr std::size_t pad = 2u;
        constexpr std::size_t ldC = n + pad;
        constexpr std::size_t ldA = k + pad;
        constexpr auto sentinel = 0x5'a5a5'a5af;

        auto Astorage = alpaka::onHost::allocUnified<Scalar>(device, n * ldA);
        auto A = alpaka::makeMdSpan(
            Astorage.data(),
            alpaka::Vec<uint32_t, 2u>{n, k},
            alpaka::Vec<std::size_t, 2u>{ldA * sizeof(Scalar), sizeof(Scalar)});
        auto Cstorage = alpaka::onHost::allocUnified<Scalar>(device, n * ldC);
        auto C = alpaka::makeMdSpan(
            Cstorage.data(),
            alpaka::Vec<uint32_t, 2u>{n, n},
            alpaka::Vec<std::size_t, 2u>{ldC * sizeof(Scalar), sizeof(Scalar)});
        fillMatrix(A, n, k);
        fillMatrix(C, n, n);
        for(std::size_t i = 0; i < n; ++i)
            for(std::size_t j = n; j < ldC; ++j)
                Cstorage.data()[i * ldC + j] = Scalar(sentinel);
        for(std::size_t i = 0; i < n; ++i)
            for(std::size_t j = k; j < ldA; ++j)
                Astorage.data()[i * ldA + j] = Scalar(sentinel);
        auto Acopy = alpaka::onHost::allocHostLike(Astorage);
        auto Ccopy = alpaka::onHost::allocHostLike(Cstorage);
        std::copy_n(Astorage.data(), n * ldA, Acopy.data());
        std::copy_n(Cstorage.data(), n * ldC, Ccopy.data());

        auto upperC = alpaka::blas::upper(C);
        alpaka::blas::onHost::syrk(queue, Scalar{1.25}, A, Scalar{-0.5}, upperC, options);
        alpaka::onHost::wait(queue);

        auto Cref = copyRaw(Ccopy, n, n, ldC);
        auto Aref = copyRaw(Acopy, n, k, ldA);
        blas::syrkRef(
            Scalar{1.25},
            Aref.data(),
            ldA,
            n,
            k,
            alpaka::blas::Transpose::none,
            Scalar{-0.5},
            Cref.data(),
            ldC,
            n,
            alpaka::blas::Triangle::upper);
        for(uint32_t i = 0; i < n; ++i)
            for(uint32_t j = 0; j < n; ++j)
            {
                if(j >= i)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(1e-4f));
                else
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[i * ldC + j]);
            }
        for(std::size_t i = 0; i < n; ++i)
            for(std::size_t j = n; j < ldC; ++j)
                CHECK(Cstorage.data()[i * ldC + j] == Ccopy.data()[i * ldC + j]);
        for(std::size_t i = 0; i < n; ++i)
            for(std::size_t j = k; j < ldA; ++j)
                CHECK(Astorage.data()[i * ldA + j] == Acopy.data()[i * ldA + j]);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS syrk degenerate n==0 is a no-op and degenerate kernels are queue-ordered",
    "[integr][blas][rankk][syrk]",
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
        using Scalar = float;
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        // n==0: no data access, no exception, C untouched.
        {
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{0u, 3u});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{0u, 0u});
            auto upperC = alpaka::blas::upper(C);
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{1.0}, upperC);
            alpaka::onHost::wait(queue);
            SUCCEED("n==0 syrk is a valid no-op.");
        }

        // Queue ordering: a vendor write (alpha!=0) followed by a degenerate scale must be serialized; the scale
        // reads C produced by the vendor call, so reordering or lost updates would change the result.
        {
            constexpr uint32_t n = 3u;
            constexpr uint32_t k = 2u;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrix(A, n, k);
            fillMatrixSentinel(C, n, n, Scalar{0.0});
            auto upperC = alpaka::blas::upper(C);
            // First: C <- 1*A*A^T + 0*C  => triangle = A*A^T (a vendor syrk call).
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{0.0}, upperC, options);
            // Second: C <- 0*A*A^T + 2*C => triangle = 2*(A*A^T). If the calls reorder or drop, the value differs.
            alpaka::blas::onHost::syrk(queue, Scalar{0.0}, A, Scalar{2.0}, upperC, options);
            alpaka::onHost::wait(queue);
            auto const ldA = ldOf(A);
            auto const Aref = copyRaw(A, n, k, ldA);
            auto const ldC = ldOf(C);
            auto Cref = std::vector<Scalar>(n * ldC, Scalar{0.0});
            blas::syrkRef(
                Scalar{1.0},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Scalar{0.0},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::upper);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i; j < n; ++j)
                    CHECK(
                        C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(2.0f * Cref[i * ldC + j]).epsilon(1e-4f));
        }

        // Producer -> syrk -> consumer chain with a single final wait: a queued fill writes C, a non-degenerate
        // syrk reads that produced value and updates the triangle, then a queued memcpy reads C back to host. No
        // intermediate wait may separate the three operations; the vendor syrk must be in-order with respect to the
        // queued fill and the queued readback.
        {
            constexpr uint32_t n = 3u;
            constexpr uint32_t k = 2u;
            auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
            auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            auto hostC = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
            fillMatrix(A, n, k);
            auto const producerValue = Scalar{7.0f};
            fillMatrixSentinel(C, n, n, producerValue);
            auto upperC = alpaka::blas::upper(C);
            // Enqueue producer -> syrk -> consumer with ONE final wait (all three ops in the same queue).
            alpaka::onHost::fill(queue, C, producerValue);
            alpaka::blas::onHost::syrk(queue, Scalar{1.5}, A, Scalar{-0.5}, upperC, options);
            alpaka::onHost::memcpy(queue, hostC, C);
            alpaka::onHost::wait(queue);
            auto const ldA = ldOf(A);
            auto const ldC = ldOf(C);
            auto const Aref = copyRaw(A, n, k, ldA);
            auto Cref = std::vector<Scalar>(n * ldC, producerValue);
            blas::syrkRef(
                Scalar{1.5},
                Aref.data(),
                ldA,
                n,
                k,
                alpaka::blas::Transpose::none,
                Scalar{-0.5},
                Cref.data(),
                ldC,
                n,
                alpaka::blas::Triangle::upper);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j < n; ++j)
                {
                    // hostC holds C through the same queue; both C and hostC must equal the reference.
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(1e-4f));
                    CHECK(hostC[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(1e-4f));
                }
        }

        // Degenerate scale with padded C (ld > cols): the selected triangle scales and the padding is untouched.
        {
            constexpr uint32_t n = 3u;
            constexpr uint32_t k = 2u;
            auto Astorage = alpaka::onHost::allocUnified<Scalar>(device, 24u);
            auto A = alpaka::makeMdSpan(
                Astorage.data(),
                alpaka::Vec<uint32_t, 2u>{n, k},
                alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(n) * sizeof(Scalar), sizeof(Scalar)});
            auto Cstorage = alpaka::onHost::allocUnified<Scalar>(device, 3u * 6u);
            auto C = alpaka::makeMdSpan(
                Cstorage.data(),
                alpaka::Vec<uint32_t, 2u>{3u, 3u},
                alpaka::Vec<std::size_t, 2u>{6u * sizeof(Scalar), sizeof(Scalar)});
            fillMatrix(A, n, k);
            fillMatrixSentinel(C, n, 6u, Scalar{5.0});
            auto lowerC = alpaka::blas::lower(C);
            alpaka::blas::onHost::syrk(queue, Scalar{0.0}, A, Scalar{2.0}, lowerC, options);
            alpaka::onHost::wait(queue);
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = 0; j <= i; ++j)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Scalar{10.0}).epsilon(1e-5f));
            for(uint32_t i = 0; i < n; ++i)
                for(uint32_t j = i + 1; j < 6u; ++j) // opposite triangle + padding untouched
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Scalar{5.0}).epsilon(1e-5f));
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS syrk oversized dimensions rejected by checkedCast",
    "[integr][blas][rankk][syrk]",
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
        using Scalar = float;
        constexpr uint32_t n = 3u;
        constexpr uint32_t k = 3u;
        // An enormous leading dimension: the pitch overflows the 32-bit vendor int parameters used by the
        // host/cuda/hip syrk dispatches (and by the degenerate scale-kernel branch).
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
        // A normal-C variants with only C's leading dimension oversized exercises the cdLd checkedCast.
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
        if constexpr(!std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::OneApi>)
        {
            auto upperC = alpaka::blas::upper(C);
            CHECK_THROWS_AS(
                alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{1.0}, upperC),
                std::invalid_argument);
            // C-ld only oversized: the A descriptor is well-formed (normal A), C's cdLd must still be rejected.
            auto upperCbig = alpaka::blas::upper(Cbig);
            CHECK_THROWS_AS(
                alpaka::blas::onHost::syrk(queue, Scalar{1.0}, Anormal, Scalar{1.0}, upperCbig),
                std::invalid_argument);
            // Degenerate branch (alpha == 0) bypasses the vendor dispatch, so an oversized C ld must be rejected by
            // the scale path itself. An oversized A ld is harmless there (A is never read) and must be accepted while
            // C is well-formed -- verified by the succeeding calls below.
            CHECK_THROWS_AS(
                alpaka::blas::onHost::syrk(queue, Scalar{0.0}, Anormal, Scalar{1.0}, upperCbig),
                std::invalid_argument);
            // Oversized A ld but well-formed C: accepted in the degenerate branch (A untouched), and non-degenerate
            // run proceeds far enough to reject via the A dispatch checkedCast.
            auto upperCnormal = alpaka::blas::upper(Cnormal);
            CHECK_THROWS_AS(
                alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{1.0}, upperCnormal),
                std::invalid_argument);
            // Metadata-only oversized n and k: the vendor int parameters overflow, so the checkedCast<int>
            // in the host/cuda/hip dispatches rejects before any data access. The claimed extents and row pitch are
            // internally consistent so the descriptor and shape validation pass; only the vendor-width narrowing
            // rejects the call. Backing storage stays tiny; nothing may be written.
            {
                // n > INT_MAX (2^31-1), k == 3. C must be (hugeN x hugeN), so its row pitch encodes hugeN.
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
                    alpaka::blas::onHost::syrk(queue, Scalar{1.0}, AhugeN, Scalar{1.0}, upperChugeN),
                    std::invalid_argument);
                // k > INT_MAX, n == 3. op(A) is n x k, so C stays 3x3; A's row pitch encodes the huge k columns.
                constexpr auto hugeK = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1u;
                auto AstorageK = alpaka::onHost::allocUnified<Scalar>(device, 16u);
                auto AhugeK = alpaka::makeMdSpan(
                    AstorageK.data(),
                    alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(n), hugeK},
                    alpaka::Vec<std::size_t, 2u>{hugeK * sizeof(Scalar), sizeof(Scalar)});
                auto CnormalK = alpaka::makeMdSpan(
                    Cst.data(),
                    alpaka::Vec<uint32_t, 2u>{n, n},
                    alpaka::Vec<std::size_t, 2u>{static_cast<std::size_t>(n) * sizeof(Scalar), sizeof(Scalar)});
                auto upperCnormalK = alpaka::blas::upper(CnormalK);
                CHECK_THROWS_AS(
                    alpaka::blas::onHost::syrk(queue, Scalar{1.0}, AhugeK, Scalar{1.0}, upperCnormalK),
                    std::invalid_argument);
            }
        }
        else
        {
            // oneAPI/oneMKL uses 64-bit dimensions, so oversized-n/k with internally-consistent extents is not
            // rejected here; it is the oneMKL fast-path's own problem domain. Keep a degenerate alpha==0 scale with a
            // huge-n C rejected by the generic scale guard (which is backend-generic).
            constexpr auto hugeN = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 2u;
            auto AstorageN = alpaka::onHost::allocUnified<Scalar>(device, 16u);
            auto CstorageN = alpaka::onHost::allocUnified<Scalar>(device, 16u);
            auto AhugeN = alpaka::makeMdSpan(
                AstorageN.data(),
                alpaka::Vec<std::size_t, 2u>{hugeN, std::size_t{0u}},
                alpaka::Vec<std::size_t, 2u>{std::size_t{0u}, sizeof(Scalar)});
            auto ChugeN = alpaka::makeMdSpan(
                CstorageN.data(),
                alpaka::Vec<std::size_t, 2u>{hugeN, hugeN},
                alpaka::Vec<std::size_t, 2u>{hugeN * sizeof(Scalar), sizeof(Scalar)});
            auto upperChugeN = alpaka::blas::upper(ChugeN);
            // k == 0 degenerates to the scale kernel; with beta != 1 the n>uint32-max guard throws.
            CHECK_THROWS_AS(
                alpaka::blas::onHost::syrk(queue, Scalar{1.0}, AhugeN, Scalar{2.0}, upperChugeN),
                std::invalid_argument);
            // The n>uint32-max guard is metadata-only and runs before any kernel, so even a beta==1 scale is
            // rejected: the selected triangle cannot be enumerated by the uint32 index domain.
            CHECK_THROWS_AS(
                alpaka::blas::onHost::syrk(queue, Scalar{1.0}, AhugeN, Scalar{1.0}, upperChugeN),
                std::invalid_argument);
        }
    }
}
