/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>

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

void fillMatrix(auto& view, std::size_t rows, std::size_t cols)
{
    using T = alpaka::GetValueType_t<std::remove_cvref_t<decltype(view)>>;
    for(std::size_t i = 0; i < rows; ++i)
        for(std::size_t j = 0; j < cols; ++j)
            view[alpaka::Vec<uint32_t, 2u>{i, j}]
                = T(static_cast<typename alpaka::blas::Real_t<T>>(i * 10 + j + 1));
}

void fillMatrixSentinel(auto& view, std::size_t rows, std::size_t cols, auto sentinel)
{
    using T = alpaka::GetValueType_t<std::remove_cvref_t<decltype(view)>>;
    for(std::size_t i = 0; i < rows; ++i)
        for(std::size_t j = 0; j < cols; ++j)
            view[alpaka::Vec<uint32_t, 2u>{i, j}] = T(sentinel);
}

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

        using Scalar = float;
        constexpr uint32_t n = 5u;
        constexpr uint32_t k = 3u;

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

        using Scalar = double;
        constexpr uint32_t n = 4u;
        constexpr uint32_t k = 6u;

        // A stored as k x n; transposed(A) is n x k.
        auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{k, n});
        auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        auto Ccopy = alpaka::onHost::allocHostLike(C);
        fillMatrix(A, k, n);
        fillMatrix(C, n, n);
        for(uint32_t i = 0; i < n; ++i)
            for(uint32_t j = 0; j < n; ++j)
                Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}] = C[alpaka::Vec<uint32_t, 2u>{i, j}];

        auto transposedA = alpaka::blas::transposed(A);
        auto lowerC = alpaka::blas::lower(C);
        alpaka::blas::onHost::syrk(queue, Scalar{0.75}, transposedA, Scalar{2.0}, lowerC, options);
        alpaka::onHost::wait(queue);

        auto const ldA = ldOf(A);
        auto const ldC = ldOf(C);
        auto const Aref = copyRaw(A, k, n, ldA);
        auto Cref = copyRaw(Ccopy, n, n, ldC);
        blas::syrkRef(
            Scalar{0.75},
            Aref.data(),
            ldA,
            k,
            n,
            alpaka::blas::Transpose::transposed,
            Scalar{2.0},
            Cref.data(),
            ldC,
            n,
            alpaka::blas::Triangle::lower);
        for(uint32_t i = 0; i < n; ++i)
            for(uint32_t j = 0; j < n; ++j)
            {
                if(j <= i)
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Cref[i * ldC + j]).epsilon(1e-12));
                else
                    CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Ccopy[alpaka::Vec<uint32_t, 2u>{i, j}]);
            }
    }
}

TEMPLATE_LIST_TEST_CASE("BLAS syrk alpha=0 and beta=0 no-read semantics", "[integr][blas][rankk][syrk]", TestBackends)
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

        auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, k});
        auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        fillMatrix(A, n, k);
        fillMatrixSentinel(C, n, n, Scalar{3.0});
        auto upperC = alpaka::blas::upper(C);
        alpaka::blas::onHost::syrk(queue, Scalar{0.0}, A, Scalar{2.0}, upperC, options);
        alpaka::onHost::wait(queue);
        for(uint32_t i = 0; i < n; ++i)
            for(uint32_t j = i; j < n; ++j)
                CHECK(C[alpaka::Vec<uint32_t, 2u>{i, j}] == Catch::Approx(Scalar{6.0}).epsilon(1e-5f));

        // beta=0: C2 = alpha * M * M^T (selected triangle), old C2 not read.
        auto C2 = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{n, n});
        fillMatrixSentinel(C2, n, n, Scalar{99.0});
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
        CHECK_THROWS_AS(
            alpaka::blas::onHost::syrk(queue, Scalar{1.0}, A, Scalar{1.0}, C),
            std::invalid_argument);
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
    }
}
