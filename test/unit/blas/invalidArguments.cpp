/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/blas.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

// Whether the public dotc entry forms for the given argument types. Expressing the call through a variable-template
// requires-expression turns unsatisfied constraints (mismatched element types, const result) into `false` instead of
// a hard compile error, so the negative cases can be asserted inside a test.
template<typename TQueue, typename TViewX, typename TViewY, typename TViewResult>
inline constexpr bool dotcCallable = requires(TQueue& queue, TViewX const& x, TViewY const& y, TViewResult& result) {
    alpaka::blas::onHost::dotc(queue, x, y, result);
};

// Whether the public dot entry forms for the given argument types. This guards that the pre-existing `dot` keeps the
// exact base-dev behavior (50b3d837): the base already accepted read-only x views and mixed x/y element types through
// its cv-stripped dispatch, and the PR-11 cv fix must not change that acceptance matrix.
template<typename TQueue, typename TViewX, typename TViewY, typename TViewResult>
inline constexpr bool dotCallable = requires(TQueue& queue, TViewX const& x, TViewY const& y, TViewResult& result) {
    alpaka::blas::onHost::dot(queue, x, y, result);
};

TEMPLATE_LIST_TEST_CASE(
    "blas invalid sizes, annotations and layouts are rejected before backend dispatch",
    "[unit][blas][invalid]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SUCCEED();
    }
    else
    {
        auto queue = device.makeQueue();

        auto A = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{2u, 3u});
        auto B = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{4u, 2u});
        auto C = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        CHECK_THROWS_AS(alpaka::blas::onHost::gemm(queue, 1.0f, A, B, 0.0f, C), std::invalid_argument);

        auto x = alpaka::onHost::allocUnified<float>(device, 4u);
        auto y = alpaka::onHost::allocUnified<float>(device, 2u);
        CHECK_THROWS_AS(alpaka::blas::onHost::copy(queue, x, y), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::swap(queue, x, y), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::axpy(queue, 1.0f, x, y), std::invalid_argument);
        auto result2 = alpaka::onHost::allocUnified<float>(device, 2u);
        auto y4 = alpaka::onHost::allocUnified<float>(device, 4u);
        CHECK_THROWS_AS(alpaka::blas::onHost::dot(queue, x, y, result2), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::dotc(queue, x, y, result2), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::nrm2(queue, x, result2), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::dot(queue, x, y4, result2), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::dotc(queue, x, y4, result2), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::gemv(queue, 1.0f, A, x, 0.0f, y), std::invalid_argument);

        auto rhs = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{2u, 1u});
        CHECK_THROWS_AS(
            alpaka::blas::onHost::trsm(queue, alpaka::blas::Side::left, 1.0f, A, rhs),
            std::invalid_argument);
        auto square = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        CHECK_THROWS_AS(
            alpaka::blas::onHost::trsm(queue, alpaka::blas::Side::left, 1.0f, square, rhs),
            std::invalid_argument);

        auto BA = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 3u});
        auto BB = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 3u>{3u, 3u, 2u});
        auto BC = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 2u});
        CHECK_THROWS_AS(
            alpaka::blas::onHost::stridedBatchedGemm(queue, 1.0f, BA, BB, 0.0f, BC),
            std::invalid_argument);

        auto storage = alpaka::onHost::allocUnified<float>(device, 16u);
        auto invalidLd = alpaka::makeMdSpan(
            storage.data(),
            alpaka::Vec<uint32_t, 2u>{2u, 3u},
            alpaka::Vec<std::size_t, 2u>{2u * sizeof(float), sizeof(float)});
        CHECK_THROWS_AS(alpaka::blas::internal::makeMatrixDescriptor(invalidLd), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::gemm(queue, 1.0f, invalidLd, C, 0.0f, C), std::invalid_argument);

        // dotc element-type and const-result contracts. x<const float> with y<float> must be accepted
        // (read-only first operand), while x<const float> with y<double> and const results are rejected at
        // compile time through the same requires clause.
        auto xf = alpaka::makeMdSpan(static_cast<float const*>(nullptr), alpaka::Vec<std::size_t, 1u>{1u});
        auto yfMut = alpaka::makeMdSpan(static_cast<float*>(nullptr), alpaka::Vec<std::size_t, 1u>{1u});
        auto yfConst = alpaka::makeMdSpan(static_cast<float const*>(nullptr), alpaka::Vec<std::size_t, 1u>{1u});
        auto resultF = alpaka::makeMdSpan(static_cast<float*>(nullptr), alpaka::Vec<std::size_t, 1u>{1u});
        auto resultFConst = alpaka::makeMdSpan(static_cast<float const*>(nullptr), alpaka::Vec<std::size_t, 1u>{1u});
        auto yd = alpaka::makeMdSpan(static_cast<double*>(nullptr), alpaka::Vec<std::size_t, 1u>{1u});
        auto resultD = alpaka::makeMdSpan(static_cast<double*>(nullptr), alpaka::Vec<std::size_t, 1u>{1u});
        auto resultDConst = alpaka::makeMdSpan(static_cast<double const*>(nullptr), alpaka::Vec<std::size_t, 1u>{1u});
        using TQueue = ALPAKA_TYPEOF(queue);
        using TViewX = decltype(xf);
        using TViewY = decltype(yfMut);
        using TViewYConst = decltype(yfConst);
        using TViewResult = decltype(resultF);
        using TViewResultConst = decltype(resultFConst);
        using TViewYDouble = decltype(yd);
        using TViewResultDouble = decltype(resultD);
        using TViewResultDoubleConst = decltype(resultDConst);
        // Positive control: const x + writable y/result.
        static_assert(dotcCallable<TQueue, TViewX, TViewY, TViewResult>);
        // Read-only first operand with writable result is accepted.
        static_assert(dotcCallable<TQueue, TViewX, TViewY, TViewResult>);
        // const-result is rejected.
        static_assert(!dotcCallable<TQueue, TViewX, TViewY, TViewResultConst>);
        // Mismatched y element type is rejected.
        static_assert(!dotcCallable<TQueue, TViewX, TViewYDouble, TViewResult>);
        static_assert(!dotcCallable<TQueue, TViewX, TViewYDouble, TViewResultDouble>);
        // Mismatched result element type is rejected.
        static_assert(!dotcCallable<TQueue, TViewX, TViewY, TViewResultDouble>);
        // Any combination involving a const result fails, even with matching types.
        static_assert(!dotcCallable<TQueue, TViewX, TViewYConst, TViewResultDoubleConst>);
        static_assert(!dotcCallable<TQueue, TViewX, TViewY, TViewResultDoubleConst>);
        // A const y view with writable result of the same element type is accepted (read-only y).
        static_assert(dotcCallable<TQueue, TViewX, TViewYConst, TViewResult>);
        SUCCEED();

        // Pre-existing `dot` mixed-type regression guard: `dot` must keep the exact base-dev (50b3d837) acceptance
        // matrix. Base dev accepted read-only x views and mixed x/y element types (both silently dispatch through the
        // cv-stripped Value_t), and the PR-11 Value_t revert must not change that. The only `dot` difference between
        // base and this branch is the added in-body writable-result static_assert, which does not alter which calls
        // form (the overload is unconstrained), so every probe below must match base-dev behavior exactly.
        static_assert(dotCallable<TQueue, TViewX, TViewY, TViewResult>);
        static_assert(dotCallable<TQueue, TViewX, TViewYConst, TViewResult>);
        static_assert(dotCallable<TQueue, TViewX, TViewYDouble, TViewResult>);
        static_assert(dotCallable<TQueue, TViewX, TViewYDouble, TViewResultDouble>);
        static_assert(dotCallable<TQueue, TViewX, TViewYDouble, TViewResultDoubleConst>);
        static_assert(dotCallable<TQueue, TViewX, TViewY, TViewResultDouble>);
        static_assert(dotCallable<TQueue, TViewX, TViewY, TViewResultConst>);
    }
}
