/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>
#include <cstdint>

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

// Whether the public dot entry form for the given argument types compiles. `dot` is deliberately unconstrained (its
// body only carries static_asserts and runtime checks), so this is a formability-only probe: it never instantiates the
// backend dispatcher and cannot detect a dispatch regression. It is a regression guard for the round-2 `Value_t`
// revert (9e2ea37 restored base behavior after the cv-stripped head dba5ddc changed dispatch): it asserts that the
// acceptance matrix is unchanged relative to base dev 50b3d837, NOT that every accepted form is correct. Base dev is
// cv-preserving (`Value_t` keeps const), so a mixed call like dot(x<const float>, y<double>) is a pre-existing
// base-dev hazard (host `OpenBlas<float const>` unspecialized / CUDA misdispatch), not supported parity behavior.
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

        // A result view with a null data pointer must be rejected before any backend dispatch. The check is a pure
        // host-side pointer comparison and therefore backend-independent.
        auto nullResult = alpaka::makeMdSpan(static_cast<float*>(nullptr), alpaka::Vec<uint32_t, 1u>{1u});
        CHECK_THROWS_AS(alpaka::blas::onHost::dot(queue, x, y4, nullResult), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::nrm2(queue, x, nullResult), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::asum(queue, x, nullResult), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::iamax(queue, x, nullResult), std::invalid_argument);

        // The result element type is part of the contract: nrm2/asum expect the real type and dot the scalar type.
        // The backend dispatch is guarded by an `if constexpr` on the result element type, so a mismatch is rejected
        // at runtime before dispatch and never instantiates an incompatible vendor call.
        auto doubleResult1 = alpaka::onHost::allocUnified<double>(device, 1u);
        CHECK_THROWS_AS(alpaka::blas::onHost::nrm2(queue, x, doubleResult1), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::asum(queue, x, doubleResult1), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::dot(queue, x, y4, doubleResult1), std::invalid_argument);
        auto intResult1 = alpaka::onHost::allocUnified<int>(device, 1u);
        CHECK_THROWS_AS(alpaka::blas::onHost::nrm2(queue, x, intResult1), std::invalid_argument);
        // iamax requires exactly a 32-bit signed integer result element. The host backend writes through `int`,
        // CUDA/HIP reinterpret the result pointer as `int*`, and oneMKL has an int32 overload, so any other element
        // width (narrow or wide) or a floating-point type is rejected at runtime before dispatch. The wider oneMKL
        // result width is tracked in issue #19.
        auto floatResult1 = alpaka::onHost::allocUnified<float>(device, 1u);
        CHECK_THROWS_AS(alpaka::blas::onHost::iamax(queue, x, floatResult1), std::invalid_argument);
        auto int8Result1 = alpaka::onHost::allocUnified<std::int8_t>(device, 1u);
        CHECK_THROWS_AS(alpaka::blas::onHost::iamax(queue, x, int8Result1), std::invalid_argument);
        auto int64Result1 = alpaka::onHost::allocUnified<std::int64_t>(device, 1u);
        CHECK_THROWS_AS(alpaka::blas::onHost::iamax(queue, x, int64Result1), std::invalid_argument);

        // A complex result view for nrm2/asum (which produce the real type) is likewise rejected before dispatch.
        using Complex = alpaka::math::Complex<float>;
        auto zx = alpaka::onHost::allocUnified<Complex>(device, 4u);
        auto complexResult1 = alpaka::onHost::allocUnified<Complex>(device, 1u);
        CHECK_THROWS_AS(alpaka::blas::onHost::nrm2(queue, zx, complexResult1), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::asum(queue, zx, complexResult1), std::invalid_argument);

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

        // Pre-existing `dot` acceptance-matrix regression guard: the probes below must match base-dev (50b3d837)
        // formability exactly. Base dev accepts read-only x views and mixed x/y element types because its dot body is
        // unconstrained (backend dispatch handles them, wrongly in the const/mixed cases, but that hazard predates
        // this branch); the round-2 Value_t revert (9e2ea37) restored that cv-preserving behavior. The only `dot`
        // difference between base and this branch is the added in-body writable-result static_assert, which does not
        // alter which calls form (the overload is unconstrained), so every probe below must match base-dev behavior
        // exactly. A dispatch-level probe is intentionally not added here: instantiating `internal::DotFn::call` with
        // const/mixed element types hard-errors on host (`OpenBlas<float const>` is unspecialized) and would flip the
        // very matrix this guard exists to pin. The healthy <float,float,float> dot form's dispatch is already
        // exercised by the runtime CHECK_THROWS_AS calls above (which call through the full wrapper body).
        static_assert(dotCallable<TQueue, TViewX, TViewY, TViewResult>);
        static_assert(dotCallable<TQueue, TViewX, TViewYConst, TViewResult>);
        static_assert(dotCallable<TQueue, TViewX, TViewYDouble, TViewResult>);
        static_assert(dotCallable<TQueue, TViewX, TViewYDouble, TViewResultDouble>);
        static_assert(dotCallable<TQueue, TViewX, TViewYDouble, TViewResultDoubleConst>);
        static_assert(dotCallable<TQueue, TViewX, TViewY, TViewResultDouble>);
        static_assert(dotCallable<TQueue, TViewX, TViewY, TViewResultConst>);
    }
}
