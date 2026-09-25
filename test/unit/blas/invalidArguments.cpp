/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <limits>

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

TEMPLATE_LIST_TEST_CASE(
    "blas centralized validation rejects non-element pitches, aliasing and oversized extents",
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
        using Api = std::remove_cvref_t<ALPAKA_TYPEOF(device.getApi())>;

        // Misaligned pitch: the descriptor layer must reject a byte pitch that is not a whole multiple of the
        // element size instead of silently truncating the element stride. The innermost (x/column and vector) axes
        // are exercised with configurable-pitch stub views in helpers.cpp because alpaka3 normalizes the innermost
        // MdSpan pitch to sizeof(value_type); the outer row/batch axes are real MdSpan pitches here.
        auto misalignedStorage = alpaka::onHost::allocUnified<float>(device, 64u);
        auto rowPitch = alpaka::makeMdSpan(
            misalignedStorage.data(),
            alpaka::Vec<std::uint32_t, 2u>{3u, 4u},
            alpaka::Vec<std::size_t, 2u>{6u * sizeof(float) + 2u, sizeof(float)});
        CHECK_THROWS_AS(alpaka::blas::internal::makeMatrixDescriptor(rowPitch), std::invalid_argument);
        auto batchPitch = alpaka::makeMdSpan(
            misalignedStorage.data(),
            alpaka::Vec<std::uint32_t, 3u>{2u, 3u, 4u},
            alpaka::Vec<std::size_t, 3u>{36u * sizeof(float) + 2u, 6u * sizeof(float), sizeof(float)});
        CHECK_THROWS_AS(alpaka::blas::internal::makeBatchedMatrixDescriptor(batchPitch), std::invalid_argument);

        // Alias/overlap rejection for the documented herk/syrk contracts: A and C backed by the same storage must be
        // rejected by the metadata validation before any dispatch. The views are tiny and are never read.
        {
            auto storage = alpaka::onHost::allocUnified<float>(device, 64u);
            auto A = alpaka::makeMdSpan(
                storage.data(),
                alpaka::Vec<std::uint32_t, 2u>{3u, 3u},
                alpaka::Vec<std::size_t, 2u>{3u * sizeof(float), sizeof(float)});
            auto Coverlap = alpaka::makeMdSpan(
                storage.data(),
                alpaka::Vec<std::uint32_t, 2u>{3u, 3u},
                alpaka::Vec<std::size_t, 2u>{3u * sizeof(float), sizeof(float)});
            auto upperCoverlap = alpaka::blas::upper(Coverlap);
            CHECK_THROWS_AS(alpaka::blas::onHost::syrk(queue, 1.0f, A, 1.0f, upperCoverlap), std::invalid_argument);
        }
        {
            using Scalar = alpaka::math::Complex<float>;
            auto storage = alpaka::onHost::allocUnified<Scalar>(device, 64u);
            auto A = alpaka::makeMdSpan(
                storage.data(),
                alpaka::Vec<std::uint32_t, 2u>{3u, 3u},
                alpaka::Vec<std::size_t, 2u>{3u * sizeof(Scalar), sizeof(Scalar)});
            auto Coverlap = alpaka::makeMdSpan(
                storage.data(),
                alpaka::Vec<std::uint32_t, 2u>{3u, 3u},
                alpaka::Vec<std::size_t, 2u>{3u * sizeof(Scalar), sizeof(Scalar)});
            auto upperCoverlap = alpaka::blas::upper(Coverlap);
            CHECK_THROWS_AS(alpaka::blas::onHost::herk(queue, 1.0f, A, 1.0f, upperCoverlap), std::invalid_argument);
        }

        // Partial overlap with a *distinct* base pointer: the same 3x3 shape one element later genuinely intersects
        // A's byte span and must be rejected. The adjacent view starting exactly one element past A's exclusive end
        // is disjoint and must be accepted (positive control). Both views are tiny and never read.
        {
            auto storage = alpaka::onHost::allocUnified<float>(device, 64u);
            auto makeView = [&](float* base)
            {
                return alpaka::makeMdSpan(
                    base,
                    alpaka::Vec<std::uint32_t, 2u>{3u, 3u},
                    alpaka::Vec<std::size_t, 2u>{3u * sizeof(float), sizeof(float)});
            };
            auto A = makeView(storage.data());
            auto Cpartial = alpaka::blas::upper(makeView(storage.data() + 1u));
            CHECK_THROWS_WITH(
                alpaka::blas::onHost::syrk(queue, 1.0f, A, 1.0f, Cpartial),
                Catch::Matchers::ContainsSubstring("must not overlap"));
            // A spans elements 0..8 (3x3 ld=3); starting at element 9 is adjacent, not overlapping.
            auto Cadjacent = alpaka::blas::upper(makeView(storage.data() + 9u));
            CHECK_NOTHROW(alpaka::blas::onHost::syrk(queue, 1.0f, A, 1.0f, Cadjacent));
            // Drain the accepted positive-control work before storage goes out of scope.
            alpaka::onHost::wait(queue);
        }

        // Zero-extent no-op contracts: an empty vector copy/scal and an empty rank-k update perform no data access
        // and must not throw. A one-element backing buffer is used so any out-of-range access would be obvious.
        {
            auto storage = alpaka::onHost::allocUnified<float>(device, 1u);
            auto emptyX = alpaka::makeMdSpan(
                storage.data(),
                alpaka::Vec<std::uint32_t, 1u>{0u},
                alpaka::Vec<std::size_t, 1u>{sizeof(float)});
            auto emptyY = alpaka::makeMdSpan(
                storage.data(),
                alpaka::Vec<std::uint32_t, 1u>{0u},
                alpaka::Vec<std::size_t, 1u>{sizeof(float)});
            CHECK_NOTHROW(alpaka::blas::onHost::copy(queue, emptyX, emptyY));
            CHECK_NOTHROW(alpaka::blas::onHost::scal(queue, 2.0f, emptyY));

            auto emptyA = alpaka::makeMdSpan(
                storage.data(),
                alpaka::Vec<std::uint32_t, 2u>{0u, 3u},
                alpaka::Vec<std::size_t, 2u>{4u * sizeof(float), sizeof(float)});
            auto emptyC = alpaka::makeMdSpan(
                storage.data(),
                alpaka::Vec<std::uint32_t, 2u>{0u, 0u},
                alpaka::Vec<std::size_t, 2u>{sizeof(float), sizeof(float)});
            auto upperEmptyC = alpaka::blas::upper(emptyC);
            CHECK_NOTHROW(alpaka::blas::onHost::syrk(queue, 1.0f, emptyA, 1.0f, upperEmptyC));
            alpaka::onHost::wait(queue);
        }

        // Oversized extents: the metadata alone (a single-element backing store) must reject before enqueue on the
        // 32-bit vendor-int backends, while oneMKL keeps the 64-bit extent. This mirrors the integration guard for
        // the descriptor layer; no backend routine is executed for the rejected case.
        {
            using Scalar = float;
            constexpr std::size_t hugeN = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 2u;
            auto storage = alpaka::onHost::allocUnified<Scalar>(device, 1u);
            // Explicit element-sized pitch keeps the descriptor well-formed (a default pitch computed from the huge
            // extent could overflow); only the extent exceeds the 32-bit vendor int.
            auto big = alpaka::makeMdSpan(
                storage.data(),
                alpaka::Vec<std::size_t, 1u>{hugeN},
                alpaka::Vec<std::size_t, 1u>{sizeof(Scalar)});
            auto desc = alpaka::blas::internal::makeVectorDescriptor(big);
            if constexpr(std::same_as<Api, alpaka::api::OneApi>)
            {
                // oneMKL descriptor integers are 64-bit: the extent is representable and preserved losslessly.
                CHECK(desc.n == static_cast<std::int64_t>(hugeN));
                CHECK(alpaka::blas::internal::checkedVendorInt<alpaka::api::OneApi>(desc.n, "n") == desc.n);
            }
            else
            {
                CHECK_THROWS_AS(alpaka::blas::internal::checkedVendorInt<Api>(desc.n, "n"), std::invalid_argument);
            }
        }
        SUCCEED();
    }
}

TEMPLATE_LIST_TEST_CASE(
    "blas host non-blocking queue surfaces oversized-extent rejection synchronously",
    "[unit][blas][invalid]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using Api = std::remove_cvref_t<ALPAKA_TYPEOF(device.getApi())>;
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SUCCEED();
    }
    else if constexpr(!std::same_as<Api, alpaka::api::Host>)
    {
        // The narrowed-int rejection path under test is host-specific; cuda/hip/oneapi have their own dispatch.
        SUCCEED();
    }
    else
    {
        // A host *non-blocking* queue defers the enqueued lambda onto the callback thread. A descriptor narrowing
        // performed inside that lambda would be captured in a future that enqueueNativeFn discards, so the caller
        // would never observe the error. The narrowing must therefore run on the caller thread before enqueue, and
        // this CHECK_THROWS_AS must fire on the calling thread. The views are lightweight: a one-element backing
        // store with a huge logical extent (no allocation proportional to the extent).
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        constexpr std::size_t hugeN = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 2u;
        auto storageX = alpaka::onHost::allocUnified<float>(device, 1u);
        auto storageY = alpaka::onHost::allocUnified<float>(device, 1u);
        auto bigX = alpaka::makeMdSpan(
            storageX.data(),
            alpaka::Vec<std::size_t, 1u>{hugeN},
            alpaka::Vec<std::size_t, 1u>{sizeof(float)});
        auto bigY = alpaka::makeMdSpan(
            storageY.data(),
            alpaka::Vec<std::size_t, 1u>{hugeN},
            alpaka::Vec<std::size_t, 1u>{sizeof(float)});
        CHECK_THROWS_AS(alpaka::blas::onHost::copy(queue, bigX, bigY), std::invalid_argument);
        // Positive control: a small pair on the same non-blocking queue is accepted and completes after wait.
        auto smallX = alpaka::onHost::allocUnified<float>(device, 4u);
        auto smallY = alpaka::onHost::allocUnified<float>(device, 4u);
        CHECK_NOTHROW(alpaka::blas::onHost::copy(queue, smallX, smallY));
        alpaka::onHost::wait(queue);
        SUCCEED();
    }
}
