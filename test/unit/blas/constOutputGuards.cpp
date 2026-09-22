/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 *
 * Const-element output contract of the BLAS writers.
 *
 * ``Value_t`` is cv-preserving: ``Value_t<MdSpan<T const>>`` is ``T const``. Writers (scal, copy, swap, axpy,
 * dot/nrm2/asum/iamax result, gemv y, gemm/trsm result) therefore reject const-element outputs at compile time
 * exactly like base dev (50b3d837): their backend bodies fail to form (e.g. the host ``OpenBlas`` specializations
 * are only defined for the unqualified scalar, so ``OpenBlas<T const>`` is an incomplete type). This is a
 * pre-existing, dev-identical contract -- this PR does not change the siblings -- so the positive half of this test
 * proves the machinery (cv-preservation, the writability guard, and that the writable forms still form), while the
 * guarded negative half instantiates the const-output calls and is compiled separately expecting failure.
 *
 * The guarded negative half is only compiled when ``ALPAKAV_BLAS_TEST_NEGATIVE_CONST_OUTPUT`` is defined (see the
 * compile-fail validation step). It is not part of the normal build.
 */

#include <catch2/catch_test_macros.hpp>

#include "alpaka/blas.hpp"
#include "alpakaTest/deviceHelper.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

namespace
{
    template<typename TQueue, typename TAlpha, typename TView, typename TOptions>
    inline constexpr bool scalWritableCallable = requires(TQueue& queue, TAlpha alpha, TView& x, TOptions options) {
        alpaka::blas::onHost::scal(queue, alpha, x, options);
    };

    template<typename TQueue, typename TViewX, typename TViewY, typename TOptions>
    inline constexpr bool copyWritableCallable = requires(TQueue& queue, TViewX& x, TViewY& y, TOptions options) {
        alpaka::blas::onHost::copy(queue, x, y, options);
    };

    template<typename TQueue, typename TViewX, typename TViewY, typename TViewR, typename TOptions>
    inline constexpr bool dotWritableCallable
        = requires(TQueue& queue, TViewX& x, TViewY& y, TViewR& result, TOptions options) {
              alpaka::blas::onHost::dot(queue, x, y, result, options);
          };

    template<typename TView>
    using ValueOf = alpaka::blas::internal::Value_t<TView>;
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "BLAS writers preserve const element types and guard writable operands",
    "[unit][blas][const-output]",
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
        auto xb = alpaka::onHost::allocUnified<float>(device, 2u);
        auto yb = alpaka::onHost::allocUnified<float>(device, 2u);
        auto rb = alpaka::onHost::allocUnified<float>(device, 1u);
        auto x = alpaka::makeMdSpan(xb.data(), alpaka::Vec<std::size_t, 1u>{2u});
        auto y = alpaka::makeMdSpan(yb.data(), alpaka::Vec<std::size_t, 1u>{2u});
        auto r = alpaka::makeMdSpan(rb.data(), alpaka::Vec<std::size_t, 1u>{1u});
        auto const xConst = alpaka::makeMdSpan(static_cast<float const*>(xb.data()), alpaka::Vec<std::size_t, 1u>{2u});

        // Value_t is cv-preserving: MdSpan<const float> yields `const float`, exactly like base dev.
        static_assert(std::same_as<ValueOf<decltype(x)>, float>);
        static_assert(std::same_as<ValueOf<decltype(xConst)>, float const>);
        static_assert(std::same_as<ValueOf<decltype(alpaka::blas::upper(x))>, float>);

        using TQ = ALPAKA_TYPEOF(queue);
        using TOptions = alpaka::blas::Options;
        using TX = ALPAKA_TYPEOF(x);
        using TY = ALPAKA_TYPEOF(y);
        using TR = ALPAKA_TYPEOF(r);

        // The writable forms must still be callable (descriptor dispatch unifies the scalar branch).
        static_assert(scalWritableCallable<TQ, float, TX, TOptions>);
        static_assert(copyWritableCallable<TQ, TX, TY, TOptions>);
        static_assert(dotWritableCallable<TQ, TX, TY, TR, TOptions>);

        // The const forms are rejected: for the guarded negative half below every const-output call must fail to
        // compile; here we only reference the const view type so the guarded block stays self-contained.
#if defined(ALPAKAV_BLAS_TEST_NEGATIVE_CONST_OUTPUT)
        using TXConst = ALPAKA_TYPEOF(xConst);
        alpaka::unused(TXConst{});
#endif

        // The guarded negative half is compiled separately and must fail:
        // - scal/copy/dot/nrm2/asum/iamax/gemm/trsm instantiated with const-element outputs
        //   must not compile (dev-identical rejection).
#if defined(ALPAKAV_BLAS_TEST_NEGATIVE_CONST_OUTPUT)
        alpaka::blas::onHost::scal(queue, 2.0f, xConst); // const output: must not form
        alpaka::blas::onHost::copy(queue, xConst, xConst); // const output: must not form
        alpaka::blas::onHost::dot(queue, x, xConst, xConst); // const result: must not form
        alpaka::blas::onHost::nrm2(queue, x, xConst); // const result: must not form
        alpaka::blas::onHost::asum(queue, x, xConst); // const result: must not form
        alpaka::blas::onHost::iamax(queue, x, xConst); // const result: must not form
        auto Ab = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<std::size_t, 2u>{2u, 2u});
        auto const AConst = alpaka::makeMdSpan(
            static_cast<float const*>(Ab.data()),
            alpaka::Vec<std::size_t, 2u>{2u, 2u},
            alpaka::Vec<std::size_t, 2u>{2u * sizeof(float), sizeof(float)});
        alpaka::blas::onHost::gemm(queue, 1.0f, Ab, Ab, 0.0f, AConst); // const C: must not form
        auto const upperA = alpaka::blas::upper(AConst);
        alpaka::blas::onHost::trsm(queue, alpaka::blas::Side::left, 1.0f, upperA, AConst); // const B: must not form
#endif
        SUCCEED();
    }
}
