/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <algorithm>
#include <alpakaTest/deviceHelper.hpp>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "../unit/blas/reference.hpp"
#include "../unit/test.hpp"
#include "alpaka/blas.hpp"

using namespace alpakaVendor::test;

template<typename T>
void fillVector(T* ptr, std::size_t n)
{
    for(std::size_t i = 0; i < n; ++i)
    {
        if constexpr(alpaka::blas::ComplexScalar<T>)
            ptr[i] = T{static_cast<typename T::value_type>(i + 1), static_cast<typename T::value_type>(2 * i + 1)};
        else
            ptr[i] = T(i + 1);
    }
}

TEMPLATE_LIST_TEST_CASE("BLAS level1 real and complex vectors", "[integr][blas][level1]", TestBackends)
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
        constexpr uint32_t n = 5u;
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<float>;
        auto x = alpaka::onHost::allocUnified<Scalar>(device, n);
        auto y = alpaka::onHost::allocUnified<Scalar>(device, n);
        auto z = alpaka::onHost::allocUnified<Scalar>(device, n);
        auto dotResult = alpaka::onHost::allocUnified<Scalar>(device, 1u);
        auto nrm2Result = alpaka::onHost::allocUnified<float>(device, 1u);
        auto asumResult = alpaka::onHost::allocUnified<float>(device, 1u);
        auto iamaxResult = alpaka::onHost::allocUnified<int>(device, 1u);

        fillVector(x.data(), n);
        fillVector(y.data(), n);
        for(uint32_t i = 0; i < n; ++i)
            z.data()[i] = {};
        auto xBefore = alpaka::onHost::allocHostLike(x);
        auto yBefore = alpaka::onHost::allocHostLike(y);
        std::copy_n(x.data(), n, xBefore.data());
        std::copy_n(y.data(), n, yBefore.data());

        alpaka::blas::onHost::copy(queue, x, z, options);
        alpaka::blas::onHost::swap(queue, x, y, options);
        alpaka::blas::onHost::scal(queue, Scalar{2, -1}, x, options);
        alpaka::blas::onHost::axpy(queue, Scalar{2, -1}, x, y, options);
        alpaka::blas::onHost::dot(queue, x, z, dotResult, options);
        alpaka::blas::onHost::nrm2(queue, x, nrm2Result, options);
        alpaka::blas::onHost::asum(queue, x, asumResult, options);
        alpaka::blas::onHost::iamax(queue, x, iamaxResult, options);
        alpaka::onHost::wait(queue);

        for(uint32_t i = 0; i < n; ++i)
        {
            CHECK(z.data()[i] == xBefore.data()[i]);
            CHECK(y.data()[i] == xBefore.data()[i] + Scalar{2, -1} * x.data()[i]);
            CHECK(x.data()[i] == Scalar{2, -1} * yBefore.data()[i]);
        }
        auto const dotExpected = blas::dotRef(x.data(), z.data(), n);
        CHECK(dotResult.data()[0].real() == Catch::Approx(dotExpected.real()).epsilon(1e-4));
        CHECK(dotResult.data()[0].imag() == Catch::Approx(dotExpected.imag()).epsilon(1e-4));
        CHECK(nrm2Result.data()[0] == Catch::Approx(blas::nrm2Ref(x.data(), n)).epsilon(1e-4));
        CHECK(asumResult.data()[0] == Catch::Approx(blas::asumRef(x.data(), n)).epsilon(1e-4));
        CHECK(iamaxResult.data()[0] == blas::iamaxRef(x.data(), n));

        // iamax must return 0 for an empty (zero-extent) vector on every backend.
        auto xEmpty = alpaka::onHost::allocUnified<Scalar>(device, 0u);
        auto iamaxEmptyResult = alpaka::onHost::allocUnified<int>(device, 1u);
        // Poison the result so the n == 0 check cannot pass by reading uninitialized memory.
        iamaxEmptyResult.data()[0] = 42;
        alpaka::blas::onHost::iamax(queue, xEmpty, iamaxEmptyResult, options);
        alpaka::onHost::wait(queue);
        CHECK(iamaxEmptyResult.data()[0] == 0);
    }
}

TEMPLATE_LIST_TEST_CASE("BLAS level1 dot accepts read-only inputs", "[integr][blas][level1][dot]", TestBackends)
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
        constexpr uint32_t n = 5u;
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        // A read-only (const-element) input view must not be mistaken for a mismatched result type. `Value_t` is
        // cv-preserving, so the dot result-type guard and the backend scalar derivation cv-strip both operands.
        auto runReadOnly = [&]<typename T>()
        {
            auto xb = alpaka::onHost::allocUnified<T>(device, n);
            auto yb = alpaka::onHost::allocUnified<T>(device, n);
            fillVector(xb.data(), n);
            fillVector(yb.data(), n);
            auto xConst = alpaka::makeMdSpan(static_cast<T const*>(xb.data()), alpaka::Vec<std::size_t, 1u>{n});
            auto yConst = alpaka::makeMdSpan(static_cast<T const*>(yb.data()), alpaka::Vec<std::size_t, 1u>{n});
            auto dotRO = alpaka::onHost::allocUnified<T>(device, 1u);
            // Must not throw std::invalid_argument("dot requires a result buffer ...").
            alpaka::blas::onHost::dot(queue, xConst, yConst, dotRO, options);
            alpaka::onHost::wait(queue);
            if constexpr(alpaka::blas::ComplexScalar<T>)
            {
                auto const expected = blas::dotRef(xb.data(), yb.data(), n);
                CHECK(dotRO.data()[0].real() == Catch::Approx(expected.real()).epsilon(1e-4));
                CHECK(dotRO.data()[0].imag() == Catch::Approx(expected.imag()).epsilon(1e-4));
            }
            else
                CHECK(dotRO.data()[0] == Catch::Approx(blas::dotRef(xb.data(), yb.data(), n)).epsilon(1e-4));
        };
        runReadOnly.template operator()<float>();
        runReadOnly.template operator()<double>();
        runReadOnly.template operator()<alpaka::math::Complex<float>>();
        runReadOnly.template operator()<alpaka::math::Complex<double>>();
    }
}

TEMPLATE_LIST_TEST_CASE("BLAS level1 dotc conjugated dot product", "[integr][blas][level1][dotc]", TestBackends)
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
        constexpr uint32_t n = 5u;
        auto const options = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::fastest};

        using Scalar = alpaka::math::Complex<float>;
        auto x = alpaka::onHost::allocUnified<Scalar>(device, n);
        auto y = alpaka::onHost::allocUnified<Scalar>(device, n);
        auto dotcResult = alpaka::onHost::allocUnified<Scalar>(device, 1u);
        auto dotResult = alpaka::onHost::allocUnified<Scalar>(device, 1u);

        fillVector(x.data(), n);
        fillVector(y.data(), n);

        alpaka::blas::onHost::dotc(queue, x, y, dotcResult, options);
        alpaka::blas::onHost::dot(queue, x, y, dotResult, options);
        alpaka::onHost::wait(queue);

        auto const dotcExpected = blas::dotcRef(x.data(), y.data(), n);
        CHECK(dotcResult.data()[0].real() == Catch::Approx(dotcExpected.real()).epsilon(1e-4));
        CHECK(dotcResult.data()[0].imag() == Catch::Approx(dotcExpected.imag()).epsilon(1e-4));

        // Verify dotc differs from dot for complex data (first operand conjugated).
        auto const dotExpected = blas::dotRef(x.data(), y.data(), n);
        CHECK(dotResult.data()[0].real() == Catch::Approx(dotExpected.real()).epsilon(1e-4));
        CHECK(dotResult.data()[0].imag() == Catch::Approx(dotExpected.imag()).epsilon(1e-4));
        // For this fillVector pattern (x[i]=(i+1)+(2i+1)i, y[i]=(i+1)+(2i+1)i), dotc != dot.
        CHECK(dotcResult.data()[0] != dotResult.data()[0]);

        // Real-valued dotc delegates to the real dot path (cblas_sdot / cblas_ddot and the equivalent vendor
        // Sdot/Ddot routines), so it must match dot exactly.
        {
            using Real = float;
            auto xr = alpaka::onHost::allocUnified<Real>(device, n);
            auto yr = alpaka::onHost::allocUnified<Real>(device, n);
            auto dotcReal = alpaka::onHost::allocUnified<Real>(device, 1u);
            fillVector(xr.data(), n);
            fillVector(yr.data(), n);
            alpaka::blas::onHost::dotc(queue, xr, yr, dotcReal, options);
            alpaka::onHost::wait(queue);
            CHECK(dotcReal.data()[0] == Catch::Approx(blas::dotRef(xr.data(), yr.data(), n)).epsilon(1e-4));
        }
        {
            using Real = double;
            auto xr = alpaka::onHost::allocUnified<Real>(device, n);
            auto yr = alpaka::onHost::allocUnified<Real>(device, n);
            auto dotcReal = alpaka::onHost::allocUnified<Real>(device, 1u);
            fillVector(xr.data(), n);
            fillVector(yr.data(), n);
            alpaka::blas::onHost::dotc(queue, xr, yr, dotcReal, options);
            alpaka::onHost::wait(queue);
            CHECK(dotcReal.data()[0] == Catch::Approx(blas::dotRef(xr.data(), yr.data(), n)).epsilon(1e-4));
        }

        // Read-only inputs: dotc must accept vector views with const element type for every supported scalar. The
        // descriptor scalar type is cv-stripped centrally, so a `MdSpan<const T>` selects the same backend branch as
        // the writable `MdSpan<T>` (regression guard for issue 8's common-element-type / cv-qualified dispatch).
        // The const view is created over unified memory, so the same block also runs for the CUDA, HIP and oneAPI
        // device APIs where the data() pointer is a backend-specific unified (managed) pointer.
        {
            auto runReadOnly = [&]<typename T>()
            {
                // MdSpan const-ness is a property of the view type, so a const buffer is sufficient; the data region
                // is written through the buffer's non-const data() before the const view is created.
                auto xb = alpaka::onHost::allocUnified<T>(device, n);
                auto yb = alpaka::onHost::allocUnified<T>(device, n);
                fillVector(xb.data(), n);
                fillVector(yb.data(), n);
                auto xConst = alpaka::makeMdSpan(static_cast<T const*>(xb.data()), alpaka::Vec<std::size_t, 1u>{n});
                auto yConst = alpaka::makeMdSpan(static_cast<T const*>(yb.data()), alpaka::Vec<std::size_t, 1u>{n});
                auto dotcRO = alpaka::onHost::allocUnified<T>(device, 1u);
                alpaka::blas::onHost::dotc(queue, xConst, yConst, dotcRO, options);
                alpaka::onHost::wait(queue);
                if constexpr(alpaka::blas::ComplexScalar<T>)
                {
                    auto const expected = blas::dotcRef(xb.data(), yb.data(), n);
                    CHECK(dotcRO.data()[0].real() == Catch::Approx(expected.real()).epsilon(1e-4));
                    CHECK(dotcRO.data()[0].imag() == Catch::Approx(expected.imag()).epsilon(1e-4));
                }
                else
                    CHECK(dotcRO.data()[0] == Catch::Approx(blas::dotcRef(xb.data(), yb.data(), n)).epsilon(1e-4));
            };
            runReadOnly.template operator()<float>();
            runReadOnly.template operator()<double>();
            runReadOnly.template operator()<alpaka::math::Complex<float>>();
            runReadOnly.template operator()<alpaka::math::Complex<double>>();
        }

        // Single-element reduction over a complex pair: dotc([a+bi],[c+di]) = conj(a+bi)*(c+di).
        {
            using C1 = alpaka::math::Complex<double>;
            auto xs = alpaka::onHost::allocUnified<C1>(device, 1u);
            auto ys = alpaka::onHost::allocUnified<C1>(device, 1u);
            auto dotcS = alpaka::onHost::allocUnified<C1>(device, 1u);
            xs.data()[0] = C1{3, -2};
            ys.data()[0] = C1{-1, 4};
            alpaka::blas::onHost::dotc(queue, xs, ys, dotcS, options);
            alpaka::onHost::wait(queue);
            auto const expected = blas::dotcRef(xs.data(), ys.data(), 1u);
            CHECK(dotcS.data()[0].real() == Catch::Approx(expected.real()).epsilon(1e-12));
            CHECK(dotcS.data()[0].imag() == Catch::Approx(expected.imag()).epsilon(1e-12));
        }

        // Leading offset: alpaka3 1D views are always contiguous (their reported element pitch is the element size),
        // so BLAS increments exposed through the descriptor are always 1. A view with a nonzero starting offset must
        // still honor its shifted base pointer; this guards the descriptor base-pointer forwarding used for n = 1 and
        // for the acceptance example below. Non-unit 1D strides are not expressible through alpaka 1D MdSpan views.
        // Unified-memory buffers make the same shifted views runnable on the CUDA/HIP/oneAPI device APIs, so this
        // block is not restricted to the host API anymore.
        {
            using Real = double;
            constexpr uint32_t off = 2u;
            constexpr uint32_t subN = 3u;
            constexpr uint32_t padN = off + subN;
            auto xb = alpaka::onHost::allocUnified<Real>(device, padN);
            auto yb = alpaka::onHost::allocUnified<Real>(device, padN);
            for(uint32_t i = 0; i < padN; ++i)
                xb.data()[i] = Real{-21.0};
            for(uint32_t i = 0; i < padN; ++i)
                yb.data()[i] = Real{-22.0};
            for(uint32_t i = 0; i < subN; ++i)
            {
                xb.data()[off + i] = static_cast<Real>(i + 1);
                yb.data()[off + i] = static_cast<Real>(2 * (i + 1));
            }
            auto xConst
                = alpaka::makeMdSpan(static_cast<Real const*>(xb.data() + off), alpaka::Vec<std::size_t, 1u>{subN});
            auto yConst
                = alpaka::makeMdSpan(static_cast<Real const*>(yb.data() + off), alpaka::Vec<std::size_t, 1u>{subN});
            auto dotcS = alpaka::onHost::allocUnified<Real>(device, 1u);
            alpaka::blas::onHost::dotc(queue, xConst, yConst, dotcS, options);
            auto expected = Real{0};
            for(uint32_t i = 0; i < subN; ++i)
                expected += xb.data()[off + i] * yb.data()[off + i];
            alpaka::onHost::wait(queue);
            CHECK(dotcS.data()[0] == Catch::Approx(expected).epsilon(1e-12));
        }

        // Empty reduction: for n = 0 the queued dotc must write exactly zero into the result buffer.
        {
            using C0 = alpaka::math::Complex<float>;
            auto xe = alpaka::onHost::allocUnified<C0>(device, 0u);
            auto ye = alpaka::onHost::allocUnified<C0>(device, 0u);
            auto dotcE = alpaka::onHost::allocUnified<C0>(device, 1u);
            dotcE.data()[0] = C0{7, -3};
            alpaka::blas::onHost::dotc(queue, xe, ye, dotcE, options);
            alpaka::onHost::wait(queue);
            CHECK(dotcE.data()[0].real() == 0.0f);
            CHECK(dotcE.data()[0].imag() == 0.0f);
        }

        // Acceptance example from the spec: x=[1+2i,3-i], y=[2-i,-1+4i] -> dotc=-7+6i, dot=5+16i.
        using C = alpaka::math::Complex<double>;
        constexpr uint32_t m = 2u;
        auto xa = alpaka::onHost::allocUnified<C>(device, m);
        auto ya = alpaka::onHost::allocUnified<C>(device, m);
        auto dotcA = alpaka::onHost::allocUnified<C>(device, 1u);
        auto dotA = alpaka::onHost::allocUnified<C>(device, 1u);
        xa.data()[0] = C{1, 2};
        xa.data()[1] = C{3, -1};
        ya.data()[0] = C{2, -1};
        ya.data()[1] = C{-1, 4};
        alpaka::blas::onHost::dotc(queue, xa, ya, dotcA, options);
        alpaka::blas::onHost::dot(queue, xa, ya, dotA, options);
        alpaka::onHost::wait(queue);
        CHECK(dotcA.data()[0].real() == Catch::Approx(-7.0).epsilon(1e-12));
        CHECK(dotcA.data()[0].imag() == Catch::Approx(6.0).epsilon(1e-12));
        CHECK(dotA.data()[0].real() == Catch::Approx(5.0).epsilon(1e-12));
        CHECK(dotA.data()[0].imag() == Catch::Approx(16.0).epsilon(1e-12));
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS dotc queue ordering: producer to dotc to consumer with one final wait",
    "[integr][blas][level1][dotc]",
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

        using Real = float;
        constexpr uint32_t n = 4u;
        // Producer buffer pair and consumer result on the same queue. The producer overwrites the input buffers with
        // a known pattern through a queued BLAS copy, dotc reads them, and the consumer (a plain host read after the
        // single final wait) must observe the ordered result. dotc must neither run before the producer nor be
        // silently dropped nor use a different stream: any of those would change the expected sum.
        auto xsrc = alpaka::onHost::allocUnified<Real>(device, n);
        auto ysrc = alpaka::onHost::allocUnified<Real>(device, n);
        auto x = alpaka::onHost::allocUnified<Real>(device, n);
        auto y = alpaka::onHost::allocUnified<Real>(device, n);
        auto dotcOut = alpaka::onHost::allocUnified<Real>(device, 1u);
        for(uint32_t i = 0; i < n; ++i)
        {
            x.data()[i] = Real(-1.0);
            y.data()[i] = Real(-2.0);
            xsrc.data()[i] = static_cast<Real>(i + 1);
            ysrc.data()[i] = static_cast<Real>(2 * (i + 1));
        }
        dotcOut.data()[0] = Real(-99.0);

        // Producer: two queued BLAS copies fill x and y from xsrc/ysrc on the same queue as the dotc.
        alpaka::blas::onHost::copy(queue, xsrc, x, options);
        alpaka::blas::onHost::copy(queue, ysrc, y, options);
        // dotc on the same queue.
        alpaka::blas::onHost::dotc(queue, x, y, dotcOut, options);
        // Consumer: exactly one wait at the end.
        alpaka::onHost::wait(queue);
        auto expected = Real{0};
        for(uint32_t i = 0; i < n; ++i)
            expected += xsrc.data()[i] * ysrc.data()[i];
        CHECK(dotcOut.data()[0] == Catch::Approx(expected).epsilon(1e-4f));
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS dotc oversized n rejected by checkedCast on 32-bit descriptor backends",
    "[integr][blas][level1][dotc]",
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
        // An extent larger than the 32-bit vendor int parameters (host/cuda) and the 32-bit rocblas_int (hip) used by
        // the dotc dispatches. The rejection must come from the metadata alone, before enqueue and without any data
        // access: the backing storage is a single scalar so any element access at that extent would overflow.
        constexpr std::size_t hugeN = static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) + 2u;
        auto xstorage = alpaka::onHost::allocUnified<Scalar>(device, 1u);
        auto ystorage = alpaka::onHost::allocUnified<Scalar>(device, 1u);
        auto dotcResult = alpaka::onHost::allocUnified<Scalar>(device, 1u);
        auto xbig = alpaka::makeMdSpan(xstorage.data(), alpaka::Vec<std::size_t, 1u>{hugeN});
        auto ybig = alpaka::makeMdSpan(ystorage.data(), alpaka::Vec<std::size_t, 1u>{hugeN});
        if constexpr(!std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::OneApi>)
        {
            CHECK_THROWS_AS(alpaka::blas::onHost::dotc(queue, xbig, ybig, dotcResult), std::invalid_argument);
        }
        else
        {
            // Rejection is claimed only where the vendor API limits the descriptor integer width. oneMKL takes 64-bit
            // descriptor integers, so the same oversized extent is representable there and no checkedCast triggers;
            // the assertion below exercises exactly that metadata path. The reduction is intentionally not enqueued:
            // executing a dotc over hugeN elements backed by a single-element buffer would read out of bounds.
            auto const xDesc = alpaka::blas::internal::makeVectorDescriptor(xbig);
            auto const yDesc = alpaka::blas::internal::makeVectorDescriptor(ybig);
            CHECK(xDesc.n == static_cast<std::int64_t>(hugeN));
            CHECK(yDesc.n == static_cast<std::int64_t>(hugeN));
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "BLAS iamax 1-based conversion honors blocking queues",
    "[integr][blas][level1][blocking]",
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
        // A blocking queue guarantees that the 1-based conversion kernel is complete when the enqueue call
        // returns, so the result is host-visible without an explicit wait().
        auto queue = device.makeQueue(alpaka::queueKind::blocking);
        constexpr uint32_t n = 5u;

        auto x = alpaka::onHost::allocUnified<float>(device, n);
        auto iamaxResult = alpaka::onHost::allocUnified<int>(device, 1u);

        for(uint32_t i = 0; i < n; ++i)
            x.data()[i] = static_cast<float>(i + 1);

        alpaka::blas::onHost::iamax(queue, x, iamaxResult);
        // intentionally no alpaka::onHost::wait(queue): the blocking queue semantics must already guarantee it.
        CHECK(iamaxResult.data()[0] == blas::iamaxRef(x.data(), n));
    }
}
