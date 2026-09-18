/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <algorithm>
#include <alpakaTest/deviceHelper.hpp>
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
        {
            auto runReadOnly = [&]<typename T>()
            {
                // MdSpan const-ness is a property of the view type, so a const buffer is sufficient; the data region
                // is written through the buffer's non-const data() before the const view is created.
                auto xb = alpaka::onHost::allocUnified<T>(device, n);
                auto yb = alpaka::onHost::allocUnified<T>(device, n);
                fillVector(xb.data(), n);
                fillVector(yb.data(), n);
                if constexpr(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::Host>)
                {
                    auto xConst
                        = alpaka::makeMdSpan(static_cast<T const*>(xb.data()), alpaka::Vec<std::size_t, 1u>{n});
                    auto yConst
                        = alpaka::makeMdSpan(static_cast<T const*>(yb.data()), alpaka::Vec<std::size_t, 1u>{n});
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
                }
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
        if constexpr(std::same_as<ALPAKA_TYPEOF(device.getApi()), alpaka::api::Host>)
        {
            using Real = double;
            constexpr uint32_t off = 2u;
            constexpr uint32_t subN = 3u;
            auto hostX = std::vector<Real>(off + subN, Real{-21.0});
            auto hostY = std::vector<Real>(off + subN, Real{-22.0});
            for(uint32_t i = 0; i < subN; ++i)
            {
                hostX[off + i] = static_cast<Real>(i + 1);
                hostY[off + i] = static_cast<Real>(2 * (i + 1));
            }
            auto xConst
                = alpaka::makeMdSpan(static_cast<Real const*>(hostX.data() + off), alpaka::Vec<std::size_t, 1u>{subN});
            auto yConst
                = alpaka::makeMdSpan(static_cast<Real const*>(hostY.data() + off), alpaka::Vec<std::size_t, 1u>{subN});
            auto dotcS = alpaka::onHost::allocUnified<Real>(device, 1u);
            alpaka::blas::onHost::dotc(queue, xConst, yConst, dotcS, options);
            auto expected = Real{0};
            for(uint32_t i = 0; i < subN; ++i)
                expected += hostX[off + i] * hostY[off + i];
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
