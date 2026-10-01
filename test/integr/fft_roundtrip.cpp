/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>
#include <numbers>

#include "../unit/test.hpp"
#include "alpaka/fft.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("FFT C2C roundtrip 1D", "[integr][fft][c2c]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto queue = device.makeQueue();
        constexpr uint32_t n = 8u;
        auto extents = n;

        auto in = alpaka::fft::onHost::allocUnified<Complex>(device, extents);
        auto tmp = alpaka::fft::onHost::allocUnified<Complex>(device, extents);
        auto out = alpaka::fft::onHost::allocUnified<Complex>(device, extents);

        for(uint32_t i = 0; i < n; ++i)
            in.data()[i] = Complex{float(static_cast<int>(i) - 3), float(static_cast<int>(i % 3u) - 1)};

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}.c2c().build(device);
        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        for(uint32_t i = 0; i < n; ++i)
        {
            CHECK(
                out.data()[i].real() == Catch::Approx(in.data()[i].real() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
            CHECK(
                out.data()[i].imag() == Catch::Approx(in.data()[i].imag() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "FFT C2C roundtrip 1D survives plan destruction before wait",
    "[integr][fft][c2c][lifetime]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto queue = device.makeQueue();
        constexpr uint32_t n = 8u;
        auto extents = n;

        auto in = alpaka::fft::onHost::allocUnified<Complex>(device, extents);
        auto tmp = alpaka::fft::onHost::allocUnified<Complex>(device, extents);
        auto out = alpaka::fft::onHost::allocUnified<Complex>(device, extents);

        for(uint32_t i = 0; i < n; ++i)
            in.data()[i] = Complex{float(static_cast<int>(i) - 3), float(static_cast<int>(i % 3u) - 1)};

        {
            auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}.c2c().build(device);
            alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
            alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        }
        alpaka::onHost::wait(queue);

        for(uint32_t i = 0; i < n; ++i)
        {
            CHECK(
                out.data()[i].real() == Catch::Approx(in.data()[i].real() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
            CHECK(
                out.data()[i].imag() == Catch::Approx(in.data()[i].imag() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("FFT C2C roundtrip 1D accepts plain alpaka buffers", "[integr][fft][c2c][plain]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto queue = device.makeQueue();
        constexpr uint32_t n = 8u;
        auto extents = n;

        auto in = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto tmp = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto out = alpaka::onHost::allocUnified<Complex>(device, extents);

        for(uint32_t i = 0; i < n; ++i)
            in.data()[i] = Complex{float(static_cast<int>(i) - 3), float(static_cast<int>(i % 3u) - 1)};

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}.c2c().build(device);
        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        for(uint32_t i = 0; i < n; ++i)
        {
            CHECK(
                out.data()[i].real() == Catch::Approx(in.data()[i].real() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
            CHECK(
                out.data()[i].imag() == Catch::Approx(in.data()[i].imag() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
        }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "FFT R2C/C2R out-of-place accepts plain alpaka buffers",
    "[integr][fft][r2c][c2r][plain]",
    TestBackends)
{
    using namespace alpaka::fft;

    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        constexpr uint32_t n = 8u;
        constexpr uint32_t complexN = r2cComplexExtent(n);

        auto in = alpaka::onHost::allocUnified<float>(device, n);
        auto spectrum = alpaka::onHost::allocUnified<alpaka::math::Complex<float>>(device, complexN);
        auto out = alpaka::onHost::allocUnified<float>(device, n);

        for(uint32_t i = 0; i < n; ++i)
            in.data()[i] = float(i + 1u);

        auto r2cPlan = alpaka::fft::onHost::PlanBuilder<float>{n}.r2c().outOfPlace().build(device);
        alpaka::fft::onHost::executeForward(queue, r2cPlan, in, spectrum);

        auto c2rPlan = alpaka::fft::onHost::PlanBuilder<float>{n}.c2r().outOfPlace().build(device);
        alpaka::fft::onHost::executeBackward(queue, c2rPlan, spectrum, out);
        alpaka::onHost::wait(queue);

        for(uint32_t i = 0; i < n; ++i)
            CHECK(out.data()[i] == Catch::Approx(float(i + 1u) * float(n)).epsilon(1.0e-4));
    }
}

TEMPLATE_LIST_TEST_CASE("FFT R2C/C2R in place 1D", "[integr][fft][r2c][c2r]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;

        auto queue = device.makeQueue();
        constexpr uint32_t n = 8u;
        auto storage = makeInPlaceRealStorage<float>(n);
        auto realBuffer = alpaka::fft::onHost::allocUnified<float>(device, n);

        for(uint32_t i = 0; i < n; ++i)
            realBuffer.data()[i] = float(i + 1u);
        for(uint32_t i = n; i < storage.physicalRealElements; ++i)
            realBuffer.data()[i] = 0.0f;

        auto r2cPlan = alpaka::fft::onHost::PlanBuilder<float>{n}.r2c().inPlace().build(device);
        auto complexBuffer = alpaka::fft::onHost::executeR2CInPlace(queue, r2cPlan, realBuffer);
        CHECK(complexBuffer.getExtents() == storage.logicalComplexExtents);

        auto c2rPlan = alpaka::fft::onHost::PlanBuilder<float>{n}.c2r().inPlace().build(device);
        auto recovered = alpaka::fft::onHost::executeC2RInPlace(queue, c2rPlan, complexBuffer);
        alpaka::onHost::wait(queue);
        CHECK(recovered.getExtents() == alpaka::fft::Extents<uint32_t, 1u>{n});
        for(uint32_t i = 0; i < n; ++i)
            CHECK(recovered.data()[i] == Catch::Approx(float(i + 1u) * float(n)).epsilon(1.0e-4));
    }
}

TEMPLATE_LIST_TEST_CASE("FFT plan can switch queues", "[integr][fft][c2c][queue]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using Complex = alpaka::math::Complex<float>;

        auto queue0 = device.makeQueue();
        auto queue1 = device.makeQueue();
        constexpr uint32_t n = 8u;
        auto extents = alpaka::fft::Extents<uint32_t, 1u>{n};
        auto in = alpaka::fft::onHost::allocUnified<Complex>(device, extents);
        auto tmp = alpaka::fft::onHost::allocUnified<Complex>(device, extents);
        auto out = alpaka::fft::onHost::allocUnified<Complex>(device, extents);

        for(uint32_t i = 0; i < n; ++i)
            in.data()[i] = Complex{float(i + 1u), float(static_cast<int>(i) - 2)};

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}.c2c().build(device);
        alpaka::fft::onHost::executeForward(queue0, plan, in, tmp);
        alpaka::onHost::wait(queue0);
        alpaka::fft::onHost::executeBackward(queue1, plan, tmp, out);
        alpaka::onHost::wait(queue1);

        for(uint32_t i = 0; i < n; ++i)
        {
            CHECK(
                out.data()[i].real() == Catch::Approx(in.data()[i].real() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
            CHECK(
                out.data()[i].imag() == Catch::Approx(in.data()[i].imag() * float(n)).epsilon(1.0e-4).margin(1.0e-5));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("FFT plan keepAlive survives scope", "[integr][fft][plan]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using Complex = alpaka::math::Complex<float>;

        auto queue = device.makeQueue();
        constexpr uint32_t n = 8u;
        auto extents = alpaka::fft::Extents<uint32_t, 1u>{n};
        auto in = alpaka::fft::onHost::allocUnified<Complex>(device, extents);
        auto tmp = alpaka::fft::onHost::allocUnified<Complex>(device, extents);
        auto out = alpaka::fft::onHost::allocUnified<Complex>(device, extents);

        for(uint32_t i = 0; i < n; ++i)
            in.data()[i] = Complex{float(i + 1u), float(2u * i + 1u)};

        {
            auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}.c2c().build(device);
            alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
            alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
            plan.keepAlive(queue);
        }
        alpaka::onHost::wait(queue);

        for(uint32_t i = 0; i < n; ++i)
        {
            CHECK(out.data()[i].real() == Catch::Approx(float(i + 1u) * float(n)).epsilon(1.0e-4));
            CHECK(out.data()[i].imag() == Catch::Approx(float(2u * i + 1u) * float(n)).epsilon(1.0e-4));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("FFT plan keepAlive after scoped execution", "[integr][fft][plan]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using Complex = alpaka::math::Complex<float>;

        auto queue = device.makeQueue();
        constexpr uint32_t n = 8u;
        auto extents = alpaka::fft::Extents<uint32_t, 1u>{n};
        auto in = alpaka::fft::onHost::allocUnified<Complex>(device, extents);
        auto out = alpaka::fft::onHost::allocUnified<Complex>(device, extents);

        in.data()[0] = Complex{1.0f, 0.0f};
        for(uint32_t i = 1; i < n; ++i)
            in.data()[i] = Complex{0.0f, 0.0f};

        {
            auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}.c2c().build(device);
            alpaka::fft::onHost::executeForward(queue, plan, in, out);
            plan.keepAlive(queue);
        }

        alpaka::onHost::wait(queue);

        for(uint32_t i = 0; i < n; ++i)
        {
            CHECK(out.data()[i].real() == Catch::Approx(1.0f).epsilon(1.0e-4));
            CHECK(out.data()[i].imag() == Catch::Approx(0.0f).margin(1.0e-4));
        }
    }
}
