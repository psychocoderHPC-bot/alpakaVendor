/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>
#include <numbers>

#include "../unit/test.hpp"
#include "alpaka/fft.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("FFT 2D C2C out-of-place roundtrip", "[integr][fft][2d][c2c]", TestBackends)
{
    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;
    using Extents2D = Extents<uint32_t, 2u>;

    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        constexpr uint32_t nx = 4u;
        constexpr uint32_t ny = 4u;

        auto in = alpaka::fft::onHost::allocUnified<Complex>(device, Extents2D{nx, ny});
        auto tmp = alpaka::fft::onHost::allocUnified<Complex>(device, Extents2D{nx, ny});
        auto out = alpaka::fft::onHost::allocUnified<Complex>(device, Extents2D{nx, ny});

        for(uint32_t ix = 0; ix < nx; ++ix)
            for(uint32_t iy = 0; iy < ny; ++iy)
                in.data()[ix * ny + iy] = Complex{float(ix * ny + iy + 1u), float(2u * (ix * ny + iy) + 1u)};

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, Extents2D>(Extents2D{nx, ny}).c2c().build(device);
        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        for(uint32_t ix = 0; ix < nx; ++ix)
            for(uint32_t iy = 0; iy < ny; ++iy)
            {
                CHECK(
                    out.data()[ix * ny + iy].real()
                    == Catch::Approx(float(ix * ny + iy + 1u) * float(nx * ny)).epsilon(1.0e-4));
                CHECK(
                    out.data()[ix * ny + iy].imag()
                    == Catch::Approx(float(2u * (ix * ny + iy) + 1u) * float(nx * ny)).epsilon(1.0e-4));
            }
    }
}

TEMPLATE_LIST_TEST_CASE("FFT 2D R2C/C2R out-of-place roundtrip", "[integr][fft][2d][r2c][c2r]", TestBackends)
{
    using namespace alpaka::fft;
    using Extents2D = Extents<uint32_t, 2u>;

    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        constexpr uint32_t nx = 4u;
        constexpr uint32_t ny = 8u;

        auto in = alpaka::fft::onHost::allocUnified<float>(device, Extents2D{nx, ny});
        auto complexExtents = r2cComplexExtent(Extents2D{nx, ny});
        auto spectrum = alpaka::fft::onHost::allocUnified<alpaka::math::Complex<float>>(device, complexExtents);
        auto out = alpaka::fft::onHost::allocUnified<float>(device, Extents2D{nx, ny});

        for(uint32_t ix = 0; ix < nx; ++ix)
            for(uint32_t iy = 0; iy < ny; ++iy)
                in.data()[ix * ny + iy] = float(ix * ny + iy + 1u);

        auto r2cPlan = alpaka::fft::onHost::PlanBuilder<float, Extents2D>(Extents2D{nx, ny}).r2c().build(device);
        alpaka::fft::onHost::executeForward(queue, r2cPlan, in, spectrum);

        auto c2rPlan = alpaka::fft::onHost::PlanBuilder<float, Extents2D>(Extents2D{nx, ny}).c2r().build(device);
        alpaka::fft::onHost::executeBackward(queue, c2rPlan, spectrum, out);
        alpaka::onHost::wait(queue);

        for(uint32_t ix = 0; ix < nx; ++ix)
            for(uint32_t iy = 0; iy < ny; ++iy)
                CHECK(
                    out.data()[ix * ny + iy]
                    == Catch::Approx(float(ix * ny + iy + 1u) * float(nx * ny)).epsilon(1.0e-4));
    }
}
