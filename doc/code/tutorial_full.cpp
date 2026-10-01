/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * Tutorial: Complete FFT example with multiple transform types
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>
#include <type_traits>

//! [tutorial-setup]
using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Tutorial: Complete FFT example", "[doc][tutorial][full]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        SKIP("No device available for selected backend.");
    auto device = devSelector.makeDevice(0);

    using Complex = alpaka::math::Complex<float>;
    using Extents2D = alpaka::fft::Extents<uint32_t, 2u>;

    // Create device and queue
    auto queue = device.makeQueue();
    //! [tutorial-setup]

    // =====================================================================
    // Example 1: 1D C2C out-of-place transform
    // =====================================================================

    //! [tutorial-c2c]
    constexpr uint32_t n1d = 16u;
    auto in1d = alpaka::fft::onHost::allocUnified<Complex>(device, n1d);
    auto out1d = alpaka::fft::onHost::allocUnified<Complex>(device, n1d);

    // Initialize: sum of two sinusoids
    for(std::size_t i = 0; i < n1d; ++i)
    {
        float t = float(i) / float(n1d);
        in1d.data()[i] = Complex{
            std::cos(2.0f * std::numbers::pi_v<float> * t) + 0.5f * std::cos(4.0f * std::numbers::pi_v<float> * t),
            0.0f};
    }

    auto plan1d = alpaka::fft::onHost::makePlan<Complex>(n1d).c2c().build(device);

    alpaka::fft::onHost::executeForward(queue, plan1d, in1d, out1d);
    alpaka::onHost::wait(queue);

    // Verify forward transform produces non-zero output
    bool hasNonZero = false;
    for(std::size_t i = 0; i < n1d; ++i)
    {
        if(std::abs(out1d.data()[i].real()) > 1.0e-6f || std::abs(out1d.data()[i].imag()) > 1.0e-6f)
        {
            hasNonZero = true;
            break;
        }
    }
    CHECK(hasNonZero);
    //! [tutorial-c2c]

    // =====================================================================
    // Example 2: 2D C2C out-of-place transform
    // =====================================================================

    //! [tutorial-2d]
    constexpr uint32_t nx = 4u;
    constexpr uint32_t ny = 4u;
    auto in2d = alpaka::fft::onHost::allocUnified<Complex>(device, Extents2D{nx, ny});
    auto out2d = alpaka::fft::onHost::allocUnified<Complex>(device, Extents2D{nx, ny});

    // Initialize: 2D Gaussian
    for(std::size_t ix = 0; ix < nx; ++ix)
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            float x = float(ix) - float(nx) / 2.0f;
            float y = float(iy) - float(ny) / 2.0f;
            in2d.data()[ix * ny + iy] = Complex{std::exp(-(x * x + y * y) / 4.0f), 0.0f};
        }

    auto plan2d = alpaka::fft::onHost::makePlan<Complex, Extents2D>(Extents2D{nx, ny}).c2c().build(device);

    alpaka::fft::onHost::executeForward(queue, plan2d, in2d, out2d);
    alpaka::onHost::wait(queue);

    // Verify 2D FFT produces non-zero output
    hasNonZero = false;
    for(std::size_t ix = 0; ix < nx; ++ix)
        for(std::size_t iy = 0; iy < ny; ++iy)
        {
            if(std::abs(out2d.data()[ix * ny + iy].real()) > 1.0e-6f
               || std::abs(out2d.data()[ix * ny + iy].imag()) > 1.0e-6f)
            {
                hasNonZero = true;
                break;
            }
        }
    CHECK(hasNonZero);
    //! [tutorial-2d]

    // =====================================================================
    // Example 3: 1D R2C/C2R in-place roundtrip
    // =====================================================================

    //! [tutorial-inplace]
    constexpr uint32_t n = 8u;
    auto storage = alpaka::fft::makeInPlaceRealStorage<float>(n);
    auto buffer = alpaka::fft::onHost::allocUnified<float>(device, n);

    // Initialize: square wave
    for(std::size_t i = 0; i < n; ++i)
        buffer.data()[i] = (i < n / 2) ? 1.0f : -1.0f;
    for(std::size_t i = n; i < storage.physicalRealElements; ++i)
        buffer.data()[i] = 0.0f;

    auto r2cPlan = alpaka::fft::onHost::makePlan<float>(n).r2c().inPlace().build(device);

    auto complexBuffer = alpaka::fft::onHost::executeR2CInPlace(queue, r2cPlan, buffer);
    alpaka::onHost::wait(queue);

    auto c2rPlan = alpaka::fft::onHost::makePlan<float>(n).c2r().inPlace().build(device);

    auto recovered = alpaka::fft::onHost::executeC2RInPlace(queue, c2rPlan, complexBuffer);
    alpaka::onHost::wait(queue);

    // Verify roundtrip (result = n * original)
    for(std::size_t i = 0; i < n; ++i)
    {
        float expected = ((i < n / 2) ? 1.0f : -1.0f) * float(n);
        CHECK(recovered.data()[i] == Catch::Approx(expected).epsilon(1.0e-4));
    }
    //! [tutorial-inplace]
}
