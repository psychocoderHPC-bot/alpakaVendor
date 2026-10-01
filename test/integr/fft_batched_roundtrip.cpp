/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>

#include "../unit/test.hpp"
#include "alpaka/fft.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("FFT batched R2C/C2R out-of-place roundtrip", "[integr][fft][batched][r2c][c2r]", TestBackends)
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
        constexpr uint32_t batchSize = 4u;
        constexpr uint32_t complexN = r2cComplexExtent(n);

        auto in = alpaka::fft::onHost::allocUnified<float>(device, batchSize * n);
        auto spectrum = alpaka::fft::onHost::allocUnified<alpaka::math::Complex<float>>(device, batchSize * complexN);
        auto out = alpaka::fft::onHost::allocUnified<float>(device, batchSize * n);

        for(uint32_t b = 0; b < batchSize; ++b)
            for(uint32_t i = 0; i < n; ++i)
                in.data()[b * n + i] = float(b * n + i + 1u);

        auto r2cPlan = alpaka::fft::onHost::PlanBuilder<float>{n}.r2c().outOfPlace().batch(batchSize).build(device);
        alpaka::fft::onHost::executeForward(queue, r2cPlan, in, spectrum);

        auto c2rPlan = alpaka::fft::onHost::PlanBuilder<float>{n}.c2r().outOfPlace().batch(batchSize).build(device);
        alpaka::fft::onHost::executeBackward(queue, c2rPlan, spectrum, out);
        alpaka::onHost::wait(queue);

        for(uint32_t b = 0; b < batchSize; ++b)
            for(uint32_t i = 0; i < n; ++i)
                CHECK(out.data()[b * n + i] == Catch::Approx(float(b * n + i + 1u) * float(n)).epsilon(1.0e-4));
    }
}

TEMPLATE_LIST_TEST_CASE("FFT batched C2C roundtrip", "[integr][fft][batched][c2c]", TestBackends)
{
    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;

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
        constexpr uint32_t batchSize = 4u;

        auto in = alpaka::fft::onHost::allocUnified<Complex>(device, batchSize * n);
        auto tmp = alpaka::fft::onHost::allocUnified<Complex>(device, batchSize * n);
        auto out = alpaka::fft::onHost::allocUnified<Complex>(device, batchSize * n);

        for(uint32_t b = 0; b < batchSize; ++b)
            for(uint32_t i = 0; i < n; ++i)
                in.data()[b * n + i] = Complex{float(b * n + i + 1u), float(2u * (b * n + i) + 1u)};

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}
                        .c2c()
                        .batch(batchSize)
                        .distances(n * sizeof(Complex), n * sizeof(Complex))
                        .build(device);
        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        for(uint32_t b = 0; b < batchSize; ++b)
            for(uint32_t i = 0; i < n; ++i)
            {
                CHECK(out.data()[b * n + i].real() == Catch::Approx(float(b * n + i + 1u) * float(n)).epsilon(1.0e-4));
                CHECK(
                    out.data()[b * n + i].imag()
                    == Catch::Approx(float(2u * (b * n + i) + 1u) * float(n)).epsilon(1.0e-4));
            }
    }
}

TEMPLATE_LIST_TEST_CASE(
    "FFT batched 1D C2C can use a 2D alpaka buffer",
    "[integr][fft][batched][c2c][2d-buffer]",
    TestBackends)
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
        constexpr uint32_t batchSize = 4u;
        constexpr uint32_t n = 8u;
        auto const extents = Extents2D{batchSize, n};

        auto in = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto tmp = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto out = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto const inPitches = in.getPitches();
        auto const outPitches = out.getPitches();
        alpaka::meta::ndLoopIncIdx(
            extents,
            [&](auto const& idx)
            {
                auto const linear = alpaka::linearize(extents, idx);
                in[idx] = Complex{float(linear + 1u), float(2u * linear + 1u)};
            });
        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}
                        .c2c()
                        .batch(batchSize)
                        .strides(inPitches[1], outPitches[1])
                        .distances(inPitches[0], outPitches[0])
                        .build(device);

        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        alpaka::meta::ndLoopIncIdx(
            extents,
            [&](auto const& idx)
            {
                auto const linear = alpaka::linearize(extents, idx);
                CHECK(out[idx].real() == Catch::Approx(float(linear + 1u) * float(n)).epsilon(1.0e-4));
                CHECK(out[idx].imag() == Catch::Approx(float(2u * linear + 1u) * float(n)).epsilon(1.0e-4));
            });
    }
}

TEMPLATE_LIST_TEST_CASE(
    "FFT batched 3D C2C can use a 4D alpaka buffer",
    "[integr][fft][batched][c2c][4d-buffer]",
    TestBackends)
{
    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;
    using Extents3D = Extents<uint32_t, 3u>;
    using Extents4D = Extents<uint32_t, 4u>;

    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        constexpr uint32_t batchSize = 3u;
        constexpr uint32_t nz = 2u;
        constexpr uint32_t ny = 3u;
        constexpr uint32_t nx = 4u; // x is the fast-moving alpaka dimension: last component
        auto const extents = Extents4D{batchSize, nz, ny, nx};

        auto in = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto tmp = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto out = alpaka::onHost::allocUnified<Complex>(device, extents);
        auto const inPitches = in.getPitches();
        auto const outPitches = out.getPitches();

        alpaka::meta::ndLoopIncIdx(
            extents,
            [&](auto const& idx)
            {
                auto const linear = alpaka::linearize(extents, idx);
                in[idx] = Complex{float(linear + 1u), float(2u * linear + 1u)};
            });
        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, Extents3D>(Extents3D{nz, ny, nx})
                        .c2c()
                        .batch(batchSize)
                        .strides(
                            alpaka::pCast<size_t>(inPitches.template rshrink<3u>()),
                            alpaka::pCast<size_t>(outPitches.template rshrink<3u>()))
                        .distances(inPitches[0], outPitches[0])
                        .build(device);

        alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
        alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
        alpaka::onHost::wait(queue);

        constexpr float fftSize = float(nz * ny * nx);
        alpaka::meta::ndLoopIncIdx(
            extents,
            [&](auto const& idx)
            {
                auto const linear = alpaka::linearize(extents, idx);
                CHECK(out[idx].real() == Catch::Approx(float(linear + 1u) * fftSize).epsilon(1.0e-4));
                CHECK(out[idx].imag() == Catch::Approx(float(2u * linear + 1u) * fftSize).epsilon(1.0e-4));
            });
    }
}
