/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * Batched transforms: N-D batched examples using alpaka pitches
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <type_traits>

using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

//! [batched-2d-from-3d]
TEMPLATE_LIST_TEST_CASE("Batched 2D C2C transform from a 3D alpaka buffer", "[doc][batched][nd][2d]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        SKIP("No device available for selected backend.");
    auto device = devSelector.makeDevice(0);

    using Complex = alpaka::math::Complex<float>;
    using Extents2D = alpaka::fft::Extents<uint32_t, 2u>;
    using Extents3D = alpaka::fft::Extents<uint32_t, 3u>;
    using namespace alpaka::fft;

    auto queue = device.makeQueue();

    constexpr uint32_t batchSize = 3u;
    constexpr uint32_t ny = 2u;
    constexpr uint32_t nx = 4u; // x is last => fast-moving alpaka dimension
    auto const extents = Extents3D{batchSize, ny, nx};

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

    auto plan = alpaka::fft::onHost::makePlan<Complex, Extents2D>(Extents2D{ny, nx})
                    .c2c()
                    .batch(batchSize)
                    .strides(
                        alpaka::pCast<size_t>(inPitches.template rshrink<2u>()),
                        alpaka::pCast<size_t>(outPitches.template rshrink<2u>()))
                    .distances(inPitches[0], outPitches[0])
                    .build(device);

    alpaka::fft::onHost::executeForward(queue, plan, in, tmp);
    alpaka::fft::onHost::executeBackward(queue, plan, tmp, out);
    alpaka::onHost::wait(queue);

    constexpr float fftSize = float(ny * nx);
    alpaka::meta::ndLoopIncIdx(
        extents,
        [&](auto const& idx)
        {
            auto const linear = alpaka::linearize(extents, idx);
            CHECK(out[idx].real() == Catch::Approx(float(linear + 1u) * fftSize).epsilon(1.0e-4));
            CHECK(out[idx].imag() == Catch::Approx(float(2u * linear + 1u) * fftSize).epsilon(1.0e-4));
        });
}

//! [batched-2d-from-3d]

//! [batched-3d-from-4d]
TEMPLATE_LIST_TEST_CASE("Batched 3D C2C transform from a 4D alpaka buffer", "[doc][batched][nd][3d]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        SKIP("No device available for selected backend.");
    auto device = devSelector.makeDevice(0);

    using Complex = alpaka::math::Complex<float>;
    using Extents3D = alpaka::fft::Extents<uint32_t, 3u>;
    using Extents4D = alpaka::fft::Extents<uint32_t, 4u>;
    using namespace alpaka::fft;

    auto queue = device.makeQueue();

    constexpr uint32_t batchSize = 3u;
    constexpr uint32_t nz = 2u;
    constexpr uint32_t ny = 3u;
    constexpr uint32_t nx = 4u; // x is last => fast-moving alpaka dimension
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

    auto plan = alpaka::fft::onHost::makePlan<Complex, Extents3D>(Extents3D{nz, ny, nx})
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

//! [batched-3d-from-4d]
