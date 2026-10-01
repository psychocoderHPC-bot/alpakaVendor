/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * Batched transforms: Batched R2C example
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>
#include <type_traits>

//! [batched-r2c]
using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Batched R2C transform", "[doc][batched][r2c]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        SKIP("No device available for selected backend.");
    auto device = devSelector.makeDevice(0);

    using Complex = alpaka::math::Complex<float>;
    using namespace alpaka::fft;

    auto queue = device.makeQueue();

    constexpr uint32_t n = 8u;
    constexpr uint32_t batchSize = 4u;
    constexpr uint32_t complexN = r2cComplexExtent(n);

    auto in = alpaka::fft::onHost::allocUnified<float>(device, batchSize * n);
    auto out = alpaka::fft::onHost::allocUnified<Complex>(device, batchSize * complexN);

    for(std::size_t b = 0; b < batchSize; ++b)
        for(std::size_t i = 0; i < n; ++i)
        {
            float freq = float(b + 1);
            in.data()[b * n + i] = std::cos(2.0f * std::numbers::pi_v<float> * freq * float(i) / float(n));
        }

    auto plan = alpaka::fft::onHost::makePlan<float>(n)
                    .r2c()
                    .batch(batchSize)
                    .distances(n * sizeof(float), complexN * sizeof(Complex))
                    .build(device);

    alpaka::fft::onHost::executeForward(queue, plan, in, out);
    alpaka::onHost::wait(queue);

    // Verify that DC component is real-valued for all batches
    for(std::size_t b = 0; b < batchSize; ++b)
    {
        CHECK(out.data()[b * complexN].imag() == Catch::Approx(0.0f).margin(1.0e-4));
    }
    //! [batched-r2c]
}
