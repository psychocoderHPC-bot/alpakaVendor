/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * Batched transforms: Batched C2C example
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <type_traits>


//! [batched-c2c]
using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Batched C2C transform", "[doc][batched][c2c]", TestBackends)
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

    auto in = alpaka::fft::onHost::allocUnified<Complex>(device, batchSize * n);
    auto out = alpaka::fft::onHost::allocUnified<Complex>(device, batchSize * n);

    for(std::size_t b = 0; b < batchSize; ++b)
        for(std::size_t i = 0; i < n; ++i)
            in.data()[b * n + i] = (i == 0u) ? Complex{1.0f, 0.0f} : Complex{0.0f, 0.0f};

    auto plan = alpaka::fft::onHost::makePlan<Complex>(n)
                    .c2c()
                    .batch(batchSize)
                    .distances(n * sizeof(Complex), n * sizeof(Complex))
                    .build(device);

    alpaka::fft::onHost::executeForward(queue, plan, in, out);
    alpaka::onHost::wait(queue);

    // Verify forward transform: batch 0 should have delta at index 0
    CHECK(out.data()[0].real() == Catch::Approx(1.0f).epsilon(1.0e-4));
    CHECK(out.data()[0].imag() == Catch::Approx(0.0f).margin(1.0e-4));

    // Execute backward transform
    alpaka::fft::onHost::executeBackward(queue, plan, out, in);
    alpaka::onHost::wait(queue);

    // Verify roundtrip: result = n * original for batch 0
    CHECK(in.data()[0].real() == Catch::Approx(float(n)).epsilon(1.0e-4));
    CHECK(in.data()[0].imag() == Catch::Approx(0.0f).margin(1.0e-4));
    //! [batched-c2c]
}
