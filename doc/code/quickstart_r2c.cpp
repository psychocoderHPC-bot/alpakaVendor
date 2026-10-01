/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * Quickstart: Basic R2C transform (backend-agnostic)
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>
#include <type_traits>

//! [quickstart-r2c-backends]
using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Quickstart: Basic R2C transform", "[doc][quickstart][r2c]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        SKIP("No device available for selected backend.");
    auto device = devSelector.makeDevice(0);

    using namespace alpaka::fft;
    constexpr uint32_t n = 8u;

    auto queue = device.makeQueue();
    //! [quickstart-r2c-backends]

    //! [quickstart-r2c-core]
    auto in = alpaka::fft::onHost::allocUnified<float>(device, n);
    auto out
        = alpaka::fft::onHost::allocUnified<alpaka::math::Complex<float>>(device, alpaka::fft::r2cComplexExtent(n));

    for(std::size_t i = 0; i < n; ++i)
        in.data()[i] = std::cos(2.0f * std::numbers::pi_v<float> * float(i) / float(n));

    auto plan = alpaka::fft::onHost::makePlan<float>(n).r2c().outOfPlace().build(device);

    alpaka::fft::onHost::executeForward(queue, plan, in, out);
    alpaka::onHost::wait(queue);

    CHECK(out.data()[0].imag() == Catch::Approx(0.0f).margin(1.0e-4));
    //! [quickstart-r2c-core]
}
