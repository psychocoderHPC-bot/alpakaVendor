/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * Quickstart: Basic C2C transform (backend-agnostic)
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <type_traits>


//! [quickstart-c2c-backends]
using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Quickstart: Basic C2C transform", "[doc][quickstart][c2c]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        SKIP("No device available for selected backend.");
    auto device = devSelector.makeDevice(0);

    using Complex = alpaka::math::Complex<float>;
    constexpr uint32_t n = 8u;

    auto queue = device.makeQueue();
    //! [quickstart-c2c-backends]

    //! [quickstart-c2c-core]
    auto in = alpaka::fft::onHost::allocUnified<Complex>(device, n);
    auto out = alpaka::fft::onHost::allocUnified<Complex>(device, n);

    for(std::size_t i = 0; i < n; ++i)
        in.data()[i] = (i == 0u) ? Complex{1.0f, 0.0f} : Complex{0.0f, 0.0f};

    auto plan = alpaka::fft::onHost::makePlan<Complex>(n).c2c().outOfPlace().build(device);

    alpaka::fft::onHost::executeForward(queue, plan, in, out);
    alpaka::fft::onHost::executeBackward(queue, plan, out, in);
    alpaka::onHost::wait(queue);

    CHECK(in.data()[0].real() == Catch::Approx(float(n)).epsilon(1.0e-4));
    CHECK(in.data()[0].imag() == Catch::Approx(0.0f).margin(1.0e-4));
    //! [quickstart-c2c-core]
}
