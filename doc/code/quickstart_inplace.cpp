/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * Quickstart: In-place R2C/C2R roundtrip (backend-agnostic)
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>
#include <type_traits>

//! [quickstart-inplace-backends]
using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Quickstart: In-place R2C/C2R roundtrip", "[doc][quickstart][inplace]", TestBackends)
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
    //! [quickstart-inplace-backends]

    //! [quickstart-inplace-core]
    auto storage = makeInPlaceRealStorage<float>(n);

    auto buffer = alpaka::fft::onHost::allocUnified<float>(device, n);

    for(std::size_t i = 0; i < n; ++i)
        buffer.data()[i] = std::sin(2.0f * std::numbers::pi_v<float> * float(i) / float(n));
    for(std::size_t i = n; i < storage.physicalRealElements; ++i)
        buffer.data()[i] = 0.0f;

    auto r2cPlan = alpaka::fft::onHost::makePlan<float>(n).r2c().inPlace().build(device);

    auto complexBuffer = alpaka::fft::onHost::executeR2CInPlace(queue, r2cPlan, buffer);
    alpaka::onHost::wait(queue);

    CHECK(complexBuffer.getExtents()[0] == storage.logicalComplexElements);

    auto c2rPlan = alpaka::fft::onHost::makePlan<float>(n).c2r().inPlace().build(device);

    auto recovered = alpaka::fft::onHost::executeC2RInPlace(queue, c2rPlan, complexBuffer);
    alpaka::onHost::wait(queue);

    CHECK(recovered.getExtents() == alpaka::fft::Extents<uint32_t, 1u>{n});
    //! [quickstart-inplace-core]
}
