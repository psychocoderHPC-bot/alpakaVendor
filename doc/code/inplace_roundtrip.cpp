/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * In-place real transforms: Complete R2C/C2R roundtrip
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>
#include <type_traits>

//! [inplace-roundtrip]
using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("In-place R2C/C2R roundtrip", "[doc][inplace][roundtrip]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        SKIP("No device available for selected backend.");
    auto device = devSelector.makeDevice(0);

    using namespace alpaka::fft;

    auto queue = device.makeQueue();

    constexpr uint32_t n = 16u;

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

    CHECK(recovered.getExtents()[0] == n);
    //! [inplace-roundtrip]
}
