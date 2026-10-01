/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE(
    "SharedBufferFFT addDestructorAction runs on last release",
    "[unit][buffer][edge]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    bool actionRan = false;
    {
        auto buffer = alpaka::fft::onHost::allocUnified<float>(device, 8u);
        buffer.addDestructorAction([&actionRan]() { actionRan = true; });
        CHECK_FALSE(actionRan);
    }
    CHECK(actionRan);
}

TEMPLATE_LIST_TEST_CASE("SharedBufferFFT multiple destructor actions all run", "[unit][buffer][edge]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    int counter = 0;
    {
        auto buffer = alpaka::fft::onHost::allocUnified<float>(device, 8u);
        buffer.addDestructorAction([&counter]() { ++counter; });
        buffer.addDestructorAction([&counter]() { ++counter; });
        buffer.addDestructorAction([&counter]() { ++counter; });
    }
    CHECK(counter == 3);
}

TEMPLATE_LIST_TEST_CASE(
    "SharedBufferFFT reinterpretBuffer throws when exceeding capacity",
    "[unit][buffer][edge]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    auto buffer = alpaka::fft::onHost::allocUnified<float>(device, 4u);
    using Complex = alpaka::math::Complex<float>;

    CHECK_THROWS_AS(buffer.template reinterpretBuffer<Complex>(uint32_t{8u}), std::invalid_argument);
}

TEMPLATE_LIST_TEST_CASE(
    "SharedBufferFFT complex buffer asReal recovers logical extents",
    "[unit][buffer][edge]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;

    auto buffer = alpaka::fft::onHost::allocUnified<Complex>(device, uint32_t{5u});
    auto realView = buffer.asReal();

    CHECK(realView.getExtents() == Extents<uint32_t, 1u>{8u});
}

TEMPLATE_LIST_TEST_CASE(
    "SharedBufferFFT real buffer asComplex returns complex view",
    "[unit][buffer][edge]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    using namespace alpaka::fft;

    auto buffer = alpaka::fft::onHost::allocUnified<float>(device, uint32_t{8u});
    auto complexView = buffer.asComplex();

    CHECK(complexView.getExtents() == Extents<uint32_t, 1u>{5u});
}

TEMPLATE_LIST_TEST_CASE("SharedBufferFFT fftMetadata accessible", "[unit][buffer][edge]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    using namespace alpaka::fft;

    auto buffer = alpaka::fft::onHost::allocUnified<float>(device, uint32_t{8u});
    auto const& meta = buffer.fftMetadata();

    CHECK(meta.extents.has_value());
    CHECK(meta.extents->logicalRealExtents == Extents<uint32_t, 1u>{8u});
    CHECK(meta.extents->logicalComplexExtents == Extents<uint32_t, 1u>{5u});
    CHECK(meta.extents->physicalRealExtents == Extents<uint32_t, 1u>{10u});
}

TEMPLATE_LIST_TEST_CASE("SharedBufferFFT getView returns valid view", "[unit][buffer][edge]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    auto buffer = alpaka::fft::onHost::allocUnified<float>(device, 8u);
    auto view = buffer.getView();

    CHECK(view.getExtents() == buffer.getExtents());
    CHECK(view.data() == buffer.data());
}
