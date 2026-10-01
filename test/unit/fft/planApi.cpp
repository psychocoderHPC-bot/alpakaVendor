/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("Plan workspaceBytes returns non-negative", "[unit][plan][api]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().build(device);
        CHECK(plan.workspaceBytes() >= std::size_t{0u});
    }
}

TEMPLATE_LIST_TEST_CASE("Plan copy preserves identity", "[unit][plan][api]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().build(device);
        auto copy = plan;

        CHECK(copy.transform() == plan.transform());
        CHECK(copy.layout().extents == plan.layout().extents);
        CHECK(copy.options().placement == plan.options().placement);
        CHECK(copy.options().workspacePolicy == plan.options().workspacePolicy);
    }
}

TEMPLATE_LIST_TEST_CASE("Plan move leaves valid state", "[unit][plan][api]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
    {
        SKIP("No FFT backend enabled for this alpaka API.");
    }
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().build(device);
        auto moved = std::move(plan);

        CHECK(moved.transform() == Transform::c2c);
        CHECK(moved.layout().extents == Extents<uint32_t, 1u>{8u});
    }
}
