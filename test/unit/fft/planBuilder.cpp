/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("PlanBuilder defaults are applied", "[unit][plan]", TestBackends)
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
        CHECK(plan.transform() == Transform::c2c);
        CHECK(plan.layout().extents == alpaka::fft::Extents<uint32_t, 1u>{8u});
        CHECK(plan.options().placement == Placement::outOfPlace);
        CHECK(plan.options().workspacePolicy == WorkspacePolicy::backendManaged);
    }
}

TEMPLATE_LIST_TEST_CASE("invalid plan configuration throws", "[unit][plan]", TestBackends)
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

        CHECK_THROWS_AS((alpaka::fft::onHost::PlanBuilder<float>{0u}.r2c().build(device)), std::invalid_argument);
        CHECK_THROWS_AS((alpaka::fft::onHost::PlanBuilder<Complex>{8u}.r2c().build(device)), std::invalid_argument);
    }
}
