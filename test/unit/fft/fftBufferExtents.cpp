/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("makeFftBufferExtents for real scalar", "[unit][helpers][bufferExtents]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    auto result = makeFftBufferExtents<float>(Extents1D{8u});

    CHECK(result.logicalRealExtents == Extents1D{8u});
    CHECK(result.physicalRealExtents == Extents1D{10u});
    CHECK(result.logicalComplexExtents == Extents1D{5u});
}

TEMPLATE_LIST_TEST_CASE(
    "makeFftBufferExtents for real scalar from scalar input",
    "[unit][helpers][bufferExtents]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;

    auto result = makeFftBufferExtents<float>(uint32_t{8u});

    CHECK(result.logicalRealExtents[0] == uint32_t{8u});
    CHECK(result.physicalRealExtents[0] == uint32_t{10u});
    CHECK(result.logicalComplexExtents[0] == uint32_t{5u});
}

TEMPLATE_LIST_TEST_CASE("makeFftBufferExtents for complex scalar", "[unit][helpers][bufferExtents]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;
    using Extents1D = Extents<uint32_t, 1u>;

    auto result = makeFftBufferExtents<Complex>(Extents1D{8u});

    CHECK(result.logicalRealExtents == Extents1D{14u});
    CHECK(result.physicalRealExtents == Extents1D{16u});
    CHECK(result.logicalComplexExtents == Extents1D{8u});
}

TEMPLATE_LIST_TEST_CASE("makeFftBufferExtents for real scalar 2D", "[unit][helpers][bufferExtents]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents2D = Extents<uint32_t, 2u>;

    auto result = makeFftBufferExtents<float>(Extents2D{4u, 8u});

    CHECK(result.logicalRealExtents == Extents2D{4u, 8u});
    CHECK(result.physicalRealExtents == Extents2D{4u, 10u});
    CHECK(result.logicalComplexExtents == Extents2D{4u, 5u});
}

TEMPLATE_LIST_TEST_CASE("makeFftBufferExtents for complex scalar 2D", "[unit][helpers][bufferExtents]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Complex = alpaka::math::Complex<float>;
    using Extents2D = Extents<uint32_t, 2u>;

    auto result = makeFftBufferExtents<Complex>(Extents2D{4u, 8u});

    CHECK(result.logicalRealExtents == Extents2D{4u, 14u});
    CHECK(result.physicalRealExtents == Extents2D{4u, 16u});
    CHECK(result.logicalComplexExtents == Extents2D{4u, 8u});
}
