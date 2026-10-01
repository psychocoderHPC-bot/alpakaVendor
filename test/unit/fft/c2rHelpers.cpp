/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("c2rLogicalRealExtent scalar", "[unit][helpers][c2r]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;

    CHECK(c2rLogicalRealExtent(uint32_t{8u}) == uint32_t{14u});
    CHECK(c2rLogicalRealExtent(uint32_t{9u}) == uint32_t{16u});
    CHECK(c2rLogicalRealExtent(uint32_t{10u}) == uint32_t{18u});
    CHECK(c2rLogicalRealExtent(uint32_t{2u}) == uint32_t{2u});
    CHECK(c2rLogicalRealExtent(uint32_t{1u}) == uint32_t{0u});
}

TEMPLATE_LIST_TEST_CASE("c2rLogicalRealExtent scalar throws on zero", "[unit][helpers][c2r]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;

    CHECK_THROWS_AS(c2rLogicalRealExtent(uint32_t{0u}), std::invalid_argument);
}

TEMPLATE_LIST_TEST_CASE("c2rLogicalRealExtent vector", "[unit][helpers][c2r]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents2D = Extents<uint32_t, 2u>;

    auto result = c2rLogicalRealExtent(Extents2D{4u, 8u});
    CHECK(result == Extents2D{4u, 14u});

    auto result2 = c2rLogicalRealExtent(Extents2D{3u, 9u});
    CHECK(result2 == Extents2D{3u, 16u});
}

TEMPLATE_LIST_TEST_CASE("c2rPaddedRealExtent scalar", "[unit][helpers][c2r]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;

    CHECK(c2rPaddedRealExtent(uint32_t{8u}) == uint32_t{16u});
    CHECK(c2rPaddedRealExtent(uint32_t{9u}) == uint32_t{18u});
    CHECK(c2rPaddedRealExtent(uint32_t{5u}) == uint32_t{10u});
}

TEMPLATE_LIST_TEST_CASE("c2rPaddedRealExtent vector", "[unit][helpers][c2r]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents2D = Extents<uint32_t, 2u>;

    auto result = c2rPaddedRealExtent(Extents2D{4u, 8u});
    CHECK(result == Extents2D{4u, 16u});

    auto result2 = c2rPaddedRealExtent(Extents2D{3u, 9u});
    CHECK(result2 == Extents2D{3u, 18u});
}

TEMPLATE_LIST_TEST_CASE(
    "c2r roundtrip consistency: r2cComplexExtent inverse of c2rLogicalRealExtent",
    "[unit][helpers][c2r]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;

    for(uint32_t n = 2u; n <= 32u; n += 2u)
    {
        auto complexN = r2cComplexExtent(n);
        auto recoveredN = c2rLogicalRealExtent(complexN);
        CHECK(recoveredN == n);
    }
}
