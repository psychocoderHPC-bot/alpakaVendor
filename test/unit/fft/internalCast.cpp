/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/fft/internal/utility.hpp"

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("castVec identity preserves values", "[unit][helpers][cast]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents2D = Extents<uint32_t, 2u>;

    auto result = internal::castVec<Extents2D>(Extents2D{4u, 8u});
    CHECK(result == Extents2D{4u, 8u});
}

TEMPLATE_LIST_TEST_CASE("castVec lossless upcast from uint16 to uint32", "[unit][helpers][cast]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using SourceVec = Extents<uint16_t, 2u>;
    using TargetVec = Extents<uint32_t, 2u>;

    auto result = internal::castVec<TargetVec>(SourceVec{4u, 8u});
    CHECK(result == TargetVec{4u, 8u});
}

TEMPLATE_LIST_TEST_CASE("normalizeVectorOrScalar from vector", "[unit][helpers][cast]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents2D = Extents<uint32_t, 2u>;

    auto result = internal::normalizeVectorOrScalar<Extents2D>(Extents2D{4u, 8u});
    CHECK(result == Extents2D{4u, 8u});
}

TEMPLATE_LIST_TEST_CASE("normalizeVectorOrScalar from scalar", "[unit][helpers][cast]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents2D = Extents<uint32_t, 2u>;

    auto result = internal::normalizeVectorOrScalar<Extents2D>(uint32_t{8u});
    CHECK(result == Extents2D{8u, 8u});
}

TEMPLATE_LIST_TEST_CASE("asExtentVec from vector returns same vector", "[unit][helpers][cast]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents2D = Extents<uint32_t, 2u>;

    auto result = internal::asExtentVec(Extents2D{4u, 8u});
    CHECK(result == Extents2D{4u, 8u});
}

TEMPLATE_LIST_TEST_CASE("asExtentVec from scalar wraps to 1D", "[unit][helpers][cast]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    auto result = internal::asExtentVec(uint32_t{8u});
    CHECK(result == Extents1D{8u});
}
