/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * In-place real transforms: Padding helpers example
 */

#include <alpaka/fft.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <type_traits>


//! [inplace-helpers]
using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Padding helpers", "[doc][inplace][helpers]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        SKIP("No device available for selected backend.");
    auto device = devSelector.makeDevice(0);

    using namespace alpaka::fft;

    // 1D examples
    CHECK(r2cComplexExtent(8u) == 5u);
    CHECK(r2cPaddedRealExtent(8u) == 10u);

    CHECK(r2cComplexExtent(9u) == 5u);
    CHECK(r2cPaddedRealExtent(9u) == 10u);

    CHECK(r2cComplexExtent(10u) == 6u);
    CHECK(r2cPaddedRealExtent(10u) == 12u);

    // 2D examples
    using Extents2D = Extents<uint32_t, 2u>;
    constexpr auto extents2d = Extents2D{4u, 9u};
    auto complexExtents2d = r2cComplexExtent(extents2d);
    auto paddedExtents2d = r2cPaddedRealExtent(extents2d);

    CHECK((complexExtents2d == Extents2D{4u, 5u}));
    CHECK((paddedExtents2d == Extents2D{4u, 10u}));

    // InPlaceRealStorage
    auto storage = makeInPlaceRealStorage<float>(Extents2D{1024u, 1024u});
    CHECK(storage.logicalRealElements == 1024u * 1024u);
    CHECK(storage.physicalRealElements == 1024u * 1026u);
    CHECK(storage.logicalComplexElements == 1024u * 513u);
    //! [inplace-helpers]
}
