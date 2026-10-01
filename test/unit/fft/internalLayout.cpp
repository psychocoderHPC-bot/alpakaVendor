/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/fft/internal/utility.hpp"

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("expectedInputExtents c2c passthrough", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInputExtents(layout, Transform::c2c, Placement::outOfPlace) == Extents1D{8u});
    CHECK(internal::expectedInputExtents(layout, Transform::c2c, Placement::inPlace) == Extents1D{8u});
}

TEMPLATE_LIST_TEST_CASE("expectedInputExtents r2c out-of-place passthrough", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInputExtents(layout, Transform::r2c, Placement::outOfPlace) == Extents1D{8u});
}

TEMPLATE_LIST_TEST_CASE(
    "expectedInputExtents r2c in-place uses padded real extents",
    "[unit][helpers][layout]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInputExtents(layout, Transform::r2c, Placement::inPlace) == Extents1D{10u});
}

TEMPLATE_LIST_TEST_CASE("expectedInputExtents c2r uses complex extents", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedInputExtents(layout, Transform::c2r, Placement::outOfPlace) == Extents1D{5u});
    CHECK(internal::expectedInputExtents(layout, Transform::c2r, Placement::inPlace) == Extents1D{5u});
}

TEMPLATE_LIST_TEST_CASE("expectedOutputExtents c2c passthrough", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedOutputExtents(layout, Transform::c2c, Placement::outOfPlace) == Extents1D{8u});
    CHECK(internal::expectedOutputExtents(layout, Transform::c2c, Placement::inPlace) == Extents1D{8u});
}

TEMPLATE_LIST_TEST_CASE("expectedOutputExtents r2c uses complex extents", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedOutputExtents(layout, Transform::r2c, Placement::outOfPlace) == Extents1D{5u});
    CHECK(internal::expectedOutputExtents(layout, Transform::r2c, Placement::inPlace) == Extents1D{5u});
}

TEMPLATE_LIST_TEST_CASE("expectedOutputExtents c2r out-of-place passthrough", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedOutputExtents(layout, Transform::c2r, Placement::outOfPlace) == Extents1D{8u});
}

TEMPLATE_LIST_TEST_CASE(
    "expectedOutputExtents c2r in-place uses padded real extents",
    "[unit][helpers][layout]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(internal::expectedOutputExtents(layout, Transform::c2r, Placement::inPlace) == Extents1D{10u});
}

TEMPLATE_LIST_TEST_CASE("expected strides are contiguous", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;
    using Complex = alpaka::math::Complex<float>;
    using ByteStrides1D = alpaka::Vec<std::size_t, 1u>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(
        internal::expectedInStrides<Complex>(layout, Transform::c2c, Placement::outOfPlace)
        == ByteStrides1D{sizeof(Complex)});
    CHECK(
        internal::expectedOutStrides<Complex>(layout, Transform::c2c, Placement::outOfPlace)
        == ByteStrides1D{sizeof(Complex)});
    CHECK(
        internal::expectedInStrides<float>(layout, Transform::r2c, Placement::outOfPlace)
        == ByteStrides1D{sizeof(float)});
    CHECK(
        internal::expectedOutStrides<float>(layout, Transform::r2c, Placement::outOfPlace)
        == ByteStrides1D{sizeof(Complex)});
    CHECK(
        internal::expectedInStrides<float>(layout, Transform::c2r, Placement::outOfPlace)
        == ByteStrides1D{sizeof(Complex)});
    CHECK(
        internal::expectedOutStrides<float>(layout, Transform::c2r, Placement::outOfPlace)
        == ByteStrides1D{sizeof(float)});
}

TEMPLATE_LIST_TEST_CASE("areZero detects zero and non-zero strides", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Strides1D = Strides<uint32_t, 1u>;
    using Strides2D = Strides<uint32_t, 2u>;

    CHECK(internal::areZero(Strides1D{0u}));
    CHECK_FALSE(internal::areZero(Strides1D{1u}));
    CHECK_FALSE(internal::areZero(Strides2D{0u, 1u}));
}

TEMPLATE_LIST_TEST_CASE("expected distances default to product of extents", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;
    using Complex = alpaka::math::Complex<float>;

    Layout<Extents1D> layout{.extents = Extents1D{8u}};

    CHECK(
        internal::expectedInDistance<Complex>(layout, Transform::c2c, Placement::outOfPlace) == 8u * sizeof(Complex));
    CHECK(
        internal::expectedOutDistance<Complex>(layout, Transform::c2c, Placement::outOfPlace) == 8u * sizeof(Complex));
    CHECK(internal::expectedInDistance<float>(layout, Transform::r2c, Placement::outOfPlace) == 8u * sizeof(float));
    CHECK(internal::expectedOutDistance<float>(layout, Transform::r2c, Placement::outOfPlace) == 5u * sizeof(Complex));
    CHECK(internal::expectedInDistance<float>(layout, Transform::c2r, Placement::outOfPlace) == 5u * sizeof(Complex));
    CHECK(internal::expectedOutDistance<float>(layout, Transform::c2r, Placement::outOfPlace) == 8u * sizeof(float));
}

TEMPLATE_LIST_TEST_CASE(
    "expected distances use custom layout distances when non-zero",
    "[unit][helpers][layout]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;
    using Extents1D = Extents<uint32_t, 1u>;

    Layout<Extents1D> layout{
        .extents = Extents1D{8u},
        .inDistance = std::size_t{16u},
        .outDistance = std::size_t{16u}};

    CHECK(internal::expectedInDistance<float>(layout, Transform::c2c, Placement::outOfPlace) == std::size_t{16u});
    CHECK(internal::expectedOutDistance<float>(layout, Transform::c2c, Placement::outOfPlace) == std::size_t{16u});
}

TEMPLATE_LIST_TEST_CASE("byte strides convert back to element strides", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;

    auto const byteStrides = alpaka::Vec<std::size_t, 3u>{96u, 24u, 8u};
    CHECK(
        internal::stridesToElements<std::size_t>(byteStrides, sizeof(double))
        == alpaka::Vec<std::size_t, 3u>{12u, 3u, 1u});
}

TEMPLATE_LIST_TEST_CASE("byte distances convert back to element distances", "[unit][helpers][layout]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    using namespace alpaka::fft;

    CHECK(internal::distanceToElements<std::size_t>(128u, sizeof(alpaka::math::Complex<float>)) == 16u);
}
