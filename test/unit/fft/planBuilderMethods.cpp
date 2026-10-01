/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE("PlanBuilder batch sets layout batch", "[unit][plan][methods]", TestBackends)
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

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().batch(4u).build(device);
        CHECK(plan.layout().batch == uint32_t{4u});
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder distances sets layout distances", "[unit][plan][methods]", TestBackends)
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

        // distances are in bytes: 16 elements * sizeof(Complex) = 16 * 8 = 128 bytes
        constexpr std::size_t distBytes = 16u * sizeof(Complex);
        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().distances(distBytes, distBytes).build(device);
        CHECK(plan.layout().inDistance == distBytes);
        CHECK(plan.layout().outDistance == distBytes);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder strides sets layout strides", "[unit][plan][methods]", TestBackends)
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

        // strides are in bytes: 1 element * sizeof(Complex) = 8 bytes
        auto inStrides = alpaka::Vec<std::size_t, 1u>{sizeof(Complex)};
        auto outStrides = alpaka::Vec<std::size_t, 1u>{sizeof(Complex)};
        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().strides(inStrides, outStrides).build(device);
        CHECK(plan.layout().inStrides == alpaka::Vec<std::size_t, 1u>{sizeof(Complex)});
        CHECK(plan.layout().outStrides == alpaka::Vec<std::size_t, 1u>{sizeof(Complex)});
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder invalid batch zero throws", "[unit][plan][methods]", TestBackends)
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

        CHECK_THROWS_AS(
            (alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().batch(0u).build(device)),
            std::invalid_argument);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder complex type with r2c throws", "[unit][plan][methods]", TestBackends)
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

        CHECK_THROWS_AS((alpaka::fft::onHost::PlanBuilder<Complex>{8u}.r2c().build(device)), std::invalid_argument);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder complex type with c2r throws", "[unit][plan][methods]", TestBackends)
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

        CHECK_THROWS_AS((alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2r().build(device)), std::invalid_argument);
    }
}

TEMPLATE_LIST_TEST_CASE("makePlan factory produces working builder", "[unit][plan][methods]", TestBackends)
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

        auto plan = alpaka::fft::onHost::makePlan<Complex>(8u).c2c().build(device);
        CHECK(plan.transform() == Transform::c2c);
        CHECK(plan.layout().extents == Extents<uint32_t, 1u>{8u});
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder r2c with real type is valid", "[unit][plan][methods]", TestBackends)
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

        auto plan = alpaka::fft::onHost::PlanBuilder<float>{8u}.r2c().build(device);
        CHECK(plan.transform() == Transform::r2c);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder c2r with real type is valid", "[unit][plan][methods]", TestBackends)
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

        auto plan = alpaka::fft::onHost::PlanBuilder<float>{8u}.c2r().build(device);
        CHECK(plan.transform() == Transform::c2r);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "PlanBuilder options default to outOfPlace and backendManaged",
    "[unit][plan][methods]",
    TestBackends)
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
        CHECK(plan.options().placement == Placement::outOfPlace);
        CHECK(plan.options().workspacePolicy == WorkspacePolicy::backendManaged);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder inPlace sets placement", "[unit][plan][methods]", TestBackends)
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

        auto plan = alpaka::fft::onHost::PlanBuilder<float>{8u}.r2c().inPlace().build(device);
        CHECK(plan.options().placement == Placement::inPlace);
    }
}

// ---------------------------------------------------------------------------
// 1D byte-based strides and distances
// ---------------------------------------------------------------------------

TEMPLATE_LIST_TEST_CASE("PlanBuilder 1D distances in bytes - C2C", "[unit][plan][methods][bytes][1d]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
        SKIP("No FFT backend enabled for this alpaka API.");
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        constexpr uint32_t n = 16u;
        constexpr uint32_t batch = 4u;
        // contiguous distance: n elements * sizeof(Complex) bytes/element
        constexpr std::size_t distBytes = n * sizeof(Complex);

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}
                        .c2c()
                        .batch(batch)
                        .distances(distBytes, distBytes)
                        .build(device);

        CHECK(plan.layout().inDistance == distBytes);
        CHECK(plan.layout().outDistance == distBytes);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder 1D strides in bytes - C2C", "[unit][plan][methods][bytes][1d]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
        SKIP("No FFT backend enabled for this alpaka API.");
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        constexpr uint32_t n = 16u;
        // contiguous stride: 1 element * sizeof(Complex) bytes/element
        constexpr std::size_t strideBytes = sizeof(Complex);

        auto inStrides = alpaka::Vec<std::size_t, 1u>{strideBytes};
        auto outStrides = alpaka::Vec<std::size_t, 1u>{strideBytes};
        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}.c2c().strides(inStrides, outStrides).build(device);

        CHECK(plan.layout().inStrides[0] == strideBytes);
        CHECK(plan.layout().outStrides[0] == strideBytes);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "PlanBuilder 1D scalar strides in bytes - C2C",
    "[unit][plan][methods][bytes][1d]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
        SKIP("No FFT backend enabled for this alpaka API.");
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        constexpr uint32_t n = 16u;
        constexpr std::size_t strideBytes = sizeof(Complex);

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{n}.c2c().strides(strideBytes, strideBytes).build(device);

        CHECK(plan.layout().inStrides[0] == strideBytes);
        CHECK(plan.layout().outStrides[0] == strideBytes);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder 1D distances in bytes - R2C", "[unit][plan][methods][bytes][1d]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
        SKIP("No FFT backend enabled for this alpaka API.");
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        constexpr uint32_t n = 16u;
        constexpr uint32_t batch = 4u;
        constexpr uint32_t complexN = r2cComplexExtent(n);
        // input: n real elements, output: complexN complex elements
        constexpr std::size_t inDistBytes = n * sizeof(float);
        constexpr std::size_t outDistBytes = complexN * sizeof(Complex);

        auto plan = alpaka::fft::onHost::PlanBuilder<float>{n}
                        .r2c()
                        .batch(batch)
                        .distances(inDistBytes, outDistBytes)
                        .build(device);

        CHECK(plan.layout().inDistance == inDistBytes);
        CHECK(plan.layout().outDistance == outDistBytes);
    }
}

// ---------------------------------------------------------------------------
// 2D byte-based strides and distances
// ---------------------------------------------------------------------------

TEMPLATE_LIST_TEST_CASE("PlanBuilder 2D distances in bytes - C2C", "[unit][plan][methods][bytes][2d]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
        SKIP("No FFT backend enabled for this alpaka API.");
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        constexpr uint32_t ny = 8u;
        constexpr uint32_t nx = 4u; // x is the fast-moving alpaka dimension: last component
        constexpr uint32_t batch = 2u;
        // 2D contiguous: product(ny, nx) * sizeof(Complex)
        constexpr std::size_t distBytes = nx * ny * sizeof(Complex);

        auto plan
            = alpaka::fft::onHost::PlanBuilder<Complex, alpaka::Vec<uint32_t, 2u>>(alpaka::Vec<uint32_t, 2u>{ny, nx})
                  .c2c()
                  .batch(batch)
                  .distances(distBytes, distBytes)
                  .build(device);

        CHECK(plan.layout().inDistance == distBytes);
        CHECK(plan.layout().outDistance == distBytes);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder 2D strides in bytes - C2C", "[unit][plan][methods][bytes][2d]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
        SKIP("No FFT backend enabled for this alpaka API.");
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        constexpr uint32_t ny = 8u;
        constexpr uint32_t nx = 4u; // x is the fast-moving alpaka dimension: last component
        // 2D contiguous strides: {nx * sizeof(Complex), 1 * sizeof(Complex)}
        // alpaka layout: last index is fast-moving
        constexpr std::size_t strideDim0 = nx * sizeof(Complex); // bytes to next y-row
        constexpr std::size_t strideDim1 = sizeof(Complex); // bytes to next x-element

        auto inStrides = alpaka::Vec<std::size_t, 2u>{strideDim0, strideDim1};
        auto outStrides = alpaka::Vec<std::size_t, 2u>{strideDim0, strideDim1};
        auto plan
            = alpaka::fft::onHost::PlanBuilder<Complex, alpaka::Vec<uint32_t, 2u>>(alpaka::Vec<uint32_t, 2u>{ny, nx})
                  .c2c()
                  .strides(inStrides, outStrides)
                  .build(device);

        CHECK(plan.layout().inStrides[0] == strideDim0);
        CHECK(plan.layout().inStrides[1] == strideDim1);
        CHECK(plan.layout().outStrides[0] == strideDim0);
        CHECK(plan.layout().outStrides[1] == strideDim1);
    }
}

// ---------------------------------------------------------------------------
// 3D byte-based strides and distances
// ---------------------------------------------------------------------------

TEMPLATE_LIST_TEST_CASE("PlanBuilder 3D distances in bytes - C2C", "[unit][plan][methods][bytes][3d]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
        SKIP("No FFT backend enabled for this alpaka API.");
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        constexpr uint32_t nz = 4u;
        constexpr uint32_t ny = 3u;
        constexpr uint32_t nx = 2u; // x is the fast-moving alpaka dimension: last component
        constexpr uint32_t batch = 2u;
        // 3D contiguous: product(nz, ny, nx) * sizeof(Complex)
        constexpr std::size_t distBytes = nx * ny * nz * sizeof(Complex);

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, alpaka::Vec<uint32_t, 3u>>(
                        alpaka::Vec<uint32_t, 3u>{nz, ny, nx})
                        .c2c()
                        .batch(batch)
                        .distances(distBytes, distBytes)
                        .build(device);

        CHECK(plan.layout().inDistance == distBytes);
        CHECK(plan.layout().outDistance == distBytes);
    }
}

TEMPLATE_LIST_TEST_CASE("PlanBuilder 3D strides in bytes - C2C", "[unit][plan][methods][bytes][3d]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
        SKIP("No FFT backend enabled for this alpaka API.");
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        constexpr uint32_t nz = 4u;
        constexpr uint32_t ny = 3u;
        constexpr uint32_t nx = 2u; // x is the fast-moving alpaka dimension: last component
        // 3D contiguous strides (alpaka: last index fast-moving):
        //   dim 0 (z): ny * nx * sizeof(Complex)
        //   dim 1 (y): nx * sizeof(Complex)
        //   dim 2 (x): sizeof(Complex)
        constexpr std::size_t strideDim0 = ny * nx * sizeof(Complex);
        constexpr std::size_t strideDim1 = nx * sizeof(Complex);
        constexpr std::size_t strideDim2 = sizeof(Complex);

        auto inStrides = alpaka::Vec<std::size_t, 3u>{strideDim0, strideDim1, strideDim2};
        auto outStrides = alpaka::Vec<std::size_t, 3u>{strideDim0, strideDim1, strideDim2};
        auto plan = alpaka::fft::onHost::PlanBuilder<Complex, alpaka::Vec<uint32_t, 3u>>(
                        alpaka::Vec<uint32_t, 3u>{nz, ny, nx})
                        .c2c()
                        .strides(inStrides, outStrides)
                        .build(device);

        CHECK(plan.layout().inStrides[0] == strideDim0);
        CHECK(plan.layout().inStrides[1] == strideDim1);
        CHECK(plan.layout().inStrides[2] == strideDim2);
        CHECK(plan.layout().outStrides[0] == strideDim0);
        CHECK(plan.layout().outStrides[1] == strideDim1);
        CHECK(plan.layout().outStrides[2] == strideDim2);
    }
}

// ---------------------------------------------------------------------------
// Default distances (0 -> contiguous) in bytes
// ---------------------------------------------------------------------------

TEMPLATE_LIST_TEST_CASE(
    "PlanBuilder default distances are zero (contiguous)",
    "[unit][plan][methods][bytes]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
        SKIP("No FFT backend enabled for this alpaka API.");
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        auto plan = alpaka::fft::onHost::PlanBuilder<Complex>{8u}.c2c().build(device);
        CHECK(plan.layout().inDistance == std::size_t{0u});
        CHECK(plan.layout().outDistance == std::size_t{0u});
    }
}

// ---------------------------------------------------------------------------
// Asymmetric distances (R2C: different input/output sizes)
// ---------------------------------------------------------------------------

TEMPLATE_LIST_TEST_CASE("PlanBuilder R2C asymmetric byte distances", "[unit][plan][methods][bytes]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    if constexpr(!isFftBackendEnabledForDevice(device))
        SKIP("No FFT backend enabled for this alpaka API.");
    else
    {
        using namespace alpaka::fft;
        using Complex = alpaka::math::Complex<float>;

        constexpr uint32_t n = 32u;
        constexpr uint32_t batch = 8u;
        constexpr uint32_t complexN = r2cComplexExtent(n);

        constexpr std::size_t inDistBytes = n * sizeof(float);
        constexpr std::size_t outDistBytes = complexN * sizeof(Complex);

        auto plan = alpaka::fft::onHost::PlanBuilder<float>{n}
                        .r2c()
                        .batch(batch)
                        .distances(inDistBytes, outDistBytes)
                        .build(device);

        CHECK(plan.layout().inDistance == inDistBytes);
        CHECK(plan.layout().outDistance == outDistBytes);
        // verify asymmetric: real input is smaller than complex output
        CHECK(plan.layout().inDistance < plan.layout().outDistance);
    }
}
