/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/mem/concepts/detail/CopyConstructableDataSource.hpp>

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/fft.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

namespace
{
    void requiresBuffer(alpaka::concepts::IBuffer auto)
    {
    }

    void requiresView(alpaka::concepts::IView auto)
    {
    }

    void requiresMdSpan(alpaka::concepts::IMdSpan auto)
    {
    }
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "SharedBufferFFT satisfies alpaka view and buffer concepts",
    "[unit][buffer][concepts]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    auto buffer = alpaka::fft::onHost::alloc<float>(device, 8u);
    using Buffer = std::decay_t<decltype(buffer)>;
    using ConstBuffer = alpaka::internal::CopyConstructableDataSource<Buffer>::InnerConst;

    STATIC_REQUIRE(alpaka::concepts::IDataSource<Buffer>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<Buffer>);
    STATIC_REQUIRE(alpaka::concepts::IView<Buffer>);
    STATIC_REQUIRE(alpaka::concepts::IBuffer<Buffer>);
    STATIC_REQUIRE(alpaka::internal::CopyConstructableDataSource<Buffer>::value);
    STATIC_REQUIRE(alpaka::internal::concepts::CopyConstructableDataSource<Buffer>);
    STATIC_REQUIRE(alpaka::internal::concepts::CopyConstructableDataSource<ConstBuffer>);

    requiresBuffer(buffer);
    requiresView(buffer);
    requiresMdSpan(buffer);

    auto constBuffer = ConstBuffer(buffer);
    requiresBuffer(constBuffer);
    requiresView(constBuffer);
    requiresMdSpan(constBuffer);
}

TEMPLATE_LIST_TEST_CASE(
    "SharedBufferFFT reinterpretation preserves lifetime and adjusts extents",
    "[unit][buffer]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    using namespace alpaka::fft;

    auto storage = makeInPlaceRealStorage<float>(8u);
    auto realBuffer = alpaka::fft::onHost::alloc<float>(device, 8u);

    CHECK(realBuffer.byteCapacity() == storage.physicalRealElements * sizeof(float));
    CHECK(realBuffer.getExtents() == storage.physicalRealExtents);

    auto complexBuffer = realBuffer.asComplex();
    CHECK(complexBuffer.byteCapacity() == realBuffer.byteCapacity());
    CHECK(complexBuffer.getUseCount() >= 2);
    CHECK(complexBuffer.getExtents() == storage.logicalComplexExtents);

    auto logicalRealBuffer = complexBuffer.asReal();
    CHECK(logicalRealBuffer.getExtents() == storage.logicalRealExtents);

    auto manualComplexBuffer = realBuffer.template reinterpretBuffer<Complex_t<float>>(storage.logicalComplexExtents);
    CHECK(manualComplexBuffer.getExtents() == storage.logicalComplexExtents);
}

TEMPLATE_LIST_TEST_CASE(
    "SharedBufferFFT same-kind reinterpretation returns a shallow copy",
    "[unit][buffer][reinterpret]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    auto realBuffer = alpaka::fft::onHost::alloc<float>(device, 8u);
    auto sameReal = realBuffer.asReal();
    CHECK(sameReal.data() == realBuffer.data());
    CHECK(sameReal.getUseCount() == realBuffer.getUseCount());

    auto complexBuffer = alpaka::fft::onHost::alloc<alpaka::math::Complex<float>>(device, 5u);
    auto sameComplex = complexBuffer.asComplex();
    CHECK(sameComplex.data() == complexBuffer.data());
    CHECK(sameComplex.getUseCount() == complexBuffer.getUseCount());
}

TEMPLATE_LIST_TEST_CASE(
    "SharedBufferFFT allocator family returns valid buffers",
    "[unit][buffer][alloc]",
    TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);

    using namespace alpaka::fft;

    auto queue = device.makeQueue();
    auto extents = Extents<uint16_t, 1u>{16u};

    auto plain = alpaka::fft::onHost::alloc<float>(device, extents);
    auto unified = alpaka::fft::onHost::allocUnified<float>(device, extents);
    auto mapped = alpaka::fft::onHost::allocMapped<float>(device, extents);
    auto deferred = alpaka::fft::onHost::allocDeferred<float>(queue, extents);

    CHECK(bool(plain));
    CHECK(bool(unified));
    CHECK(bool(mapped));
    CHECK(bool(deferred));
    CHECK(plain.byteCapacity() == 18u * sizeof(float));
}
