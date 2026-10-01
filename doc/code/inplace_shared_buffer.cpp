/*
 * SPDX-FileCopyrightText: Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * In-place real transforms: SharedBufferFFT example
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>
#include <alpaka/math/Complex.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <type_traits>


//! [inplace-shared-buffer]
using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("SharedBufferFFT example", "[doc][inplace][shared_buffer]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        SKIP("No device available for selected backend.");
    auto device = devSelector.makeDevice(0);

    using namespace alpaka::fft;

    // Compute storage requirements for N=16
    constexpr uint32_t n = 16u;
    auto storage = makeInPlaceRealStorage<float>(n);

    // Allocate physical storage
    auto realBuffer = alpaka::fft::onHost::allocUnified<float>(device, n);

    // Verify buffer properties
    CHECK(realBuffer);
    CHECK(realBuffer.byteCapacity() == storage.physicalRealElements * sizeof(float));
    CHECK(realBuffer.getUseCount() >= 1);

    // Reinterpret as complex
    auto complexBuffer = realBuffer.asComplex();

    // Verify reinterpretation
    CHECK(complexBuffer);
    CHECK(complexBuffer.byteCapacity() == realBuffer.byteCapacity());
    CHECK(complexBuffer.getUseCount() >= 2);

    // Both buffers share the same memory
    CHECK(realBuffer.data() == reinterpret_cast<float*>(complexBuffer.data()));
    //! [inplace-shared-buffer]
}
