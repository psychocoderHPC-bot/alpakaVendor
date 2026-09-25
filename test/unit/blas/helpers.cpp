/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "alpaka/blas.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

namespace
{
    /** Minimal 1-D view stub exposing a configurable byte pitch.
     *
     * alpaka3 normalizes the innermost byte pitch of every MdSpan to ``sizeof(value_type)``, so a real 1-D view can
     * never report a non-element-multiple pitch. This stub mimics the byte pitch a padded/adapted 1-D view would
     * report so the vector x-axis divisibility check can be exercised. It only needs the accessors that
     * ``makeVectorDescriptor`` uses.
     */
    struct PaddedVectorView
    {
        using value_type = float;

        float* ptr = nullptr;
        std::size_t pitchBytes = sizeof(float);

        static consteval uint32_t dim()
        {
            return 1u;
        }

        auto getExtents() const
        {
            return alpaka::Vec<uint32_t, 1u>{4u};
        }

        auto getPitches() const
        {
            return alpaka::Vec<std::size_t, 1u>{pitchBytes};
        }

        float* data() const
        {
            return ptr;
        }
    };

    /** Minimal 2-D view stub exposing a configurable innermost (column) byte pitch. */
    struct PaddedMatrixView
    {
        using value_type = float;

        float* ptr = nullptr;
        std::size_t columnPitchBytes = sizeof(float);

        static consteval uint32_t dim()
        {
            return 2u;
        }

        auto getExtents() const
        {
            return alpaka::Vec<uint32_t, 2u>{3u, 4u};
        }

        auto getPitches() const
        {
            return alpaka::Vec<std::size_t, 2u>{columnPitchBytes, 6u * sizeof(float)};
        }

        float* data() const
        {
            return ptr;
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("blas annotations and metadata", "[unit][blas][annotations]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    auto buffer = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{2u, 3u});

    auto At = alpaka::blas::transposed(buffer);
    auto Ac = alpaka::blas::conjTransposed(buffer);
    auto Au = alpaka::blas::upper(buffer);
    auto Al = alpaka::blas::unitDiag(alpaka::blas::lower(buffer));
    auto An = alpaka::blas::nonUnitDiag(buffer);

    CHECK(alpaka::blas::detail::getTranspose(At) == alpaka::blas::Transpose::transposed);
    CHECK(alpaka::blas::detail::getTranspose(Ac) == alpaka::blas::Transpose::conjugateTransposed);
    CHECK(alpaka::blas::detail::getTriangle(Au) == alpaka::blas::Triangle::upper);
    CHECK(alpaka::blas::detail::getTriangle(Al) == alpaka::blas::Triangle::lower);
    CHECK(alpaka::blas::detail::getDiagonal(Al) == alpaka::blas::Diagonal::unit);
    CHECK(alpaka::blas::detail::getDiagonal(An) == alpaka::blas::Diagonal::nonUnit);
    CHECK_THROWS_AS(alpaka::blas::upper(alpaka::blas::lower(buffer)), std::invalid_argument);
    CHECK_THROWS_AS(alpaka::blas::lower(alpaka::blas::upper(buffer)), std::invalid_argument);
}

TEMPLATE_LIST_TEST_CASE(
    "blas descriptor extraction supports padded row major matrices and batches",
    "[unit][blas][layout]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    auto storage = alpaka::onHost::allocUnified<float>(device, 64u);

    auto matrixView = alpaka::makeMdSpan(
        storage.data(),
        alpaka::Vec<uint32_t, 2u>{3u, 4u},
        alpaka::Vec<std::size_t, 2u>{6u * sizeof(float), sizeof(float)});
    auto matrixDesc = alpaka::blas::internal::makeMatrixDescriptor(matrixView);
    CHECK(matrixDesc.rows == 3);
    CHECK(matrixDesc.cols == 4);
    CHECK(matrixDesc.ld == 6);

    auto batchView = alpaka::makeMdSpan(
        storage.data(),
        alpaka::Vec<uint32_t, 3u>{2u, 3u, 4u},
        alpaka::Vec<std::size_t, 3u>{24u * sizeof(float), 6u * sizeof(float), sizeof(float)});
    auto batchDesc = alpaka::blas::internal::makeBatchedMatrixDescriptor(batchView);
    CHECK(batchDesc.rows == 3);
    CHECK(batchDesc.cols == 4);
    CHECK(batchDesc.ld == 6);
    CHECK(batchDesc.batchCount == 2);
    CHECK(batchDesc.batchStride == 24);
}

TEMPLATE_LIST_TEST_CASE(
    "blas descriptors reject pitches that are not multiples of the element size",
    "[unit][blas][layout]",
    TestBackends)
{
    // Row pitch of a 3x4 matrix is 6*sizeof(float)+2 bytes and the batched batch pitch is
    // 24*sizeof(float)+4 bytes: both are not multiples of sizeof(float).
    constexpr auto storage = 128u;
    auto buffer = std::vector<float>(storage);

    auto misalignedRowPitch = alpaka::makeMdSpan(
        buffer.data(),
        alpaka::Vec<uint32_t, 2u>{3u, 4u},
        alpaka::Vec<std::size_t, 2u>{6u * sizeof(float) + 2u, sizeof(float)});
    CHECK_THROWS_AS(alpaka::blas::internal::makeMatrixDescriptor(misalignedRowPitch), std::invalid_argument);

    auto misalignedBatchPitch = alpaka::makeMdSpan(
        buffer.data(),
        alpaka::Vec<uint32_t, 3u>{2u, 3u, 4u},
        alpaka::Vec<std::size_t, 3u>{24u * sizeof(float) + 4u, 6u * sizeof(float) + 2u, sizeof(float)});
    CHECK_THROWS_AS(alpaka::blas::internal::makeBatchedMatrixDescriptor(misalignedBatchPitch), std::invalid_argument);

    // Element-multiple row pitch but a non-multiple batch pitch: the row pitch check passes and the
    // batch pitch branch must reject the view.
    auto misalignedBatchOnlyPitch = alpaka::makeMdSpan(
        buffer.data(),
        alpaka::Vec<uint32_t, 3u>{2u, 3u, 4u},
        alpaka::Vec<std::size_t, 3u>{36u * sizeof(float) + 2u, 6u * sizeof(float), sizeof(float)});
    CHECK_THROWS_AS(
        alpaka::blas::internal::makeBatchedMatrixDescriptor(misalignedBatchOnlyPitch),
        std::invalid_argument);

    // Vector x-axis pitch that is not a multiple of the element size: the vector descriptor must reject it instead
    // of silently truncating the element stride (generalizes the former vector-only PR #52 guard). A real 1-D
    // MdSpan normalizes its innermost pitch to sizeof(value_type), so a stub view exposes the padded pitch.
    auto validVector = PaddedVectorView{buffer.data(), 2u * sizeof(float)};
    CHECK(alpaka::blas::internal::makeVectorDescriptor(validVector).inc == 2);
    auto misalignedVector = PaddedVectorView{buffer.data(), 2u * sizeof(float) + 2u};
    CHECK_THROWS_AS(alpaka::blas::internal::makeVectorDescriptor(misalignedVector), std::invalid_argument);

    // Matrix x-axis (column) pitch that is not a multiple of the element size: checked before the strideCol == 1
    // dense-layout rule.
    auto misalignedColumn = PaddedMatrixView{buffer.data(), sizeof(float) + 2u};
    CHECK_THROWS_AS(alpaka::blas::internal::makeMatrixDescriptor(misalignedColumn), std::invalid_argument);
}

// Pure metadata tests for the centralized validation helpers. These do not touch a device, so they run once (not per
// backend) and cover the pitch divisibility, vendor-width narrowing and alias/overlap contracts directly.
TEST_CASE("blas validation helpers reject non-multiple pitches and unsafe narrowing", "[unit][blas][layout]")
{
    using namespace alpaka::blas::internal;
    using alpaka::api::Cuda;
    using alpaka::api::Hip;
    using alpaka::api::Host;
    using alpaka::api::OneApi;

    // pitchToElements: a whole multiple is converted losslessly; a remainder is rejected for every axis name.
    CHECK(pitchToElements(std::size_t{4u * sizeof(float)}, sizeof(float), "test pitch") == 4);
    CHECK_THROWS_AS(
        pitchToElements(std::size_t{4u * sizeof(float) + 2u}, sizeof(float), "test pitch x axis"),
        std::invalid_argument);
    CHECK_THROWS_AS(
        pitchToElements(std::size_t{3u * sizeof(double) + 1u}, sizeof(double), "test pitch y axis"),
        std::invalid_argument);

    // checkedVendorInt: 32-bit for host/cuda/hip with the signed negative and upper bound rejected; oneMKL keeps
    // the full int64 value (identity, no artificial rejection).
    constexpr auto intMax = static_cast<std::int64_t>(std::numeric_limits<int>::max());
    CHECK(checkedVendorInt<Host>(intMax, "host") == std::numeric_limits<int>::max());
    CHECK(checkedVendorInt<Cuda>(intMax, "cuda") == std::numeric_limits<int>::max());
    CHECK(checkedVendorInt<Hip>(intMax, "hip") == std::numeric_limits<int>::max());
    CHECK_THROWS_AS(checkedVendorInt<Host>(intMax + 1, "host"), std::invalid_argument);
    CHECK_THROWS_AS(checkedVendorInt<Cuda>(intMax + 1, "cuda"), std::invalid_argument);
    CHECK_THROWS_AS(checkedVendorInt<Hip>(intMax + 1, "hip"), std::invalid_argument);
    CHECK_THROWS_AS(checkedVendorInt<Host>(-1, "host"), std::invalid_argument);
    // An unsigned value above INT64_MAX cast to int64_t underflows negative and is rejected by the signed branch.
    CHECK_THROWS_AS(
        checkedVendorInt<Hip>(static_cast<std::int64_t>(std::numeric_limits<std::uint64_t>::max()), "hip"),
        std::invalid_argument);
    constexpr auto int64Max = std::numeric_limits<std::int64_t>::max();
    CHECK(checkedVendorInt<OneApi>(int64Max, "oneapi") == int64Max);

    // Alias/overlap detection: identical/overlapping byte spans are rejected, disjoint spans pass, and a
    // zero-extent operand can never overlap.
    auto const base = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x1000));
    auto const other = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0x1000 + 64));
    MatrixDescriptor a{};
    a.constPtr = base;
    a.rows = 2;
    a.cols = 2;
    a.ld = 2;
    MatrixDescriptor same{};
    same.constPtr = base;
    same.rows = 2;
    same.cols = 2;
    same.ld = 2;
    MatrixDescriptor disjoint{};
    disjoint.constPtr = other;
    disjoint.rows = 2;
    disjoint.cols = 2;
    disjoint.ld = 2;
    MatrixDescriptor empty{};
    empty.constPtr = base;
    empty.rows = 0;
    empty.cols = 0;
    empty.ld = 0;
    CHECK_THROWS_AS(validateNoOverlap(a, sizeof(float), same, sizeof(float), "test"), std::invalid_argument);
    CHECK_NOTHROW(validateNoOverlap(a, sizeof(float), disjoint, sizeof(float), "test"));
    CHECK_NOTHROW(validateNoOverlap(a, sizeof(float), empty, sizeof(float), "test"));

    // A vector increment other than 1 expands the worst-case span and is accounted for.
    VectorDescriptor v{};
    v.constPtr = base;
    v.n = 4;
    v.inc = 4; // touches base .. base + 12 elements
    VectorDescriptor near{};
    near.constPtr = base;
    near.n = 4;
    near.inc = 4;
    CHECK_THROWS_AS(validateNoOverlap(v, sizeof(float), near, sizeof(float), "vec"), std::invalid_argument);
}
