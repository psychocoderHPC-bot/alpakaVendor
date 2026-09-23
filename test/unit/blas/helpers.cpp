/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "alpaka/blas.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

namespace
{
    /** Minimal 1-D view stub used to exercise the vector pitch validation.
     *
     * alpaka3 normalizes the innermost byte pitch of 1-D MdSpan/View instances to `sizeof(value_type)`,
     * therefore a real 1-D mdspan can never expose a non-element-multiple pitch. This stub mimics the byte
     * pitch a padded/adapted 1-D view would report.
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
}

TEMPLATE_LIST_TEST_CASE(
    "blas vector descriptors reject pitches that are not multiples of the element size",
    "[unit][blas][layout]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    auto storage = alpaka::onHost::allocUnified<float>(device, 16u);

    // Positive control: an element-multiple pitch is accepted and converted to an element stride.
    auto validView = PaddedVectorView{storage.data(), 2u * sizeof(float)};
    auto validDesc = alpaka::blas::internal::makeVectorDescriptor(validView);
    CHECK(validDesc.n == 4);
    CHECK(validDesc.inc == 2);

    // A pitch of 2*sizeof(float)+1 bytes cannot be expressed as a whole element stride.
    auto misalignedView = PaddedVectorView{storage.data(), 2u * sizeof(float) + 1u};
    CHECK_THROWS_AS(alpaka::blas::internal::makeVectorDescriptor(misalignedView), std::invalid_argument);
}
