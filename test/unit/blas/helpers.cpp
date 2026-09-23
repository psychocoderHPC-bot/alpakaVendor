/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/blas.hpp"
#include "test.hpp"

#include <stdexcept>
#include <vector>

using namespace alpakaVendor::test;

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
    auto constexpr storage = 128u;
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
    CHECK_THROWS_AS(
        alpaka::blas::internal::makeBatchedMatrixDescriptor(misalignedBatchPitch),
        std::invalid_argument);

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
