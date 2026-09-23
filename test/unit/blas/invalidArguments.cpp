/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpakaTest/deviceHelper.hpp>

#include "alpaka/blas.hpp"
#include "test.hpp"

using namespace alpakaVendor::test;

TEMPLATE_LIST_TEST_CASE(
    "blas invalid sizes, annotations and layouts are rejected before backend dispatch",
    "[unit][blas][invalid]",
    TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SUCCEED();
    }
    else
    {
        auto queue = device.makeQueue();

        auto A = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{2u, 3u});
        auto B = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{4u, 2u});
        auto C = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        CHECK_THROWS_AS(alpaka::blas::onHost::gemm(queue, 1.0f, A, B, 0.0f, C), std::invalid_argument);

        auto x = alpaka::onHost::allocUnified<float>(device, 4u);
        auto y = alpaka::onHost::allocUnified<float>(device, 2u);
        CHECK_THROWS_AS(alpaka::blas::onHost::copy(queue, x, y), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::swap(queue, x, y), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::axpy(queue, 1.0f, x, y), std::invalid_argument);
        auto result2 = alpaka::onHost::allocUnified<float>(device, 2u);
        auto y4 = alpaka::onHost::allocUnified<float>(device, 4u);
        CHECK_THROWS_AS(alpaka::blas::onHost::dot(queue, x, y, result2), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::nrm2(queue, x, result2), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::dot(queue, x, y4, result2), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::gemv(queue, 1.0f, A, x, 0.0f, y), std::invalid_argument);

        auto rhs = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{2u, 1u});
        CHECK_THROWS_AS(
            alpaka::blas::onHost::trsm(queue, alpaka::blas::Side::left, 1.0f, A, rhs),
            std::invalid_argument);
        auto square = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        CHECK_THROWS_AS(
            alpaka::blas::onHost::trsm(queue, alpaka::blas::Side::left, 1.0f, square, rhs),
            std::invalid_argument);
        // A full-triangle annotation (the default state which upper()/lower() overwrite) must be rejected as well.
        CHECK_THROWS_AS(
            alpaka::blas::onHost::trsm(queue, alpaka::blas::Side::left, 1.0f, alpaka::blas::unitDiag(square), rhs),
            std::invalid_argument);
        CHECK_THROWS_AS(
            alpaka::blas::onHost::trsm(queue, alpaka::blas::Side::left, 1.0f, alpaka::blas::transposed(square), rhs),
            std::invalid_argument);

        auto BA = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 3u});
        auto BB = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 3u>{3u, 3u, 2u});
        auto BC = alpaka::onHost::allocUnified<float>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 2u});
        CHECK_THROWS_AS(
            alpaka::blas::onHost::stridedBatchedGemm(queue, 1.0f, BA, BB, 0.0f, BC),
            std::invalid_argument);

        auto storage = alpaka::onHost::allocUnified<float>(device, 16u);
        auto invalidLd = alpaka::makeMdSpan(
            storage.data(),
            alpaka::Vec<uint32_t, 2u>{2u, 3u},
            alpaka::Vec<std::size_t, 2u>{2u * sizeof(float), sizeof(float)});
        CHECK_THROWS_AS(alpaka::blas::internal::makeMatrixDescriptor(invalidLd), std::invalid_argument);
        CHECK_THROWS_AS(alpaka::blas::onHost::gemm(queue, 1.0f, invalidLd, C, 0.0f, C), std::invalid_argument);
    }
}

#if ALPAKAV_DEP_OPENBLAS && ALPAKAV_HAS_OPENBLAS
TEST_CASE("blas host uplo mapper rejects Triangle::full", "[unit][blas][invalid]")
{
    CHECK(alpaka::blas::internal::toCblasUplo(alpaka::blas::Triangle::upper) == CblasUpper);
    CHECK(alpaka::blas::internal::toCblasUplo(alpaka::blas::Triangle::lower) == CblasLower);
    CHECK_THROWS_AS(alpaka::blas::internal::toCblasUplo(alpaka::blas::Triangle::full), std::invalid_argument);
}
#endif
