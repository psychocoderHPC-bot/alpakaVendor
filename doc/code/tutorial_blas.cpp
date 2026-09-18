/*
 * Copyright 2026 Rene Widera
 * SPDX-License-Identifier: ISC
 *
 * Tutorial: BLAS walkthrough with small, tested examples
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/blas.hpp>
#include <alpaka/math/Complex.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <type_traits>

#include "test.hpp"

using namespace alpakaVendor::test;

//! [blas-tutorial-setup]
TEMPLATE_LIST_TEST_CASE("Tutorial: BLAS walkthrough", "[doc][tutorial][blas]", TestBackends)
{
    auto device = getDeviceOrSkipTest(TestType::makeDict());
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API.");
    }
    else
    {
        auto queue = device.makeQueue();
        using Scalar = float;
        using Complex = alpaka::math::Complex<float>;
        //! [blas-tutorial-setup]

        //! [blas-tutorial-level1]
        auto x = alpaka::onHost::allocUnified<Scalar>(device, 3u);
        auto y = alpaka::onHost::allocUnified<Scalar>(device, 3u);
        auto z = alpaka::onHost::allocUnified<Scalar>(device, 3u);
        auto dotResult = alpaka::onHost::allocUnified<Scalar>(device, 1u);
        auto dotcResult = alpaka::onHost::allocUnified<Scalar>(device, 1u);
        auto nrm2Result = alpaka::onHost::allocUnified<Scalar>(device, 1u);
        auto asumResult = alpaka::onHost::allocUnified<Scalar>(device, 1u);
        auto iamaxResult = alpaka::onHost::allocUnified<int>(device, 1u);

        x.data()[0] = 1.0f;
        x.data()[1] = 2.0f;
        x.data()[2] = 3.0f;
        y.data()[0] = 4.0f;
        y.data()[1] = 5.0f;
        y.data()[2] = 6.0f;

        alpaka::blas::onHost::copy(queue, x, z);
        alpaka::blas::onHost::swap(queue, x, y);
        alpaka::blas::onHost::scal(queue, 2.0f, z);
        alpaka::blas::onHost::axpy(queue, -1.0f, y, x);
        alpaka::blas::onHost::dot(queue, y, z, dotResult);
        alpaka::blas::onHost::dotc(queue, y, z, dotcResult);
        alpaka::blas::onHost::nrm2(queue, y, nrm2Result);
        alpaka::blas::onHost::asum(queue, x, asumResult);
        alpaka::blas::onHost::iamax(queue, x, iamaxResult);
        alpaka::onHost::wait(queue);

        CHECK(z.data()[0] == Catch::Approx(2.0f));
        CHECK(z.data()[1] == Catch::Approx(4.0f));
        CHECK(z.data()[2] == Catch::Approx(6.0f));
        CHECK(x.data()[0] == Catch::Approx(3.0f));
        CHECK(x.data()[1] == Catch::Approx(3.0f));
        CHECK(x.data()[2] == Catch::Approx(3.0f));
        CHECK(y.data()[0] == Catch::Approx(1.0f));
        CHECK(y.data()[1] == Catch::Approx(2.0f));
        CHECK(y.data()[2] == Catch::Approx(3.0f));
        CHECK(dotResult.data()[0] == Catch::Approx(28.0f));
        // For real-valued vectors dotc is identical to dot.
        CHECK(dotcResult.data()[0] == Catch::Approx(28.0f));
        CHECK(nrm2Result.data()[0] == Catch::Approx(std::sqrt(14.0f)).epsilon(1.0e-5));
        CHECK(asumResult.data()[0] == Catch::Approx(9.0f));
        CHECK(iamaxResult.data()[0] == 1);
        //! [blas-tutorial-level1]

        //! [blas-tutorial-gemv]
        auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 3u});
        auto vx = alpaka::onHost::allocUnified<Scalar>(device, 3u);
        auto vy = alpaka::onHost::allocUnified<Scalar>(device, 2u);
        auto vxShort = alpaka::onHost::allocUnified<Scalar>(device, 2u);
        auto vyLong = alpaka::onHost::allocUnified<Scalar>(device, 3u);

        A[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 1.0f;
        A[alpaka::Vec<uint32_t, 2u>{0u, 1u}] = 2.0f;
        A[alpaka::Vec<uint32_t, 2u>{0u, 2u}] = 3.0f;
        A[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 4.0f;
        A[alpaka::Vec<uint32_t, 2u>{1u, 1u}] = 5.0f;
        A[alpaka::Vec<uint32_t, 2u>{1u, 2u}] = 6.0f;

        vx.data()[0] = 1.0f;
        vx.data()[1] = 0.0f;
        vx.data()[2] = -1.0f;
        vy.data()[0] = 10.0f;
        vy.data()[1] = 20.0f;

        alpaka::blas::onHost::gemv(queue, 2.0f, A, vx, 0.5f, vy);

        vxShort.data()[0] = 1.0f;
        vxShort.data()[1] = 2.0f;
        vyLong.data()[0] = 0.0f;
        vyLong.data()[1] = 0.0f;
        vyLong.data()[2] = 0.0f;

        alpaka::blas::onHost::gemv(queue, 1.0f, alpaka::blas::transposed(A), vxShort, 0.0f, vyLong);
        alpaka::onHost::wait(queue);

        CHECK(vy.data()[0] == Catch::Approx(1.0f));
        CHECK(vy.data()[1] == Catch::Approx(6.0f));
        CHECK(vyLong.data()[0] == Catch::Approx(9.0f));
        CHECK(vyLong.data()[1] == Catch::Approx(12.0f));
        CHECK(vyLong.data()[2] == Catch::Approx(15.0f));
        //! [blas-tutorial-gemv]

        //! [blas-tutorial-gemm]
        auto B = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{3u, 2u});
        auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});

        B[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 7.0f;
        B[alpaka::Vec<uint32_t, 2u>{0u, 1u}] = 8.0f;
        B[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 9.0f;
        B[alpaka::Vec<uint32_t, 2u>{1u, 1u}] = 10.0f;
        B[alpaka::Vec<uint32_t, 2u>{2u, 0u}] = 11.0f;
        B[alpaka::Vec<uint32_t, 2u>{2u, 1u}] = 12.0f;
        C[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 0.0f;
        C[alpaka::Vec<uint32_t, 2u>{0u, 1u}] = 0.0f;
        C[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 0.0f;
        C[alpaka::Vec<uint32_t, 2u>{1u, 1u}] = 0.0f;

        alpaka::blas::Options options{
            .precision = alpaka::blas::Precision::backendDefault,
            .algorithm = alpaka::blas::Algorithm::deterministic};
        alpaka::blas::onHost::gemm(queue, 1.0f, A, B, 0.0f, C, options);
        alpaka::onHost::wait(queue);

        CHECK(C[alpaka::Vec<uint32_t, 2u>{0u, 0u}] == Catch::Approx(58.0f));
        CHECK(C[alpaka::Vec<uint32_t, 2u>{0u, 1u}] == Catch::Approx(64.0f));
        CHECK(C[alpaka::Vec<uint32_t, 2u>{1u, 0u}] == Catch::Approx(139.0f));
        CHECK(C[alpaka::Vec<uint32_t, 2u>{1u, 1u}] == Catch::Approx(154.0f));

        auto H = alpaka::onHost::allocUnified<Complex>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        auto rhs = alpaka::onHost::allocUnified<Complex>(device, alpaka::Vec<uint32_t, 2u>{2u, 1u});
        auto out = alpaka::onHost::allocUnified<Complex>(device, alpaka::Vec<uint32_t, 2u>{2u, 1u});

        H[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = Complex{1.0f, 1.0f};
        H[alpaka::Vec<uint32_t, 2u>{0u, 1u}] = Complex{2.0f, 0.0f};
        H[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = Complex{0.0f, 3.0f};
        H[alpaka::Vec<uint32_t, 2u>{1u, 1u}] = Complex{4.0f, -1.0f};
        rhs[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = Complex{1.0f, 0.0f};
        rhs[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = Complex{1.0f, 0.0f};
        out[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = Complex{};
        out[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = Complex{};

        alpaka::blas::onHost::gemm(
            queue,
            Complex{1.0f, 0.0f},
            alpaka::blas::conjTransposed(H),
            rhs,
            Complex{0.0f, 0.0f},
            out);
        alpaka::onHost::wait(queue);

        CHECK(out[alpaka::Vec<uint32_t, 2u>{0u, 0u}].real() == Catch::Approx(1.0f));
        CHECK(out[alpaka::Vec<uint32_t, 2u>{0u, 0u}].imag() == Catch::Approx(-4.0f));
        CHECK(out[alpaka::Vec<uint32_t, 2u>{1u, 0u}].real() == Catch::Approx(6.0f));
        CHECK(out[alpaka::Vec<uint32_t, 2u>{1u, 0u}].imag() == Catch::Approx(1.0f));
        //! [blas-tutorial-gemm]

        //! [blas-tutorial-trsm]
        auto triangular = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        auto rhsSolve = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 1u});

        triangular[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 99.0f;
        triangular[alpaka::Vec<uint32_t, 2u>{0u, 1u}] = 0.0f;
        triangular[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 3.0f;
        triangular[alpaka::Vec<uint32_t, 2u>{1u, 1u}] = 77.0f;
        rhsSolve[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 5.0f;
        rhsSolve[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 11.0f;

        alpaka::blas::onHost::trsm(
            queue,
            alpaka::blas::Side::left,
            1.0f,
            alpaka::blas::unitDiag(alpaka::blas::lower(triangular)),
            rhsSolve,
            options);
        alpaka::onHost::wait(queue);

        CHECK(rhsSolve[alpaka::Vec<uint32_t, 2u>{0u, 0u}] == Catch::Approx(5.0f));
        CHECK(rhsSolve[alpaka::Vec<uint32_t, 2u>{1u, 0u}] == Catch::Approx(-4.0f));
        //! [blas-tutorial-trsm]

        //! [blas-tutorial-batched-gemm]
        auto batchA = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 2u});
        auto batchB = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 2u});
        auto batchC = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 2u});

        for(uint32_t b = 0; b < 2u; ++b)
            for(uint32_t r = 0; r < 2u; ++r)
                for(uint32_t c = 0; c < 2u; ++c)
                {
                    auto idx = b * 4u + r * 2u + c;
                    batchA[alpaka::Vec<uint32_t, 3u>{b, r, c}] = float(idx + 1u);
                    batchB[alpaka::Vec<uint32_t, 3u>{b, r, c}] = float((idx % 4u) + 1u);
                    batchC[alpaka::Vec<uint32_t, 3u>{b, r, c}] = 0.0f;
                }

        alpaka::blas::onHost::stridedBatchedGemm(queue, 1.0f, batchA, batchB, 0.0f, batchC, options);
        alpaka::onHost::wait(queue);

        CHECK(batchC[alpaka::Vec<uint32_t, 3u>{0u, 0u, 0u}] == Catch::Approx(7.0f));
        CHECK(batchC[alpaka::Vec<uint32_t, 3u>{0u, 0u, 1u}] == Catch::Approx(10.0f));
        CHECK(batchC[alpaka::Vec<uint32_t, 3u>{1u, 1u, 0u}] == Catch::Approx(31.0f));
        CHECK(batchC[alpaka::Vec<uint32_t, 3u>{1u, 1u, 1u}] == Catch::Approx(46.0f));
        //! [blas-tutorial-batched-gemm]
    }
}
