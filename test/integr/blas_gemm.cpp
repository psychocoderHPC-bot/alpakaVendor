/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <algorithm>
#include <alpakaTest/deviceHelper.hpp>
#include <concepts>
#include <type_traits>
#include <utility>

#include "../unit/blas/reference.hpp"
#include "../unit/test.hpp"
#include "alpaka/blas.hpp"

using namespace alpakaVendor::test;

template<typename T_View>
auto ldOf(T_View const& view)
{
    return view.getPitches().y() / sizeof(alpaka::GetValueType_t<T_View>);
}

template<typename T_Device>
consteval bool supportsDoubleGemm(T_Device const&)
{
    using Api = std::remove_cvref_t<decltype(alpaka::getApi(std::declval<T_Device>()))>;
    using DeviceKind = std::remove_cvref_t<decltype(alpaka::getDeviceKind(std::declval<T_Device>()))>;
    return !std::same_as<Api, alpaka::api::OneApi> || std::same_as<DeviceKind, alpaka::deviceKind::Cpu>;
}

TEMPLATE_LIST_TEST_CASE("BLAS GEMM/GEMV/TRSM and batched GEMM", "[integr][blas][gemm]", TestBackends)
{
    auto deviceExec = getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = getDevice(deviceExec);
    if constexpr(!isBlasBackendEnabledForDevice(device))
    {
        SKIP("No BLAS backend enabled for this alpaka API " << device.getApi().getName() << " " << device.getName());
    }
    else
    {
        auto queue = device.makeQueue();
        using Scalar = float;
        auto const deterministicOptions = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::backendDefault,
            .algorithm = alpaka::blas::Algorithm::deterministic};
        auto const fastestOptions = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::backendDefault,
            .algorithm = alpaka::blas::Algorithm::fastest};
        auto const exactOptions = alpaka::blas::Options{
            .precision = alpaka::blas::Precision::exact,
            .algorithm = alpaka::blas::Algorithm::backendDefault};

        auto A = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 3u});
        auto B = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{3u, 2u});
        auto C = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        auto Cref = alpaka::onHost::allocHostLike(C);
        auto x = alpaka::onHost::allocUnified<Scalar>(device, 3u);
        auto y = alpaka::onHost::allocUnified<Scalar>(device, 2u);

        Scalar aVals[] = {1, 2, 3, 4, 5, 6};
        Scalar bVals[] = {7, 8, 9, 10, 11, 12};
        Scalar cVals[] = {0, 1, 2, 3};
        Scalar xVals[] = {1, 2, 3};
        Scalar yVals[] = {4, 5};
        for(uint32_t r = 0; r < 2u; ++r)
            for(uint32_t c = 0; c < 3u; ++c)
                A[alpaka::Vec<uint32_t, 2u>{r, c}] = aVals[r * 3u + c];
        for(uint32_t r = 0; r < 3u; ++r)
            for(uint32_t c = 0; c < 2u; ++c)
                B[alpaka::Vec<uint32_t, 2u>{r, c}] = bVals[r * 2u + c];
        for(uint32_t r = 0; r < 2u; ++r)
            for(uint32_t c = 0; c < 2u; ++c)
            {
                C[alpaka::Vec<uint32_t, 2u>{r, c}] = cVals[r * 2u + c];
                Cref[alpaka::Vec<uint32_t, 2u>{r, c}] = cVals[r * 2u + c];
            }
        std::copy(std::begin(xVals), std::end(xVals), x.data());
        std::copy(std::begin(yVals), std::end(yVals), y.data());

        blas::gemmRef(
            1.5f,
            A.data(),
            ldOf(A),
            2u,
            3u,
            alpaka::blas::Transpose::none,
            B.data(),
            ldOf(B),
            3u,
            2u,
            alpaka::blas::Transpose::none,
            0.5f,
            Cref.data(),
            ldOf(Cref),
            2u,
            2u);
        alpaka::blas::onHost::gemm(queue, 1.5f, A, B, 0.5f, C, exactOptions);
        alpaka::onHost::wait(queue);

        for(uint32_t r = 0; r < 2u; ++r)
            for(uint32_t c = 0; c < 2u; ++c)
                CHECK(
                    C[alpaka::Vec<uint32_t, 2u>{r, c}]
                    == Catch::Approx(Cref[alpaka::Vec<uint32_t, 2u>{r, c}]).epsilon(1e-5));

        if constexpr(supportsDoubleGemm(device))
        {
            auto Ad = alpaka::onHost::allocUnified<double>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
            auto Bd = alpaka::onHost::allocUnified<double>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
            auto Cd = alpaka::onHost::allocUnified<double>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
            auto CdRef = alpaka::onHost::allocHostLike(Cd);
            double adVals[] = {1.0, 2.0, 3.0, 4.0};
            double bdVals[] = {5.0, 6.0, 7.0, 8.0};
            for(uint32_t r = 0; r < 2u; ++r)
                for(uint32_t c = 0; c < 2u; ++c)
                {
                    Ad[alpaka::Vec<uint32_t, 2u>{r, c}] = adVals[r * 2u + c];
                    Bd[alpaka::Vec<uint32_t, 2u>{r, c}] = bdVals[r * 2u + c];
                    Cd[alpaka::Vec<uint32_t, 2u>{r, c}] = 0.0;
                    CdRef[alpaka::Vec<uint32_t, 2u>{r, c}] = 0.0;
                }
            blas::gemmRef(
                1.0,
                Ad.data(),
                ldOf(Ad),
                2u,
                2u,
                alpaka::blas::Transpose::none,
                Bd.data(),
                ldOf(Bd),
                2u,
                2u,
                alpaka::blas::Transpose::none,
                0.0,
                CdRef.data(),
                ldOf(CdRef),
                2u,
                2u);
            alpaka::blas::onHost::gemm(queue, 1.0, Ad, Bd, 0.0, Cd, deterministicOptions);
            alpaka::onHost::wait(queue);
            for(uint32_t r = 0; r < 2u; ++r)
                for(uint32_t c = 0; c < 2u; ++c)
                    CHECK(
                        Cd[alpaka::Vec<uint32_t, 2u>{r, c}]
                        == Catch::Approx(CdRef[alpaka::Vec<uint32_t, 2u>{r, c}]).epsilon(1e-12));
        }


        auto Atm = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{3u, 2u});
        auto Bt = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{3u, 2u});
        auto Ct = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        auto CtRef = alpaka::onHost::allocHostLike(Ct);
        Scalar atVals[] = {1, 2, 3, 4, 5, 6};
        for(uint32_t r = 0; r < 3u; ++r)
            for(uint32_t c = 0; c < 2u; ++c)
            {
                Atm[alpaka::Vec<uint32_t, 2u>{r, c}] = atVals[r * 2u + c];
                Bt[alpaka::Vec<uint32_t, 2u>{r, c}] = bVals[r * 2u + c];
            }
        for(uint32_t r = 0; r < 2u; ++r)
            for(uint32_t c = 0; c < 2u; ++c)
            {
                Ct[alpaka::Vec<uint32_t, 2u>{r, c}] = 0.0f;
                CtRef[alpaka::Vec<uint32_t, 2u>{r, c}] = 0.0f;
            }
        blas::gemmRef(
            1.0f,
            Atm.data(),
            ldOf(Atm),
            3u,
            2u,
            alpaka::blas::Transpose::transposed,
            Bt.data(),
            ldOf(Bt),
            3u,
            2u,
            alpaka::blas::Transpose::none,
            0.0f,
            CtRef.data(),
            ldOf(CtRef),
            2u,
            2u);
        alpaka::blas::onHost::gemm(queue, 1.0f, alpaka::blas::transposed(Atm), Bt, 0.0f, Ct, deterministicOptions);
        alpaka::onHost::wait(queue);
        for(uint32_t r = 0; r < 2u; ++r)
            for(uint32_t c = 0; c < 2u; ++c)
                CHECK(
                    Ct[alpaka::Vec<uint32_t, 2u>{r, c}]
                    == Catch::Approx(CtRef[alpaka::Vec<uint32_t, 2u>{r, c}]).epsilon(1e-5));

        auto yRef = alpaka::onHost::allocHostLike(y);
        yRef.data()[0] = yVals[0];
        yRef.data()[1] = yVals[1];
        blas::gemvRef(2.0f, A.data(), ldOf(A), 2u, 3u, alpaka::blas::Transpose::none, x.data(), 0.5f, yRef.data());
        alpaka::blas::onHost::gemv(queue, 2.0f, A, x, 0.5f, y, deterministicOptions);
        alpaka::onHost::wait(queue);
        CHECK(y.data()[0] == Catch::Approx(yRef.data()[0]).epsilon(1e-5));
        CHECK(y.data()[1] == Catch::Approx(yRef.data()[1]).epsilon(1e-5));

        auto xt = alpaka::onHost::allocUnified<Scalar>(device, 2u);
        auto yt = alpaka::onHost::allocUnified<Scalar>(device, 3u);
        auto ytRef = alpaka::onHost::allocHostLike(yt);
        xt.data()[0] = 2.0f;
        xt.data()[1] = -1.0f;
        yt.data()[0] = 1.0f;
        yt.data()[1] = 0.0f;
        yt.data()[2] = -2.0f;
        std::copy_n(yt.data(), 3u, ytRef.data());
        blas::gemvRef(
            1.25f,
            A.data(),
            ldOf(A),
            2u,
            3u,
            alpaka::blas::Transpose::transposed,
            xt.data(),
            -0.5f,
            ytRef.data());
        alpaka::blas::onHost::gemv(queue, 1.25f, alpaka::blas::transposed(A), xt, -0.5f, yt, deterministicOptions);
        alpaka::onHost::wait(queue);
        for(uint32_t i = 0; i < 3u; ++i)
            CHECK(yt.data()[i] == Catch::Approx(ytRef.data()[i]).epsilon(1e-5));

        using Complex = alpaka::math::Complex<float>;
        auto Ac = alpaka::onHost::allocUnified<Complex>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        auto Bc = alpaka::onHost::allocUnified<Complex>(device, alpaka::Vec<uint32_t, 2u>{2u, 1u});
        auto Cc = alpaka::onHost::allocUnified<Complex>(device, alpaka::Vec<uint32_t, 2u>{2u, 1u});
        auto CcRef = alpaka::onHost::allocHostLike(Cc);
        Ac[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = Complex{1.0f, 1.0f};
        Ac[alpaka::Vec<uint32_t, 2u>{0u, 1u}] = Complex{2.0f, -1.0f};
        Ac[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = Complex{0.0f, 3.0f};
        Ac[alpaka::Vec<uint32_t, 2u>{1u, 1u}] = Complex{4.0f, 0.5f};
        Bc[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = Complex{1.0f, -1.0f};
        Bc[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = Complex{2.0f, 0.5f};
        for(uint32_t i = 0; i < 2u; ++i)
        {
            Cc[alpaka::Vec<uint32_t, 2u>{i, 0u}] = Complex{0.5f * float(i), -float(i)};
            CcRef[alpaka::Vec<uint32_t, 2u>{i, 0u}] = Cc[alpaka::Vec<uint32_t, 2u>{i, 0u}];
        }
        blas::gemmRef(
            Complex{1.0f, 0.0f},
            Ac.data(),
            ldOf(Ac),
            2u,
            2u,
            alpaka::blas::Transpose::conjugateTransposed,
            Bc.data(),
            ldOf(Bc),
            2u,
            1u,
            alpaka::blas::Transpose::none,
            Complex{0.25f, 0.0f},
            CcRef.data(),
            ldOf(CcRef),
            2u,
            1u);
        alpaka::blas::onHost::gemm(
            queue,
            Complex{1.0f, 0.0f},
            alpaka::blas::conjTransposed(Ac),
            Bc,
            Complex{0.25f, 0.0f},
            Cc,
            deterministicOptions);
        alpaka::onHost::wait(queue);
        for(uint32_t i = 0; i < 2u; ++i)
        {
            CHECK(
                Cc[alpaka::Vec<uint32_t, 2u>{i, 0u}].real()
                == Catch::Approx(CcRef[alpaka::Vec<uint32_t, 2u>{i, 0u}].real()).epsilon(1e-5));
            CHECK(
                Cc[alpaka::Vec<uint32_t, 2u>{i, 0u}].imag()
                == Catch::Approx(CcRef[alpaka::Vec<uint32_t, 2u>{i, 0u}].imag()).epsilon(1e-5));
        }

        auto Tm = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        auto Rhs = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 1u});
        Tm[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 2;
        Tm[alpaka::Vec<uint32_t, 2u>{0u, 1u}] = 0;
        Tm[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 3;
        Tm[alpaka::Vec<uint32_t, 2u>{1u, 1u}] = 4;
        Rhs[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 2;
        Rhs[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 11;
        alpaka::blas::onHost::trsm(
            queue,
            alpaka::blas::Side::left,
            1.0f,
            alpaka::blas::lower(Tm),
            Rhs,
            deterministicOptions);
        alpaka::onHost::wait(queue);
        CHECK(Rhs[alpaka::Vec<uint32_t, 2u>{0u, 0u}] == Catch::Approx(1.0f).epsilon(1e-5));
        CHECK(Rhs[alpaka::Vec<uint32_t, 2u>{1u, 0u}] == Catch::Approx(2.0f).epsilon(1e-5));

        auto Tu = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        auto RightRhs = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        Tu[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 9.0f;
        Tu[alpaka::Vec<uint32_t, 2u>{0u, 1u}] = 2.0f;
        Tu[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 0.0f;
        Tu[alpaka::Vec<uint32_t, 2u>{1u, 1u}] = 7.0f;
        RightRhs[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 5.0f;
        RightRhs[alpaka::Vec<uint32_t, 2u>{0u, 1u}] = 16.0f;
        RightRhs[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 7.0f;
        RightRhs[alpaka::Vec<uint32_t, 2u>{1u, 1u}] = 22.0f;
        alpaka::blas::onHost::trsm(
            queue,
            alpaka::blas::Side::right,
            1.0f,
            alpaka::blas::unitDiag(alpaka::blas::upper(Tu)),
            RightRhs,
            deterministicOptions);
        alpaka::onHost::wait(queue);
        CHECK(RightRhs[alpaka::Vec<uint32_t, 2u>{0u, 0u}] == Catch::Approx(5.0f).epsilon(1e-5));
        CHECK(RightRhs[alpaka::Vec<uint32_t, 2u>{0u, 1u}] == Catch::Approx(6.0f).epsilon(1e-5));
        CHECK(RightRhs[alpaka::Vec<uint32_t, 2u>{1u, 0u}] == Catch::Approx(7.0f).epsilon(1e-5));
        CHECK(RightRhs[alpaka::Vec<uint32_t, 2u>{1u, 1u}] == Catch::Approx(8.0f).epsilon(1e-5));

        auto Tlt = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 2u});
        auto RhsLt = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 2u>{2u, 1u});
        Tlt[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 2.0f;
        Tlt[alpaka::Vec<uint32_t, 2u>{0u, 1u}] = 0.0f;
        Tlt[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 3.0f;
        Tlt[alpaka::Vec<uint32_t, 2u>{1u, 1u}] = 4.0f;
        RhsLt[alpaka::Vec<uint32_t, 2u>{0u, 0u}] = 11.0f;
        RhsLt[alpaka::Vec<uint32_t, 2u>{1u, 0u}] = 8.0f;
        alpaka::blas::onHost::trsm(
            queue,
            alpaka::blas::Side::left,
            1.0f,
            alpaka::blas::nonUnitDiag(alpaka::blas::transposed(alpaka::blas::lower(Tlt))),
            RhsLt,
            deterministicOptions);
        alpaka::onHost::wait(queue);
        CHECK(RhsLt[alpaka::Vec<uint32_t, 2u>{0u, 0u}] == Catch::Approx(2.5f).epsilon(1e-5));
        CHECK(RhsLt[alpaka::Vec<uint32_t, 2u>{1u, 0u}] == Catch::Approx(2.0f).epsilon(1e-5));

        auto BA = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 2u});
        auto BB = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 2u});
        auto BC = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 2u});
        for(uint32_t b = 0; b < 2u; ++b)
            for(uint32_t r = 0; r < 2u; ++r)
                for(uint32_t c = 0; c < 2u; ++c)
                {
                    auto idx = b * 4u + r * 2u + c;
                    BA[alpaka::Vec<uint32_t, 3u>{b, r, c}] = float(idx + 1);
                    BB[alpaka::Vec<uint32_t, 3u>{b, r, c}] = float((idx % 4u) + 1);
                    BC[alpaka::Vec<uint32_t, 3u>{b, r, c}] = 0.0f;
                }
        alpaka::blas::onHost::stridedBatchedGemm(queue, 1.0f, BA, BB, 0.0f, BC, fastestOptions);
        alpaka::onHost::wait(queue);
        CHECK(BC[alpaka::Vec<uint32_t, 3u>{0u, 0u, 0u}] == Catch::Approx(7.0f).epsilon(1e-5));
        CHECK(BC[alpaka::Vec<uint32_t, 3u>{0u, 0u, 1u}] == Catch::Approx(10.0f).epsilon(1e-5));
        CHECK(BC[alpaka::Vec<uint32_t, 3u>{1u, 1u, 0u}] == Catch::Approx(31.0f).epsilon(1e-5));
        CHECK(BC[alpaka::Vec<uint32_t, 3u>{1u, 1u, 1u}] == Catch::Approx(46.0f).epsilon(1e-5));

        auto BCt = alpaka::onHost::allocUnified<Scalar>(device, alpaka::Vec<uint32_t, 3u>{2u, 2u, 2u});
        for(uint32_t b = 0; b < 2u; ++b)
            for(uint32_t r = 0; r < 2u; ++r)
                for(uint32_t c = 0; c < 2u; ++c)
                    BCt[alpaka::Vec<uint32_t, 3u>{b, r, c}] = 0.0f;
        alpaka::blas::onHost::stridedBatchedGemm(
            queue,
            1.0f,
            alpaka::blas::transposed(BA),
            BB,
            0.0f,
            BCt,
            fastestOptions);
        alpaka::onHost::wait(queue);
        CHECK(BCt[alpaka::Vec<uint32_t, 3u>{0u, 0u, 0u}] == Catch::Approx(10.0f).epsilon(1e-5));
        CHECK(BCt[alpaka::Vec<uint32_t, 3u>{0u, 1u, 1u}] == Catch::Approx(20.0f).epsilon(1e-5));
        CHECK(BCt[alpaka::Vec<uint32_t, 3u>{1u, 0u, 1u}] == Catch::Approx(38.0f).epsilon(1e-5));
        CHECK(BCt[alpaka::Vec<uint32_t, 3u>{1u, 1u, 0u}] == Catch::Approx(30.0f).epsilon(1e-5));
    }
}
