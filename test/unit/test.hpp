/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <concepts>
#include <type_traits>

namespace alpakaVendor::test
{
    using TestBackends = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

    template<typename T_Api>
    consteval bool isFftBackendEnabledForApi()
    {
        using Api = std::remove_cvref_t<T_Api>;
        if constexpr(std::same_as<Api, alpaka::api::Host>)
            return ALPAKAV_DEP_FFTW;
        else if constexpr(std::same_as<Api, alpaka::api::Cuda>)
            return ALPAKAV_DEP_CUFFT;
        else if constexpr(std::same_as<Api, alpaka::api::Hip>)
            return ALPAKAV_DEP_ROCFFT;
        else if constexpr(std::same_as<Api, alpaka::api::OneApi>)
            return ALPAKAV_DEP_ONEMKL;
        else
            return false;
    }

    template<typename T_Api>
    consteval bool isBlasBackendEnabledForApi()
    {
        using Api = std::remove_cvref_t<T_Api>;
        if constexpr(std::same_as<Api, alpaka::api::Host>)
            return ALPAKAV_DEP_OPENBLAS;
        else if constexpr(std::same_as<Api, alpaka::api::Cuda>)
            return ALPAKAV_DEP_CUBLAS;
        else if constexpr(std::same_as<Api, alpaka::api::Hip>)
            return ALPAKAV_DEP_ROCBLAS;
        else if constexpr(std::same_as<Api, alpaka::api::OneApi>)
            return ALPAKAV_DEP_ONEMKL;
        else
            return false;
    }

    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    consteval bool isFftBackendEnabledForDevice(alpaka::onHost::Device<T_Api, T_DeviceKind> const&)
    {
        return isFftBackendEnabledForApi<T_Api>();
    }

    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    consteval bool isBlasBackendEnabledForDevice(alpaka::onHost::Device<T_Api, T_DeviceKind> const&)
    {
        return isBlasBackendEnabledForApi<T_Api>();
    }

    template<typename T_Type>
    inline void checkValue(T_Type actual, T_Type expected, double epsilon = 1.0e-5, double margin = 1.0e-6)
    {
        if constexpr(std::floating_point<T_Type>)
            CHECK(actual == Catch::Approx(expected).epsilon(epsilon).margin(margin));
        else
            CHECK(actual == expected);
    }
} // namespace alpakaVendor::test
