/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/fft/common.hpp"
#include "alpaka/fft/internal/api/config.hpp"
#include "alpaka/fft/internal/utility.hpp"

namespace alpaka::fft::internal
{
    template<typename T_Api, typename T_Value, alpaka::concepts::Vector T_Extents>
    struct PlanImpl
    {
        PlanImpl(auto&, Transform, Layout<T_Extents>, PlanOptions)
        {
            static_assert(!sizeof(T_Api), "Unsupported alpaka::fft FFT backend for this API.");
        }
    };
} // namespace alpaka::fft::internal

#include "alpaka/fft/internal/api/cuda/fft.hpp"
#include "alpaka/fft/internal/api/hip/fft.hpp"
#include "alpaka/fft/internal/api/host/fft.hpp"
#include "alpaka/fft/internal/api/oneapi/fft.hpp"
