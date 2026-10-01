/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if __has_include(<fftw3.h>)
#    include <fftw3.h>
#    define ALPAKAV_HAS_FFTW 1
#else
#    define ALPAKAV_HAS_FFTW 0
#endif

#if __has_include(<cufft.h>)
#    include <cufft.h>
#    define ALPAKAV_HAS_CUFFT 1
#else
#    define ALPAKAV_HAS_CUFFT 0
#endif

#if __has_include(<rocfft/rocfft.h>)
#    include <rocfft/rocfft.h>
#    define ALPAKAV_HAS_ROCFFT 1
#else
#    define ALPAKAV_HAS_ROCFFT 0
#endif

#if ALPAKAV_DEP_ONEMKL && ALPAKA_LANG_ONEAPI && __has_include(<oneapi/mkl/dft.hpp>)
#    include <oneapi/mkl/dft.hpp>
#    include <sycl/sycl.hpp>
#    define ALPAKAV_HAS_ONEMKL 1
#else
#    define ALPAKAV_HAS_ONEMKL 0
#endif
