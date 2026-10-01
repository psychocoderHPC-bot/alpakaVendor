/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if ALPAKAV_DEP_OPENBLAS && __has_include(<cblas.h>)
#    include <cblas.h>
#    define ALPAKAV_HAS_OPENBLAS 1
#elif ALPAKAV_DEP_OPENBLAS && __has_include(<openblas/cblas.h>)
#    include <openblas/cblas.h>
#    define ALPAKAV_HAS_OPENBLAS 1
#else
#    define ALPAKAV_HAS_OPENBLAS 0
#endif

#if __has_include(<cublas_v2.h>)
#    include <cublas_v2.h>
#    define ALPAKAV_HAS_CUBLAS 1
#else
#    define ALPAKAV_HAS_CUBLAS 0
#endif

#if __has_include(<rocblas/rocblas.h>)
#    include <rocblas/rocblas.h>
#    define ALPAKAV_HAS_ROCBLAS 1
#elif __has_include(<rocblas.h>)
#    include <rocblas.h>
#    define ALPAKAV_HAS_ROCBLAS 1
#else
#    define ALPAKAV_HAS_ROCBLAS 0
#endif

#if ALPAKAV_DEP_ONEMKL && ALPAKA_LANG_ONEAPI && __has_include(<oneapi/mkl/blas.hpp>)
/* oneMKL's C++ BLAS header includes mkl_cblas.h via oneapi/mkl/types.hpp.
 * OpenBLAS and oneMKL both define the global CBLAS enums and function
 * declarations, so avoid re-including oneMKL's CBLAS declarations after an
 * OpenBLAS CBLAS header was selected for the host backend.  The oneMKL C++
 * enums used below are defined by oneapi/mkl/types.hpp independently of
 * mkl_cblas.h.
 */
#    if ALPAKAV_HAS_OPENBLAS && !defined(__MKL_CBLAS_H__)
#        define __MKL_CBLAS_H__
#    endif
#    include <oneapi/mkl/blas.hpp>
#    include <sycl/sycl.hpp>
#    define ALPAKAV_HAS_ONEMKL_BLAS 1
#else
#    define ALPAKAV_HAS_ONEMKL_BLAS 0
#endif
