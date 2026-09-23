/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/blas/common.hpp"
#include "alpaka/blas/internal/api/config.hpp"
#include "alpaka/blas/internal/utility.hpp"

namespace alpaka::blas::internal
{
    ALPAKA_FN_SYMBOL(CopyFn);
    ALPAKA_FN_SYMBOL(SwapFn);
    ALPAKA_FN_SYMBOL(ScalFn);
    ALPAKA_FN_SYMBOL(AxpyFn);
    ALPAKA_FN_SYMBOL(DotFn);
    ALPAKA_FN_SYMBOL(DotcFn);
    ALPAKA_FN_SYMBOL(Nrm2Fn);
    ALPAKA_FN_SYMBOL(AsumFn);
    ALPAKA_FN_SYMBOL(IamaxFn);
    ALPAKA_FN_SYMBOL(GemvFn);
    ALPAKA_FN_SYMBOL(GemmFn);
    ALPAKA_FN_SYMBOL(StridedBatchedGemmFn);
    ALPAKA_FN_SYMBOL(TrsmFn);
    ALPAKA_FN_SYMBOL(SyrkFn);
} // namespace alpaka::blas::internal

#include "alpaka/blas/internal/api/cuda/blas.hpp"
#include "alpaka/blas/internal/api/hip/blas.hpp"
#include "alpaka/blas/internal/api/host/blas.hpp"
#include "alpaka/blas/internal/api/oneapi/blas.hpp"
