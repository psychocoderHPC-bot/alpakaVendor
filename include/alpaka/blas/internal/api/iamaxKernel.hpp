/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/core/common.hpp"

namespace alpaka::blas::internal
{
    /** Enforce the documented zero iamax result for an empty vector.
     *
     * The functor is launched as a regular alpaka kernel on the same queue as the vendor reduction. Therefore it
     * is sequenced after the vendor call and inherits the queue kind semantics (e.g. blocking). The vendor's
     * already 1-based result is left unchanged for n > 0. For n <= 0 the result is set to 0 so the empty-vector
     * contract is enforced independently of the vendor's behavior.
     */
    struct IamaxZeroForEmptyKernel
    {
        template<typename TAcc, typename T>
        ALPAKA_FN_ACC void operator()(TAcc const&, T* resultPtr, int n) const
        {
            if(n <= 0)
                *resultPtr = 0;
        }
    };
} // namespace alpaka::blas::internal
