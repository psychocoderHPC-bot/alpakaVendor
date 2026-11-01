/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/blas.hpp>

#include <cmath>
#include <cstdint>

int main()
{
    auto device = alpaka::onHost::makeHostDevice();
    auto queue = device.makeQueue();

    auto a = alpaka::onHost::allocHost<float>(alpaka::Vec{2u, 2u});
    auto b = alpaka::onHost::allocHost<float>(alpaka::Vec{2u, 2u});
    auto c = alpaka::onHost::allocHost<float>(alpaka::Vec{2u, 2u});

    a[alpaka::Vec{0u, 0u}] = 1.0f;
    a[alpaka::Vec{0u, 1u}] = 2.0f;
    a[alpaka::Vec{1u, 0u}] = 3.0f;
    a[alpaka::Vec{1u, 1u}] = 4.0f;
    b[alpaka::Vec{0u, 0u}] = 1.0f;
    b[alpaka::Vec{0u, 1u}] = 0.0f;
    b[alpaka::Vec{1u, 0u}] = 0.0f;
    b[alpaka::Vec{1u, 1u}] = 1.0f;

    alpaka::blas::onHost::gemm(queue, 1.0f, a, b, 0.0f, c);
    alpaka::onHost::wait(queue);

    auto const ok = std::fabs(c[alpaka::Vec{1u, 1u}] - 4.0f) < 1.0e-6f;
    return ok ? 0 : 1;
}
