/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <memory>
#include <type_traits>
#include <utility>

#include "SharedBufferFFT.hpp"

namespace alpaka::fft::onHost::internal
{
    template<typename T_Type, alpaka::concepts::Vector T_FftExtents>
    [[nodiscard]] auto wrapBuffer(auto&& buffer, FftBufferExtents<T_FftExtents> const& fftExtents)
    {
        using Buffer = ALPAKA_TYPEOF(buffer);
        using Api = decltype(alpaka::getApi(std::declval<Buffer>()));
        using ExtentsVec = decltype(buffer.getExtents());
        auto rawOwner = std::make_shared<Buffer>(ALPAKA_FORWARD(buffer));
        auto managedDeleter = std::make_shared<alpaka::onHost::internal::ManagedDealloc>([owner = rawOwner]() mutable
                                                                                         { owner.reset(); });
        auto metadata = std::make_shared<FftBufferMetadata<ExtentsVec>>();
        metadata->extents = FftBufferExtents<ExtentsVec>{
            .logicalRealExtents = alpaka::fft::internal::castVec<ExtentsVec>(fftExtents.logicalRealExtents),
            .physicalRealExtents = alpaka::fft::internal::castVec<ExtentsVec>(fftExtents.physicalRealExtents),
            .logicalComplexExtents = alpaka::fft::internal::castVec<ExtentsVec>(fftExtents.logicalComplexExtents)};
        std::size_t bytes = sizeof(T_Type);
        for(uint32_t i = 0u; i < ExtentsVec::dim(); ++i)
            bytes *= static_cast<std::size_t>(rawOwner->getExtents()[i]);
        return SharedBufferFFT<Api, T_Type, ExtentsVec, alpaka::Alignment<>>{
            Api{},
            const_cast<T_Type*>(rawOwner->data()),
            rawOwner->getExtents(),
            rawOwner->getPitches(),
            std::move(managedDeleter),
            std::move(metadata),
            bytes,
            alpaka::Alignment<>{}};
    }

    template<typename T_Type>
    [[nodiscard]] auto fftAllocationExtents(alpaka::concepts::Vector auto const& logicalExtents)
    {
        auto storage = makeFftBufferExtents<T_Type>(logicalExtents);
        if constexpr(RealScalar<T_Type>)
            return storage.physicalRealExtents;
        else
            return storage.logicalComplexExtents;
    }
} // namespace alpaka::fft::onHost::internal

namespace alpaka::fft::onHost
{
    /** Allocate an FFT-managed buffer.
     *
     * Real-valued allocations reserve padded physical storage so the returned buffer can be reinterpreted between
     * real and complex FFT views. Complex-valued allocations keep their logical complex extents and can be
     * reinterpreted back to their matching real FFT view.
     *
     * This allocation is required for in-place FFT transforms where the same buffer is used for both input and
     * output with automatic padding management.
     *
     * For manual, non-FFT-managed storage use `alpaka::onHost::alloc()` directly.
     */
    template<typename T_Type>
    [[nodiscard]] auto alloc(
        alpaka::onHost::internal::concepts::Device auto const& device,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        auto logicalExtents = alpaka::fft::internal::asExtentVec(extents);
        auto storage = makeFftBufferExtents<T_Type>(logicalExtents);
        auto fftExtents = internal::fftAllocationExtents<T_Type>(logicalExtents);
        return internal::wrapBuffer<T_Type>(alpaka::onHost::alloc<T_Type>(device, fftExtents), storage);
    }

    /** Allocate a unified FFT-managed buffer.
     *
     * Real-valued allocations reserve padded physical storage so the returned buffer can be reinterpreted between
     * real and complex FFT views. Complex-valued allocations keep their logical complex extents and can be
     * reinterpreted back to their matching real FFT view.
     *
     * This allocation is required for in-place FFT transforms where the same buffer is used for both input and
     * output with automatic padding management.
     *
     * For manual, non-FFT-managed storage use `alpaka::onHost::allocUnified()` directly.
     */
    template<typename T_Type>
    [[nodiscard]] auto allocUnified(
        alpaka::onHost::internal::concepts::Device auto const& device,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        auto logicalExtents = alpaka::fft::internal::asExtentVec(extents);
        auto storage = makeFftBufferExtents<T_Type>(logicalExtents);
        auto fftExtents = internal::fftAllocationExtents<T_Type>(logicalExtents);
        return internal::wrapBuffer<T_Type>(alpaka::onHost::allocUnified<T_Type>(device, fftExtents), storage);
    }

    /** Allocate a mapped FFT-managed buffer.
     *
     * The returned buffer stores FFT reinterpretation metadata and may be converted between matching real and complex
     * FFT views.
     *
     * This allocation is required for in-place FFT transforms where the same buffer is used for both input and
     * output with automatic padding management.
     */
    template<typename T_Type>
    [[nodiscard]] auto allocMapped(
        alpaka::onHost::internal::concepts::Device auto const& device,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        auto logicalExtents = alpaka::fft::internal::asExtentVec(extents);
        auto storage = makeFftBufferExtents<T_Type>(logicalExtents);
        auto fftExtents = internal::fftAllocationExtents<T_Type>(logicalExtents);
        return internal::wrapBuffer<T_Type>(alpaka::onHost::allocMapped<T_Type>(device, fftExtents), storage);
    }

    /** Allocate a deferred FFT-managed buffer from a queue.
     *
     * This allocation is required for in-place FFT transforms where the same buffer is used for both input and
     * output with automatic padding management.
     */
    template<typename T_Type>
    [[nodiscard]] auto allocDeferred(
        alpaka::onHost::internal::concepts::Queue auto const& queue,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        auto logicalExtents = alpaka::fft::internal::asExtentVec(extents);
        auto storage = makeFftBufferExtents<T_Type>(logicalExtents);
        auto fftExtents = internal::fftAllocationExtents<T_Type>(logicalExtents);
        return internal::wrapBuffer<T_Type>(alpaka::onHost::allocDeferred<T_Type>(queue, fftExtents), storage);
    }
} // namespace alpaka::fft::onHost
