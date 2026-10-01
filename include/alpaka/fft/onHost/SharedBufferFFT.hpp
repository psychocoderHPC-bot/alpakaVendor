/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/fft/common.hpp"
#include "alpaka/fft/internal/utility.hpp"

#include <alpaka/mem/concepts/detail/InnerTypeAllowedCast.hpp>
#include <alpaka/onHost/mem/ManagedDealloc.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace alpaka::fft::onHost
{
    template<alpaka::concepts::Vector T_Extents>
    struct FftBufferMetadata
    {
        std::optional<FftBufferExtents<T_Extents>> extents{};
    };

    template<alpaka::concepts::Api T_Api, typename T_Type, alpaka::concepts::Vector T_Extents>
    struct SharedBufferFftState
    {
        std::shared_ptr<alpaka::onHost::internal::ManagedDealloc> deleter{};
        std::shared_ptr<FftBufferMetadata<T_Extents>> metadata{};
    };

    [[nodiscard]] constexpr auto getRealExtents(auto const& metadata)
    {
        if(!metadata.extents.has_value())
            throw std::invalid_argument("SharedBufferFFT does not contain FFT extent metadata.");
        return metadata.extents->logicalRealExtents;
    }

    [[nodiscard]] constexpr auto getComplexExtents(auto const& metadata)
    {
        if(!metadata.extents.has_value())
            throw std::invalid_argument("SharedBufferFFT does not contain FFT extent metadata.");
        return metadata.extents->logicalComplexExtents;
    }

    [[nodiscard]] constexpr auto getPhysicalRealExtents(auto const& metadata)
    {
        if(!metadata.extents.has_value())
            throw std::invalid_argument("SharedBufferFFT does not contain FFT extent metadata.");
        return metadata.extents->physicalRealExtents;
    }

    /**
     * Shared alpaka view with FFT storage metadata.
     *
     * The same allocation can be reinterpreted between matching real and complex FFT views while keeping shared
     * lifetime management and padded-storage information.
     */
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment = alpaka::Alignment<>>
    struct SharedBufferFFT : alpaka::View<T_Api, T_Type, T_Extents, T_MemAlignment>
    {
        using BaseView = alpaka::View<T_Api, T_Type, T_Extents, T_MemAlignment>;
        using value_type = T_Type;
        using extents_type = T_Extents;

        std::shared_ptr<alpaka::onHost::internal::ManagedDealloc> m_deleter{};
        std::shared_ptr<FftBufferMetadata<T_Extents>> m_metadata{};
        std::size_t m_capacityBytes = 0u;

        SharedBufferFFT() = default;

        SharedBufferFFT(
            alpaka::concepts::HasApi auto const& any,
            T_Type* data,
            alpaka::concepts::Vector auto const& extents,
            alpaka::concepts::Vector auto const& pitches,
            std::shared_ptr<alpaka::onHost::internal::ManagedDealloc> deleter,
            std::shared_ptr<FftBufferMetadata<T_Extents>> metadata,
            std::size_t capacityBytes,
            T_MemAlignment const memAlignment = T_MemAlignment{})
            : BaseView{any, data, extents, pitches, memAlignment}
            , m_deleter{std::move(deleter)}
            , m_metadata{std::move(metadata)}
            , m_capacityBytes{capacityBytes}
        {
        }

        template<typename T_OtherType>
        requires alpaka::internal::concepts::InnerTypeAllowedCast<T_Type, T_OtherType>
        SharedBufferFFT(SharedBufferFFT<T_Api, T_OtherType, T_Extents, T_MemAlignment> const& other)
            : BaseView{static_cast<alpaka::View<T_Api, T_OtherType, T_Extents, T_MemAlignment> const&>(other)}
            , m_deleter{other.m_deleter}
            , m_metadata{other.m_metadata}
            , m_capacityBytes{other.m_capacityBytes}
        {
        }

        template<typename T_OtherType>
        requires alpaka::internal::concepts::InnerTypeAllowedCast<T_Type, T_OtherType>
        SharedBufferFFT(SharedBufferFFT<T_Api, T_OtherType, T_Extents, T_MemAlignment>&& other)
            : BaseView{std::move(static_cast<alpaka::View<T_Api, T_OtherType, T_Extents, T_MemAlignment>&>(other))}
            , m_deleter{std::move(other.m_deleter)}
            , m_metadata{std::move(other.m_metadata)}
            , m_capacityBytes{std::exchange(other.m_capacityBytes, 0u)}
        {
        }

        SharedBufferFFT(SharedBufferFFT const&) = default;
        SharedBufferFFT(SharedBufferFFT&&) = default;
        SharedBufferFFT& operator=(SharedBufferFFT const&) = default;
        SharedBufferFFT& operator=(SharedBufferFFT&&) = default;

        [[nodiscard]] auto getView() const
        {
            return BaseView::getConstView();
        }

        [[nodiscard]] auto getView()
        {
            return static_cast<BaseView>(*this);
        }

        /**
         * Register work that runs when the last shared view releases the allocation.
         *
         * Actions execute during destruction, so they should not depend on temporaries that may already be gone.
         */
        void addDestructorAction(std::function<void()>&& action)
        {
            m_deleter->addAction(ALPAKA_FORWARD(action));
        }

        /**
         * Delay final destruction until `alpaka::onHost::wait(any)` completes.
         *
         * This is useful when the buffer may still be referenced by asynchronous work at the moment the last host
         * handle disappears.
         */
        void destructorWaitFor(auto const& any)
        {
            addDestructorAction([any]() { alpaka::onHost::wait(any); });
        }

        [[nodiscard]] auto byteCapacity() const noexcept
        {
            return m_capacityBytes;
        }

        [[nodiscard]] auto getUseCount() const noexcept
        {
            return m_deleter.use_count();
        }

        [[nodiscard]] explicit operator bool() const noexcept
        {
            return static_cast<bool>(m_deleter);
        }

        [[nodiscard]] auto const& fftMetadata() const
        {
            return *m_metadata;
        }

        /**
         * Reinterpret the same bytes as another element type and extents.
         *
         * No data is rearranged. The caller must provide extents whose addressed byte range fits into the original
         * allocation and whose layout matches the way the backend produced the data.
         */
        template<typename T_Other>
        [[nodiscard]] auto reinterpretBuffer(alpaka::concepts::VectorOrScalar auto const& extents) const
        {
            using OtherExtents = typename T_Extents::UniVec;
            auto const extentsVec = internal::normalizeVectorOrScalar<OtherExtents>(extents);
            auto const neededBytes = [&]()
            {
                std::size_t r = sizeof(T_Other);
                for(uint32_t i = 0u; i < OtherExtents::dim(); ++i)
                    r *= static_cast<std::size_t>(extentsVec[i]);
                return r;
            }();
            if(neededBytes > m_capacityBytes)
                throw std::invalid_argument("SharedBufferFFT reinterpretation exceeds byte capacity.");
            auto pitches = alpaka::calculatePitchesFromExtents<T_Other>(extentsVec);
            return SharedBufferFFT<T_Api, T_Other, OtherExtents, T_MemAlignment>{
                T_Api{},
                reinterpret_cast<T_Other*>(internal::removeCvPtr(this->data())),
                extentsVec,
                pitches,
                m_deleter,
                m_metadata,
                m_capacityBytes,
                T_MemAlignment{}};
        }

        /**
         * Return the logical complex FFT view for this storage.
         *
         * For real allocations this hides any padded tail elements that exist only to satisfy in-place R2C layout
         * requirements.
         */
        [[nodiscard]] auto asComplex() const
        {
            if constexpr(ComplexScalar<T_Type>)
                return *this;
            else
                return this->template reinterpretBuffer<Complex_t<T_Type>>(getComplexExtents(*m_metadata));
        }

        /**
         * Return the logical real FFT view for this storage.
         *
         * For complex allocations this reconstructs the matching real extents from Hermitian-packed storage, so the
         * visible extent may be smaller than the underlying physical allocation.
         */
        [[nodiscard]] auto asReal() const
        {
            if constexpr(RealScalar<T_Type>)
                return *this;
            else
                return this->template reinterpretBuffer<Real_t<T_Type>>(getRealExtents(*m_metadata));
        }
    };
} // namespace alpaka::fft::onHost

namespace alpaka::internal
{
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct GetApi::Op<alpaka::fft::onHost::SharedBufferFFT<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        constexpr auto operator()(auto&&) const
        {
            return T_Api{};
        }
    };

    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct CopyConstructableDataSource<alpaka::fft::onHost::SharedBufferFFT<T_Api, T_Type, T_Extents, T_MemAlignment>>
        : std::true_type
    {
        using InnerMutable
            = alpaka::fft::onHost::SharedBufferFFT<T_Api, std::remove_const_t<T_Type>, T_Extents, T_MemAlignment>;
        using InnerConst
            = alpaka::fft::onHost::SharedBufferFFT<T_Api, std::add_const_t<T_Type>, T_Extents, T_MemAlignment>;
    };
} // namespace alpaka::internal

namespace alpaka::onHost
{
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct MakeAccessibleOnAcc::Op<alpaka::fft::onHost::SharedBufferFFT<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        auto operator()(auto&& any) const
        {
            return any.getView();
        }
    };
} // namespace alpaka::onHost
