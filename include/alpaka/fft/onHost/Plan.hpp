/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/internal/api/fft.hpp"
#include "alpaka/fft/onHost/alloc.hpp"

#include <memory>

#include "SharedBufferFFT.hpp"

namespace alpaka::fft::onHost
{
    template<typename T_Value, alpaka::concepts::Vector T_Extents = alpaka::fft::Extents<uint32_t, 1u>>
    class PlanBuilder;

    template<typename T_Api, typename T_Value, alpaka::concepts::Vector T_Extents>
    class Plan
    {
    public:
        using api_type = T_Api;
        using value_type = T_Value;
        using extents_type = T_Extents;
        using index_type = alpaka::GetValueType_t<T_Extents>;
        static constexpr uint32_t dim = T_Extents::dim();
        using impl_type = alpaka::fft::internal::PlanImpl<T_Api, T_Value, T_Extents>;

        Plan() = delete;

        Plan(auto& queue, Transform transform, Layout<T_Extents> layout, PlanOptions options)
            : m_transform{transform}
            , m_layout{layout}
            , m_options{options}
            , m_impl{std::make_shared<impl_type>(queue, transform, layout, options)}
        {
        }

        Plan(Plan const&) = default;
        Plan& operator=(Plan const&) = default;
        Plan(Plan&&) noexcept = default;
        Plan& operator=(Plan&&) noexcept = default;

        [[nodiscard]] auto transform() const noexcept
        {
            return m_transform;
        }

        [[nodiscard]] auto const& layout() const noexcept
        {
            return m_layout;
        }

        [[nodiscard]] auto const& options() const noexcept
        {
            return m_options;
        }

        [[nodiscard]] auto workspaceBytes() const noexcept
        {
            return m_impl->workspaceBytes();
        }

        /** Bind user-managed workspace to this plan.
         *
         * The storage must remain valid for every later execution that uses this plan. Rebinding replaces the
         * previously configured workspace for subsequent launches.
         *
         * @param span An alpaka mdspan-like object whose byte capacity is at least `workspaceBytes()`.
         */
        void setWorkspace(alpaka::concepts::IMdSpan auto& span)
        {
            auto const bytes
                = alpaka::onHost::getExtents(span).product() * sizeof(alpaka::GetValueType_t<ALPAKA_TYPEOF(span)>);
            m_impl->setWorkspace(span.data(), static_cast<std::size_t>(bytes));
        }

        /** Execute the transform without implicit normalization.
         *
         * Forward and backward transforms use the raw vendor-library convention. A round-trip therefore usually
         * needs an explicit division by the logical FFT size to recover the original values.
         */
        void execute(auto& queue, auto const& in, auto& out, Direction direction)
        {
            m_impl->execute(queue, in, out, direction);
            m_impl->trackCompletion(queue);
        }

        /** Keep the plan alive until the queue reaches this point.
         *
         * This mirrors alpaka's buffer `keepAlive()` helper and is useful when work using the plan has already been
         * enqueued but the last host-side plan handle is about to leave scope.
         */
        void keepAlive(auto& queue) const
        {
            queue.enqueueHostFnDeferred([impl = m_impl] {});
        }

    private:
        Transform m_transform;
        Layout<T_Extents> m_layout;
        PlanOptions m_options;
        std::shared_ptr<impl_type> m_impl;
    };

    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    class PlanBuilder
    {
    public:
        using extents_type = T_Extents;
        using index_type = alpaka::GetValueType_t<T_Extents>;
        static constexpr uint32_t dim = T_Extents::dim();

        /** Construct a plan builder with the given transform extents.
         *
         * Extents are required and must be non-zero. All other parameters have sensible defaults and can be
         * configured through the builder methods.
         */
        PlanBuilder(alpaka::concepts::VectorOrScalar auto const& extents)
            : m_layout{.extents = alpaka::fft::internal::normalizeVectorOrScalar<T_Extents>(extents)}
        {
        }

        /** Select a complex-to-complex transform. */
        PlanBuilder& c2c()
        {
            m_transform = Transform::c2c;
            return *this;
        }

        /** Select a real-to-complex forward transform. */
        PlanBuilder& r2c()
        {
            m_transform = Transform::r2c;
            return *this;
        }

        /** Select a complex-to-real backward transform. */
        PlanBuilder& c2r()
        {
            m_transform = Transform::c2r;
            return *this;
        }

        /** Set the number of transforms stored in the input/output buffers.
         *
         * Values below `1` are invalid and rejected during `build()`.
         */
        PlanBuilder& batch(index_type value)
        {
            m_layout.batch = value;
            return *this;
        }

        /** Override the byte-strides passed to the backend.
         *
         * Strides are expressed in **bytes**, matching the alpaka pitch convention returned by `getPitches()`.
         * Each value is the number of bytes to advance to reach the next element along the corresponding
         * dimension. The last dimension is the fast-moving one. When a backend library requires element strides,
         * the conversion (dividing by `sizeof(value_type)`) is performed automatically.
         *
         * Call order relative to `extents()` does not matter.
         */
        PlanBuilder& strides(
            alpaka::concepts::VectorOrScalar auto const& in,
            alpaka::concepts::VectorOrScalar auto const& out)
        {
            using byte_strides_type = alpaka::Vec<typename Layout<T_Extents>::byte_type, T_Extents::dim()>;
            m_layout.inStrides = alpaka::fft::internal::normalizeVectorOrScalar<byte_strides_type>(in);
            m_layout.outStrides = alpaka::fft::internal::normalizeVectorOrScalar<byte_strides_type>(out);
            return *this;
        }

        /** Set the byte distance between consecutive batches.
         *
         * The distance is the number of **bytes** between the start of batch `i` and batch `i+1` in the
         * input/output buffers. This matches the byte-level layout of alpaka buffers and views. When a backend
         * library requires an element count, the conversion (dividing by `sizeof(value_type)`) is performed
         * automatically.
         *
         * A distance of 0 means the batches are contiguous (distance = product of extents * sizeof(value_type)).
         * Distances smaller than the transform footprint can make batches overlap, which is only safe if that
         * aliasing is intentional.
         *
         * Call order relative to `extents()` does not matter.
         */
        PlanBuilder& distances(std::size_t inDistanceBytes, std::size_t outDistanceBytes)
        {
            m_layout.inDistance = inDistanceBytes;
            m_layout.outDistance = outDistanceBytes;
            return *this;
        }

        /** Request in-place execution.
         *
         * For real/complex transforms the buffer must use the padded FFT storage layout returned by the
         * `alpaka::fft::onHost::alloc*()` helpers or an equivalent manual layout.
         */
        PlanBuilder& inPlace()
        {
            m_options.placement = Placement::inPlace;
            return *this;
        }

        /** Request distinct input and output buffers. */
        PlanBuilder& outOfPlace()
        {
            m_options.placement = Placement::outOfPlace;
            return *this;
        }

        /** Let the backend allocate and own any temporary workspace. */
        PlanBuilder& backendManagedWorkspace()
        {
            m_options.workspacePolicy = WorkspacePolicy::backendManaged;
            return *this;
        }

        /** Require the caller to provide workspace explicitly.
         *
         * Build succeeds only on backends that support external work areas. Use `workspaceBytes()` on the built
         * plan and call `setWorkspace()` before execution.
         */
        PlanBuilder& userProvidedWorkspace()
        {
            m_options.workspacePolicy = WorkspacePolicy::userProvided;
            return *this;
        }

        /** Create a backend-specific plan for the selected device.
         *
         * The device selects the backend used for plan creation and later execution.
         */
        [[nodiscard]] auto build(alpaka::onHost::internal::concepts::Device auto& device) const
        {
            auto queue = device.makeQueue();
            return makePlanFromQueue(queue);
        }

    private:
        [[nodiscard]] auto makePlanFromQueue(alpaka::onHost::internal::concepts::Queue auto& queue) const
        {
            using Api = decltype(alpaka::getApi(queue));
            return Plan<Api, T_Value, T_Extents>{queue, m_transform, m_layout, m_options};
        }

        Transform m_transform = Transform::c2c;
        Layout<T_Extents> m_layout{};
        PlanOptions m_options{};
    };

    /** Factory function to create a PlanBuilder with the given extents.
     *
     * Equivalent to `PlanBuilder<T_Value, T_Extents>{extents}` but shorter and more consistent with alpaka3
     * naming conventions.
     */
    template<typename T_Value, alpaka::concepts::Vector T_Extents = alpaka::fft::Extents<uint32_t, 1u>>
    [[nodiscard]] auto makePlan(alpaka::concepts::VectorOrScalar auto const& extents)
    {
        return PlanBuilder<T_Value, T_Extents>{extents};
    }

    /** Convenience wrapper for `Direction::forward`.
     *
     * Depending on the backend the first invocation could have a higher latency do to deferred plan creation.
     */
    void executeForward(auto& queue, auto& plan, auto const& in, auto& out)
    {
        plan.execute(queue, in, out, Direction::forward);
    }

    /** Convenience wrapper for `Direction::backward`.
     *
     * Depending on the backend the first invocation could have a higher latency do to deferred plan creation.
     */
    void executeBackward(auto& queue, auto& plan, auto const& in, auto& out)
    {
        plan.execute(queue, in, out, Direction::backward);
    }

    /** Run an in-place real-to-complex transform and return the complex view of the same storage.
     *
     * `buffer` is reinterpreted after execution; the returned view aliases the original allocation. Do not keep
     * using the old real extents as if they still described the transform output.
     * Depending on the backend the first invocation could have a higher latency do to deferred plan creation.
     */
    auto executeR2CInPlace(auto& queue, auto& plan, auto& buffer) requires requires {
        typename ALPAKA_TYPEOF(buffer)::value_type;
        requires RealScalar<typename ALPAKA_TYPEOF(buffer)::value_type>;
    }
    {
        auto complexBuffer = buffer.asComplex();
        plan.execute(queue, buffer, complexBuffer, Direction::forward);
        return complexBuffer;
    }

    /** Run an in-place complex-to-real transform and return the logical real view of the same storage.
     *
     * The returned view aliases `buffer` and exposes only the logical real extents, not the padded physical
     * storage that may still be present in memory.
     * Depending on the backend the first invocation could have a higher latency do to deferred plan creation.
     */
    auto executeC2RInPlace(auto& queue, auto& plan, auto& buffer) requires requires {
        typename ALPAKA_TYPEOF(buffer)::value_type;
        requires ComplexScalar<typename ALPAKA_TYPEOF(buffer)::value_type>;
    }
    {
        auto realBuffer = buffer.asReal();
        plan.execute(queue, buffer, realBuffer, Direction::backward);
        return realBuffer;
    }
} // namespace alpaka::fft::onHost
