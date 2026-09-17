/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/internal/api/config.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#if ALPAKAV_HAS_ROCFFT
namespace alpaka::fft::internal
{
    template<typename T>
    struct RocfftTraits;

    template<>
    struct RocfftTraits<float>
    {
        static constexpr rocfft_precision precision = rocfft_precision_single;
    };

    template<>
    struct RocfftTraits<double>
    {
        static constexpr rocfft_precision precision = rocfft_precision_double;
    };

    inline void check(rocfft_status status, char const* what)
    {
        if(status != rocfft_status_success)
            throw std::invalid_argument(
                std::string{what} + " failed with rocFFT error code " + std::to_string(int(status)));
    }

    inline void ensureRocfftSetup()
    {
        static std::once_flag setupFlag;
        std::call_once(
            setupFlag,
            []()
            {
                check(rocfft_setup(), "rocfft_setup");
                std::atexit([]() { static_cast<void>(rocfft_cleanup()); });
            });
    }

    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    struct PlanImpl<alpaka::api::Hip, T_Value, T_Extents>
    {
        using value_type = T_Value;
        using real_type = Real_t<T_Value>;
        using traits = RocfftTraits<real_type>;

        Transform m_transform;
        static constexpr uint32_t T_dim = T_Extents::dim();
        using index_type = alpaka::GetValueType_t<T_Extents>;

        Layout<T_Extents> m_layout;
        PlanOptions m_options;
        rocfft_plan m_forwardPlan = nullptr;
        rocfft_plan m_inversePlan = nullptr;
        rocfft_execution_info m_execInfo = nullptr;
        std::size_t m_workspaceBytes = 0u;

        PlanImpl(auto&, Transform transform, Layout<T_Extents> layout, PlanOptions options)
            : m_transform{transform}
            , m_layout{layout}
            , m_options{options}
        {
            validateConfig();
            ensureRocfftSetup();
            createPlans();
            check(rocfft_execution_info_create(&m_execInfo), "rocfft_execution_info_create");
        }

        PlanImpl(PlanImpl&& other) noexcept
            : m_transform{other.m_transform}
            , m_layout{other.m_layout}
            , m_options{other.m_options}
            , m_forwardPlan{std::exchange(other.m_forwardPlan, nullptr)}
            , m_inversePlan{std::exchange(other.m_inversePlan, nullptr)}
            , m_execInfo{std::exchange(other.m_execInfo, nullptr)}
            , m_workspaceBytes{std::exchange(other.m_workspaceBytes, 0u)}
        {
            std::lock_guard lock(other.m_pendingWaitsMutex);
            m_pendingWaits = std::move(other.m_pendingWaits);
        }

        PlanImpl& operator=(PlanImpl&& other) noexcept
        {
            if(this != &other)
            {
                destroy();
                m_transform = other.m_transform;
                m_layout = other.m_layout;
                m_options = other.m_options;
                m_forwardPlan = std::exchange(other.m_forwardPlan, nullptr);
                m_inversePlan = std::exchange(other.m_inversePlan, nullptr);
                m_execInfo = std::exchange(other.m_execInfo, nullptr);
                m_workspaceBytes = std::exchange(other.m_workspaceBytes, 0u);
                {
                    std::scoped_lock lock(m_pendingWaitsMutex, other.m_pendingWaitsMutex);
                    m_pendingWaits = std::move(other.m_pendingWaits);
                }
            }
            return *this;
        }

        ~PlanImpl()
        {
            waitForPending();
            destroy();
        }

        void destroy() noexcept
        {
            if(m_execInfo != nullptr)
                rocfft_execution_info_destroy(m_execInfo);
            if(m_forwardPlan != nullptr)
                rocfft_plan_destroy(m_forwardPlan);
            if(m_inversePlan != nullptr && m_inversePlan != m_forwardPlan)
                rocfft_plan_destroy(m_inversePlan);
            m_execInfo = nullptr;
            m_forwardPlan = nullptr;
            m_inversePlan = nullptr;
        }

        void validateConfig() const
        {
            validate(T_dim >= 1u && T_dim <= 3u, "FFT only supports dimensions 1..3.");
            validate(m_layout.batch >= 1u, "FFT batch must be >= 1.");
            for(uint32_t i = 0u; i < T_dim; ++i)
                validate(m_layout.extents[i] > static_cast<index_type>(0u), "FFT extents must be non-zero.");
            validate(m_options.normalization == Normalization::none, "Only Normalization::none is supported.");
            if constexpr(ComplexScalar<T_Value>)
                validate(m_transform == Transform::c2c, "Complex plan value type only supports C2C.");
            else
                validate(m_transform != Transform::c2c, "Real plan value type only supports R2C/C2R.");
        }

        [[nodiscard]] auto lengths() const
        {
            std::array<std::size_t, T_dim> result{};
            for(uint32_t i = 0u; i < T_dim; ++i)
                result[i] = static_cast<std::size_t>(m_layout.extents[T_dim - 1u - i]);
            return result;
        }

        [[nodiscard]] static auto reverseToSizeT(alpaka::concepts::Vector auto const& vec)
        {
            std::array<std::size_t, T_dim> result{};
            for(uint32_t i = 0u; i < T_dim; ++i)
                result[i] = static_cast<std::size_t>(vec[T_dim - 1u - i]);
            return result;
        }

        [[nodiscard]] auto inStrides() const
        {
            return reverseToSizeT(
                stridesToElements<index_type>(
                    resolvedInStrides<T_Value>(m_layout, m_transform, m_options.placement),
                    inputElementBytes<T_Value>(m_transform)));
        }

        [[nodiscard]] auto outStrides() const
        {
            return reverseToSizeT(
                stridesToElements<index_type>(
                    resolvedOutStrides<T_Value>(m_layout, m_transform, m_options.placement),
                    outputElementBytes<T_Value>(m_transform)));
        }

        [[nodiscard]] rocfft_result_placement placement() const
        {
            return m_options.placement == Placement::inPlace ? rocfft_placement_inplace : rocfft_placement_notinplace;
        }

        [[nodiscard]] rocfft_array_type inputArrayType() const
        {
            if constexpr(ComplexScalar<T_Value>)
                return rocfft_array_type_complex_interleaved;
            if(m_transform == Transform::r2c)
                return rocfft_array_type_real;
            return rocfft_array_type_hermitian_interleaved;
        }

        [[nodiscard]] rocfft_array_type outputArrayType() const
        {
            if constexpr(ComplexScalar<T_Value>)
                return rocfft_array_type_complex_interleaved;
            if(m_transform == Transform::r2c)
                return rocfft_array_type_hermitian_interleaved;
            return rocfft_array_type_real;
        }

        [[nodiscard]] static rocfft_transform_type transformType(Transform transform, Direction direction)
        {
            if(transform == Transform::c2c)
                return direction == Direction::forward ? rocfft_transform_type_complex_forward
                                                       : rocfft_transform_type_complex_inverse;
            if(transform == Transform::r2c)
                return rocfft_transform_type_real_forward;
            return rocfft_transform_type_real_inverse;
        }

        [[nodiscard]] auto makeDescription() const
        {
            rocfft_plan_description description = nullptr;
            check(rocfft_plan_description_create(&description), "rocfft_plan_description_create");
            auto const inStridesVals = inStrides();
            auto const outStridesVals = outStrides();
            auto const inDistance = distanceToElements<std::size_t>(
                expectedInDistance<T_Value>(m_layout, m_transform, m_options.placement),
                inputElementBytes<T_Value>(m_transform));
            auto const outDistance = distanceToElements<std::size_t>(
                expectedOutDistance<T_Value>(m_layout, m_transform, m_options.placement),
                outputElementBytes<T_Value>(m_transform));
            check(
                rocfft_plan_description_set_data_layout(
                    description,
                    inputArrayType(),
                    outputArrayType(),
                    nullptr,
                    nullptr,
                    T_dim,
                    inStridesVals.data(),
                    inDistance,
                    T_dim,
                    outStridesVals.data(),
                    outDistance),
                "rocfft_plan_description_set_data_layout");
            return description;
        }

        [[nodiscard]] auto createPlanFor(Direction direction) const
        {
            auto dimsVals = lengths();
            auto description = makeDescription();
            rocfft_plan plan = nullptr;
            auto cleanup = [&]()
            {
                if(description != nullptr)
                    rocfft_plan_description_destroy(description);
            };
            check(
                rocfft_plan_create(
                    &plan,
                    placement(),
                    transformType(m_transform, direction),
                    traits::precision,
                    T_dim,
                    dimsVals.data(),
                    static_cast<std::size_t>(m_layout.batch),
                    description),
                "rocfft_plan_create");
            cleanup();
            return plan;
        }

        void createPlans()
        {
            if constexpr(ComplexScalar<T_Value>)
            {
                m_forwardPlan = createPlanFor(Direction::forward);
                m_inversePlan = createPlanFor(Direction::backward);
            }
            else if(m_transform == Transform::r2c)
            {
                m_forwardPlan = createPlanFor(Direction::forward);
                m_inversePlan = nullptr;
            }
            else
            {
                m_forwardPlan = createPlanFor(Direction::backward);
                m_inversePlan = nullptr;
            }

            size_t workSize = 0u;
            check(rocfft_plan_get_work_buffer_size(m_forwardPlan, &workSize), "rocfft_plan_get_work_buffer_size");
            m_workspaceBytes = workSize;
            if(m_inversePlan != nullptr)
            {
                size_t inverseWorkSize = 0u;
                check(
                    rocfft_plan_get_work_buffer_size(m_inversePlan, &inverseWorkSize),
                    "rocfft_plan_get_work_buffer_size");
                m_workspaceBytes = std::max<std::size_t>(m_workspaceBytes, inverseWorkSize);
            }
        }

        [[nodiscard]] std::size_t workspaceBytes() const noexcept
        {
            return m_workspaceBytes;
        }

        void setWorkspace(void* ptr, std::size_t bytes)
        {
            validate(
                m_options.workspacePolicy == WorkspacePolicy::userProvided,
                "Plan does not use user-provided workspace.");
            validate(ptr != nullptr, "Workspace pointer must not be nullptr.");
            validate(bytes >= m_workspaceBytes, "Provided rocFFT workspace is too small.");
            check(
                rocfft_execution_info_set_work_buffer(m_execInfo, ptr, bytes),
                "rocfft_execution_info_set_work_buffer");
        }

        void trackCompletion(auto& queue)
        {
            auto event = queue.getDevice().makeEvent();
            queue.enqueue(event);
            std::lock_guard lock(m_pendingWaitsMutex);
            m_pendingWaits.emplace_back([event]() mutable { alpaka::onHost::wait(event); });
        }

        void execute(auto& queue, auto const& in, auto& out, Direction direction)
        {
            auto* rawInPtr = removeCvPtr(in.data());
            auto* rawOutPtr = removeCvPtr(out.data());
            queue.enqueueNativeFn(
                [=, this](auto cudaStream)
                {
                    check(
                        rocfft_execution_info_set_stream(m_execInfo, reinterpret_cast<void*>(cudaStream)),
                        "rocfft_execution_info_set_stream");

                    rocfft_plan activePlan = nullptr;
                    if constexpr(ComplexScalar<T_Value>)
                    {
                        activePlan = direction == Direction::forward ? m_forwardPlan : m_inversePlan;
                    }
                    else if(m_transform == Transform::r2c)
                    {
                        validate(direction == Direction::forward, "R2C only supports forward execution.");
                        activePlan = m_forwardPlan;
                    }
                    else
                    {
                        validate(direction == Direction::backward, "C2R only supports backward execution.");
                        activePlan = m_forwardPlan;
                    }

                    void* inBuffers[] = {reinterpret_cast<void*>(rawInPtr)};
                    void* outBuffers[] = {reinterpret_cast<void*>(rawOutPtr)};
                    check(
                        rocfft_execute(
                            activePlan,
                            inBuffers,
                            m_options.placement == Placement::inPlace ? nullptr : outBuffers,
                            m_execInfo),
                        "rocfft_execute");
                });
        }

    private:
        void waitForPending() noexcept
        {
            std::vector<std::function<void()>> waits;
            {
                std::lock_guard lock(m_pendingWaitsMutex);
                waits.swap(m_pendingWaits);
            }
            for(auto& waitFn : waits)
            {
                try
                {
                    waitFn();
                }
                catch(...)
                {
                }
            }
        }

        std::mutex m_pendingWaitsMutex;
        std::vector<std::function<void()>> m_pendingWaits;
    };
} // namespace alpaka::fft::internal
#endif
