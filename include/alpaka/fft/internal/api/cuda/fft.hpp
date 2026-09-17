/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/fft/internal/api/config.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#if ALPAKAV_HAS_CUFFT
namespace alpaka::fft::internal
{
    template<typename T>
    struct CufftTraits;

    template<>
    struct CufftTraits<float>
    {
        using complex_type = cufftComplex;
        static constexpr cufftType c2c = CUFFT_C2C;
        static constexpr cufftType r2c = CUFFT_R2C;
        static constexpr cufftType c2r = CUFFT_C2R;
    };

    template<>
    struct CufftTraits<double>
    {
        using complex_type = cufftDoubleComplex;
        static constexpr cufftType c2c = CUFFT_Z2Z;
        static constexpr cufftType r2c = CUFFT_D2Z;
        static constexpr cufftType c2r = CUFFT_Z2D;
    };

    template<typename T_Value, alpaka::concepts::Vector T_Extents>
    struct PlanImpl<alpaka::api::Cuda, T_Value, T_Extents>
    {
        using value_type = T_Value;
        using real_type = Real_t<T_Value>;
        using traits = CufftTraits<real_type>;

        Transform m_transform;
        static constexpr uint32_t T_dim = T_Extents::dim();
        using index_type = alpaka::GetValueType_t<T_Extents>;

        Layout<T_Extents> m_layout;
        PlanOptions m_options;
        cufftHandle m_handle = 0;
        std::size_t m_workspaceBytes = 0u;

        PlanImpl(auto& queue, Transform transform, Layout<T_Extents> layout, PlanOptions options)
            : m_transform{transform}
            , m_layout{layout}
            , m_options{options}
        {
            validateConfig();
            createPlan(queue);
        }

        PlanImpl(PlanImpl&& other) noexcept
            : m_transform{other.m_transform}
            , m_layout{other.m_layout}
            , m_options{other.m_options}
            , m_handle{std::exchange(other.m_handle, 0)}
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
                m_handle = std::exchange(other.m_handle, 0);
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
            if(m_handle != 0)
                cufftDestroy(m_handle);
            m_handle = 0;
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

        [[nodiscard]] auto dims() const
        {
            std::array<long long, T_dim> result{};
            for(uint32_t i = 0u; i < T_dim; ++i)
                result[i] = static_cast<long long>(m_layout.extents[i]);
            return result;
        }

        [[nodiscard]] auto inEmbed() const
        {
            auto const ext = expectedInputExtents(m_layout, m_transform, m_options.placement);
            auto const strides = stridesToElements<long long>(
                resolvedInStrides<T_Value>(m_layout, m_transform, m_options.placement),
                inputElementBytes<T_Value>(m_transform));
            return embedsFromStrides(strides, ext);
        }

        [[nodiscard]] auto outEmbed() const
        {
            auto const ext = expectedOutputExtents(m_layout, m_transform, m_options.placement);
            auto const strides = stridesToElements<long long>(
                resolvedOutStrides<T_Value>(m_layout, m_transform, m_options.placement),
                outputElementBytes<T_Value>(m_transform));
            return embedsFromStrides(strides, ext);
        }

        [[nodiscard]] auto inputStride() const
        {
            auto const strides = stridesToElements<long long>(
                resolvedInStrides<T_Value>(m_layout, m_transform, m_options.placement),
                inputElementBytes<T_Value>(m_transform));
            return strides[T_dim - 1u];
        }

        [[nodiscard]] auto outputStride() const
        {
            auto const strides = stridesToElements<long long>(
                resolvedOutStrides<T_Value>(m_layout, m_transform, m_options.placement),
                outputElementBytes<T_Value>(m_transform));
            return strides[T_dim - 1u];
        }

        [[nodiscard]] cufftType type() const
        {
            if(m_transform == Transform::c2c)
                return traits::c2c;
            if(m_transform == Transform::r2c)
                return traits::r2c;
            return traits::c2r;
        }

        static void check(cufftResult result, char const* what)
        {
            if(result != CUFFT_SUCCESS)
                throw std::invalid_argument(
                    std::string{what} + " failed with cuFFT error code " + std::to_string(int(result)));
        }

        void createPlan(auto& queue)
        {
            auto dimsVals = dims();
            auto inEmbedVals = inEmbed();
            auto outEmbedVals = outEmbed();
            auto const inStride = inputStride();
            auto const outStride = outputStride();
            check(cufftCreate(&m_handle), "cufftCreate");
            check(cufftSetStream(m_handle, queue.getNativeHandle()), "cufftSetStream");
            check(
                cufftSetAutoAllocation(m_handle, m_options.workspacePolicy == WorkspacePolicy::backendManaged ? 1 : 0),
                "cufftSetAutoAllocation");
            size_t workSize = 0u;
            check(
                cufftMakePlanMany64(
                    m_handle,
                    static_cast<int>(T_dim),
                    dimsVals.data(),
                    inEmbedVals.data(),
                    inStride,
                    static_cast<long long>(distanceToElements<long long>(
                        expectedInDistance<T_Value>(m_layout, m_transform, m_options.placement),
                        inputElementBytes<T_Value>(m_transform))),
                    outEmbedVals.data(),
                    outStride,
                    static_cast<long long>(distanceToElements<long long>(
                        expectedOutDistance<T_Value>(m_layout, m_transform, m_options.placement),
                        outputElementBytes<T_Value>(m_transform))),
                    type(),
                    static_cast<long long>(m_layout.batch),
                    &workSize),
                "cufftMakePlanMany64");
            m_workspaceBytes = workSize;
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
            validate(bytes >= m_workspaceBytes, "Provided cuFFT workspace is too small.");
            check(cufftSetWorkArea(m_handle, ptr), "cufftSetWorkArea");
        }

        void trackCompletion(auto& queue)
        {
            auto event = queue.getDevice().makeEvent();
            queue.enqueue(event);
            std::lock_guard lock(m_pendingWaitsMutex);
            m_pendingWaits.emplace_back([event]() mutable { alpaka::onHost::wait(event); });
        }

        static void executeComplex(cufftHandle handle, auto* rawInPtr, auto* rawOutPtr, Direction direction)
        {
            if constexpr(std::same_as<real_type, float>)
                check(
                    cufftExecC2C(
                        handle,
                        reinterpret_cast<cufftComplex*>(rawInPtr),
                        reinterpret_cast<cufftComplex*>(rawOutPtr),
                        direction == Direction::forward ? CUFFT_FORWARD : CUFFT_INVERSE),
                    "cufftExecC2C");
            else
                check(
                    cufftExecZ2Z(
                        handle,
                        reinterpret_cast<cufftDoubleComplex*>(rawInPtr),
                        reinterpret_cast<cufftDoubleComplex*>(rawOutPtr),
                        direction == Direction::forward ? CUFFT_FORWARD : CUFFT_INVERSE),
                    "cufftExecZ2Z");
        }

        static void executeReal(cufftHandle handle, auto* rawInPtr, auto* rawOutPtr, Direction direction)
        {
            using InValue = std::remove_cv_t<std::remove_pointer_t<ALPAKA_TYPEOF(rawInPtr)>>;
            if constexpr(std::same_as<InValue, real_type>)
            {
                validate(direction == Direction::forward, "R2C only supports forward execution.");
                if constexpr(std::same_as<real_type, float>)
                    check(cufftExecR2C(handle, rawInPtr, reinterpret_cast<cufftComplex*>(rawOutPtr)), "cufftExecR2C");
                else
                    check(
                        cufftExecD2Z(handle, rawInPtr, reinterpret_cast<cufftDoubleComplex*>(rawOutPtr)),
                        "cufftExecD2Z");
            }
            else
            {
                validate(direction == Direction::backward, "C2R only supports backward execution.");
                if constexpr(std::same_as<real_type, float>)
                    check(cufftExecC2R(handle, reinterpret_cast<cufftComplex*>(rawInPtr), rawOutPtr), "cufftExecC2R");
                else
                    check(
                        cufftExecZ2D(handle, reinterpret_cast<cufftDoubleComplex*>(rawInPtr), rawOutPtr),
                        "cufftExecZ2D");
            }
        }

        void execute(auto& queue, auto const& in, auto& out, Direction direction)
        {
            auto* rawInPtr = removeCvPtr(in.data());
            auto* rawOutPtr = removeCvPtr(out.data());
            using QueueNative = decltype(queue.getNativeHandle());
            if constexpr(ComplexScalar<T_Value>)
            {
                queue.enqueueNativeFn(
                    [=, this](QueueNative cudaStream)
                    {
                        check(cufftSetStream(m_handle, cudaStream), "cufftSetStream");
                        // Calling a helper function instead of the fft function directly is required because the nvcc
                        // frontend is ignoring `if constexpr`.
                        executeComplex(m_handle, rawInPtr, rawOutPtr, direction);
                    });
            }
            else
            {
                // Calling a helper function instead of the fft function directly is required because the nvcc frontend
                // is ignoring `if constexpr`.
                queue.enqueueNativeFn(
                    [=, this](QueueNative cudaStream)
                    {
                        check(cufftSetStream(m_handle, cudaStream), "cufftSetStream");
                        executeReal(m_handle, rawInPtr, rawOutPtr, direction);
                    });
            }
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
