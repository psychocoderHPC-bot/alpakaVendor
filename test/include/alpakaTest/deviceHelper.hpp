/* SPDX-FileCopyrightText: Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/api/trait.hpp>
#include <alpaka/onHost/Device.hpp>
#include <alpaka/onHost/DeviceSelector.hpp>
#include <alpaka/onHost/interface.hpp>

#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <tuple>

namespace alpakaVendor::test
{
    /** Takes a test configuration and create the device 0.
     *
     * Prints information about the device Spec, API and device.
     *
     * @attention This function should only be used with the Catch2 macro `TEMPLATE_LIST_TEST_CASE`.
     * See following code snippet:
     *
     * @code
     * using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs,
     * exec::enabledExecutors))>;
     *
     * TEMPLATE_LIST_TEST_CASE("device analysis", "", TestBackends)
     * {
     *    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());
     *    // ...
     * }
     * @endcode
     *
     * @param cfg Test configuration. An entry of the list returned from alpaka::onHost::allBackends().
     * @return The device 0 if available. Otherwise, SKIP() the test.
     */
    [[nodiscard]] auto getDeviceOrSkipTest(auto const& cfg)
        -> decltype(alpaka::onHost::makeDeviceSelector(cfg).makeDevice(0))
    {
        auto deviceSpec = alpaka::onHost::DeviceSpec{cfg};
        auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        UNSCOPED_INFO("DeviceSpec: " << alpaka::onHost::getName(deviceSpec));
        UNSCOPED_INFO("API: " << deviceSpec.getApi().getName());

        if(!devSelector.isAvailable())
        {
            SKIP("No device available for " << deviceSpec.getName());
        }

        alpaka::onHost::Device device = devSelector.makeDevice(0);
        UNSCOPED_INFO("Device: " << device.getName());
        return device;
    }

    /** Takes a test configuration and create the device 0 and extract the executor.
     *
     * Prints information about the device Spec, API, device and executor.
     *
     * @attention This function should only be used with the Catch2 macro `TEMPLATE_LIST_TEST_CASE`.
     * See following code snippet:
     *
     * @code
     * using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs,
     * exec::enabledExecutors))>;
     *
     * TEMPLATE_LIST_TEST_CASE("device analysis", "", TestBackends)
     * {
     *    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
     *    onHost::Device device = test::getDevice(deviceExec);
     *    concepts::Executor auto exec = test::getExecutor(deviceExec);
     *    // ...
     * }
     * @endcode
     *
     * @param cfg Test configuration. An entry of the list returned from alpaka::onHost::allBackends().
     * @return A std::tuple with the device 0 and an executor. If no device is available, SKIP() the test.
     */
    [[nodiscard]] auto getDeviceExecutorOrSkipTest(auto const& cfg) -> std::
        tuple<decltype(alpaka::onHost::makeDeviceSelector(cfg).makeDevice(0)), decltype(cfg[alpaka::object::exec])>
    {
        auto device = getDeviceOrSkipTest(cfg);
        alpaka::concepts::Executor auto executor = cfg[alpaka::object::exec];
        UNSCOPED_INFO("Executor: " << executor.getName());

        return std::make_tuple(device, executor);
    }

    /** Takes a tuple with an alpaka::onHost::Device and an executor and returns the device. */
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind, alpaka::concepts::Executor T_Exec>
    [[nodiscard]] alpaka::onHost::Device<T_Api, T_DeviceKind> getDevice(
        std::tuple<alpaka::onHost::Device<T_Api, T_DeviceKind>, T_Exec> const& deviceExec)
    {
        return std::get<0>(deviceExec);
    }

    /** Takes a tuple with an alpaka::onHost::Device and an executor and returns the executor. */
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind, alpaka::concepts::Executor T_Exec>
    [[nodiscard]] T_Exec getExecutor(std::tuple<alpaka::onHost::Device<T_Api, T_DeviceKind>, T_Exec> const& deviceExec)
    {
        return std::get<1>(deviceExec);
    }
} // namespace alpakaVendor::test
