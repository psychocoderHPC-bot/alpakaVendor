#
# SPDX-FileCopyrightText: René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Wrapper project for the CI-less "extern alpaka provider" verification. It adds one
# external alpaka checkout, then adds alpakaVendor with alpakaV_ALPAKA_PROVIDER=extern.
# A consumer such as PIConGPU does exactly this so that only one alpaka exists.
#
#   cmake -S test/externProvider -B build/externProvider \
#         -DALPAKA_SOURCE_DIR=<path-to-alpaka> -DALPAKAVENDOR_SOURCE_DIR=<path-to-alpakaVendor>

cmake_minimum_required(VERSION 3.25)

project(alpakaVendorExternProvider LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT ALPAKA_SOURCE_DIR)
    set(ALPAKA_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../extern/alpaka3")
endif()
if(NOT ALPAKAVENDOR_SOURCE_DIR)
    set(ALPAKAVENDOR_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()

set(alpaka_TESTS OFF CACHE BOOL "" FORCE)
set(alpaka_DOCS OFF CACHE BOOL "" FORCE)
set(alpaka_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(alpaka_EXAMPLES OFF CACHE BOOL "" FORCE)
set(alpaka_HEADERCHECKS OFF CACHE BOOL "" FORCE)
if(DEFINED alpaka_DEP_CUDA AND alpaka_DEP_CUDA)
    enable_language(CUDA)
endif()

# Exactly one alpaka: added here, before alpakaVendor.
add_subdirectory("${ALPAKA_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/alpaka-external")

set(alpakaV_TESTS OFF CACHE BOOL "" FORCE)
set(alpakaV_DOCS OFF CACHE BOOL "" FORCE)
set(alpakaV_ALPAKA_PROVIDER extern CACHE STRING "" FORCE)
add_subdirectory("${ALPAKAVENDOR_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/alpakaVendor-extern")

add_executable(alpakaVendorExternProvider main.cpp)
target_link_libraries(alpakaVendorExternProvider PRIVATE alpaka::alpaka alpakaVendor::alpakaVendor)
alpaka_finalize(alpakaVendorExternProvider)
