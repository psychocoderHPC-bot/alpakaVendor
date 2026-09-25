#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

alpaka_install_catch2()
function(alpakaV_add_test_target target_prefix folder)
    file(GLOB_RECURSE test_sources CONFIGURE_DEPENDS "${folder}/*.cpp")
    foreach(source_file ${test_sources})
        get_filename_component(source_file_name ${source_file} NAME_WE)
        get_filename_component(test_folder_name ${source_file} DIRECTORY)
        get_filename_component(test_folder_name ${test_folder_name} NAME)

        set(test_target_name "${target_prefix}${test_folder_name}_${source_file_name}")
        add_executable(${test_target_name} ${source_file})
        target_link_libraries(
            ${test_target_name}
            PRIVATE Catch2::Catch2WithMain alpakaVendor::alpakaVendor alpaka::alpaka
        )
        target_include_directories(
            ${test_target_name}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        if(NOT alpakaV_DEP_FFTW AND NOT alpakaV_DEP_OPENBLAS)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Host_Cpu ALPAKA_DISABLE_Host_NumaCpu)
        endif()
        if(NOT alpakaV_DEP_CUFFT)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Cuda_NvidiaGpu)
        endif()
        if(NOT alpakaV_DEP_CUBLAS)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Cuda_NvidiaGpu)
        endif()
        if(NOT alpakaV_DEP_ROCFFT)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Hip_AmdGpu)
        endif()
        if(NOT alpakaV_DEP_ROCBLAS)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Hip_AmdGpu)
        endif()
        if(NOT alpakaV_DEP_ONEMKL)
            target_compile_definitions(
                ${test_target_name}
                PRIVATE
                    ALPAKA_DISABLE_OneApi_Cpu
                    ALPAKA_DISABLE_OneApi_IntelGpu
                    ALPAKA_DISABLE_OneApi_NvidiaGpu
                    ALPAKA_DISABLE_OneApi_AmdGpu
            )
        endif()
        alpaka_finalize(${test_target_name})
        catch_discover_tests(${test_target_name})
    endforeach()
endfunction()

# The const-element output contract of the BLAS writers is a negative compile-time guarantee: the guarded half of
# constOutputGuards.cpp must NOT compile (dev-identical rejection, see the file header). A dedicated target compiles
# that translation unit with ALPAKAV_BLAS_TEST_NEGATIVE_CONST_OUTPUT defined and is expected to fail, so ctest
# enforces the const-output regression class machine-checked (e.g. a scal accepting a const output would silently
# "fix" the compile and fail the WILL_FAIL test below).
function(alpakaV_add_negative_const_output_check)
    set(test_target_name "unit_test_blas_constOutputGuards_neg")
    add_executable(${test_target_name} EXCLUDE_FROM_ALL "${CMAKE_CURRENT_SOURCE_DIR}/unit/blas/constOutputGuards.cpp")
    target_link_libraries(${test_target_name} PRIVATE Catch2::Catch2WithMain alpakaVendor::alpakaVendor alpaka::alpaka)
    target_include_directories(
        ${test_target_name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(${test_target_name} PRIVATE ALPAKAV_BLAS_TEST_NEGATIVE_CONST_OUTPUT)
    if(NOT alpakaV_DEP_FFTW AND NOT alpakaV_DEP_OPENBLAS)
        target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Host_Cpu ALPAKA_DISABLE_Host_NumaCpu)
    endif()
    if(NOT alpakaV_DEP_CUFFT)
        target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Cuda_NvidiaGpu)
    endif()
    if(NOT alpakaV_DEP_CUBLAS)
        target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Cuda_NvidiaGpu)
    endif()
    if(NOT alpakaV_DEP_ROCFFT)
        target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Hip_AmdGpu)
    endif()
    if(NOT alpakaV_DEP_ROCBLAS)
        target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Hip_AmdGpu)
    endif()
    if(NOT alpakaV_DEP_ONEMKL)
        target_compile_definitions(
            ${test_target_name}
            PRIVATE
                ALPAKA_DISABLE_OneApi_Cpu
                ALPAKA_DISABLE_OneApi_IntelGpu
                ALPAKA_DISABLE_OneApi_NvidiaGpu
                ALPAKA_DISABLE_OneApi_AmdGpu
        )
    endif()
    alpaka_finalize(${test_target_name})
    # The compile is expected to fail: WILL_FAIL turns the build failure into a passing test, so a regression that
    # makes a const output form flips the result to "failed" and the CI turns red.
    add_test(
        NAME blas_const_output_negative_compile
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${test_target_name} --config $<CONFIG>
    )
    set_tests_properties(blas_const_output_negative_compile PROPERTIES WILL_FAIL TRUE)
endfunction()

# Regression guard against re-introducing bare 32-bit narrowing on a BLAS descriptor field in any backend
# dispatch. Every descriptor value must go through checkedVendorInt/checkedCast; a bare `int(xd.n)` or
# `static_cast<int>(ad.ld)` silently truncates. The guard scans the backend headers for an `int(` or
# `static_cast<int>(` immediately applied to a descriptor field (xd/yd/ad/bd/cd) and fails the test if one
# appears. The only allowed `int(...)` uses are error-message formatting such as `int(status)`, which does not
# name a descriptor field and therefore does not match.
function(alpakaV_add_blas_descriptor_narrowing_guard)
    set(guard_script "${CMAKE_CURRENT_BINARY_DIR}/check_blas_descriptor_narrowing.cmake")
    file(
        WRITE "${guard_script}"
        [==[
# Generated by test/CMakeLists.txt; scans the checked-in backend headers for bare descriptor narrowing.
file(GLOB_RECURSE headers "${ALPAKAV_SOURCE_DIR}/include/alpaka/blas/internal/api/*.hpp")
if(NOT headers)
    message(FATAL_ERROR "No BLAS backend headers found under ${ALPAKAV_SOURCE_DIR}")
endif()
set(violations "")
foreach(header ${headers})
    file(READ "${header}" content)
    # `int(` / `static_cast<int>(` optionally separated by whitespace from a descriptor field access.
    string(
        REGEX MATCHALL
        "(int|static_cast[ \t]*<[ \t]*int[ \t]*>)[ \t\r\n]*\([ \t\r\n]*(xd|yd|ad|bd|cd)\."
        matches
        "${content}"
    )
    foreach(match ${matches})
        string(REGEX REPLACE ".*\n" "" last_line "${content}")
        set(violations "${violations}
  ${header}: ${match}")
    endforeach()
endforeach()
if(NOT violations STREQUAL "")
    message(FATAL_ERROR "Bare 32-bit narrowing on a BLAS descriptor field (use checkedVendorInt/checkedCast):${violations}")
endif()
message(STATUS "blas descriptor narrowing guard: no bare int()/static_cast<int> found")
]==]
    )
    add_test(
        NAME blas_descriptor_narrowing_guard
        COMMAND ${CMAKE_COMMAND} -DALPAKAV_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/.. -P "${guard_script}"
    )
endfunction()

alpakav_add_test_target(unit_test_ ${CMAKE_CURRENT_SOURCE_DIR}/unit)
alpakav_add_test_target(integr_test_ ${CMAKE_CURRENT_SOURCE_DIR}/integr)
if(alpakaV_TESTS)
    alpakav_add_negative_const_output_check()
    alpakaV_add_blas_descriptor_narrowing_guard()
endif()
