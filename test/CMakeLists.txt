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

alpakav_add_test_target(unit_test_ ${CMAKE_CURRENT_SOURCE_DIR}/unit)
alpakav_add_test_target(integr_test_ ${CMAKE_CURRENT_SOURCE_DIR}/integr)
if(alpakaV_TESTS)
    alpakav_add_negative_const_output_check()
endif()
