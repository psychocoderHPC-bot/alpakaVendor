#
# SPDX-FileCopyrightText: Rene Widera
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.25)

alpaka_install_catch2()

# Collect all code snippets
file(GLOB DOC_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/code/*.cpp")

foreach(source_file ${DOC_SOURCES})
    get_filename_component(source_name ${source_file} NAME_WE)

    # Backend-agnostic target - compiles for all enabled backends
    set(target_name "doc_${source_name}")
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE Catch2::Catch2WithMain alpakaVendor::alpakaVendor alpaka::alpaka)
    target_include_directories(
        ${target_name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../test/unit ${CMAKE_CURRENT_SOURCE_DIR}/../test/include
    )
    alpaka_finalize(${target_name})
    set_target_properties(
        ${target_name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/doc" CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    )

    # Register as a test
    catch_discover_tests(${target_name})
endforeach()

# Sphinx documentation build
find_program(SPHINX_EXECUTABLE sphinx-build)
if(SPHINX_EXECUTABLE)
    add_custom_target(
        docs
        COMMAND ${SPHINX_EXECUTABLE} -b html "${CMAKE_CURRENT_SOURCE_DIR}/source" "${CMAKE_BINARY_DIR}/doc/html"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Building HTML documentation with Sphinx"
        VERBATIM
    )
else()
    message(STATUS "sphinx-build not found, 'docs' target will not be available")
endif()
