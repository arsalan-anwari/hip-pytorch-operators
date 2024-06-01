cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
cmake_policy(VERSION 3.16...3.25)

# Install prefix
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

# rocPRIM project
project(hip-pytorch-operators LANGUAGES CXX)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(HIP_PYTORCH_OPERATORS_PROJECT_IS_TOP_LEVEL TRUE)
else()
  set(HIP_PYTORCH_OPERATORS_PROJECT_IS_TOP_LEVEL FALSE)
endif()

#Adding CMAKE_PREFIX_PATH
if(WIN32)
  set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
  set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()

# Build options
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_NAIVE_BENCHMARK "Build naive benchmarks" OFF)
option(BUILD_EXAMPLE "Build examples" OFF)
option(USE_HIP_CPU "Prefer HIP-CPU runtime instead of HW acceleration" OFF)

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${ROCM_PATH}/lib/cmake/hip ${HIP_PATH}/cmake ${ROCM_ROOT}/lib/cmake/hip ${ROCM_ROOT}/hip/cmake # FindHIP.cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# Set CXX flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF) # don't build client dependencies as shared
if(NOT USE_HIP_CPU)
  # Get dependencies (required here to get rocm-cmake)
  include(cmake/Dependencies.cmake)
  # Use target ID syntax if supported for GPU_TARGETS
  if (NOT DEFINED AMDGPU_TARGETS)
    set(GPU_TARGETS "all" CACHE STRING "GPU architectures to compile for")
  else()
    set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for")
  endif()
  set_property(CACHE GPU_TARGETS PROPERTY STRINGS "all")

  if(GPU_TARGETS STREQUAL "all")
    rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
      TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102"
    )
    set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
  endif()

  # TODO: Fix VerifyCompiler for HIP on Windows
  if (NOT WIN32)
    include(cmake/VerifyCompiler.cmake)
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/llvm ${ROCM_ROOT}/llvm ${ROCM_ROOT} ${ROCM_ROOT}/hip)
  find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)
endif()

if(USE_HIP_CPU)
  # Get dependencies
  include(cmake/Dependencies.cmake)
endif()

# Setup VERSION
set(VERSION_STRING "3.2.0")
rocm_setup_version(VERSION ${VERSION_STRING})

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# rocPRIM library
add_subdirectory(hip-pytorch-operators)

if(HIP_PYTORCH_OPERATORS_PROJECT_IS_TOP_LEVEL AND (BUILD_TEST OR BUILD_BENCHMARK))
  rocm_package_setup_component(clients)
endif()

# Tests
if(BUILD_TEST)
  if (HIP_PYTORCH_OPERATORS_PROJECT_IS_TOP_LEVEL)
    rocm_package_setup_client_component(tests)
  endif()
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks
if(BUILD_BENCHMARK)
  if (HIP_PYTORCH_OPERATORS_PROJECT_IS_TOP_LEVEL)
    rocm_package_setup_client_component(benchmarks)
  endif()
  add_subdirectory(benchmark)
endif()

# Examples
if(BUILD_EXAMPLE)
  add_subdirectory(example)
endif()

# Package
if (HIP_PYTORCH_OPERATORS_PROJECT_IS_TOP_LEVEL)
  set(BUILD_SHARED_LIBS ON) # Build as though shared library for naming
  if(BUILD_ADDRESS_SANITIZER)
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
  else()
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
  endif()

  rocm_package_add_dependencies(DEPENDS "${DEPENDS_HIP_RUNTIME} >= 4.5.0")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT")

  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

  rocm_create_package(
    NAME hip-pytorch-operators
    DESCRIPTION "HIP C++ Pytorch operators"
    MAINTAINER "Maintainer <arsalan.anwari@outlook.com>"
    HEADER_ONLY
  )
endif()
