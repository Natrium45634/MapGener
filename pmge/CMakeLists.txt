cmake_minimum_required(VERSION 3.20)
project(pmge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PMGE_BUILD_TESTS "Build the test executable" ON)
option(PMGE_BUILD_TOOLS "Build the command line tool" ON)

set(PMGE_SOURCES src/plan.cpp src/sampler.cpp src/hex.cpp src/erosion.cpp src/c_api.cpp)

# The shared library is what other programs load through the C ABI.
add_library(pmge SHARED ${PMGE_SOURCES})
target_include_directories(pmge PUBLIC include PRIVATE src)
target_compile_definitions(pmge PRIVATE PMGE_BUILD)
# Bit-identical results need the compiler to keep IEEE semantics: no fast-math.
if(MSVC)
  target_compile_options(pmge PRIVATE /W4 /fp:strict)
else()
  target_compile_options(pmge PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

if(PMGE_BUILD_TOOLS)
  add_executable(pmge_cli tools/pmge_cli.cpp)
  target_link_libraries(pmge_cli PRIVATE pmge)
endif()

if(PMGE_BUILD_TESTS)
  enable_testing()
  add_executable(pmge_tests tests/test_pmge.cpp)
  target_link_libraries(pmge_tests PRIVATE pmge)
  add_test(NAME pmge_tests COMMAND pmge_tests)
endif()
