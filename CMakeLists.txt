cmake_minimum_required(VERSION 3.18)
project(lobx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LOBX_BUILD_TESTS "Build the C++ unit tests and benchmark" OFF)
# AddressSanitizer catches out-of-bounds access, use-after-free and leaks; UndefinedBehaviorSanitizer
# catches signed overflow, bad shifts and similar. Both slow the code down several times, so this is
# for testing, never for the benchmark numbers.
option(LOBX_SANITIZE "Build the engine and its tests with ASan and UBSan" OFF)
if(LOBX_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=undefined
                      -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# The engine itself: plain C++17 with no dependencies.
add_library(lobx_engine STATIC cpp/src/order_book.cpp)
target_include_directories(lobx_engine PUBLIC cpp/include)
set_target_properties(lobx_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(lobx_engine PRIVATE -Wall -Wextra -Werror)
endif()

# The Python extension is built only when scikit-build-core drives CMake (pip install).
if(SKBUILD)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native cpp/src/bindings.cpp)
  target_link_libraries(_native PRIVATE lobx_engine)
  install(TARGETS _native DESTINATION lobx)
endif()

if(LOBX_BUILD_TESTS)
  enable_testing()
  add_executable(lobx_tests cpp/tests/test_order_book.cpp)
  target_link_libraries(lobx_tests PRIVATE lobx_engine)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lobx_tests PRIVATE -Wall -Wextra -Werror)
  endif()
  add_test(NAME lobx_tests COMMAND lobx_tests)

  add_executable(lobx_bench cpp/bench/bench_order_book.cpp)
  target_link_libraries(lobx_bench PRIVATE lobx_engine)
  # A short run of the benchmark stream (random orders and cancels) doubles as a stress test, most
  # useful under LOBX_SANITIZE.
  add_test(NAME lobx_stress COMMAND lobx_bench 200000)
endif()
