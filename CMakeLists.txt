cmake_minimum_required(VERSION 3.20)
project(stat_arb_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wall -Wextra -Werror
        -Wpedantic
        -Wshadow
        -Wnull-dereference
        -O3
        -march=native
        -fno-omit-frame-pointer    # keep frame pointers for profiling
    )
    # Enable LTO in Release
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /WX /O2 /std:c++20)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

find_package(benchmark REQUIRED)

# Optional: enable AddressSanitizer in Debug builds
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

# ---------------------------------------------------------------------------
# Core library: arb_lib
# ---------------------------------------------------------------------------

add_library(arb_lib STATIC
    cpp/src/arb/risk_manager.cpp
    cpp/src/arb/sim_exchange.cpp
    cpp/src/arb/position_manager.cpp
    cpp/src/arb/order_router.cpp
)

target_include_directories(arb_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_compile_features(arb_lib PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Benchmark binary: bm_arb
# ---------------------------------------------------------------------------

add_executable(bm_arb
    cpp/bench/bm_arb.cpp
)

target_link_libraries(bm_arb
    PRIVATE
        arb_lib
        benchmark::benchmark
)

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install(TARGETS arb_lib bm_arb
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY cpp/include/
    DESTINATION include
)

# ---------------------------------------------------------------------------
# CTest integration (placeholder for future unit tests)
# ---------------------------------------------------------------------------

enable_testing()
