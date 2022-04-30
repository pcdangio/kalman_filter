cmake_minimum_required(VERSION 3.2)

# Set up project.
project(kalman_filter
        VERSION 1.0
        DESCRIPTION "A C++ library for Kalman Filters.")

# Find dependencies.
find_package(Eigen3 REQUIRED)

# Add source files.
file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Create library.
add_library(kalman_filter STATIC ${sources})

# Link library to dependencies.
target_link_libraries(kalman_filter
    Eigen3::Eigen)

# Set include directories for internal/external use.
target_include_directories(kalman_filter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)