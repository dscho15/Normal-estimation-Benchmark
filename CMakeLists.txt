cmake_minimum_required(VERSION 3.10)
project(pc_benmark)

set(CMAKE_BUILD_TYPE "Debug")

option(ENABLE_NATIVE_BUILD_OPTIMIZATIONS "Enable native build optimization flags" ON)
option(ENABLE_NON_DETERMINISTIC_PARALLELISM "Enable parallelization that may produce non-deterministic results" ON)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(cilantro)
find_package(PCL 1.10 REQUIRED 
                      COMPONENTS 
                        common
                        io
                        registration
                        kdtree
                        filters
                        features
                        visualization)

include_directories(
    include
    ${Pangolin_INCLUDE_DIRS}
    ${cilantro_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} src/main.cc)
target_link_libraries(${PROJECT_NAME} ${cilantro_LIBRARIES} ${Pangolin_LIBRARIES} ${PCL_LIBRARIES})

if (ENABLE_NATIVE_BUILD_OPTIMIZATIONS)
    target_compile_options(${PROJECT_NAME} PUBLIC -O3 -march=native -mtune=native)
else()
    target_compile_options(${PROJECT_NAME} PUBLIC -O3)
endif() 