cmake_minimum_required(VERSION 3.31)
project(juliacl C)

set(CMAKE_C_STANDARD 11)

find_package(OpenCL REQUIRED)

set(STB_IMAGE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/libs/stb_image/include)

add_executable(${PROJECT_NAME} src/main.c)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

target_include_directories(${PROJECT_NAME} PUBLIC ${STB_IMAGE_INCLUDE_DIRS})
