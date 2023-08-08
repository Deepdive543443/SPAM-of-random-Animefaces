cmake_minimum_required(VERSION 3.10.0)
project(Progan_cpp VERSION 0.1.0 LANGUAGES C CXX)


# Source files
add_executable(Progan_cpp main.cpp utils.h)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_SOURCE_DIR D:/cpp_lib)
set(Vulkan_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/vulkan/Include)
set(Vulkan_LIBS ${CMAKE_SOURCE_DIR}/vulkan/Lib/*.lib)

set(NCNN_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/ncnn/x64/include/ncnn)
set(NCNN_LIBS ${CMAKE_SOURCE_DIR}/ncnn/x64/lib/*.lib)

#======== Lib and dependency========
# find_package(OpenCV REQUIRED core imgproc)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( Progan_cpp ${OpenCV_LIBS} )

find_package(Vulkan REQUIRED)
include_directories( ${Vulkan_INCLUDE_DIR} )
target_link_libraries(Progan_cpp ${Vulkan_LIBS} )

find_package(NCNN REQUIRED)
include_directories( ${NCNN_INCLUDE_DIRS} )
target_link_libraries( Progan_cpp ${NCNN_LIBS} )
#===================================

message(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
message(OpenCV_LIBS="${OpenCV_LIBS}")
message(NCNN_INCLUDE_DIRS="${NCNN_INCLUDE_DIRS}")
message(NCNN_LIBS="${NCNN_LIBS}")