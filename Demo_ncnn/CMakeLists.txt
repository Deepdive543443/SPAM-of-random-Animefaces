cmake_minimum_required(VERSION 3.0.0)
project(TWNE_MINI VERSION 0.1.0 LANGUAGES C CXX)

# include(CTest)
# enable_testing()


add_executable(TWNE_MINI main.cpp)


set(OpenCV_DIR D:/cpp_libs/opencv-mobile-4.8.1-windows-vs2022/x64/x64/vc17/staticlib)
find_package(OpenCV REQUIRED)
target_link_libraries( TWNE_MINI ${OpenCV_LIBS} )


set(ncnn_DIR D:/cpp_libs/ncnn-20231027-windows-vs2022/x64/lib/cmake/ncnn CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)
target_link_libraries(TWNE_MINI ncnn)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()