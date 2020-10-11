cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(GMCP)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(gmcp gmcp.cpp utils.cpp video.cpp tracklet.cpp node.cpp)
target_link_libraries(gmcp ${OpenCV_LIBS})
