cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)

project(Tracker)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(tracker tracker.cpp utils.cpp video.cpp tracklet.cpp node.cpp edge.cpp iou.cpp hik.cpp location.cpp clique.cpp)
target_link_libraries(tracker ${OpenCV_LIBS})
