cmake_minimum_required(VERSION 3.10)
project(visual)

include_directories(
    ${PROJECT_SOURCE_DIR}/../)

add_library(${PROJECT_NAME} SHARED
    tracking.cc
    camera.cc
    visual_drawer.cc
    visual_frame.cc
    visual_map.cc
    visual_mappoint.cc)

## Subdirectory

# common
target_link_libraries(${PROJECT_NAME} common)

## Libraries

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# yaml-cpp
find_package(yaml-cpp REQUIRED)
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})

# glog
find_package(Glog REQUIRED)
target_link_libraries(${PROJECT_NAME} ${GLOG_LIBRARIES})

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

# TBB
find_package(TBB REQUIRED)
target_link_libraries(${PROJECT_NAME}
    TBB::tbb)

# abseil
target_link_libraries(${PROJECT_NAME}
    absl::strings
    absl::str_format)
