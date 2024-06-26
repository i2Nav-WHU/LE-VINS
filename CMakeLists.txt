cmake_minimum_required(VERSION 3.10)
project(le_vins)

## Build Options

set(CMAKE_CXX_STANDARD 17)
if (CMAKE_BUILD_TYPE STREQUAL Release)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
endif()

# glog
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# LE-VINS Library
add_subdirectory(le_vins)

# abseil
add_subdirectory(thirdparty/abseil-cpp-20230125.0)

## Executable

# ROS interface
find_package(catkin REQUIRED COMPONENTS
    message_generation roscpp std_msgs pcl_conversions livox_ros_driver)
catkin_package(
    LIBRARIES
    CATKIN_DEPENDS roscpp std_msgs pcl_conversions livox_ros_driver)
include_directories(
    ${catkin_INCLUDE_DIRS})

# LE-VINS ROS
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/le_vins)
add_executable(${PROJECT_NAME}_ros
    ROS/main.cc
    ROS/fusion.cc
    ROS/visual/visual_drawer_rviz.cc
    ROS/lidar/lidar_converter.cc)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_ros
    ${catkin_LIBRARIES}
    le_vins
    stdc++fs)

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}_ros ${PCL_LIBRARIES})

# abseil
target_link_libraries(${PROJECT_NAME}_ros
    absl::strings
    absl::time)
