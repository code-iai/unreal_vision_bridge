cmake_minimum_required(VERSION 2.8.3)
project(unreal_vision_bridge)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBT_USE_DOUBLE_PRECISION")
# Unused warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self -Wunused-function -Wunused-label -Wunused-variable -Wunused-but-set-variable -Wunused-but-set-parameter")
# Additional warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Warray-bounds -Wtype-limits -Wreturn-type -Wsequence-point -Wparentheses -Wmissing-braces -Wchar-subscripts -Wswitch -Wwrite-strings -Wenum-compare -Wempty-body -Wlogical-op")

# Check for c++11 support
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

#Check for f16c support
CHECK_CXX_COMPILER_FLAG("-mf16c" COMPILER_SUPPORTS_F16C)

IF(COMPILER_SUPPORTS_F16C)
  # Enable F16C
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mf16c")
ELSE()
  MESSAGE(ERROR "The compiler ${CMAKE_CXX_COMPILER} has no F16C support. Please use a different C++ compiler and check that your hardware supports F16C instructions.")
ENDIF()

IF(COMPILER_SUPPORTS_CXX11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ELSEIF(COMPILER_SUPPORTS_CXX0X)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  MESSAGE(ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  nodelet
  roscpp
  rostime
  sensor_msgs
  std_msgs
  tf
  message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files (
  FILES
  ObjectColorMap.msg)

generate_messages(
  DEPENDENCIES
  std_msgs)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS
    std_msgs
    message_runtime
)

###########
## Build ##
###########

# Display additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME}
  SOURCES
  ${FILES_LIST}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_library(unreal_vision_nodelet SHARED src/unreal_vision_bridge.cpp)
target_link_libraries(unreal_vision_nodelet
  ${catkin_LIBRARIES}
)
add_dependencies(unreal_vision_nodelet ${${PROJECT_NAME}_EXPORTED_TARGETS})

add_executable(unreal_vision_bridge src/unreal_vision_bridge.cpp)
target_link_libraries(unreal_vision_bridge
  ${catkin_LIBRARIES}
)
add_dependencies(unreal_vision_bridge ${${PROJECT_NAME}_EXPORTED_TARGETS})

#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS unreal_vision_bridge unreal_vision_nodelet
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY
  launch
  # myfile2
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES
  nodelet_plugins.xml
  # myfile2
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
