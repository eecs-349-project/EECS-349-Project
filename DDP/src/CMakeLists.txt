cmake_minimum_required(VERSION 2.8)

find_package(Eigen3 REQUIRED)
find_package(Matlab REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${MATLAB_INCLUDE_DIR})

include_directories(headers)

if (CMAKE_BUILD_TYPE)
  message("The build type is " ${CMAKE_BUILD_TYPE})
  if (CMAKE_BUILD_TYPE  STREQUAL Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -fkeep-inline-functions")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
  endif()
endif()

message(${MATLAB_INCLUDE_DIR})
message(${MATLAB_LIBRARIES})
message(${CMAKE_CXX_FLAGS})

add_executable(quadrotor_ddp quadrotor_ddp.cpp)
target_link_libraries(quadrotor_ddp ${MATLAB_LIBRARIES})
