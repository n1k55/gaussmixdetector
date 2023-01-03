# CMakeList.txt 
#
cmake_minimum_required( VERSION 2.8 )

project( "gaussmixdetector" LANGUAGES CXX )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( gaussmixdetector "main.cpp" "GaussMixDetector.cpp" "GaussMixDetector.h" )

target_link_libraries( gaussmixdetector ${OpenCV_LIBS} )

if( CMAKE_VERSION VERSION_GREATER 3.12 )
  set_property( TARGET gaussmixdetector PROPERTY CXX_STANDARD 20 )
endif()