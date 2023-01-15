# CMakeList.txt 
#
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

if(CMAKE_VERSION VERSION_GREATER 3.6)
   set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")
endif()

project( "gaussmixdetector" LANGUAGES CXX )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( gaussmixdetector "main.cpp" "GaussMixDetector.cpp" "GaussMixDetector.h" )

target_link_libraries( gaussmixdetector ${OpenCV_LIBS} )

target_compile_options(gaussmixdetector PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
