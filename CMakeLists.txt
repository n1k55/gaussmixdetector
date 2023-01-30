# CMakeList.txt 
#
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

if(CMAKE_VERSION VERSION_GREATER 3.6)
# suppressed warnings:
# disputable value checks: -modernize-use-trailing-return-type,-fuchsia-default-arguments-declarations,-google-runtime-references,-llvm-header-guard,-google-readability-todo, -fuchsia-default-arguments-calls
# external code warnings: -hicpp-signed-bitwise,-hicpp-no-array-decay,-cppcoreguidelines-pro-bounds-array-to-pointer-decay
# fix later: -readability-magic-numbers,-cppcoreguidelines-avoid-magic-numbers,-cppcoreguidelines-pro-bounds-pointer-arithmetic
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-header-filter=.*;-checks=*,-modernize-use-trailing-return-type,-fuchsia-default-arguments-declarations,-google-runtime-references,-llvm-header-guard,-google-readability-todo,-fuchsia-default-arguments-calls,-hicpp-signed-bitwise,-hicpp-no-array-decay,-cppcoreguidelines-pro-bounds-array-to-pointer-decay,-readability-magic-numbers,-cppcoreguidelines-avoid-magic-numbers,-cppcoreguidelines-pro-bounds-pointer-arithmetic")
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
