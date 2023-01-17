# CMakeList.txt 
#
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

if(CMAKE_VERSION VERSION_GREATER 3.6)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-header-filter=.*;-checks=*,-fuchsia-default-arguments-calls,-modernize-use-trailing-return-type,-cppcoreguidelines-macro-usage,-readability-magic-numbers,-cppcoreguidelines-avoid-magic-numbers,-cppcoreguidelines-pro-bounds-constant-array-index,-hicpp-signed-bitwise,-cppcoreguidelines-pro-bounds-pointer-arithmetic,-bugprone-too-small-loop-variable,-hicpp-avoid-c-arrays,-modernize-avoid-c-arrays,-cppcoreguidelines-avoid-c-arrays,-google-runtime-int,-hicpp-no-array-decay,-cppcoreguidelines-pro-bounds-array-to-pointer-decay,-performance-unnecessary-value-param,-cppcoreguidelines-narrowing-conversions,-bugprone-narrowing-conversions,-modernize-use-emplace,-hicpp-use-emplace,-cert-err58-cpp,-fuchsia-statically-constructed-objects,-performance-inefficient-string-concatenation,-performance-faster-string-find,-modernize-redundant-void-arg,-fuchsia-default-arguments-declarations,-google-runtime-references,-readability-named-parameter,-hicpp-explicit-conversions,-google-explicit-constructor,-hicpp-special-member-functions,-cppcoreguidelines-special-member-functions,-llvm-header-guard")
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
