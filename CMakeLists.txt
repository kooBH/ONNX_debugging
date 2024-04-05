# CMAKE version of Ubuntu 16.04 is 3.5.
cmake_minimum_required(VERSION 3.5)
project(Tester)

set(SRC)
set(INCL)
set(LINKLIBS)
set(COMPILE_OPTION)

# For ARM arch
if(NOT MSVC)
	EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
	message( STATUS "Architecture: ${ARCHITECTURE}" )

	if(${ARCHITECTURE} STREQUAL "aarch64")
		SET(AARCH64 TRUE)
	endif()
endif()

list(APPEND INCL
  ${CMAKE_CURRENT_LIST_DIR}/include
  )

# LINKLIBS
if(MSVC)
list(APPEND LINKLIBS
	${CMAKE_CURRENT_LIST_DIR}/build/onnxruntime.lib
	${CMAKE_CURRENT_LIST_DIR}/build/onnxruntime_providers_shared.lib
  )
elseif(AARCH64)
list(APPEND LINKLIBS
	${CMAKE_CURRENT_LIST_DIR}/build/aarch64/libonnxruntime.so.1.14.1
  m
  #  sudo apt install libtbb-dev
  tbb
)
else()
list(APPEND LINKLIBS
	${CMAKE_CURRENT_LIST_DIR}/build/libonnxruntime.so.1.14.1
  m
  #  sudo apt install libtbb-dev
  tbb
)
endif()

if(MSVC)
  list(APPEND COMPILE_OPTION
  /D_EXPORT_
  )
else()
  list(APPEND COMPILE_OPTION
    -D_EXPORT_
    -pthread
  )
endif()
############ TEST

add_executable(test ${SRC}
${CMAKE_CURRENT_LIST_DIR}/test.cpp
${CMAKE_CURRENT_LIST_DIR}/Tester.h
${CMAKE_CURRENT_LIST_DIR}/Tester.cpp
)

target_include_directories(test PRIVATE    ${INCL}  )
target_link_libraries(test ${LINKLIBS})
target_compile_options(test PRIVATE ${COMPILE_OPTION} )
set_target_properties(test PROPERTIES  CXX_STANDARD 17 )
