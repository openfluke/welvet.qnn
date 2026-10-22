cmake_minimum_required(VERSION 3.16)
project(welvet_qnn VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WELVET_QNN_BUILD_TOOLS "Build smoke + probe executables" ON)

# QAIRT / QNN SDK (optional — without it, library still builds in stub mode)
set(QAIRT_SDK_ROOT "$ENV{QAIRT_SDK_ROOT}" CACHE PATH "Qualcomm QAIRT/QNN SDK root")
if(NOT QAIRT_SDK_ROOT AND DEFINED ENV{QNN_SDK_ROOT})
  set(QAIRT_SDK_ROOT "$ENV{QNN_SDK_ROOT}" CACHE PATH "Qualcomm QAIRT/QNN SDK root" FORCE)
endif()

set(WELVET_QNN_HAS_SDK 0)
if(QAIRT_SDK_ROOT AND EXISTS "${QAIRT_SDK_ROOT}/include/QNN/QnnInterface.h")
  set(WELVET_QNN_HAS_SDK 1)
  message(STATUS "welvet.qnn: QAIRT SDK ${QAIRT_SDK_ROOT}")
else()
  message(STATUS "welvet.qnn: no QAIRT SDK — stub build (detect filesystem only)")
endif()

add_library(welvet_qnn
  src/detect.cpp
  src/session.cpp
  src/layers.cpp
)
add_library(welvet_qnn::welvet_qnn ALIAS welvet_qnn)

target_include_directories(welvet_qnn
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(welvet_qnn PUBLIC WELVET_QNN_HAS_SDK=${WELVET_QNN_HAS_SDK})

if(WELVET_QNN_HAS_SDK)
  target_include_directories(welvet_qnn PRIVATE
    "${QAIRT_SDK_ROOT}/include/QNN"
    "${QAIRT_SDK_ROOT}/include"
  )
endif()

if(WIN32)
  target_compile_definitions(welvet_qnn PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# libstdc++ / libc++ filesystem (GCC < 9)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  target_link_libraries(welvet_qnn PUBLIC stdc++fs)
endif()

if(WELVET_QNN_BUILD_TOOLS)
  add_executable(welvet_qnn_smoke tools/smoke/main.cpp)
  target_link_libraries(welvet_qnn_smoke PRIVATE welvet_qnn)

  add_executable(welvet_qnn_probe tools/probe/main.cpp)
  target_compile_definitions(welvet_qnn_probe PRIVATE WELVET_QNN_HAS_SDK=${WELVET_QNN_HAS_SDK})
  if(WELVET_QNN_HAS_SDK)
    target_include_directories(welvet_qnn_probe PRIVATE
      "${QAIRT_SDK_ROOT}/include/QNN"
      "${QAIRT_SDK_ROOT}/include"
    )
  endif()
  if(WIN32)
    target_compile_definitions(welvet_qnn_probe PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
endif()

install(TARGETS welvet_qnn
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/welvet_qnn DESTINATION include)
