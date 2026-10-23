cmake_minimum_required(VERSION 3.16)
project(welvet_qnn VERSION 0.2.1 LANGUAGES CXX)

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

set(WQNN_SOURCES
  src/detect.cpp
  src/dtype.cpp
  src/session.cpp
  src/layers.cpp
  src/sequential.cpp
  src/c_abi.cpp
)

# Static lib (tools / embed)
add_library(welvet_qnn STATIC ${WQNN_SOURCES})
add_library(welvet_qnn::welvet_qnn ALIAS welvet_qnn)

# Shared DLL/so for Welvet Go cgo (libwelvet_qnn.dll)
add_library(welvet_qnn_shared SHARED ${WQNN_SOURCES})
set_target_properties(welvet_qnn_shared PROPERTIES
  OUTPUT_NAME welvet_qnn
  WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

foreach(tgt welvet_qnn welvet_qnn_shared)
  target_include_directories(${tgt}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_compile_definitions(${tgt} PUBLIC WELVET_QNN_HAS_SDK=${WELVET_QNN_HAS_SDK})
  if(WELVET_QNN_HAS_SDK)
    target_include_directories(${tgt} PRIVATE
      "${QAIRT_SDK_ROOT}/include/QNN"
      "${QAIRT_SDK_ROOT}/include"
    )
  endif()
  if(WIN32)
    target_compile_definitions(${tgt} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(${tgt} PUBLIC stdc++fs)
  endif()
endforeach()

target_compile_definitions(welvet_qnn_shared PRIVATE WELVET_QNN_BUILD_SHARED)

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

install(TARGETS welvet_qnn welvet_qnn_shared
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/welvet_qnn DESTINATION include)
install(FILES include/welvet_qnn_c.h DESTINATION include)
