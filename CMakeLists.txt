cmake_minimum_required(VERSION 2.6)

#project info
project(KissNN)

#set version
set(KissNN_VERSION_MAJOR 1)
set(KissNN_VERSION_MINOR 0)

#set environment variables
message("CMAKE_CURRENT_SOURCE_DIR = " ${CMAKE_CURRENT_SOURCE_DIR})
message("CMAKE_CURRENT_BINARY_DIR = " ${CMAKE_CURRENT_BINARY_DIR})
message("CMAKE_INSTALL_PREFIX = " ${CMAKE_INSTALL_PREFIX})

#headers
include_directories (./include)

#library directory

add_definitions(-O3 -std=c++11 -Wall)

set(SOURCE
    src/KissNN.cpp
    src/learn.cpp
    src/print.cpp
)

add_library(libKissNN
    ${SOURCE}
)
# change lib_target properties
SET_TARGET_PROPERTIES( libKissNN PROPERTIES
# create *nix style library versions + symbolic links
    VERSION ${${PROJECT_NAME}_VERSION}
    SOVERSION ${${PROJECT_NAME}_SOVERSION}
    # allow creating static and shared libs without conflicts
    CLEAN_DIRECT_OUTPUT 1
    # avoid conflicts between library and binary target names
    OUTPUT_NAME KissNN)


target_link_libraries(libKissNN)

add_executable(kissNN
    src/main.cpp
)

target_link_libraries(kissNN libKissNN)


INSTALL(TARGETS libKissNN
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
