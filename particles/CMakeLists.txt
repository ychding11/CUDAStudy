cmake_minimum_required (VERSION 2.8)

project (cuda_study)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON) 

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Configuration types" FORCE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

# set execuatable name of current project
set(EXE_NAME "particles_test")

file(GLOB SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB CU_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/*.cu
	${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/ )

#CMake built-in module
find_package(CUDA QUIET REQUIRED)

#------------------------------------------------------------------------------------------------
# group source code 
#------------------------------------------------------------------------------------------------
foreach(f ${SRC_FILES})
    # Get the path of the file relative to ${DIRECTORY},
    # then alter it (not compulsory)
    file(RELATIVE_PATH SRCGR ${CMAKE_CURRENT_SOURCE_DIR} ${f})
    set(SRCGR "Cpp/${SRCGR}")

    # Extract the folder, ie remove the filename part
    string(REGEX REPLACE "(.*)(/[^/]*)$" "\\1" SRCGR ${SRCGR})

    # Source_group expects \\ (double antislash), not / (slash)
    string(REPLACE / \\ SRCGR ${SRCGR})
    source_group("${SRCGR}" FILES ${f})
endforeach()

#------------------------------------------------------------------------------------------------
# preproc
#------------------------------------------------------------------------------------------------
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-D__STDC_FORMAT_MACROS)
add_definitions(-DBX_CONFIG_ENABLE_MSVC_LEVEL4_WARNINGS=1)
add_definitions(-D__STDC_LIMIT_MACROS)
add_definitions(-D__STDC_CONSTANT_MACROS)
add_definitions(-DIMGUI_DISABLE_OBSOLETE_FUNCTIONS)

add_definitions(-DUSE_DL_PREFIX)
add_definitions(-DGLEW_STATIC)

if(MSVC)
endif()

if(WINDOWS)
  set(GUI_TYPE WIN32)
elseif(MACOSX)
  set(GUI_TYPE MACOSX_BUNDLE)
endif()

#------------------------------------------------------------------------------------------------
# compliler option 
#------------------------------------------------------------------------------------------------
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()



SET(LINK_OPTIONS " ")

#------------------------------------------------------------------------------------------------
# add executable & set output dirs
#------------------------------------------------------------------------------------------------

#Use static cuda link
cuda_add_executable(${EXE_NAME} ${CU_FILES} ${SRC_FILES})

set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin )
set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin )
set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_SOURCE_DIR}/bin )
set_target_properties(${EXE_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${EXE_NAME} PROPERTIES RELWITHDEBINFO_POSTFIX "RelWithDebInfo")
set_target_properties(${EXE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

#------------------------------------------------------------------------------------------------
# Open the console window in visual studio projects
#------------------------------------------------------------------------------------------------
if(MSVC)
	set_target_properties(${EXE_NAME} PROPERTIES LINK_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE")
	set_target_properties(${EXE_NAME} PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:CONSOLE")
endif()


#------------------------------------------------------------------------------------------------
# 
# https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#target-properties
#------------------------------------------------------------------------------------------------
