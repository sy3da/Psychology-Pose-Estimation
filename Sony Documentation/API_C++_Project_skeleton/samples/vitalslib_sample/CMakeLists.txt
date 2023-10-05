cmake_minimum_required(VERSION 2.8)

project(vitalslib_sample)

## ----------------------------------------------------------------------------
## Dependencies
## ----------------------------------------------------------------------------
set(SENT_DEPENDENCIES_DIR ${CMAKE_CURRENT_LIST_DIR}/../../deps)

# load all directories with find<pkg>.cmake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
	${CMAKE_CURRENT_LIST_DIR}/../../cmake)

find_package(vitalslib REQUIRED)
#find_package(iu456 REQUIRED) <- Example


## ----------------------------------------------------------------------------
## Project
## ----------------------------------------------------------------------------
 
include_directories(
	${CMAKE_CURRENT_LIST_DIR}/../../vitalslib_lib/)


set(vitalslib_HEADERS
	${CMAKE_CURRENT_LIST_DIR}/../../vitalslib_api.h
	${CMAKE_CURRENT_LIST_DIR}/../../vitalslib_lib/vitalslib_types.h
	${CMAKE_CURRENT_LIST_DIR}/../../vitalslib_lib/vitalslib_platform.h)

set(vitalslib_sample_SOURCE_FILES 
	main.cpp 
	vitalslib_sample.h
	vitalslib_sample.cpp)


source_group("sources" FILES ${vitalslib_sample_SOURCE_FILES})

## ----------------------------------------------------------------------------
## Targets
## ----------------------------------------------------------------------------
add_executable(vitalslib_sample ${vitalslib_sample_SOURCE_FILES})

target_link_libraries(vitalslib_sample vitalslib_lib
	#${ui456_LIBRARIES}
	)

target_include_directories(vitalslib_sample
	PRIVATE
	#${ui456_INCLUDE_DIRS}
	${vitalslib_HEADERS}
	)

set_target_properties(vitalslib_sample PROPERTIES FOLDER "samples" DEBUG_POSTFIX "d")
	
	
# foreach(file ${ui456_BIN_FILES} )
	# add_custom_command(TARGET vitalslib_sample POST_BUILD
		# COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file} $<TARGET_FILE_DIR:vitalslib_sample>)
# endforeach()

add_custom_command(TARGET vitalslib_sample POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE_DIR:vitalslib_lib>/vitalslib_lib.dll $<TARGET_FILE_DIR:vitalslib_sample>)


# # ----------------------------------------------------------------------------
# # Install
# # ----------------------------------------------------------------------------

install(FILES ${vitalslib_sample_SOURCE_FILES} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../../dist/vitalslib_sample/src)

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows" AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
	set(vitalslib_SAMPLE_PACKAGE_NAME "vitalslib_Sample-${vitalslib_VERSION_MAJOR}.${vitalslib_VERSION_MINOR}.${vitalslib_VERSION_PATCH}.${vitalslib_VERSION_EXTRA}-win64.zip")
	install(CODE "
			execute_process(COMMAND ${ZIP_EXECUTABLE} a -tzip ${CMAKE_CURRENT_SOURCE_DIR}/../../dist/${vitalslib_SAMPLE_PACKAGE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/../../dist/vitalslib_sample)
			")
endif()

