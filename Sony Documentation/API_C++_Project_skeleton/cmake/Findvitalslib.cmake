# - Try to find vitalslib_lib
# Once done, this will define
#
#  vitalslib_lib_FOUND                 System has vitalslib_lib
#  vitalslib_lib_INCLUDE_DIRS          The vitalslib_lib include directories
#  vitalslib_lib_LIBRARIES             Link these to use vitalslib_lib
#  vitalslib_lib_BINARIES              The vitalslib_lib binaries
#
#
# Required variables that must be set before calling this:
#
# SK_DEPENDENCIES_DIR   The root directory for the dependencies
# vitalslib_lib_PLATFORM_NAME      The name of the platform you are targeting (e.g. "Windows_Win7_x86_VS2010", "Linux_ARM_CortexA9_linaro", etc)
#
#
# Example usage:
#  set(SK_DEPENDENCIES_DIR ${CMAKE_SOURCE_DIR}/dependencies)
#  set(SK_DEPENDENCIES_DIR "Windows_Win7_x86_VS2010")
#  find_package(vitalslib_lib)
#
#  include_directories(${vitalslib_lib_INCLUDE_DIRS})
#  link_directories(${vitalslib_lib_LIB_DIRS})
#  add_executable(my_skv_based_app ${vitalslib_lib_LIBRARIES})


unset(vitalslib_lib_FOUND)

if(NOT vitalslib_lib_DIR)
	message(ERROR "Could not find a suitable root directory")
endif()

set(vitalslib_lib_INCLUDE_DIRS ${vitalslib_lib_DIR}/include)
set(vitalslib_lib_LIBRARIES ${vitalslib_lib_DIR}/lib/vitalslib_lib.lib)
set(vitalslib_lib_BINARIES ${vitalslib_lib_DIR}/bin/vitalslib_lib.dll)

set(vitalslib_lib_FOUND true)