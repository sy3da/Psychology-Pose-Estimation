project(vitalslib_lib)

if(MSVC)
	add_definitions(-DVITALSLIB_EXPORTS)

endif()

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------

set(vitalslib_PUBLIC_SRCFILES 
	vitalslib_api.h
	vitalslib_platform.h
	vitalslib_types.h
)

set(vitalslib_PRIVATE_SRCFILES 
	vitalslib_api.cpp
	version.h)
	
set(vitalslib_CORE 
	core/pipeline.h)

set(vitalslib_SRCFILES 
	${vitalslib_PUBLIC_SRCFILES}
	${vitalslib_PRIVATE_SRCFILES}
)

source_group("sources" FILES ${vitalslib_SRCFILES})
source_group("core" FILES ${vitalslib_CORE})

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------

add_library(vitalslib_lib SHARED ${vitalslib_SRCFILES} ${vitalslib_CORE})

target_link_libraries(vitalslib_lib PRIVATE)

target_include_directories(vitalslib_lib PRIVATE)


install(TARGETS vitalslib_lib 
		ARCHIVE DESTINATION vitalslib_lib/lib/${CMAKE_SYSTEM_NAME}
		RUNTIME DESTINATION vitalslib_lib/bin/${CMAKE_SYSTEM_NAME}
		LIBRARY DESTINATION vitalslib_lib/lib/${CMAKE_SYSTEM_NAME})

install(FILES ${vitalslib_PUBLIC_SRCFILES} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../dist/vitalslib_lib/include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Findvitalslib.cmake DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../dist/vitalslib_lib/cmake)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../config/vitalslib.config.xml DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../dist/vitalslib_lib/config)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../dist/vitalslib_lib DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../dist/vitalslib_sample/deps)

target_compile_definitions(vitalslib_lib PRIVATE)

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows" AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
	set(vitalslib_PACKAGE_NAME "vitalslib-${vitalslib_VERSION_MAJOR}.${vitalslib_VERSION_MINOR}.${vitalslib_VERSION_PATCH}.${vitalslib_VERSION_EXTRA}-win64.zip")
	install(CODE "
			execute_process(COMMAND ${ZIP_EXECUTABLE} a -tzip ${CMAKE_CURRENT_SOURCE_DIR}/../dist/${vitalslib_PACKAGE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/../dist/vitalslib_lib)
			")
endif()