## ----------------------------------------------------------------------------
## Sources
## ----------------------------------------------------------------------------
set(test_files
	main.cpp
	test_example.cpp	
)

source_group(tests FILES ${test_files})

## ----------------------------------------------------------------------------
## Target
## ----------------------------------------------------------------------------
add_definitions(-DNOMINMAX)

add_executable(tests
	${test_files}
)

target_include_directories(tests PRIVATE
)

target_link_libraries(tests PRIVATE
)

target_compile_definitions(tests PRIVATE
)

## ----------------------------------------------------------------------------
## Tests
## ----------------------------------------------------------------------------
add_test(  NAME tests
	COMMAND $<TARGET_FILE:tests> -r junit --out=${TESTRESULTS_OUTPUT_DIRECTORY}/tests_$<CONFIGURATION>.junit.xml)
