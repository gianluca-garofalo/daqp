add_executable(verify_set verify_set.cpp)
target_link_libraries(verify_set PRIVATE daqp_eigen)
target_include_directories(verify_set PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_custom_command(TARGET verify_set POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        $<TARGET_FILE_DIR:verify_set>/data
    COMMENT "Copying test data files for verify_set"
)

add_test(NAME test_verify_set COMMAND verify_set)
set_tests_properties(test_verify_set PROPERTIES 
    WORKING_DIRECTORY $<TARGET_FILE_DIR:verify_set>)
