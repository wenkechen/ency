project("cpp17.exe" LANGUAGES CXX CUDA)
file(GLOB_RECURSE exe_source_files "${PROJECT_SOURCE_DIR}/*.cpp" "${PROJECT_SOURCE_DIR}/*.cc" "${PROJECT_SOURCE_DIR}/*.cu")
add_executable(${PROJECT_NAME} ${exe_source_files})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES
	POSITION_INDEPENDENT_CODE ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIRES}")
