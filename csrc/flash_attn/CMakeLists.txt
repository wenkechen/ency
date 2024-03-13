project("flash_attn_ops" LANGUAGES CXX CUDA)
file(GLOB_RECURSE ops_source_files "${PROJECT_SOURCE_DIR}/*.cpp" "${PROJECT_SOURCE_DIR}/*.cc" "${PROJECT_SOURCE_DIR}/*.cu")

add_library(${PROJECT_NAME} SHARED ${ops_source_files})

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
												-U__CUDA_NO_HALF_OPERATORS__
												-U__CUDA_NO_HALF_CONVERSIONS__
												_U__CUDA_NO_HALF2_OPERATORS__
												-U__CUDA_NO_BFLOAT16_CONVERSIONS__
												>)

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES
	POSITION_INDEPENDENT_CODE ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIRES}")
