cmake_minimum_required(VERSION 3.25)
project(ency LANGUAGES CXX CUDA)

get_filename_component(PROJECT_PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_definitions(-D_GLIBCXX_USE_CXX_ABI=0)
add_definitions(-DBUILD_CUTLASS_MIXED_GEMM)
# add_definitions(-DENABLE_BF16)
add_definitions(-DUSE_NVTX)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DWMMA")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWMMA")

find_package(CUDAToolkit REQUIRED)
set(COMMON_HEADER_DIRS
	${CUDAToolkit_INCLUDE_DIRS}
)
set(COMMON_LIB_DIRS ${CUDAToolkit_LIBRARY_DIR})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
string(REGEX MATCH "([0-9]+\\.[0-9]+)" CUDA_VERSION "${CUDAToolkit_VERSION}")
if (${CUDA_VERSION} GREATER_EQUAL 11.1)
	set(CMAKE_CUDA_DIRS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
	set(CMAKE_CUDA_DIRS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")
	set(ENV{TORCH_CUDA_ARCH_LIST} "8.0;8.6")
	set(CMAKE_CUDA_ARCHITECTURES 80 86)
else()
	message(FATAL_ERROR "Light is only supported on CUDA 11.1 and above")
endif()


if (${CUDA_VERSION} GREATER_EQUAL 11.3)
	set(CMAKE_CUDA_DIRS "${CMAKE_CUDA_FLAGS} -t0")
endif()


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math") # Make use of fast math library. implies --ftz=true --prec-div=false --prec-sqrt=false --fmad=true
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -ldl")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-strict-aliasing") #Disable cutlass strict-aliasing warning
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-enum-compare") #Disable flash-attn enum-compare warning
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-warn-double-usage") #Warning if double(s) are used in an instruction
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-warn-spills") #Warning if registers are spilled to local memory
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-warn-lmem-usage") #Warning if local memory is used
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Lineinto") #For nsight compute analysis
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep") #For keeping .ptx files

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
list(APPEND COMMON_HEADER_DIRS ${Python3_INCLUDE_DIRS})

execute_process(
	COMMAND ${Python3_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
	OUTPUT_STRIP_TRAINING_WHITESPACE
	OUTPUT_VARIABLE DCMAKE_PREFIX_PATH
)
set(CMAKE_PREFIX_PATH ${DCMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
list(APPEND COMMON_HEADER_DIRS ${TORCH_INCLUDE_DIRS})
list(APPEND COMMON_HEADER_DIRS "${Python3_SITEARCH}/torch/include/THC")

list(APPEND COMMON_LIB_DIRS "${Python3_SITEARCH}/torch/lib")

include_directories(${COMMON_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})

add_subdirectory(trial)
