cmake_minimum_required(VERSION 3.24)

if(ENABLE_HIP)
    set(CCK_GPU_LANGUAGE HIP)
else()
    set(CCK_GPU_LANGUAGE CUDA)
endif()

# before project(), which would otherwise fill in the compiler's own default
if(NOT ENABLE_HIP AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100;120")
endif()

project(colabfold_cuda_kernels LANGUAGES CXX ${CCK_GPU_LANGUAGE})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/CckGpuArch.cmake)

# XLA FFI headers. A handler runs on its own XLA_FFI_API and newer, so older headers keep the jax floor low.
set(CCK_FFI_INCLUDE_DIR "" CACHE PATH "XLA FFI headers (jax.ffi.include_dir()); empty asks the python on PATH")
if(CCK_FFI_INCLUDE_DIR STREQUAL "")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import jax; print(jax.ffi.include_dir())"
                    OUTPUT_VARIABLE CCK_FFI_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
endif()
if(NOT EXISTS "${CCK_FFI_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    message(FATAL_ERROR "cck: no xla/ffi/api/ffi.h under CCK_FFI_INCLUDE_DIR='${CCK_FFI_INCLUDE_DIR}'")
endif()
file(STRINGS "${CCK_FFI_INCLUDE_DIR}/xla/ffi/api/c_api.h" ffi_api REGEX "^#define XLA_FFI_API_(MAJOR|MINOR) ")
string(REGEX REPLACE "[^0-9;]" "" ffi_api "${ffi_api}")
string(REPLACE ";" "." ffi_api "${ffi_api}")
message(STATUS "cck: XLA FFI headers ${CCK_FFI_INCLUDE_DIR} (api ${ffi_api})")

set(CCK_SOURCES diffusion/diffusion.cu)
set_source_files_properties(${CCK_SOURCES} PROPERTIES LANGUAGE ${CCK_GPU_LANGUAGE})
if(NOT ENABLE_HIP)
    # the FP8 triangle contraction through cuBLASLt, CUDA only; host code, so no pass per architecture
    list(APPEND CCK_SOURCES contract/fp8_contract.cpp)
endif()

add_library(colabfold_cuda_kernels SHARED ${CCK_SOURCES})
target_include_directories(colabfold_cuda_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CCK_FFI_INCLUDE_DIR})
target_compile_definitions(colabfold_cuda_kernels PRIVATE MK_FFI)

if(ENABLE_HIP)
    cck_resolve_hip_architectures(CCK_HIP_ARCHS)
    message(STATUS "cck: HIP architectures ${CCK_HIP_ARCHS}")
    set_target_properties(colabfold_cuda_kernels PROPERTIES HIP_ARCHITECTURES "${CCK_HIP_ARCHS}")
    target_compile_options(colabfold_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-value>
        $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-result>
    )
else()
    find_package(CUDAToolkit REQUIRED)

    # CUTLASS, required: CuTe runs the wide GEMMs and wraps the cp.async, ldmatrix and mma.sync instructions; headers only
    set(CUTLASS_DIR "" CACHE PATH "CUTLASS checkout; empty fetches the pinned release")
    if(CUTLASS_DIR STREQUAL "")
        include(FetchContent)
        # v4.2.1; SOURCE_SUBDIR names no CMakeLists, so only the headers are fetched, not built
        FetchContent_Declare(cutlass
            URL https://github.com/NVIDIA/cutlass/archive/f3fde58372d33e9a5650ba7b80fc48b3b49d40c8.tar.gz
            URL_HASH SHA256=2e5890306557bd87136e1d6914e4ee42536050b9116433848175520fd15991f2
            SOURCE_SUBDIR headers-only)
        FetchContent_MakeAvailable(cutlass)
        set(CUTLASS_DIR "${cutlass_SOURCE_DIR}")
    endif()

    # --threads=0 compiles the architectures in parallel: six take as long as one
    cck_resolve_cuda_architectures(CCK_CUDA_ARCHS)
    cck_cuda_gencode("${CCK_CUDA_ARCHS}" CCK_GENCODE)
    message(STATUS "cck: CUDA architectures ${CCK_CUDA_ARCHS}")

    set_target_properties(colabfold_cuda_kernels PROPERTIES CUDA_ARCHITECTURES OFF CUDA_RUNTIME_LIBRARY Static)
    # the library links to the C library only (a HIP build that does this segfaults on load into python)
    target_link_options(colabfold_cuda_kernels PRIVATE -static-libstdc++ -static-libgcc)
    target_include_directories(colabfold_cuda_kernels PRIVATE "${CUTLASS_DIR}/include")
    # cuBLASLt's header only: the library is opened at run time, the one jax loaded
    target_include_directories(colabfold_cuda_kernels PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(colabfold_cuda_kernels PRIVATE ${CMAKE_DL_LIBS})
    target_compile_options(colabfold_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-diag-suppress=177,1556;--threads=0;${CCK_GENCODE}>
    )
endif()

if(DEFINED SKBUILD_PROJECT_NAME)
    install(TARGETS colabfold_cuda_kernels LIBRARY DESTINATION colabfold_cuda_kernels)
endif()
