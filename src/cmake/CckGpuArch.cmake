# colabfold-cuda-kernels GPU architecture resolution
#
# CUDA: turns the requested CMAKE_CUDA_ARCHITECTURES into gencode flags:
#   - SASS for every requested architecture; a cubin also runs on the later minors of its major
#   - PTX on the highest one, for forward JIT to newer GPUs
#
# HIP builds for gfx11 and newer wave32 parts only; it is buildable but not shipped.

set(CCK_SUPPORTED_ARCHS "80;86;87;89;90;100;103;120;121"
    CACHE STRING "Supported CUDA architectures (used to resolve all/all-major)")

set(CCK_SUPPORTED_HIP_ARCHS "gfx1151"
    CACHE STRING "Default AMD/HIP architectures")

# Resolve CMAKE_CUDA_ARCHITECTURES into a filtered, de-duplicated integer list.
function(cck_resolve_cuda_architectures out_var)
    set(requested "${CMAKE_CUDA_ARCHITECTURES}")
    if(requested STREQUAL "native")
        set(archs "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    elseif(requested STREQUAL "all" OR requested STREQUAL "all-major")
        set(archs "${CCK_SUPPORTED_ARCHS}")
    else()
        set(archs "${requested}")
    endif()

    set(result "")
    foreach(arch IN LISTS archs)
        string(REGEX REPLACE "-(real|virtual)$" "" arch "${arch}")
        if(arch MATCHES "^[0-9]+$" AND NOT arch LESS 80)
            list(APPEND result "${arch}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES result)
    list(SORT result COMPARE NATURAL)

    if(result STREQUAL "")
        message(FATAL_ERROR "cck: no supported CUDA architectures (>= 80) after resolving CMAKE_CUDA_ARCHITECTURES='${requested}'")
    endif()
    set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

# gencode flags: SASS for each architecture, PTX for the highest
function(cck_cuda_gencode archs out_var)
    set(result "")
    foreach(arch IN LISTS archs)
        list(APPEND result "-gencode=arch=compute_${arch},code=sm_${arch}")
    endforeach()
    list(GET archs -1 top)
    list(APPEND result "-gencode=arch=compute_${top},code=compute_${top}")
    set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

function(cck_resolve_hip_architectures out_var)
    if(CMAKE_HIP_ARCHITECTURES)
        set(archs "${CMAKE_HIP_ARCHITECTURES}")
    else()
        set(archs "${CCK_SUPPORTED_HIP_ARCHS}")
    endif()
    set(${out_var} "${archs}" PARENT_SCOPE)
endfunction()
