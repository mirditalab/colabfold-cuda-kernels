// Device-side differences between CUDA and HIP that are not a rename.
#ifndef CUDA_HIP_COMPATIBILITY_
#define CUDA_HIP_COMPATIBILITY_

#include "cuda_hip_rename.h"

#if defined(__HIPCC__)
#define CCK_AMD 1
#else
#define CCK_AMD 0
#endif

// a full-warp xor shuffle: HIP's unsynchronised one, CUDA's synchronised one
template <typename T> __device__ __forceinline__ T shfl_xor(T v, int o, int w) {
#if CCK_AMD
    return __shfl_xor(v, o, w);
#else
    return __shfl_xor_sync(0xffffffffu, v, o, w);
#endif
}

namespace cg = cooperative_groups;

#endif
