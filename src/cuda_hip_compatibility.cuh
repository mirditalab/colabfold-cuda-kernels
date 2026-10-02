// Device-side differences between CUDA and HIP that are not a rename.
#ifndef CUDA_HIP_COMPATIBILITY_
#define CUDA_HIP_COMPATIBILITY_

#include "cuda_hip_rename.h"

#if defined(__HIPCC__)
#define CCK_AMD 1
#else
#define CCK_AMD 0
// HIP's unsynchronised warp shuffle, as a full-warp synchronised one
#define __shfl_xor(v, o, w) __shfl_xor_sync(0xffffffffu, v, o, w)
#endif

namespace cg = cooperative_groups;

#endif
