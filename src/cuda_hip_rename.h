// The CUDA runtime names the kernels use, mapped to HIP when hipcc compiles them.
#ifndef CUDA_HIP_RENAME_H_
#define CUDA_HIP_RENAME_H_

#if defined(__HIPCC__)

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaError_t hipError_t
#define cudaDeviceProp hipDeviceProp_t

#define cudaSuccess hipSuccess
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemGetInfo hipMemGetInfo
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDestroy hipEventDestroy
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount

// HIP names a __constant__ symbol through HIP_SYMBOL
#define cudaMemcpyToSymbolAsync(symbol, ...) hipMemcpyToSymbolAsync(HIP_SYMBOL(symbol), __VA_ARGS__)

#else

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#endif

#endif
