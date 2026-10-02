// The vendor fragment layer: everything that depends on the MMA's register layout.
// A warp computes 16x16x16 bf16 products into an f32 accumulator of 8 values per lane.
#pragma once

#include <cstring>

#include "cuda_hip_compatibility.cuh"
#include "diffusion/dims.cuh"

#if !CCK_AMD
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cuda_bf16.h>
#endif

namespace mk {

constexpr int kMma = 16;    // tile edge of one product
constexpr int kAccVals = 8; // accumulator values per lane

struct Acc {
    float v[kAccVals];
};
struct FragA {
    uint32_t r[4];
};
struct FragB {
    uint32_t r[4];
};

__device__ __forceinline__ int lane_id() {
    return threadIdx.x % kWarp;
}

// bf16 round-to-nearest-even in integer form: the intrinsics measured slower
__host__ __device__ __forceinline__ uint32_t to_bf(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
__host__ __device__ __forceinline__ float from_bf(uint32_t b) {
    uint32_t u = b << 16;
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
    return to_bf(lo) | (to_bf(hi) << 16);
}
__device__ __forceinline__ uint2 pack_bf4(float4 v) {
    return make_uint2(pack_bf2(v.x, v.y), pack_bf2(v.z, v.w));
}
__device__ __forceinline__ float4 unpack_bf4(uint2 u) {
    return make_float4(from_bf(u.x & 0xFFFFu), from_bf(u.x >> 16), from_bf(u.y & 0xFFFFu),
                       from_bf(u.y >> 16));
}

#if !CCK_AMD

constexpr int kRowSlots = 2; // distinct accumulator rows per lane

// value i of this lane sits at (acc_row(i), acc_col(i)) of the tile
__device__ __forceinline__ int acc_row(int i) {
    return lane_id() / 4 + 8 * ((i >> 1) & 1);
}
__device__ __forceinline__ int acc_col(int i) {
    return 8 * (i >> 2) + 2 * (lane_id() % 4) + (i & 1);
}
// which of the lane's rows value i belongs to
__host__ __device__ constexpr int row_slot(int i) {
    return (i >> 1) & 1;
}
// the inverse, to write a swizzled table: element (r, c) is value acc_slot of lane acc_lane
__host__ __device__ __forceinline__ int acc_lane(int r, int c) {
    return (r % 8) * 4 + (c % 8) / 2;
}
__host__ __device__ __forceinline__ int acc_slot(int r, int c) {
    return (c / 8) * 4 + (r / 8) * 2 + (c & 1);
}

__device__ __forceinline__ void acc_zero(Acc& a) {
#pragma unroll
    for (int i = 0; i < kAccVals; i++) {
        a.v[i] = 0.f;
    }
}

__device__ __forceinline__ uint32_t ld32(const bf16* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

// rows p[r * ld + k] from global memory
__device__ __forceinline__ void load_a(FragA& f, const bf16* p, int ld) {
    const int g = lane_id() / 4, t = 2 * (lane_id() % 4);
    f.r[0] = ld32(p + g * ld + t);
    f.r[1] = ld32(p + (g + 8) * ld + t);
    f.r[2] = ld32(p + g * ld + t + 8);
    f.r[3] = ld32(p + (g + 8) * ld + t + 8);
}
// columns p[n * ld + k] (W in [N][K] order) from global memory
__device__ __forceinline__ void load_b(FragB& f, const bf16* p, int ld) {
    const int g = lane_id() / 4, t = 2 * (lane_id() % 4);
    f.r[0] = ld32(p + g * ld + t);
    f.r[1] = ld32(p + g * ld + t + 8);
    f.r[2] = ld32(p + (g + 8) * ld + t);
    f.r[3] = ld32(p + (g + 8) * ld + t + 8);
}

__device__ __forceinline__ const cute::uint128_t& lds_row(const bf16* p) {
    return *reinterpret_cast<const cute::uint128_t*>(p);
}
__device__ __forceinline__ void load_a_lds(FragA& f, const bf16* s, int ld) {
    const int l = lane_id();
    cute::SM75_U32x4_LDSM_N::copy(lds_row(s + (l % 16) * ld + 8 * (l / 16)), f.r[0], f.r[1], f.r[2],
                                  f.r[3]);
}
__device__ __forceinline__ void load_b_lds(FragB& f, const bf16* s, int ld) {
    const int l = lane_id();
    cute::SM75_U32x4_LDSM_N::copy(lds_row(s + (l % 8 + 8 * (l / 16)) * ld + 8 * ((l / 8) % 2)),
                                  f.r[0], f.r[1], f.r[2], f.r[3]);
}
// B from a [K][N] tile: V in P·V needs no transposed copy
__device__ __forceinline__ void load_b_lds_t(FragB& f, const bf16* s, int ld) {
    const int l = lane_id();
    cute::SM75_U16x8_LDSM_T::copy(lds_row(s + (l % 16) * ld + 8 * (l / 16)), f.r[0], f.r[1], f.r[2],
                                  f.r[3]);
}

__device__ __forceinline__ void mma(Acc& d, const FragA& a, const FragB& b) {
    using Op = cute::SM80_16x8x16_F32BF16BF16F32_TN;
    Op::fma(d.v[0], d.v[1], d.v[2], d.v[3], a.r[0], a.r[1], a.r[2], a.r[3], b.r[0], b.r[1], d.v[0],
            d.v[1], d.v[2], d.v[3]);
    Op::fma(d.v[4], d.v[5], d.v[6], d.v[7], a.r[0], a.r[1], a.r[2], a.r[3], b.r[2], b.r[3], d.v[4],
            d.v[5], d.v[6], d.v[7]);
}

// an accumulator as the A operand of the next product over its columns: a register repack, with one
// cvt per pair (the same rounding as to_bf, at a fifth of the instructions)
__device__ __forceinline__ FragA acc_to_a(const Acc& c) {
    FragA f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const __nv_bfloat162 h = __floats2bfloat162_rn(c.v[2 * j], c.v[2 * j + 1]);
        f.r[j] = *reinterpret_cast<const uint32_t*>(&h);
    }
    return f;
}

// over the lanes sharing a row
__device__ __forceinline__ void row_reduce_max(float (&v)[kRowSlots]) {
#pragma unroll
    for (int j = 0; j < kRowSlots; j++) {
        v[j] = fmaxf(v[j], __shfl_xor(v[j], 1, kWarp));
        v[j] = fmaxf(v[j], __shfl_xor(v[j], 2, kWarp));
    }
}
__device__ __forceinline__ void row_reduce_sum(float (&v)[kRowSlots]) {
#pragma unroll
    for (int j = 0; j < kRowSlots; j++) {
        v[j] += __shfl_xor(v[j], 1, kWarp);
        v[j] += __shfl_xor(v[j], 2, kWarp);
    }
}

__device__ __forceinline__ void async_copy16(void* smem, const void* g, bool ok) {
    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
        *reinterpret_cast<const cute::uint128_t*>(g), *reinterpret_cast<cute::uint128_t*>(smem),
        ok);
}
__device__ __forceinline__ void async_commit() {
    cute::cp_async_fence();
}
template <int N> __device__ __forceinline__ void async_wait() {
    cute::cp_async_wait<N>();
}

// base-2 exponent without denormals: the instruction __expf ends in
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

#endif

} // namespace mk
