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
#if CCK_AMD
// gfx11 WMMA: a lane holds one accumulator column and a whole row of A (column of B)
constexpr int kFragRegs = 8;   // bf16 pairs per A or B fragment
constexpr int kRowLanes = 16;  // lanes sharing an accumulator row
constexpr int kAdj = 1;        // a lane's consecutive values side by side in one row
constexpr int kCopyStages = 1; // shared-memory buffers per staged operand: copies are synchronous
constexpr int kLdsPad = 0;     // bf16 of padding per staged row
#else
constexpr int kFragRegs = 4, kRowLanes = 4, kAdj = 2;
constexpr int kCopyStages = 2; // the next chunk's copy overlaps this one's products
constexpr int kLdsPad = 8;     // conflict-free ldmatrix
#endif
constexpr int kRowSlots = kMma * kRowLanes / kWarp; // distinct accumulator rows per lane
constexpr int kAccTile = kMma * kMma;               // bf16 of a warp's shared tile for acc_to_a
constexpr int kTileVals = kWarp * kAccVals; // values per 16x16 tile, swizzled as [lane][value]

struct Acc {
    float v[kAccVals];
};
struct FragA {
    uint32_t r[kFragRegs];
};
struct FragB {
    uint32_t r[kFragRegs];
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

// kAdj elements side by side in memory, read or written in one access
template <class T> struct alignas(kAdj * sizeof(T)) Adj {
    T v[kAdj];
};
template <class T> __device__ __forceinline__ Adj<T>& adj(T* p) {
    return *reinterpret_cast<Adj<T>*>(p);
}
template <class T> __device__ __forceinline__ const Adj<T>& adj(const T* p) {
    return *reinterpret_cast<const Adj<T>*>(p);
}

// kAdj values as bf16 side by side, rounded as to_bf: one cvt per pair on NVIDIA
__device__ __forceinline__ Adj<bf16> adj_bf(const float (&x)[kAdj]) {
    Adj<bf16> v;
#if CCK_AMD
    v.v[0] = bf16(to_bf(x[0]));
#else
    const __nv_bfloat162 h = __floats2bfloat162_rn(x[0], x[1]);
    memcpy(&v, &h, sizeof v);
#endif
    return v;
}
// x rounded to bf16 and back, a pair at a time
__device__ __forceinline__ void round_bf2(float& a, float& b) {
#if CCK_AMD
    a = from_bf(to_bf(a)), b = from_bf(to_bf(b));
#else
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    a = __low2float(h), b = __high2float(h);
#endif
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

#if !CCK_AMD

// value i of lane l sits at (acc_row_of(l, i), acc_col_of(l, i)) of the tile
__host__ __device__ __forceinline__ int acc_row_of(int l, int i) {
    return l / 4 + 8 * ((i >> 1) & 1);
}
__host__ __device__ __forceinline__ int acc_col_of(int l, int i) {
    return 8 * (i >> 2) + 2 * (l % 4) + (i & 1);
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
// a GEMM slab of 16-k rows: column c of row r staged at slab_col(r, c), and fragments read back;
// NVIDIA pads the rows instead of permuting them
__host__ __device__ __forceinline__ int slab_col(int, int c) {
    return c;
}
__device__ __forceinline__ void load_a_slab(FragA& f, const bf16* s, int ld) {
    load_a_lds(f, s, ld);
}
__device__ __forceinline__ void load_b_slab(FragB& f, const bf16* s, int ld) {
    load_b_lds(f, s, ld);
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
// cvt per pair (the same rounding as to_bf, at a fifth of the instructions; the tile is for AMD)
__device__ __forceinline__ FragA acc_to_a(const Acc& c, bf16*) {
    FragA f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const __nv_bfloat162 h = __floats2bfloat162_rn(c.v[2 * j], c.v[2 * j + 1]);
        f.r[j] = *reinterpret_cast<const uint32_t*>(&h);
    }
    return f;
}

__device__ __forceinline__ void async_copy16(void* smem, const void* g, bool ok) {
    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
        *reinterpret_cast<const cute::uint128_t*>(g), *reinterpret_cast<cute::uint128_t*>(smem),
        ok);
}
__device__ __forceinline__ void async_commit() {
    cute::cp_async_fence();
}
// a copy in two halves (see the AMD side); cp.async needs no second
struct Copy16 {};
__device__ __forceinline__ void copy_start(Copy16&, void* smem, const void* g, bool ok) {
    async_copy16(smem, g, ok);
}
__device__ __forceinline__ void copy_land(const Copy16&, void*) {
}
template <int N> __device__ __forceinline__ void async_wait() {
    cute::cp_async_wait<N>();
}

// v of the lane O apart among those sharing a row
template <int O> __device__ __forceinline__ float row_xor(float v) {
    return shfl_xor(v, O, kWarp);
}

// base-2 exponent without denormals: the instruction __expf ends in
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

#else

__host__ __device__ __forceinline__ int acc_row_of(int l, int i) {
    return 2 * i + l / 16;
}
__host__ __device__ __forceinline__ int acc_col_of(int l, int) {
    return l % 16;
}
__host__ __device__ constexpr int row_slot(int i) {
    return i;
}
__host__ __device__ __forceinline__ int acc_lane(int r, int c) {
    return c % 16 + 16 * (r % 2);
}
__host__ __device__ __forceinline__ int acc_slot(int r, int) {
    return r % 16 / 2;
}

// the lane's 16 contiguous values: row (lane % 16) of A, or column (lane % 16) of B as [N][K]
__device__ __forceinline__ void load_row(uint32_t (&r)[kFragRegs], const bf16* p, int ld) {
    const uint4* q = reinterpret_cast<const uint4*>(p + (lane_id() % 16) * ld);
    const uint4 lo = q[0], hi = q[1];
    r[0] = lo.x, r[1] = lo.y, r[2] = lo.z, r[3] = lo.w;
    r[4] = hi.x, r[5] = hi.y, r[6] = hi.z, r[7] = hi.w;
}
__device__ __forceinline__ void load_a(FragA& f, const bf16* p, int ld) {
    load_row(f.r, p, ld);
}
__device__ __forceinline__ void load_b(FragB& f, const bf16* p, int ld) {
    load_row(f.r, p, ld);
}
__device__ __forceinline__ void load_a_lds(FragA& f, const bf16* s, int ld) {
    load_row(f.r, s, ld);
}
__device__ __forceinline__ void load_b_lds(FragB& f, const bf16* s, int ld) {
    load_row(f.r, s, ld);
}
// a GEMM slab of 16-k rows: rows 8..15 of each 16 hold their two 16-byte halves swapped, so the
// 16 rows a fragment reads at once spread over all banks without padding
__host__ __device__ __forceinline__ int slab_col(int r, int c) {
    return c ^ (r >> 3 & 1) * 8;
}
__device__ __forceinline__ void load_slab_row(uint32_t (&r)[kFragRegs], const bf16* s, int ld) {
    const int l = lane_id() % 16, x = l >> 3 & 1;
    const uint4* q = reinterpret_cast<const uint4*>(s + l * ld);
    const uint4 lo = q[x], hi = q[x ^ 1];
    r[0] = lo.x, r[1] = lo.y, r[2] = lo.z, r[3] = lo.w;
    r[4] = hi.x, r[5] = hi.y, r[6] = hi.z, r[7] = hi.w;
}
__device__ __forceinline__ void load_a_slab(FragA& f, const bf16* s, int ld) {
    load_slab_row(f.r, s, ld);
}
__device__ __forceinline__ void load_b_slab(FragB& f, const bf16* s, int ld) {
    load_slab_row(f.r, s, ld);
}
// B from a [K][N] tile: column (lane % 16), one element per row
__device__ __forceinline__ void load_b_lds_t(FragB& f, const bf16* s, int ld) {
    const bf16* p = s + lane_id() % 16;
#pragma unroll
    for (int j = 0; j < kFragRegs; j++) {
        f.r[j] = p[2 * j * ld] | uint32_t(p[(2 * j + 1) * ld]) << 16;
    }
}

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void mma(Acc& d, const FragA& a, const FragB& b) {
    v16s va, vb;
    v8f c;
    memcpy(&va, a.r, sizeof va);
    memcpy(&vb, b.r, sizeof vb);
    memcpy(&c, d.v, sizeof c);
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(va, vb, c);
    memcpy(d.v, &c, sizeof c);
}

// an accumulator as the A operand of the next product over its columns: a lane holds a column of
// the one and a row of the other, so it goes through t, kAccTile of the warp's own shared memory
__device__ __forceinline__ FragA acc_to_a(const Acc& c, bf16* t) {
#pragma unroll
    for (int i = 0; i < kAccVals; i++) {
        t[acc_row_of(lane_id(), i) * kMma + acc_col_of(lane_id(), i)] = bf16(to_bf(c.v[i]));
    }
    __syncwarp();
    FragA f;
    load_a_lds(f, t, kMma);
    __syncwarp(); // the next call overwrites the tile
    return f;
}

// gfx11 has no asynchronous copy to shared memory: a plain one, with nothing to wait for
__device__ __forceinline__ void async_copy16(void* smem, const void* g, bool ok) {
    *reinterpret_cast<uint4*>(smem) =
        ok ? *reinterpret_cast<const uint4*>(g) : make_uint4(0u, 0u, 0u, 0u);
}
__device__ __forceinline__ void async_commit() {
}
template <int N> __device__ __forceinline__ void async_wait() {
}
// a copy in two halves: the load into registers at the start, the store at landing, so the
// global latency overlaps the work issued in between
struct Copy16 {
    uint4 v;
};
__device__ __forceinline__ void copy_start(Copy16& c, void*, const void* g, bool ok) {
    c.v = ok ? *reinterpret_cast<const uint4*>(g) : make_uint4(0u, 0u, 0u, 0u);
}
__device__ __forceinline__ void copy_land(const Copy16& c, void* smem) {
    *reinterpret_cast<uint4*>(smem) = c.v;
}

__device__ __forceinline__ float ex2(float x) {
    return __builtin_amdgcn_exp2f(x);
}

// v of the lane O apart among those sharing a row: a DPP row_xmask, no trip through the LDS unit
template <int O> __device__ __forceinline__ float row_xor(float v) {
    return __int_as_float(
        __builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 | O, 0xf, 0xf, false));
}

#endif

// over the lanes sharing a row
__device__ __forceinline__ void row_reduce_max(float (&v)[kRowSlots]) {
#pragma unroll
    for (int j = 0; j < kRowSlots; j++) {
        v[j] = fmaxf(v[j], row_xor<1>(v[j]));
        v[j] = fmaxf(v[j], row_xor<2>(v[j]));
        if (kRowLanes > 4) {
            v[j] = fmaxf(v[j], row_xor<4>(v[j]));
            v[j] = fmaxf(v[j], row_xor<8>(v[j]));
        }
    }
}
__device__ __forceinline__ void row_reduce_sum(float (&v)[kRowSlots]) {
#pragma unroll
    for (int j = 0; j < kRowSlots; j++) {
        v[j] += row_xor<1>(v[j]);
        v[j] += row_xor<2>(v[j]);
        if (kRowLanes > 4) {
            v[j] += row_xor<4>(v[j]);
            v[j] += row_xor<8>(v[j]);
        }
    }
}

__device__ __forceinline__ int acc_row(int i) {
    return acc_row_of(lane_id(), i);
}

// an accumulator as two A operands whose sum carries its f32 value to ~16 bits: a product over both
// keeps an f32 input's precision where AlphaFold 3 runs the layer in f32
__device__ __forceinline__ void acc_to_a_split(const Acc& c, FragA& hi, FragA& lo, bf16* tile) {
    Acc r;
#pragma unroll
    for (int i = 0; i < kAccVals; i++) {
        r.v[i] = c.v[i] - from_bf(to_bf(c.v[i]));
    }
    hi = acc_to_a(c, tile);
    lo = acc_to_a(r, tile);
}
__device__ __forceinline__ int acc_col(int i) {
    return acc_col_of(lane_id(), i);
}

} // namespace mk
