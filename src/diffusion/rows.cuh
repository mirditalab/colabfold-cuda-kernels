// Row phases, a warp per row: LayerNorm, adaLN and q/k norms, the noise vector, the single
// conditioning's row ops, and the small per-atom glue between blocks.
#pragma once

#include <cfloat>

#include "diffusion/gemm.cuh"

namespace mk {

constexpr int kRowV4 = kTokC / (4 * kWarp); // float4s per lane of the widest row (768)

__device__ __forceinline__ int warp_id() {
    return threadIdx.x / kWarp;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = kWarp / 2; o; o /= 2) {
        v += __shfl_xor(v, o, kWarp);
    }
    return v;
}

// over the CTA, through red[kWarps] of shared memory
__device__ inline float block_sum(float v, float* red) {
    v = warp_sum(v);
    __syncthreads();
    if (lane_id() == 0) {
        red[warp_id()] = v;
    }
    __syncthreads();
    float s = 0.f;
    for (int w = 0; w < kWarps; w++) {
        s += red[w];
    }
    return s;
}

// rows of a phase go kWarps to a CTA item, continuing the phase's round-robin
__device__ __forceinline__ int row_first(const Deal& d) {
    return d.first() * kWarps + warp_id();
}
__device__ __forceinline__ int row_step() {
    return gridDim.x * kWarps;
}
__device__ __forceinline__ void row_done(Deal& d, int rows) {
    d.advance(cdiv(rows, kWarps));
}

// A row of up to 768 values: lane l holds columns 4l + 128j for j < nv
struct Row {
    float4 v[kRowV4];
    int nv;
};

__device__ __forceinline__ Row load_row(const float* p, int n) {
    Row r;
    r.nv = n / (4 * kWarp);
#pragma unroll
    for (int j = 0; j < kRowV4; j++) {
        if (j < r.nv) {
            r.v[j] = ld4(p + 4 * (lane_id() + kWarp * j));
        }
    }
    return r;
}
__device__ __forceinline__ Row load_row_bf(const bf16* p, int n) {
    Row r;
    r.nv = n / (4 * kWarp);
#pragma unroll
    for (int j = 0; j < kRowV4; j++) {
        if (j < r.nv) {
            r.v[j] = ldbf4(p + 4 * (lane_id() + kWarp * j));
        }
    }
    return r;
}
__device__ __forceinline__ void store_row(float* p, const Row& r) {
#pragma unroll
    for (int j = 0; j < kRowV4; j++) {
        if (j < r.nv) {
            st4(p + 4 * (lane_id() + kWarp * j), r.v[j]);
        }
    }
}
__device__ __forceinline__ void store_row_bf(bf16* p, const Row& r) {
#pragma unroll
    for (int j = 0; j < kRowV4; j++) {
        if (j < r.nv) {
            stbf4(p + 4 * (lane_id() + kWarp * j), r.v[j]);
        }
    }
}
__device__ __forceinline__ float4 col4(const float* p, int j) {
    return ld4(p + 4 * (lane_id() + kWarp * j));
}
__device__ __forceinline__ float4 col4(const bf16* p, int j) {
    return ldbf4(p + 4 * (lane_id() + kWarp * j));
}

__device__ __forceinline__ float sum4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

// two-pass LayerNorm without affine, in place
__device__ inline void layernorm(Row& r, float eps) {
    const float n = float(r.nv * 4 * kWarp);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < kRowV4; j++) {
        if (j < r.nv) {
            s += sum4(r.v[j]);
        }
    }
    const float mean = warp_sum(s) / n;
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < kRowV4; j++) {
        if (j < r.nv) {
            r.v[j] =
                make_float4(r.v[j].x - mean, r.v[j].y - mean, r.v[j].z - mean, r.v[j].w - mean);
            q += sum4(mul4(r.v[j], r.v[j]));
        }
    }
    const float rstd = 1.f / sqrtf(warp_sum(q) / n + eps);
#pragma unroll
    for (int j = 0; j < kRowV4; j++) {
        r.v[j] = make_float4(r.v[j].x * rstd, r.v[j].y * rstd, r.v[j].z * rstd, r.v[j].w * rstd);
    }
}
__device__ inline void rmsnorm(Row& r, float eps) {
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < kRowV4; j++) {
        if (j < r.nv) {
            q += sum4(mul4(r.v[j], r.v[j]));
        }
    }
    const float inv = 1.f / sqrtf(warp_sum(q) / float(r.nv * 4 * kWarp) + eps);
#pragma unroll
    for (int j = 0; j < kRowV4; j++) {
        r.v[j] = make_float4(r.v[j].x * inv, r.v[j].y * inv, r.v[j].z * inv, r.v[j].w * inv);
    }
}
// y = x·scale + offset per column
template <class S, class O>
__device__ inline Row affine(const Row& x, const S* scale, const O* offset) {
    Row y = x;
#pragma unroll
    for (int j = 0; j < kRowV4; j++) {
        if (j < x.nv) {
            y.v[j] = add4(mul4(x.v[j], col4(scale, j)), col4(offset, j));
        }
    }
    return y;
}

__device__ __forceinline__ void atom_norm(Row& r) {
    if (c_kp.flags & ESM) {
        rmsnorm(r, c_kp.atom_eps);
    } else {
        layernorm(r, c_kp.atom_eps);
    }
}

// ---- conditioning

// ntv = LN(cos(2π(¼ ln(t/σ_d) w + b))) @ WN, on one CTA
__device__ void noise_vector(Deal& d, float t) {
    if (d.first() == 0) {
        float* e = reinterpret_cast<float*>(smem_base());
        float* red = e + kFourier;
        const TokenW& w = c_kp.tw;
        const float lt = 0.25f * logf(t / kSigmaData);
        float s = 0.f;
        for (int k = threadIdx.x; k < kFourier; k += kThreads) {
            e[k] = cosf(2.f * float(M_PI) * (lt * w.fourier_w[k] + w.fourier_b[k]));
            s += e[k];
        }
        const float mean = block_sum(s, red) / kFourier;
        float q = 0.f;
        for (int k = threadIdx.x; k < kFourier; k += kThreads) {
            q += (e[k] - mean) * (e[k] - mean);
        }
        const float rstd = 1.f / sqrtf(block_sum(q, red) / kFourier + kLnEps);
        for (int k = threadIdx.x; k < kFourier; k += kThreads) {
            e[k] = (e[k] - mean) * rstd * w.noise_ln[k] + w.noise_lno[k];
        }
        __syncthreads();
        for (int c = threadIdx.x; c < c_kp.cs; c += kThreads) {
            float acc = 0.f;
            for (int k = 0; k < kFourier; k++) {
                acc += e[k] * w.wn[k * c_kp.cs + c];
            }
            c_kp.buf.ntv[c] = acc;
        }
        __syncthreads();
    }
    d.advance(1);
}

// SCT = LN(sc)·ST_S + ST_O; the first one also forms sc = SC_BASE + ntv
__device__ void sc_layernorm(Deal& d, int i, bool first) {
    const int cs = c_kp.cs;
    for (int n = row_first(d); n < c_kp.N; n += row_step()) {
        Row x;
        if (first) {
            x = load_row(c_kp.tw.sc_base + size_t(n) * cs, cs);
            const Row ntv = load_row(c_kp.buf.ntv, cs);
#pragma unroll
            for (int j = 0; j < kRowV4; j++) {
                x.v[j] = add4(x.v[j], ntv.v[j]);
            }
            store_row(c_kp.buf.sc + size_t(n) * cs, x);
        } else {
            x = load_row(c_kp.buf.sc + size_t(n) * cs, cs);
        }
        layernorm(x, kLnEps);
        store_row_bf(c_kp.buf.sct + size_t(n) * cs, affine(x, c_kp.tw.st_s[i], c_kp.tw.st_o[i]));
    }
    row_done(d, c_kp.N);
}

// SCFINAL's affine LayerNorm, then the two bf16 views: SCLN0 = LN(sc), SCBF = sc
__device__ void sc_prep(Deal& d) {
    const int cs = c_kp.cs;
    for (int n = row_first(d); n < c_kp.N; n += row_step()) {
        Row x = load_row(c_kp.buf.sc + size_t(n) * cs, cs);
        if (c_kp.flags & SCFINAL) {
            layernorm(x, kLnEps);
            x = affine(x, c_kp.tw.scf, c_kp.tw.scf + cs);
        }
        store_row_bf(c_kp.buf.scbf + size_t(n) * cs, x);
        layernorm(x, kLnEps);
        store_row_bf(c_kp.buf.scln0 + size_t(n) * cs, x);
    }
    row_done(d, c_kp.N);
}

// ---- atom rows

__device__ __forceinline__ const bf16* atom_mods(const AtomW& w, int b, int a) {
    return w.amods + (size_t(b) * c_kp.as + a) * kAtomModSlots * kAtomC;
}

// AX = norm(qa)·am0 + am1; AXK = norm(qa or bf16(AX))·am2 + am3
__device__ void atom_adaln(Deal& d, const AtomW& w, int b) {
    const int rows = c_kp.S * c_kp.A;
    for (int r = row_first(d); r < rows; r += row_step()) {
        const bf16* am = atom_mods(w, b, r % c_kp.A);
        Row x = load_row(c_kp.buf.qa + size_t(r) * kAtomC, kAtomC);
        atom_norm(x);
        Row xq = affine(x, am + AM_Q_SCALE * kAtomC, am + AM_Q_BIAS * kAtomC);
        store_row_bf(c_kp.buf.ax + size_t(r) * kAtomC, xq);
        if (c_kp.flags & CHAIN_K) {
            x = load_row_bf(c_kp.buf.ax + size_t(r) * kAtomC, kAtomC);
            atom_norm(x);
        }
        store_row_bf(c_kp.buf.axk + size_t(r) * kAtomC,
                     affine(x, am + AM_K_SCALE * kAtomC, am + AM_K_BIAS * kAtomC));
    }
    row_done(d, rows);
}

// QKNORM: LayerNorm of q and of k over their 128 channels; ESM: per-head RMSNorm and rotary
__device__ void atom_qknorm(Deal& d, const AtomW& w, int b) {
    const int rows = c_kp.S * c_kp.A;
    for (int r = row_first(d); r < 2 * rows; r += row_step()) {
        const bool is_k = r >= rows;
        const int row = r - (is_k ? rows : 0);
        bf16* p = (is_k ? c_kp.buf.akv : c_kp.buf.aqg) + size_t(row) * 2 * kAtomC;
        Row x = load_row_bf(p, kAtomC);
        if (c_kp.flags & QKNORM) {
            const float* qkn =
                w.qkn + (b * kQkNormRows + (is_k ? QKN_K_SCALE : QKN_Q_SCALE)) * kAtomC;
            layernorm(x, kLnEps);
            x = affine(x, qkn, qkn + kAtomC);
        } else {
            // a head is 8 lanes of 4 channels
            float4& v = x.v[0];
            float q = sum4(mul4(v, v));
            for (int o = 1; o < kAtomHeadD / 4; o *= 2) {
                q += __shfl_xor(q, o, kWarp);
            }
            const float inv = 1.f / sqrtf(q / kAtomHeadD + FLT_EPSILON);
            v = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
            constexpr int kHalf = kAtomHeadD / 2;
            const int d0 = 4 * lane_id() % kAtomHeadD;
            const float* rope = c_kp.geo.rope + size_t(row % c_kp.A) * kAtomHeadD + d0 % kHalf;
            const float4 cs = ld4(rope), sn = ld4(rope + kHalf);
            const int partner = kHalf / 4; // the lane holding channel d ± 16
            float4 o =
                make_float4(__shfl_xor(v.x, partner, kWarp), __shfl_xor(v.y, partner, kWarp),
                            __shfl_xor(v.z, partner, kWarp), __shfl_xor(v.w, partner, kWarp));
            if (d0 < kHalf) {
                o = make_float4(-o.x, -o.y, -o.z, -o.w);
            }
            v = add4(mul4(v, cs), mul4(o, sn));
        }
        store_row_bf(p, x);
    }
    row_done(d, 2 * rows);
}

// skip = qa·QMASK, kept in f32 and as bf16 for the aggregation GEMM
__device__ void atom_skip(Deal& d) {
    const int rows = c_kp.S * c_kp.A;
    for (int r = row_first(d); r < rows; r += row_step()) {
        const float m = c_kp.geo.qmask[r % c_kp.A];
        Row x = load_row(c_kp.buf.qa + size_t(r) * kAtomC, kAtomC);
        x.v[0] = make_float4(x.v[0].x * m, x.v[0].y * m, x.v[0].z * m, x.v[0].w * m);
        store_row(c_kp.buf.skip + size_t(r) * kAtomC, x);
        store_row_bf(c_kp.buf.abf + size_t(r) * kAtomC, x);
    }
    row_done(d, rows);
}

// qa = (tb[token of a] + skip)·QMASK
__device__ void decoder_init(Deal& d) {
    const int rows = c_kp.S * c_kp.A;
    for (int r = row_first(d); r < rows; r += row_step()) {
        const int s = r / c_kp.A, a = r % c_kp.A;
        const int slot = c_kp.geo.tok2q[a];
        const float m = c_kp.geo.qmask[a];
        float4 x = col4(c_kp.buf.skip + size_t(r) * kAtomC, 0);
        if (slot >= 0) {
            x = add4(x, col4(c_kp.buf.tb + (size_t(s) * c_kp.N + slot / kSlots) * kAtomC, 0));
        }
        st4(c_kp.buf.qa + size_t(r) * kAtomC + 4 * lane_id(),
            make_float4(x.x * m, x.y * m, x.z * m, x.w * m));
    }
    row_done(d, rows);
}

// upd = LN(qa·QMASK) @ WUPDᵀ + UPD_B
__device__ void atom_update(Deal& d) {
    const int rows = c_kp.S * c_kp.A;
    for (int r = row_first(d); r < rows; r += row_step()) {
        const float m = c_kp.geo.qmask[r % c_kp.A];
        Row x = load_row(c_kp.buf.qa + size_t(r) * kAtomC, kAtomC);
        x.v[0] = make_float4(x.v[0].x * m, x.v[0].y * m, x.v[0].z * m, x.v[0].w * m);
        layernorm(x, kLnEps);
        for (int j = 0; j < kXyz; j++) {
            const float u = warp_sum(sum4(mul4(x.v[0], col4(c_kp.geo.wupd + j * kAtomC, 0))));
            if (lane_id() == 0) {
                c_kp.buf.upd[size_t(r) * kXyz + j] = u + c_kp.geo.upd_b[j];
            }
        }
    }
    row_done(d, rows);
}

// ---- token rows

__device__ __forceinline__ const bf16* tok_mods(int b, int slot) {
    return c_kp.buf.mods + (size_t(b) * kTokModSlots + slot) * c_kp.N * kTokC;
}

// out = LN(act)·mods[slot] + mods[slot+1]; with out2, also the transition's input from the same LN
__device__ void token_adaln(Deal& d, int b, int slot, bf16* out, bf16* out2) {
    const int rows = c_kp.S * c_kp.N;
    const bf16* m = tok_mods(b, slot);
    const bf16* m2 = tok_mods(b, TM_TRANS_SCALE); // PARALLEL: the transition's input, same LN
    for (int r = row_first(d); r < rows; r += row_step()) {
        const size_t n = r % c_kp.N;
        Row x = load_row(c_kp.buf.act + size_t(r) * kTokC, kTokC);
        layernorm(x, c_kp.ln_eps);
        const size_t mo = n * kTokC;
        store_row_bf(out + size_t(r) * kTokC, affine(x, m + mo, m + c_kp.N * kTokC + mo));
        if (out2) {
            store_row_bf(out2 + size_t(r) * kTokC, affine(x, m2 + mo, m2 + c_kp.N * kTokC + mo));
        }
    }
    row_done(d, rows);
}

// QKNORM: LayerNorm of q and of k over their 768 flattened channels
__device__ void token_qknorm(Deal& d, int b) {
    const int rows = c_kp.S * c_kp.N;
    for (int r = row_first(d); r < 2 * rows; r += row_step()) {
        const bool is_k = r >= rows;
        bf16* p = c_kp.buf.qkvg + size_t(r - (is_k ? rows : 0)) * kQKVG + (is_k ? kTokC : 0);
        const float* qkn =
            c_kp.tw.tqkn + (b * kQkNormRows + (is_k ? QKN_K_SCALE : QKN_Q_SCALE)) * kTokC;
        Row x = load_row_bf(p, kTokC);
        layernorm(x, kLnEps);
        store_row_bf(p, affine(x, qkn, qkn + kTokC));
    }
    row_done(d, 2 * rows);
}

// TX = LN(act): the output norm's scale and offset are folded into WTB and TB_B
__device__ void token_final_ln(Deal& d) {
    const int rows = c_kp.S * c_kp.N;
    for (int r = row_first(d); r < rows; r += row_step()) {
        Row x = load_row(c_kp.buf.act + size_t(r) * kTokC, kTokC);
        layernorm(x, kLnEps);
        store_row_bf(c_kp.buf.tx + size_t(r) * kTokC, x);
    }
    row_done(d, rows);
}

} // namespace mk
