// The fused atom MLP: out-projection residual, adaLN, GLU transition, all chained in registers.
// One warp owns 16 atom rows.
#pragma once

#include "diffusion/rows.cuh"

namespace mk {

constexpr int kAtomHidden = 2 * kAtomC; // transition width
constexpr int kChanT = kAtomC / kMma;   // k (and n) tiles over the 128 channels
constexpr int kAtomUnits = kAtomHidden / kGluHalf;
constexpr int kMlpRows = kWarps * kMma; // rows per CTA item

// d = a · W[n0 + 16 nt][k]ᵀ over K = 16 KT, W row-major with ld; B fragments straight from global
template <int NT, int KT>
__device__ __forceinline__ void warp_gemm(Acc (&d)[NT], const FragA (&a)[KT], const bf16* w,
                                          int ld) {
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
#pragma unroll
        for (int kt = 0; kt < KT; kt++) {
            FragB f;
            load_b(f, w + nt * kMma * ld + kt * kMma, ld);
            mma(d[nt], a[kt], f);
        }
    }
}

// the lane's two values (columns c, c + 1) of row r of an [rows][ld] bf16 matrix
__device__ __forceinline__ float2 bf_pair(const bf16* p) {
    const uint32_t u = ld32(p);
    return make_float2(from_bf(u & 0xFFFFu), from_bf(u >> 16));
}

// LayerNorm (or RMSNorm) of the warp's 16 x 128 accumulator rows, then ·scale + offset (am slots)
__device__ __forceinline__ void mlp_adaln(Acc (&x)[kChanT], const bf16* const (&am)[kRowSlots]) {
    const bool rms = c_kp.flags & ESM;
    float mean[kRowSlots] = {}, var[kRowSlots] = {};
    if (!rms) {
#pragma unroll
        for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
            for (int i = 0; i < kAccVals; i++) {
                mean[row_slot(i)] += x[nt].v[i];
            }
        }
        row_reduce_sum(mean);
#pragma unroll
        for (int j = 0; j < kRowSlots; j++) {
            mean[j] /= kAtomC;
        }
    }
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int i = 0; i < kAccVals; i++) {
            const float c = x[nt].v[i] - mean[row_slot(i)];
            var[row_slot(i)] += c * c;
        }
    }
    row_reduce_sum(var);
    float rstd[kRowSlots];
#pragma unroll
    for (int j = 0; j < kRowSlots; j++) {
        rstd[j] = 1.f / sqrtf(var[j] / kAtomC + c_kp.atom_eps);
    }
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int i = 0; i < kAccVals; i += 2) {
            const int c = nt * kMma + acc_col(i), rs = row_slot(i);
            const float2 sc = bf_pair(am[rs] + AM_T_SCALE * kAtomC + c),
                         of = bf_pair(am[rs] + AM_T_BIAS * kAtomC + c);
            x[nt].v[i] = (x[nt].v[i] - mean[rs]) * rstd[rs] * sc.x + of.x;
            x[nt].v[i + 1] = (x[nt].v[i + 1] - mean[rs]) * rstd[rs] * sc.y + of.y;
        }
    }
}

// rows r0..r0+15 of the residual stream through the block's attention output and transition
__device__ __noinline__ void atom_mlp_rows(const AtomW& w, int b, int r0) {
    const int A = c_kp.A, unit = c_kp.flags & ATOB ? 3 * kGluHalf : 2 * kGluHalf;
    const bf16* wo = w.wo + size_t(b) * kAtomC * kAtomC;
    const bf16* w1 = w.w1 + size_t(b) * c_kp.aw1 * kAtomC;
    const bf16* w2 = w.w2 + size_t(b) * kAtomC * kAtomHidden;
    const bool parallel = c_kp.flags & PARALLEL;
    float* qa = c_kp.buf.qa;
    const bf16* am[kRowSlots];
    float qmask[kRowSlots];
    for (int i = 0; i < kAccVals; i++) {
        const int r = r0 + acc_row(i);
        am[row_slot(i)] = atom_mods(w, b, r % A);
        qmask[row_slot(i)] = c_kp.geo.qmask[r % A];
    }

    // 1. T = AO·WOᵀ; tin = qa + am6 ⊙ T back to QA; x holds the transition's input
    Acc x[kChanT];
    {
        FragA a[kChanT];
#pragma unroll
        for (int kt = 0; kt < kChanT; kt++) {
            load_a(a[kt], c_kp.buf.ao + size_t(r0) * kAtomC + kt * kMma, kAtomC);
            acc_zero(x[kt]);
        }
        warp_gemm(x, a, wo, kAtomC);
    }
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int i = 0; i < kAccVals; i += 2) {
            const int c = nt * kMma + acc_col(i);
            float2* p = reinterpret_cast<float2*>(qa + size_t(r0 + acc_row(i)) * kAtomC + c);
            const float2 in = *p, g = bf_pair(am[row_slot(i)] + AM_ATTN_GATE * kAtomC + c);
            const float2 tin = make_float2(in.x + g.x * x[nt].v[i], in.y + g.y * x[nt].v[i + 1]);
            *p = tin;
            x[nt].v[i] = parallel ? in.x : tin.x;
            x[nt].v[i + 1] = parallel ? in.y : tin.y;
        }
    }

    // 2. adaLN, then the A operands of the transition
    mlp_adaln(x, am);
    FragA xa[kChanT];
#pragma unroll
    for (int kt = 0; kt < kChanT; kt++) {
        xa[kt] = acc_to_a(x[kt]);
    }

    // 3. per GLU unit: [gate | proj (| a_to_b)] = x·W1ᵀ, c = silu(gate)·proj(·a_to_b), y += c·W2ᵀ
    Acc y[kChanT];
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
        acc_zero(y[nt]);
    }
    constexpr int kHalves = kGluHalf / kMma;
#pragma unroll 1
    for (int j = 0; j < kAtomUnits; j++) {
        const bf16* wu = w1 + size_t(j) * unit * kAtomC;
        Acc g[kHalves], p[kHalves];
#pragma unroll
        for (int h = 0; h < kHalves; h++) {
            acc_zero(g[h]), acc_zero(p[h]);
        }
        warp_gemm(g, xa, wu, kAtomC);
        warp_gemm(p, xa, wu + kGluHalf * kAtomC, kAtomC);
#pragma unroll
        for (int h = 0; h < kHalves; h++) {
#pragma unroll
            for (int i = 0; i < kAccVals; i++) {
                p[h].v[i] *= silu(g[h].v[i]);
            }
        }
        if (unit > 2 * kGluHalf) {
#pragma unroll
            for (int h = 0; h < kHalves; h++) {
                acc_zero(g[h]);
            }
            warp_gemm(g, xa, wu + 2 * kGluHalf * kAtomC, kAtomC);
#pragma unroll
            for (int h = 0; h < kHalves; h++) {
#pragma unroll
                for (int i = 0; i < kAccVals; i++) {
                    p[h].v[i] *= g[h].v[i];
                }
            }
        }
        FragA ca[kHalves];
#pragma unroll
        for (int h = 0; h < kHalves; h++) {
            ca[h] = acc_to_a(p[h]);
        }
        warp_gemm(y, ca, w2 + j * kGluHalf, kAtomHidden);
    }

    // 4. qa = tin + QMASK·am7 ⊙ y
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int i = 0; i < kAccVals; i += 2) {
            const int c = nt * kMma + acc_col(i), rs = row_slot(i);
            float2* p = reinterpret_cast<float2*>(qa + size_t(r0 + acc_row(i)) * kAtomC + c);
            const float2 tin = *p, g = bf_pair(am[rs] + AM_T_GATE * kAtomC + c);
            *p = make_float2(tin.x + qmask[rs] * g.x * y[nt].v[i],
                             tin.y + qmask[rs] * g.y * y[nt].v[i + 1]);
        }
    }
}

__device__ void atom_mlp(Deal& d, const AtomW& w, int b) {
    const int rows = c_kp.S * c_kp.A, items = cdiv(rows, kMlpRows);
    for (int it = d.first(); it < items; it += gridDim.x) {
        const int r0 = it * kMlpRows + warp_id() * kMma;
        if (r0 < rows) {
            atom_mlp_rows(w, b, r0);
        }
    }
    d.advance(items);
}

} // namespace mk
