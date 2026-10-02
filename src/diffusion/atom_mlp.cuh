// The fused atom MLP: out-projection residual, adaLN, GLU transition, chained in registers.
// One warp owns 16 atom rows; the CTA's warps share the weights through a shared-memory ring.
#pragma once

#include "diffusion/rows.cuh"

namespace mk {

constexpr int kAtomHidden = 2 * kAtomC; // transition width
constexpr int kChanT = kAtomC / kMma;   // k (and n) tiles over the 128 channels
constexpr int kAtomUnits = kAtomHidden / kGluHalf;
constexpr int kMlpRows = kWarps * kMma; // rows per CTA item

// The weights stream through a ring of stages: WO in pieces of 32 output rows, then half-units of
// 16 hidden channels (each GLU part's W1 rows and the matching W2 columns)
constexpr int kWoRows = 2 * kMma;
constexpr int kWoPieces = kAtomC / kWoRows;
constexpr int kHalfUnit = kMma;
constexpr int kHalfUnits = kAtomHidden / kHalfUnit;
constexpr int kMlpStages = kWoPieces + kHalfUnits;
constexpr int kW1Ld = kAtomC + kLdsPad; // smem rows: conflict-free ldmatrix
constexpr int kW2Ld = kHalfUnit + kLdsPad;
constexpr int kMaxParts = 3; // gate, proj, a_to_b
__host__ __device__ constexpr int mlp_slot(int parts) {
    return parts * kHalfUnit * kW1Ld + kAtomC * kW2Ld;
}
constexpr int kMlpSmemElems = 3 * mlp_slot(2); // also fits 2 slots of three parts
static_assert(2 * mlp_slot(kMaxParts) <= kMlpSmemElems && kWoRows * kW1Ld <= mlp_slot(2), "");

// stage i of the ring into slot st
__device__ __forceinline__ void mlp_stage(bf16* st, int i, const bf16* wo, const bf16* w1,
                                          const bf16* w2, int parts) {
    constexpr int kRowCopies = kAtomC / 8;
    if (i < kWoPieces) {
        for (int k = threadIdx.x; k < kWoRows * kRowCopies; k += kThreads) {
            const int r = k / kRowCopies, c = k % kRowCopies * 8;
            async_copy16(st + r * kW1Ld + c, wo + (i * kWoRows + r) * kAtomC + c, true);
        }
        return;
    }
    const int u = (i - kWoPieces) / 2, h = (i - kWoPieces) % 2;
    for (int k = threadIdx.x; k < parts * kHalfUnit * kRowCopies; k += kThreads) {
        const int r = k / kRowCopies, c = k % kRowCopies * 8;
        const int src =
            u * parts * kGluHalf + r / kHalfUnit * kGluHalf + h * kHalfUnit + r % kHalfUnit;
        async_copy16(st + r * kW1Ld + c, w1 + src * kAtomC + c, true);
    }
    bf16* s2 = st + parts * kHalfUnit * kW1Ld;
    constexpr int kW2Copies = kHalfUnit / 8;
    for (int k = threadIdx.x; k < kAtomC * kW2Copies; k += kThreads) {
        const int r = k / kW2Copies, c = k % kW2Copies * 8;
        async_copy16(s2 + r * kW2Ld + c, w2 + r * kAtomHidden + u * kGluHalf + h * kHalfUnit + c,
                     true);
    }
}

// d[nt] += a · Ws[16 nt..][k]ᵀ over K = 16 KT, from a [rows][ld] smem tile
template <int NT, int KT>
__device__ __forceinline__ void warp_gemm_lds(Acc (&d)[NT], const FragA (&a)[KT], const bf16* ws,
                                              int ld) {
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
#pragma unroll
        for (int kt = 0; kt < KT; kt++) {
            FragB f;
            load_b_lds(f, ws + nt * kMma * ld + kt * kMma, ld);
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

// advances the ring to stage i: waits for it, refills the slot every warp has finished, returns it
__device__ __forceinline__ const bf16* mlp_ring(bf16* sm, int i, int ring, int slot, const bf16* wo,
                                                const bf16* w1, const bf16* w2, int parts) {
    if (ring > 2) {
        async_wait<1>();
    } else {
        async_wait<0>();
    }
    __syncthreads();
    if (i + ring - 1 < kMlpStages) {
        mlp_stage(sm + (i + ring - 1) % ring * slot, i + ring - 1, wo, w1, w2, parts);
    }
    async_commit();
    return sm + i % ring * slot;
}

// one CTA item: rows row0.. (16 per warp) through the block's attention output and transition
__device__ __noinline__ void atom_mlp_item(const AtomW& w, int b, int row0, int rows) {
    const int A = c_kp.A, parts = c_kp.flags & ATOB ? kMaxParts : 2;
    const int slot = mlp_slot(parts), ring = kMlpSmemElems / slot;
    const bf16* wo = w.wo + size_t(b) * kAtomC * kAtomC;
    const bf16* w1 = w.w1 + size_t(b) * c_kp.aw1 * kAtomC;
    const bf16* w2 = w.w2 + size_t(b) * kAtomC * kAtomHidden;
    bf16* sm = reinterpret_cast<bf16*>(smem_base());
    const int r0 = row0 + warp_id() * kMma;
    const bool active = r0 < rows; // every warp keeps to the barriers
    float* qa = c_kp.buf.qa;
    const bf16* am[kRowSlots];
    float qmask[kRowSlots];
    for (int i = 0; i < kAccVals; i++) {
        const int r = (active ? r0 : 0) + acc_row(i);
        am[row_slot(i)] = atom_mods(w, b, r % A);
        qmask[row_slot(i)] = c_kp.geo.qmask[r % A];
    }
    for (int i = 0; i < ring - 1; i++) {
        mlp_stage(sm + i * slot, i, wo, w1, w2, parts);
        async_commit();
    }

    // 1. T = AO·WOᵀ, 32 output channels per stage
    FragA a[kChanT]; // AO, then the transition's input
    Acc x[kChanT];
#pragma unroll
    for (int kt = 0; kt < kChanT; kt++) {
        if (active) {
            load_a(a[kt], c_kp.buf.ao + size_t(r0) * kAtomC + kt * kMma, kAtomC);
        }
        acc_zero(x[kt]);
    }
    constexpr int kPieceT = kWoRows / kMma;
#pragma unroll
    for (int i = 0; i < kWoPieces; i++) {
        const bf16* st = mlp_ring(sm, i, ring, slot, wo, w1, w2, parts);
        Acc t[kPieceT];
#pragma unroll
        for (int j = 0; j < kPieceT; j++) {
            t[j] = x[i * kPieceT + j];
        }
        warp_gemm_lds(t, a, st, kW1Ld);
#pragma unroll
        for (int j = 0; j < kPieceT; j++) {
            x[i * kPieceT + j] = t[j];
        }
    }

    // 2. tin = qa + am6 ⊙ T, kept in registers; adaLN of tin (or of the block input) as A operands
    Acc tin[kChanT];
    if (active) {
        const bool parallel = c_kp.flags & PARALLEL;
#pragma unroll
        for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
            for (int v = 0; v < kAccVals; v += 2) {
                const int c = nt * kMma + acc_col(v);
                const float2 in =
                    *reinterpret_cast<const float2*>(qa + size_t(r0 + acc_row(v)) * kAtomC + c);
                const float2 g = bf_pair(am[row_slot(v)] + AM_ATTN_GATE * kAtomC + c);
                tin[nt].v[v] = in.x + g.x * x[nt].v[v];
                tin[nt].v[v + 1] = in.y + g.y * x[nt].v[v + 1];
                x[nt].v[v] = parallel ? in.x : tin[nt].v[v];
                x[nt].v[v + 1] = parallel ? in.y : tin[nt].v[v + 1];
            }
        }
        mlp_adaln(x, am);
#pragma unroll
        for (int kt = 0; kt < kChanT; kt++) {
            a[kt] = acc_to_a(x[kt]);
        }
    }

    // 3. per half-unit: [gate | proj (| a_to_b)] = x·W1ᵀ, c = silu(gate)·proj(·a_to_b), y += c·W2ᵀ
    Acc y[kChanT];
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
        acc_zero(y[nt]);
    }
#pragma unroll 1
    for (int i = kWoPieces; i < kMlpStages; i++) {
        const bf16* st = mlp_ring(sm, i, ring, slot, wo, w1, w2, parts);
        if (!active) {
            continue;
        }
        Acc g[1], p[1];
        acc_zero(g[0]), acc_zero(p[0]);
        warp_gemm_lds(g, a, st, kW1Ld);
        warp_gemm_lds(p, a, st + kHalfUnit * kW1Ld, kW1Ld);
#pragma unroll
        for (int v = 0; v < kAccVals; v++) {
            p[0].v[v] *= silu(g[0].v[v]);
        }
        if (parts == kMaxParts) {
            acc_zero(g[0]);
            warp_gemm_lds(g, a, st + 2 * kHalfUnit * kW1Ld, kW1Ld);
#pragma unroll
            for (int v = 0; v < kAccVals; v++) {
                p[0].v[v] *= g[0].v[v];
            }
        }
        const FragA c[1] = {acc_to_a(p[0])};
        warp_gemm_lds(y, c, st + parts * kHalfUnit * kW1Ld, kW2Ld);
    }
    __syncthreads(); // the next item refills the ring from slot 0
    if (!active) {
        return;
    }
    // 4. qa = tin + QMASK·am7 ⊙ y
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int v = 0; v < kAccVals; v += 2) {
            const int c = nt * kMma + acc_col(v), rs = row_slot(v);
            const float2 g = bf_pair(am[rs] + AM_T_GATE * kAtomC + c);
            *reinterpret_cast<float2*>(qa + size_t(r0 + acc_row(v)) * kAtomC + c) =
                make_float2(tin[nt].v[v] + qmask[rs] * g.x * y[nt].v[v],
                            tin[nt].v[v + 1] + qmask[rs] * g.y * y[nt].v[v + 1]);
        }
    }
}

__device__ void atom_mlp(Deal& d, const AtomW& w, int b) {
    const int rows = c_kp.S * c_kp.A, items = cdiv(rows, kMlpRows);
    for (int it = d.first(); it < items; it += gridDim.x) {
        atom_mlp_item(w, b, it * kMlpRows, rows);
    }
    d.advance(items);
}

} // namespace mk
