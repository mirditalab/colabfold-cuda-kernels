// The fused atom row passes, chained in registers: adaLN with the q/k/v/g projections and q/k
// norms, and the MLP (out-projection residual, adaLN, GLU transition). One warp owns 16 atom rows;
// the CTA's warps share each block's weights through a shared-memory ring.
#pragma once

#include "diffusion/rows.cuh"

namespace mk {

constexpr int kAtomHidden = 2 * kAtomC;      // transition width
constexpr int kChanT = kAtomC / kMma;        // k (and n) tiles over the 128 channels
constexpr int kAtomRowsItem = kWarps * kMma; // rows per CTA item

// Weight pieces: 32 rows of a [rows][128] matrix (WQG, WKV, WO), or an MLP half-unit of 16 hidden
// channels (each GLU part's W1 rows and the matching W2 columns)
constexpr int kPieceRows = 2 * kMma;
constexpr int kPieceT = kPieceRows / kMma;
constexpr int kHalfUnit = kMma;
constexpr int kW1Ld = kAtomC + kLdsPad; // smem rows: conflict-free ldmatrix
constexpr int kW2Ld = kHalfUnit + kLdsPad;
constexpr int kMaxParts = 3; // gate, proj, a_to_b
constexpr int kPieceElems = kPieceRows * kW1Ld;
__host__ __device__ constexpr int half_unit_elems(int parts) {
    return parts * kHalfUnit * kW1Ld + kAtomC * kW2Ld;
}
// ring stages per half-unit: with one slot (synchronous copies) its W1 rows and W2 columns are two
// stages, so the slot holds the larger, not both
constexpr int kUnitStages = kCopyStages > 1 ? 1 : 2;
__host__ __device__ constexpr int unit_stage_elems(int parts) {
    return kUnitStages == 1                             ? half_unit_elems(parts)
           : parts * kHalfUnit * kW1Ld > kAtomC * kW2Ld ? parts * kHalfUnit * kW1Ld
                                                        : kAtomC * kW2Ld;
}
// the ring: three slots of two parts (two of three, five pieces) where copies overlap work; one
// slot of the model's own where they are synchronous
__host__ __device__ constexpr int ring_elems(int parts) {
    return kCopyStages > 1 ? 3 * half_unit_elems(2) : unit_stage_elems(parts);
}
static_assert(kCopyStages * unit_stage_elems(kMaxParts) <= ring_elems(kMaxParts) &&
                  kPieceElems <= ring_elems(2),
              "the ring holds kCopyStages stages");
// shared memory of the atom row passes: the ring, then the warps' acc_to_a tiles
__host__ __device__ constexpr size_t atom_rows_smem(int parts) {
    return (ring_elems(parts) + kWarps * kAccTile) * sizeof(bf16);
}
__device__ __forceinline__ int atom_parts() {
    return c_kp.flags & ATOB ? kMaxParts : 2;
}
__device__ __forceinline__ bf16* atom_rows_tile() {
    return reinterpret_cast<bf16*>(smem_base()) + ring_elems(atom_parts()) + warp_id() * kAccTile;
}

__device__ __forceinline__ void copy_piece(bf16* st, const bf16* w) {
    constexpr int kRowCopies = kAtomC / 8;
    for (int k = threadIdx.x; k < kPieceRows * kRowCopies; k += kThreads) {
        const int r = k / kRowCopies, c = k % kRowCopies * 8;
        async_copy16(st + r * kW1Ld + c, w + r * kAtomC + c, true);
    }
}

__device__ __forceinline__ void async_wait_upto(int pending) {
    switch (pending) {
    case 0:
        async_wait<0>();
        break;
    case 1:
        async_wait<1>();
        break;
    case 2:
        async_wait<2>();
        break;
    default:
        async_wait<3>();
    }
}

// A ring of `ring` slots of `slot` elements over a Stager's stages; one barrier per stage
template <class Stager> struct Ring {
    const Stager& stager;
    bf16* sm;
    int slot, ring;
    __device__ void start() const {
        for (int i = 0; i < ring - 1; i++) {
            stager.copy(sm + i * slot, i);
            async_commit();
        }
    }
    // stage i has landed, and every warp is done with the slot that stage i + ring - 1 refills
    __device__ const bf16* at(int i) const {
        async_wait_upto(ring - 2);
        __syncthreads();
        if (i + ring - 1 < Stager::kStages) {
            stager.copy(sm + (i + ring - 1) % ring * slot, i + ring - 1);
        }
        async_commit();
        if (ring == 1) { // the one slot was refilled just now, synchronously
            __syncthreads();
        }
        return sm + i % ring * slot;
    }
};

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

__device__ __forceinline__ float as_f32(bf16 x) {
    return from_bf(x);
}
__device__ __forceinline__ float as_f32(float x) {
    return x;
}

// the lane's row pointers, each moved by off
template <class T>
__device__ __forceinline__ void rows_at(const T* (&out)[kRowSlots],
                                        const T* const (&rows)[kRowSlots], int off) {
#pragma unroll
    for (int j = 0; j < kRowSlots; j++) {
        out[j] = rows[j] + off;
    }
}
template <class T>
__device__ __forceinline__ void rows_at(const T* (&out)[kRowSlots], const T* all, int off) {
#pragma unroll
    for (int j = 0; j < kRowSlots; j++) {
        out[j] = all + off;
    }
}

// ---- rows held as accumulators: lane values at (r0 + acc_row(i), 16 nt + acc_col(i))

__device__ __forceinline__ void acc_zero_all(Acc (&x)[kChanT]) {
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
        acc_zero(x[nt]);
    }
}

// x = rows of a [rows][kAtomC] f32 matrix
__device__ __forceinline__ void acc_load(Acc (&x)[kChanT], const float* p) {
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int i = 0; i < kAccVals; i += kAdj) {
            const Adj<float> v = adj(p + acc_row(i) * kAtomC + nt * kMma + acc_col(i));
#pragma unroll
            for (int j = 0; j < kAdj; j++) {
                x[nt].v[i + j] = v.v[j];
            }
        }
    }
}

// bf16 into columns col0.. of rows of an [rows][ld] matrix
__device__ __forceinline__ void acc_store_bf(const Acc (&x)[kChanT], bf16* p, int ld, int col0) {
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int i = 0; i < kAccVals; i += kAdj) {
            Adj<bf16> v;
#pragma unroll
            for (int j = 0; j < kAdj; j++) {
                v.v[j] = bf16(to_bf(x[nt].v[i + j]));
            }
            adj(p + acc_row(i) * ld + col0 + nt * kMma + acc_col(i)) = v;
        }
    }
}

// two-pass LayerNorm (or RMSNorm) of each row, in place
__device__ __forceinline__ void acc_norm(Acc (&x)[kChanT], bool rms, float eps) {
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
            x[nt].v[i] -= mean[row_slot(i)];
            var[row_slot(i)] += x[nt].v[i] * x[nt].v[i];
        }
    }
    row_reduce_sum(var);
    float rstd[kRowSlots];
#pragma unroll
    for (int j = 0; j < kRowSlots; j++) {
        rstd[j] = 1.f / sqrtf(var[j] / kAtomC + eps);
    }
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int i = 0; i < kAccVals; i++) {
            x[nt].v[i] *= rstd[row_slot(i)];
        }
    }
}

// y = x·scale + offset, per row (atom adaLN slots) or per column (q/k norm affines)
template <class T>
__device__ __forceinline__ void acc_affine(Acc (&y)[kChanT], const Acc (&x)[kChanT],
                                           const T* const (&scale)[kRowSlots],
                                           const T* const (&offset)[kRowSlots]) {
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int i = 0; i < kAccVals; i += kAdj) {
            const int c = nt * kMma + acc_col(i), rs = row_slot(i);
            const Adj<T> sc = adj(scale[rs] + c), of = adj(offset[rs] + c);
#pragma unroll
            for (int j = 0; j < kAdj; j++) {
                y[nt].v[i + j] = x[nt].v[i + j] * as_f32(sc.v[j]) + as_f32(of.v[j]);
            }
        }
    }
}

__device__ __forceinline__ void acc_to_a_all(FragA (&a)[kChanT], const Acc (&x)[kChanT],
                                             bf16* tile) {
#pragma unroll
    for (int kt = 0; kt < kChanT; kt++) {
        a[kt] = acc_to_a(x[kt], tile);
    }
}

// ESM: each 32-channel head gets an RMSNorm and the tiled rotary of its row's atom; channels d and
// d ± 16 sit in tiles 2h and 2h + 1 at the same lane value
__device__ __forceinline__ void acc_head_rope(Acc (&x)[kChanT],
                                              const float* const (&rope)[kRowSlots]) {
    constexpr int kHalf = kAtomHeadD / 2;
#pragma unroll
    for (int h = 0; h < kAtomHeads; h++) {
        Acc& lo = x[2 * h];
        Acc& hi = x[2 * h + 1];
        float ss[kRowSlots] = {};
#pragma unroll
        for (int i = 0; i < kAccVals; i++) {
            ss[row_slot(i)] += lo.v[i] * lo.v[i] + hi.v[i] * hi.v[i];
        }
        row_reduce_sum(ss);
#pragma unroll
        for (int i = 0; i < kAccVals; i++) {
            const int rs = row_slot(i);
            const float inv = 1.f / sqrtf(ss[rs] / kAtomHeadD + FLT_EPSILON);
            const float c = rope[rs][acc_col(i)], s = rope[rs][kHalf + acc_col(i)];
            const float a = lo.v[i] * inv, b = hi.v[i] * inv;
            lo.v[i] = a * c - b * s;
            hi.v[i] = b * c + a * s;
        }
    }
}

// ---- adaLN, projections and q/k norms

struct ProjStager {
    static constexpr int kStages = 2 * 2 * kAtomC / kPieceRows; // [q|g] rows, then [k|v]
    const bf16 *wqg, *wkv;
    __device__ void copy(bf16* st, int i) const {
        constexpr int kHalf = kStages / 2;
        copy_piece(st, (i < kHalf ? wqg : wkv) + (i % kHalf) * kPieceRows * kAtomC);
    }
};

// 128 output columns of a projection: four pieces of the ring from stage i0
__device__ __forceinline__ void proj_columns(Acc (&o)[kChanT], const FragA (&a)[kChanT],
                                             const Ring<ProjStager>& ring, int i0) {
    acc_zero_all(o);
#pragma unroll
    for (int p = 0; p < kChanT / kPieceT; p++) {
        Acc t[kPieceT];
#pragma unroll
        for (int j = 0; j < kPieceT; j++) {
            t[j] = o[p * kPieceT + j];
        }
        warp_gemm_lds(t, a, ring.at(i0 + p), kW1Ld);
#pragma unroll
        for (int j = 0; j < kPieceT; j++) {
            o[p * kPieceT + j] = t[j];
        }
    }
}

// QKNORM's LayerNorm or ESM's per-head RMSNorm and rotary of a q (or k) block
__device__ __forceinline__ void qk_norm(Acc (&o)[kChanT], const float* qkn, int scale, int offset,
                                        const float* const (&rope)[kRowSlots]) {
    if (c_kp.flags & QKNORM) {
        acc_norm(o, false, kLnEps);
        const float *s[kRowSlots], *f[kRowSlots];
        rows_at(s, qkn, scale * kAtomC), rows_at(f, qkn, offset * kAtomC);
        acc_affine(o, o, s, f);
    } else if (c_kp.flags & ESM) {
        acc_head_rope(o, rope);
    }
}

// one CTA item: rows row0.. (16 per warp) to AQG = [q|g] and AKV = [k|v]
__device__ __noinline__ void atom_proj_item(const AtomW& w, int b, int row0, int rows) {
    const int A = c_kp.A;
    // a warp past the end redoes the last rows: the same values, and it keeps to the barriers
    const int r0 = min(row0 + warp_id() * kMma, rows - kMma);
    const ProjStager stager{w.wqg + size_t(b) * 2 * kAtomC * kAtomC,
                            w.wkv + size_t(b) * 2 * kAtomC * kAtomC};
    const Ring<ProjStager> ring{stager, reinterpret_cast<bf16*>(smem_base()), kPieceElems,
                                ring_elems(atom_parts()) / kPieceElems};
    bf16* tile = atom_rows_tile();
    ring.start();
    const float* am[kRowSlots];
    const float* rope[kRowSlots];
    for (int i = 0; i < kAccVals; i++) {
        const int a = (r0 + acc_row(i)) % A;
        am[row_slot(i)] = atom_mods(w, b, a);
        rope[row_slot(i)] = c_kp.geo.rope + size_t(a) * kAtomHeadD;
    }
    const bool esm = c_kp.flags & ESM;

    // xq = norm(qa)·am0 + am1; xk = norm(qa, or bf16(xq) with CHAIN_K)·am2 + am3
    FragA aq[kChanT], ak[kChanT];
    {
        Acc x[kChanT], y[kChanT];
        acc_load(x, c_kp.buf.qa + size_t(r0) * kAtomC);
        acc_norm(x, esm, c_kp.atom_eps);
        const float *qs[kRowSlots], *qb[kRowSlots];
        rows_at(qs, am, AM_Q_SCALE * kAtomC), rows_at(qb, am, AM_Q_BIAS * kAtomC);
        acc_affine(y, x, qs, qb);
        acc_to_a_all(aq, y, tile);
        if (c_kp.flags & CHAIN_K) {
#pragma unroll
            for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
                for (int i = 0; i < kAccVals; i++) {
                    x[nt].v[i] = from_bf(to_bf(y[nt].v[i])); // bf16(xq), as aq holds it
                }
            }
            acc_norm(x, esm, c_kp.atom_eps);
        }
        const float *ks[kRowSlots], *kb[kRowSlots];
        rows_at(ks, am, AM_K_SCALE * kAtomC), rows_at(kb, am, AM_K_BIAS * kAtomC);
        acc_affine(y, x, ks, kb);
        acc_to_a_all(ak, y, tile);
    }

    constexpr int kQuarter = ProjStager::kStages / 4;
    bf16* aqg = c_kp.buf.aqg + size_t(r0) * 2 * kAtomC;
    bf16* akv = c_kp.buf.akv + size_t(r0) * 2 * kAtomC;
    const float* qkn = w.qkn + b * kQkNormRows * kAtomC;
    const float* bq = w.bq + b * kAtomC;
    Acc o[kChanT];
    proj_columns(o, aq, ring, 0); // q
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int i = 0; i < kAccVals; i++) {
            o[nt].v[i] += bq[nt * kMma + acc_col(i)];
        }
    }
    qk_norm(o, qkn, QKN_Q_SCALE, QKN_Q_OFFSET, rope);
    acc_store_bf(o, aqg, 2 * kAtomC, 0);
    proj_columns(o, aq, ring, kQuarter); // g
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int i = 0; i < kAccVals; i++) {
            o[nt].v[i] = sigmoid(o[nt].v[i]);
        }
    }
    acc_store_bf(o, aqg, 2 * kAtomC, kAtomC);
    proj_columns(o, ak, ring, 2 * kQuarter); // k
    qk_norm(o, qkn, QKN_K_SCALE, QKN_K_OFFSET, rope);
    acc_store_bf(o, akv, 2 * kAtomC, 0);
    proj_columns(o, ak, ring, 3 * kQuarter); // v
    acc_store_bf(o, akv, 2 * kAtomC, kAtomC);
    __syncthreads(); // the next item refills the ring from slot 0
}

__device__ void atom_proj(Deal& d, const AtomW& w, int b) {
    const int rows = c_kp.S * c_kp.A, items = cdiv(rows, kAtomRowsItem);
    for (int it = d.first(); it < items; it += gridDim.x) {
        atom_proj_item(w, b, it * kAtomRowsItem, rows);
    }
    d.advance(items);
}

// ---- the MLP

struct MlpStager {
    static constexpr int kWoPieces = kAtomC / kPieceRows;
    static constexpr int kStages = kWoPieces + kUnitStages * kAtomHidden / kHalfUnit;
    const bf16 *wo, *w1, *w2;
    int parts;
    __device__ void copy(bf16* st, int i) const {
        if (i < kWoPieces) {
            copy_piece(st, wo + i * kPieceRows * kAtomC);
            return;
        }
        constexpr int kRowCopies = kAtomC / 8;
        const int j = i - kWoPieces, u = j / kUnitStages / 2, h = j / kUnitStages % 2;
        const bool w1_part = kUnitStages == 1 || j % 2 == 0, w2_part = kUnitStages == 1 || j % 2;
        for (int k = threadIdx.x; w1_part && k < parts * kHalfUnit * kRowCopies; k += kThreads) {
            const int r = k / kRowCopies, c = k % kRowCopies * 8;
            const int src =
                u * parts * kGluHalf + r / kHalfUnit * kGluHalf + h * kHalfUnit + r % kHalfUnit;
            async_copy16(st + r * kW1Ld + c, w1 + src * kAtomC + c, true);
        }
        bf16* s2 = kUnitStages == 1 ? st + parts * kHalfUnit * kW1Ld : st;
        constexpr int kW2Copies = kHalfUnit / 8;
        for (int k = threadIdx.x; w2_part && k < kAtomC * kW2Copies; k += kThreads) {
            const int r = k / kW2Copies, c = k % kW2Copies * 8;
            async_copy16(s2 + r * kW2Ld + c,
                         w2 + r * kAtomHidden + u * kGluHalf + h * kHalfUnit + c, true);
        }
    }
};

// what the last block of a stack also writes from the new rows, saving a phase that rereads them
enum MlpTail {
    TAIL_NONE,
    TAIL_SKIP,   // the encoder's: skip = qa·QMASK in f32 and as bf16 for the aggregation
    TAIL_UPDATE, // the decoder's: upd = LN(qa·QMASK) @ WUPDᵀ + UPD_B
};

// one CTA item: rows row0.. (16 per warp) through the block's attention output and transition
__device__ __noinline__ void atom_mlp_item(const AtomW& w, int b, int row0, int rows, int tail) {
    const int A = c_kp.A, parts = atom_parts();
    const MlpStager stager{w.wo + size_t(b) * kAtomC * kAtomC, w.w1 + size_t(b) * c_kp.aw1 * kAtomC,
                           w.w2 + size_t(b) * kAtomC * kAtomHidden, parts};
    const int slot = unit_stage_elems(parts);
    const Ring<MlpStager> ring{stager, reinterpret_cast<bf16*>(smem_base()), slot,
                               ring_elems(parts) / slot};
    bf16* tile = atom_rows_tile();
    const int r0 = row0 + warp_id() * kMma;
    const bool active = r0 < rows; // every warp keeps to the barriers
    float* qa = c_kp.buf.qa + size_t(active ? r0 : 0) * kAtomC;
    const float* am[kRowSlots];
    float qmask[kRowSlots];
    for (int i = 0; i < kAccVals; i++) {
        const int a = ((active ? r0 : 0) + acc_row(i)) % A;
        am[row_slot(i)] = atom_mods(w, b, a);
        qmask[row_slot(i)] = c_kp.geo.qmask[a];
    }
    ring.start();

    // 1. T = AO·WOᵀ, 32 output channels per stage
    FragA a[kChanT];    // AO, then the transition's input
    FragA a_lo[kChanT]; // the transition input's low part: AF3 runs the atom transformers in f32
    Acc x[kChanT];
#pragma unroll
    for (int kt = 0; kt < kChanT; kt++) {
        if (active) {
            load_a(a[kt], c_kp.buf.ao + size_t(r0) * kAtomC + kt * kMma, kAtomC);
        }
    }
    acc_zero_all(x);
#pragma unroll
    for (int i = 0; i < MlpStager::kWoPieces; i++) {
        const bf16* st = ring.at(i);
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
            for (int v = 0; v < kAccVals; v += kAdj) {
                const int c = nt * kMma + acc_col(v);
                const Adj<float> in = adj(qa + acc_row(v) * kAtomC + c);
                const Adj<float> g = adj(am[row_slot(v)] + AM_ATTN_GATE * kAtomC + c);
#pragma unroll
                for (int j = 0; j < kAdj; j++) {
                    tin[nt].v[v + j] = in.v[j] + g.v[j] * x[nt].v[v + j];
                    x[nt].v[v + j] = parallel ? in.v[j] : tin[nt].v[v + j];
                }
            }
        }
        acc_norm(x, c_kp.flags & ESM, c_kp.atom_eps);
        const float *ts[kRowSlots], *tb[kRowSlots];
        rows_at(ts, am, AM_T_SCALE * kAtomC), rows_at(tb, am, AM_T_BIAS * kAtomC);
        acc_affine(x, x, ts, tb);
#pragma unroll
        for (int kt = 0; kt < kChanT; kt++) {
            acc_to_a_split(x[kt], a[kt], a_lo[kt], tile);
        }
    }

    // 3. per half-unit: [gate | proj (| a_to_b)] = x·W1ᵀ, c = silu(gate)·proj(·a_to_b), y += c·W2ᵀ
    Acc y[kChanT];
    acc_zero_all(y);
#pragma unroll 1
    for (int i = MlpStager::kWoPieces; i < MlpStager::kStages; i += kUnitStages) {
        const bf16* st = ring.at(i);
        if (!active) {
            if (kUnitStages > 1) {
                ring.at(i + 1); // every warp keeps to the ring's barriers
            }
            continue;
        }
        Acc g[1], p[1];
        acc_zero(g[0]), acc_zero(p[0]);
        warp_gemm_lds(g, a, st, kW1Ld);
        warp_gemm_lds(g, a_lo, st, kW1Ld);
        warp_gemm_lds(p, a, st + kHalfUnit * kW1Ld, kW1Ld);
        warp_gemm_lds(p, a_lo, st + kHalfUnit * kW1Ld, kW1Ld);
#pragma unroll
        for (int v = 0; v < kAccVals; v++) {
            p[0].v[v] *= silu(g[0].v[v]);
        }
        if (parts == kMaxParts) {
            acc_zero(g[0]);
            warp_gemm_lds(g, a, st + 2 * kHalfUnit * kW1Ld, kW1Ld);
            warp_gemm_lds(g, a_lo, st + 2 * kHalfUnit * kW1Ld, kW1Ld);
#pragma unroll
            for (int v = 0; v < kAccVals; v++) {
                p[0].v[v] *= g[0].v[v];
            }
        }
        const FragA c[1] = {acc_to_a(p[0], tile)};
        const bf16* s2 = kUnitStages > 1 ? ring.at(i + 1) : st + parts * kHalfUnit * kW1Ld;
        warp_gemm_lds(y, c, s2, kW2Ld);
    }
    __syncthreads(); // the next item refills the ring from slot 0
    if (!active) {
        return;
    }
    // 4. qa = tin + QMASK·am7 ⊙ y
#pragma unroll
    for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
        for (int v = 0; v < kAccVals; v += kAdj) {
            const int c = nt * kMma + acc_col(v), rs = row_slot(v);
            const Adj<float> g = adj(am[rs] + AM_T_GATE * kAtomC + c);
            Adj<float> out;
#pragma unroll
            for (int j = 0; j < kAdj; j++) {
                tin[nt].v[v + j] += qmask[rs] * g.v[j] * y[nt].v[v + j];
                out.v[j] = tin[nt].v[v + j];
                tin[nt].v[v + j] *= qmask[rs];
            }
            adj(qa + acc_row(v) * kAtomC + c) = out;
        }
    }
    if (tail == TAIL_SKIP) {
        const size_t off = size_t(r0) * kAtomC;
#pragma unroll
        for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
            for (int v = 0; v < kAccVals; v += kAdj) {
                const int at = acc_row(v) * kAtomC + nt * kMma + acc_col(v);
                Adj<float> out;
#pragma unroll
                for (int j = 0; j < kAdj; j++) {
                    out.v[j] = tin[nt].v[v + j];
                }
                adj(c_kp.buf.skip + off + at) = out;
            }
        }
        acc_store_bf(tin, c_kp.buf.abf + off, kAtomC, 0);
    } else if (tail == TAIL_UPDATE) {
        acc_norm(tin, false, kLnEps);
        for (int j = 0; j < kXyz; j++) {
            float u[kRowSlots] = {};
#pragma unroll
            for (int nt = 0; nt < kChanT; nt++) {
#pragma unroll
                for (int v = 0; v < kAccVals; v++) {
                    u[row_slot(v)] +=
                        tin[nt].v[v] * c_kp.geo.wupd[j * kAtomC + nt * kMma + acc_col(v)];
                }
            }
            row_reduce_sum(u);
            // every lane holding a row writes it: the values agree, and no layout is assumed
#pragma unroll
            for (int v = 0; v < kAccVals; v++) {
                c_kp.buf.upd[size_t(r0 + acc_row(v)) * kXyz + j] =
                    u[row_slot(v)] + c_kp.geo.upd_b[j];
            }
        }
    }
}

__device__ void atom_mlp(Deal& d, const AtomW& w, int b, MlpTail tail = TAIL_NONE) {
    const int rows = c_kp.S * c_kp.A, items = cdiv(rows, kAtomRowsItem);
    for (int it = d.first(); it < items; it += gridDim.x) {
        atom_mlp_item(w, b, it * kAtomRowsItem, rows, tail);
    }
    d.advance(items);
}

} // namespace mk
