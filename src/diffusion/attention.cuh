// Token attention (16 heads x 48, pair-logit bias) and atom attention (4 heads x 32 over gathered
// keys), FlashAttention-2 style with an online base-2 softmax.
#pragma once

#include "diffusion/rows.cuh"

namespace mk {

// Online softmax state for the rows of R query tiles
template <int R> struct Softmax {
    float m[R][kRowSlots], l[R][kRowSlots];
    __device__ void init() {
#pragma unroll
        for (int r = 0; r < R; r++) {
#pragma unroll
            for (int j = 0; j < kRowSlots; j++) {
                m[r][j] = -INFINITY, l[r][j] = 0.f;
            }
        }
    }
    // s: one chunk of base-2 logits for query tile r, KT key tiles; rescales o and turns s into P
    template <int KT, int DT> __device__ void step(int r, Acc (&s)[KT], Acc (&o)[DT]) {
        float mx[kRowSlots];
#pragma unroll
        for (int j = 0; j < kRowSlots; j++) {
            mx[j] = m[r][j];
        }
#pragma unroll
        for (int kt = 0; kt < KT; kt++) {
#pragma unroll
            for (int i = 0; i < kAccVals; i++) {
                mx[row_slot(i)] = fmaxf(mx[row_slot(i)], s[kt].v[i]);
            }
        }
        row_reduce_max(mx);
        float corr[kRowSlots];
#pragma unroll
        for (int j = 0; j < kRowSlots; j++) {
            corr[j] = ex2(m[r][j] - mx[j]);
            m[r][j] = mx[j];
            l[r][j] *= corr[j];
        }
#pragma unroll
        for (int dt = 0; dt < DT; dt++) {
#pragma unroll
            for (int i = 0; i < kAccVals; i++) {
                o[dt].v[i] *= corr[row_slot(i)];
            }
        }
#pragma unroll
        for (int kt = 0; kt < KT; kt++) {
#pragma unroll
            for (int i = 0; i < kAccVals; i++) {
                s[kt].v[i] = ex2(s[kt].v[i] - mx[row_slot(i)]);
                l[r][row_slot(i)] += s[kt].v[i];
            }
        }
    }
    // the row sums were kept per lane: reduce them once, at the end
    __device__ void finish() {
#pragma unroll
        for (int r = 0; r < R; r++) {
            row_reduce_sum(l[r]);
        }
    }
};

// out[row][col] = o / l · gate for one accumulator tile, two bf16 columns per store
__device__ __forceinline__ void store_gated(const Acc& o, const float (&l)[kRowSlots], bf16* out,
                                            int ldo, const bf16* gate, int ldg, int rows_left) {
#pragma unroll
    for (int i = 0; i < kAccVals; i += kAdj) {
        const int r = acc_row(i), c = acc_col(i);
        if (r < rows_left) {
            const Adj<bf16> g = adj(gate + size_t(r) * ldg + c);
            const float inv = 1.f / l[row_slot(i)];
            Adj<bf16> v;
#pragma unroll
            for (int j = 0; j < kAdj; j++) {
                v.v[j] = bf16(to_bf(o.v[i + j] * inv * from_bf(g.v[j])));
            }
            adj(out + size_t(r) * ldo + c) = v;
        }
    }
}

// a lane's 8 values of a swizzled bias tile, one 16-byte load
__device__ __forceinline__ void bias_tile(const bf16* tile, float (&v)[kAccVals]) {
    const uint4 u = *reinterpret_cast<const uint4*>(tile + lane_id() * kAccVals);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        v[2 * j] = from_bf(w[j] & 0xFFFFu), v[2 * j + 1] = from_bf(w[j] >> 16);
    }
}

// ---- token
constexpr int kTokLd = kHeadD + kLdsPad; // smem row: conflict-free ldmatrix
constexpr int kTokDT = kHeadD / kMma;    // head-dim tiles

// MI query tiles per warp, KT key tiles per staged chunk, STAGES chunks in flight
template <int MI, int KT, int STAGES> struct TokAttn {
    static constexpr int kMI = MI, kKT = KT, kStages = STAGES;
    static constexpr int kKeys = KT * kMma;
    static constexpr int kKV = kKeys * kTokLd;
    static constexpr int kBias = MI * KT * kTileVals; // one warp's pair-logit tiles of a chunk
    static constexpr int kStage = 2 * kKV + kWarps * kBias;
    static constexpr int kQueries = kWarps * MI * kMma; // per item
    static constexpr int kTiles = STAGES * kStage;      // the warps' acc_to_a tiles follow
    static constexpr size_t kSmem = (kTiles + kWarps * kAccTile) * sizeof(bf16);
    static_assert(kKeyPad % kKeys == 0, "chunks tile the padded keys");
};

// keys k0.. into one stage: K and V of the head (zero past the real tokens), and each warp's
// pair-logit tiles from row tile qt on (zero past the table)
template <class T>
__device__ __forceinline__ void stage_token_chunk(bf16* st, const bf16* rows, const bf16* plg,
                                                  int k0, int qt) {
    constexpr int kPieces = kHeadD / 8;
    for (int i = threadIdx.x; i < T::kKeys * kPieces; i += kThreads) {
        const int j = i / kPieces, p = i % kPieces, key = k0 + j;
        const bool ok = key < c_kp.N;
        const bf16* src = rows + size_t(ok ? key : 0) * kQKVG + p * 8;
        async_copy16(st + j * kTokLd + p * 8, src + kTokC, ok);
        async_copy16(st + T::kKV + j * kTokLd + p * 8, src + 2 * kTokC, ok);
    }
    const int tiles = c_kp.np / kMma;
    constexpr int kRow = T::kKT * kTileVals; // a tile row's KT tiles are contiguous
    bf16* bs = st + 2 * T::kKV + warp_id() * T::kBias;
    for (int i = lane_id() * 8; i < T::kBias; i += kWarp * 8) {
        const int mi = i / kRow;
        const bool ok = qt + mi < tiles;
        const bf16* src = plg + (size_t(ok ? qt + mi : 0) * tiles + k0 / kMma) * kTileVals;
        async_copy16(bs + i, src + i % kRow, ok);
    }
}

// item (head, query block, sample): each warp owns 16·MI queries
template <class T>
__device__ __noinline__ void token_attention_item(int slot, int h, int qb, int s) {
    constexpr int MI = T::kMI, KT = T::kKT, STAGES = T::kStages;
    bf16* sm = reinterpret_cast<bf16*>(smem_base());
    const int N = c_kp.N, np = c_kp.np, tiles = np / kMma;
    const int q0 = qb * T::kQueries + warp_id() * MI * kMma;
    const bf16* rows = c_kp.buf.qkvg + size_t(s) * N * kQKVG + h * kHeadD;
    const bf16* plg = c_kp.buf.plg + (size_t(slot) * kHeads + h) * tiles * tiles * kTileVals;
    const float scale = kLog2e / sqrtf(float(kHeadD));

    FragA fq[MI][kTokDT];
    Acc o[MI][kTokDT];
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
#pragma unroll
        for (int kd = 0; kd < kTokDT; kd++) {
            load_a(fq[mi][kd], rows + size_t(q0 + mi * kMma) * kQKVG + kd * kMma, kQKVG);
            acc_zero(o[mi][kd]);
        }
    }
    Softmax<MI> sm_state;
    sm_state.init();

    const int chunks = np / T::kKeys;
    for (int c = 0; c < STAGES - 1; c++) {
        stage_token_chunk<T>(sm + c * T::kStage, rows, plg, c * T::kKeys, q0 / kMma);
        async_commit();
    }
#pragma unroll 1
    for (int c = 0; c < chunks; c++) {
        // one barrier per chunk: chunk c has landed and every warp is done with the slot refilled
        async_wait<STAGES - 2>();
        __syncthreads();
        if (c + STAGES - 1 < chunks) {
            stage_token_chunk<T>(sm + (c + STAGES - 1) % STAGES * T::kStage, rows, plg,
                                 (c + STAGES - 1) * T::kKeys, q0 / kMma);
        }
        async_commit();
        if (STAGES == 1) { // the one stage was refilled just now, synchronously
            __syncthreads();
        }
        const bf16* st = sm + c % STAGES * T::kStage;
        const bf16* bs = st + 2 * T::kKV + warp_id() * T::kBias;
        Acc sc[MI][KT];
#pragma unroll
        for (int kt = 0; kt < KT; kt++) {
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                acc_zero(sc[mi][kt]);
            }
#pragma unroll
            for (int kd = 0; kd < kTokDT; kd++) {
                FragB fk;
                load_b_lds(fk, st + kt * kMma * kTokLd + kd * kMma, kTokLd);
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    mma(sc[mi][kt], fq[mi][kd], fk);
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
#pragma unroll
            for (int kt = 0; kt < KT; kt++) {
                float b[kAccVals];
                bias_tile(bs + (mi * KT + kt) * kTileVals, b);
#pragma unroll
                for (int i = 0; i < kAccVals; i++) {
                    sc[mi][kt].v[i] = sc[mi][kt].v[i] * scale + b[i];
                }
            }
            sm_state.step(mi, sc[mi], o[mi]);
        }
#pragma unroll
        for (int kt = 0; kt < KT; kt++) {
            FragA p[MI];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                p[mi] = acc_to_a(sc[mi][kt], sm + T::kTiles + warp_id() * kAccTile);
            }
#pragma unroll
            for (int dt = 0; dt < kTokDT; dt++) {
                FragB fv;
                load_b_lds_t(fv, st + T::kKV + kt * kMma * kTokLd + dt * kMma, kTokLd);
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    mma(o[mi][dt], p[mi], fv);
                }
            }
        }
    }
    __syncthreads(); // the next item restages from the first slot
    sm_state.finish();
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
        const int q = q0 + mi * kMma;
        const size_t row = size_t(s) * N + q;
#pragma unroll
        for (int dt = 0; dt < kTokDT; dt++) {
            const int col = h * kHeadD + dt * kMma;
            store_gated(o[mi][dt], sm_state.l[mi], c_kp.buf.to + row * kTokC + col, kTokC,
                        c_kp.buf.qkvg + row * kQKVG + 3 * kTokC + col, kQKVG, N - q);
        }
    }
}

template <class T> __device__ void token_attention_items(Deal& d, int b) {
    // samples fastest: they run side by side and share each pair-logit tile through L2
    const int S = c_kp.S, qblocks = cdiv(c_kp.N, T::kQueries);
    const int items = kHeads * qblocks * S;
    for (int it = d.first(); it < items; it += gridDim.x) {
        token_attention_item<T>(b % c_kp.plg, it / S % kHeads, it / (S * kHeads), it % S);
    }
    d.advance(items);
}

// one query tile per warp for small structures (more items to spread), three when there are rounds
// to spare: K/V staged once serve three tiles, which made the dimer's attention 18% faster
#if CCK_AMD
using TokAttnNarrow = TokAttn<1, 2, kCopyStages>; // shapes within 16 KB of shared memory
using TokAttnWide = TokAttn<2, 2, kCopyStages>;
#else
using TokAttnNarrow = TokAttn<1, 4, kCopyStages>;
using TokAttnWide = TokAttn<3, 2, kCopyStages>;
#endif
constexpr size_t kTokAttnSmem =
    TokAttnNarrow::kSmem > TokAttnWide::kSmem ? TokAttnNarrow::kSmem : TokAttnWide::kSmem;
constexpr float kWideItemCost = 2.5f; // a wide item's time in narrow items, measured on the dimer

// the wide shape when its whole rounds over the grid cost no more
__host__ __device__ inline bool token_attention_wide(int S, int N, int grid) {
    const int narrow = kHeads * S * cdiv(N, TokAttnNarrow::kQueries);
    const int wide = kHeads * S * cdiv(N, TokAttnWide::kQueries);
    return cdiv(wide, grid) * kWideItemCost <= cdiv(narrow, grid);
}

// TO = softmax(q kᵀ/√48 + L_b) v · g over the items (head, query block, sample)
__device__ void token_attention(Deal& d, int b) {
    if (c_kp.attn_wide) {
        token_attention_items<TokAttnWide>(d, b);
    } else {
        token_attention_items<TokAttnNarrow>(d, b);
    }
}

// ---- atom
constexpr int kAtomKeys = kSubset; // keys per gathered chunk
constexpr int kAtomLd = kAtomHeadD + kLdsPad;
constexpr int kAtomDT = kAtomHeadD / kMma;
constexpr int kAtomKeyT = kAtomKeys / kMma;
constexpr int kAtomQT = kSubset / kMma;
constexpr int kAtomStage = kAtomKeys * kAtomLd;
constexpr size_t kAtomAttnSmem = kWarps * kCopyStages * 2 * kAtomStage * sizeof(bf16);

// the warp gathers its head's keys and values by Q2K into a private stage (zero for -1)
__device__ __forceinline__ void gather_atom_kv(bf16* ks, bf16* vs, const bf16* akv,
                                               const int* q2k) {
    constexpr int kPieces = kAtomHeadD / 8;
    const int key = q2k[lane_id()];
    const bf16* src = akv + size_t(key < 0 ? 0 : key) * 2 * kAtomC;
#pragma unroll
    for (int p = 0; p < kPieces; p++) {
        async_copy16(ks + lane_id() * kAtomLd + p * 8, src + p * 8, key >= 0);
        async_copy16(vs + lane_id() * kAtomLd + p * 8, src + kAtomC + p * 8, key >= 0);
    }
}

// item (subset, sample); warp h is head h: 32 queries x NK keys in chunks of 32
template <int NK>
__device__ __noinline__ void atom_attention_item(const AtomW& w, int b, int u, int s) {
    constexpr int kChunks = NK / kAtomKeys;
    const float scale = 1.f / sqrtf(float(kAtomHeadD));
    const int h = warp_id(), A = c_kp.A;
    bf16* sm = reinterpret_cast<bf16*>(smem_base()) + h * kCopyStages * 2 * kAtomStage;
    const size_t row0 = size_t(s) * A + u * kSubset;
    const bf16* akv = c_kp.buf.akv + size_t(s) * A * 2 * kAtomC + h * kAtomHeadD;
    const int* q2k = c_kp.geo.q2k + u * NK;
    const float* kmask = c_kp.geo.kmask + u * NK;
    const bf16* apl =
        w.apl + ((size_t(b) * c_kp.nss + u) * kAtomHeads + h) * kAtomQT * (NK / kMma) * kTileVals;

    FragA fq[kAtomQT][kAtomDT];
    float qm[kAtomQT][kRowSlots]; // QMASK - 1 of the lane's query rows
#pragma unroll
    for (int qt = 0; qt < kAtomQT; qt++) {
#pragma unroll
        for (int kd = 0; kd < kAtomDT; kd++) {
            load_a(fq[qt][kd],
                   c_kp.buf.aqg + (row0 + qt * kMma) * 2 * kAtomC + h * kAtomHeadD + kd * kMma,
                   2 * kAtomC);
        }
#pragma unroll
        for (int i = 0; i < kAccVals; i++) {
            qm[qt][row_slot(i)] = c_kp.geo.qmask[u * kSubset + qt * kMma + acc_row(i)] - 1.f;
        }
    }
    Acc o[kAtomQT][kAtomDT];
#pragma unroll
    for (int qt = 0; qt < kAtomQT; qt++) {
#pragma unroll
        for (int dt = 0; dt < kAtomDT; dt++) {
            acc_zero(o[qt][dt]);
        }
    }
    Softmax<kAtomQT> sm_state;
    sm_state.init();

    gather_atom_kv(sm, sm + kAtomStage, akv, q2k);
    async_commit();
#pragma unroll 1
    for (int c = 0; c < kChunks; c++) {
        bf16* ks = sm + (c % kCopyStages) * 2 * kAtomStage;
        bf16* vs = ks + kAtomStage;
        bf16* kn = sm + ((c + 1) % kCopyStages) * 2 * kAtomStage;
        const bool next = c + 1 < kChunks;
        if (next && kCopyStages > 1) {
            gather_atom_kv(kn, kn + kAtomStage, akv, q2k + (c + 1) * kAtomKeys);
        }
        async_commit();
        async_wait<1>();
        __syncwarp();
        Acc sc[kAtomQT][kAtomKeyT];
#pragma unroll
        for (int kt = 0; kt < kAtomKeyT; kt++) {
#pragma unroll
            for (int qt = 0; qt < kAtomQT; qt++) {
                acc_zero(sc[qt][kt]);
            }
#pragma unroll
            for (int kd = 0; kd < kAtomDT; kd++) {
                FragB fk;
                load_b_lds(fk, ks + kt * kMma * kAtomLd + kd * kMma, kAtomLd);
#pragma unroll
                for (int qt = 0; qt < kAtomQT; qt++) {
                    mma(sc[qt][kt], fq[qt][kd], fk);
                }
            }
        }
#pragma unroll
        for (int qt = 0; qt < kAtomQT; qt++) {
#pragma unroll
            for (int kt = 0; kt < kAtomKeyT; kt++) {
                const int ktile = c * kAtomKeyT + kt;
                float b[kAccVals];
                bias_tile(apl + (qt * (NK / kMma) + ktile) * kTileVals, b);
#pragma unroll
                for (int i = 0; i < kAccVals; i++) {
                    // AF3's mask: only a padded query against a padded key is penalised
                    const float km = kmask[ktile * kMma + acc_col(i)] - 1.f;
                    const float x =
                        sc[qt][kt].v[i] * scale + kMaskBias * qm[qt][row_slot(i)] * km + b[i];
                    sc[qt][kt].v[i] = x * kLog2e;
                }
            }
            sm_state.step(qt, sc[qt], o[qt]);
        }
#pragma unroll
        for (int kt = 0; kt < kAtomKeyT; kt++) {
            FragA p[kAtomQT];
#pragma unroll
            for (int qt = 0; qt < kAtomQT; qt++) {
                p[qt] = acc_to_a(sc[qt][kt], ks); // K is read: its stage holds the tile
            }
#pragma unroll
            for (int dt = 0; dt < kAtomDT; dt++) {
                FragB fv;
                load_b_lds_t(fv, vs + kt * kMma * kAtomLd + dt * kMma, kAtomLd);
#pragma unroll
                for (int qt = 0; qt < kAtomQT; qt++) {
                    mma(o[qt][dt], p[qt], fv);
                }
            }
        }
        __syncwarp();
        if (next && kCopyStages == 1) { // one stage: refilled once read
            gather_atom_kv(kn, kn + kAtomStage, akv, q2k + (c + 1) * kAtomKeys);
        }
    }
    sm_state.finish();
#pragma unroll
    for (int qt = 0; qt < kAtomQT; qt++) {
        const size_t row = row0 + qt * kMma;
#pragma unroll
        for (int dt = 0; dt < kAtomDT; dt++) {
            const int col = h * kAtomHeadD + dt * kMma;
            store_gated(o[qt][dt], sm_state.l[qt], c_kp.buf.ao + row * kAtomC + col, kAtomC,
                        c_kp.buf.aqg + row * 2 * kAtomC + kAtomC + col, 2 * kAtomC, kMma);
        }
    }
}

// AO = softmax(q kᵀ/√32 + mask + APL) v · g over the items (subset, sample)
__device__ void atom_attention(Deal& d, const AtomW& w, int b) {
    const int items = c_kp.NS * c_kp.S;
    for (int it = d.first(); it < items; it += gridDim.x) {
        const int u = it % c_kp.NS, s = it / c_kp.NS;
        if (c_kp.nk == kSubsetKeysEsm) {
            atom_attention_item<kSubsetKeysEsm>(w, b, u, s);
        } else {
            atom_attention_item<kSubsetKeys>(w, b, u, s);
        }
    }
    d.advance(items);
}

} // namespace mk
