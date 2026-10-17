// GEMM tiles (portable and CuTe), their epilogues, the tile chooser and run_gemms.
#pragma once

#include "diffusion/frag.cuh"
#include "diffusion/params.cuh"

#if !CCK_AMD
#include <cute/tensor.hpp>
#endif

namespace mk {

// k per pipeline stage, rows staged per epilogue pass (a multiple of a token's 32 gathered rows)
// and floats of padding per staged row
#if CCK_AMD
// operands wait in registers; a tile fits 16 KB, which four CTAs per WGP leave; the slab swizzle
// (frag.cuh) swaps a row's two halves, so a slab is 16 wide
constexpr int kBK = 16, kEpiRows = 32, kStagePad = 0;
#else
constexpr int kBK = 32, kEpiRows = 64, kStagePad = 4;
#endif

// the reference's tile codes, which the harness passes to mk_gemm_test
enum TileCode : int { T64 = 64, T64x128 = 128, T128 = 256, T128x96 = 296, T96 = 396 };
struct TileInfo {
    int code, bm, bn;
    float speed;  // relative throughput on this vendor; 0 where it does not run
    bool gathers; // has the portable implementation, which can gather rows
};
constexpr int kNumTiles = 5;
__host__ __device__ constexpr TileInfo tile_info(int i) {
    switch (i) {
    case 0:
        return {T128, 128, 128, 1.0f, true};
    case 1:
        return {T128x96, 128, 96, CCK_AMD ? 0.f : 0.95f, false}; // CuTe's
    case 2:
        return {T96, 96, 96, CCK_AMD ? 0.f : 0.9f, false};
    case 3:
        return {T64x128, 64, 128, CCK_AMD ? 0.97f : 0.f, true};
    default:
        return {T64, 64, 64, CCK_AMD ? 0.6f : 0.55f, true};
    }
}
__host__ __device__ constexpr TileInfo tile_of(int code) {
    for (int i = 0; i < kNumTiles; i++) {
        if (tile_info(i).code == code) {
            return tile_info(i);
        }
    }
    return tile_info(kNumTiles - 1);
}

__host__ __device__ inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// the tile with the fewest whole rounds over the grid, weighted by speed; gathered rows run on the
// portable tiles, and a GLU tile must hold whole units
__host__ __device__ inline int choose_tile(int m, int n, int unit, bool gathered, int grid) {
    int best = T64;
    float best_cost = 3.4e38f;
    for (int i = 0; i < kNumTiles; i++) {
        const TileInfo t = tile_info(i);
        if (!t.speed || n % t.bn || (unit && t.bn % unit) || (gathered && !t.gathers)) {
            continue;
        }
        const float cost = cdiv(cdiv(m, t.bm) * (n / t.bn), grid) * float(t.bm * t.bn) / t.speed;
        if (cost < best_cost) {
            best_cost = cost, best = t.code;
        }
    }
    return best;
}

__host__ __device__ inline int gemm_tiles(const Gemm& g) {
    const TileInfo t = tile_of(g.tile);
    return cdiv(g.m, t.bm) * cdiv(g.n, t.bn);
}

// a first item past any phase's work: a stopped debug run deals nothing
constexpr int kIdle = 1 << 28;

// Work dealing: item i of a phase goes to CTA (start + i) mod grid, so consecutive work lists
// continue the round-robin instead of all starting on CTA 0
struct Deal {
    int start = 0;
    bool live = true;
    __device__ int first() const {
        return live ? (int(blockIdx.x) - start + int(gridDim.x)) % int(gridDim.x) : kIdle;
    }
    __device__ void advance(int items) {
        start = (start + items) % int(gridDim.x);
    }
};

// the only shared memory: a static __shared__ anywhere would shift it off CuTe's 128-byte swizzle
__device__ __forceinline__ char* smem_base() {
    extern __shared__ __align__(128) char smem[];
    return smem;
}

// fast forms: every sigmoid and silu result is rounded to bf16 (a few ulp of f32 do not reach it)
#if CCK_AMD
__device__ __forceinline__ float sigmoid(float x) {
    return __fdividef(1.f, 1.f + __expf(-x));
}
__device__ __forceinline__ float silu(float x) {
    return x * sigmoid(x);
}
#else
// one MUFU.TANH instead of an ex2 and a reciprocal: sigmoid(x) = (1 + tanh(x / 2)) / 2
__device__ __forceinline__ float tanh_approx(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float sigmoid(float x) {
    return fmaf(0.5f, tanh_approx(0.5f * x), 0.5f);
}
__device__ __forceinline__ float silu(float x) {
    const float h = 0.5f * x;
    return fmaf(h, tanh_approx(h), h);
}
#endif
__device__ __forceinline__ float4 sigmoid4(float4 v) {
    return make_float4(sigmoid(v.x), sigmoid(v.y), sigmoid(v.z), sigmoid(v.w));
}
__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 mul4(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__device__ __forceinline__ float4 ld4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ void st4(float* p, float4 v) {
    *reinterpret_cast<float4*>(p) = v;
}
__device__ __forceinline__ float4 ldbf4(const bf16* p) {
    return unpack_bf4(*reinterpret_cast<const uint2*>(p));
}
__device__ __forceinline__ void stbf4(bf16* p, float4 v) {
    *reinterpret_cast<uint2*>(p) = pack_bf4(v);
}

// the pair (q, k) of row R of the pair-logit GEMM: tile R / 256 of the NP x NP table, then its
// swizzled [lane][value] order
__device__ __forceinline__ void pair_of_row(int row, int np, int& q, int& k) {
    const int tiles = np / kMma, t = row / kTileVals, w = row % kTileVals;
    q = t / tiles * kMma + acc_row_of(w / kAccVals, w % kAccVals);
    k = t % tiles * kMma + acc_col_of(w / kAccVals, w % kAccVals);
}

// ---- epilogues

// Each maps a staged chunk of the accumulator tile: rows row0.. of the GEMM, columns
// col0..col0+cols, row r of the chunk at st[r * kld]; 4 columns per thread.

// one kind per instance: the dispatch is outside the loop, and the chunk width a constant
template <int E, int kld, int kCols>
__device__ void epi_columns(const Gemm& g, const float* st, int row0, int rows, int col0) {
    constexpr int kC4 = kCols / 4;
    for (int i = threadIdx.x; i < rows * kC4; i += kThreads) {
        const int r = i / kC4, c = col0 + 4 * (i % kC4), row = row0 + r;
        if (row >= g.m || c >= g.n) {
            continue;
        }
        float4 v = ld4(st + r * kld + c - col0);
        float* outf = static_cast<float*>(g.out) + size_t(row) * g.ldo + c;
        bf16* outb = static_cast<bf16*>(g.out) + size_t(row) * g.ldo + c;
        if constexpr (E == EPI_STORE || E == EPI_QKVG) {
            if (g.bias && c < g.bias_cols) {
                v = add4(v, ld4(g.bias + c));
            }
            if (c >= g.sig_from) { // a branch, not a select: only the gate columns pay for it
                v = sigmoid4(v);
            }
            stbf4(outb, v);
        } else if constexpr (E == EPI_F32) {
            st4(outf, g.bias ? add4(v, ld4(g.bias + c)) : v);
        } else if constexpr (E == EPI_SCRES) {
            st4(outf, add4(ld4(outf), v));
        } else if constexpr (E == EPI_RESID) {
            const int p = row % g.period;
            const float* in = g.resid ? g.resid + size_t(row) * g.ldo + c : outf;
            float4 d = mul4(ldbf4(g.gate + size_t(p) * g.n + c), v);
            if (g.rowmask) {
                const float m = g.rowmask[p];
                d = make_float4(d.x * m, d.y * m, d.z * m, d.w * m);
            }
            st4(outf, add4(ld4(in), d));
        } else {
            // MODS: column c is slot slot0 + (c / kTokC) % slots of block c / (slots kTokC)
            const int blk = c / (g.slots * kTokC), slot = g.slot0 + (c / kTokC) % g.slots;
            v = add4(v, ld4(g.bias + c));
            if (g.sig_slots >> slot & 1) {
                v = sigmoid4(v);
            }
            bf16* o = static_cast<bf16*>(g.out) +
                      ((size_t(blk) * kTokModSlots + slot) * g.period + row) * kTokC + c % kTokC;
            stbf4(o, v);
        }
    }
}

// per unit of 64 (or 96) columns: [gate 32 | proj 32 (| a_to_b 32)] -> 32 outputs
template <int kld>
__device__ void epi_glu(const Gemm& g, const float* st, int row0, int rows, int col0, int cols) {
    const int o4s = cols / g.unit * kGluHalf / 4;
    const bool ab = g.unit > 2 * kGluHalf;
    for (int i = threadIdx.x; i < rows * o4s; i += kThreads) {
        const int r = i / o4s, o = 4 * (i % o4s), row = row0 + r;
        if (row >= g.m) {
            continue;
        }
        const float* s = st + r * kld + (o / kGluHalf) * g.unit + o % kGluHalf;
        const float4 gate = ld4(s), proj = ld4(s + kGluHalf);
        float4 v = make_float4(silu(gate.x) * proj.x, silu(gate.y) * proj.y, silu(gate.z) * proj.z,
                               silu(gate.w) * proj.w);
        if (ab) {
            v = mul4(v, ld4(s + 2 * kGluHalf));
        }
        const int oc = col0 / g.unit * kGluHalf + o;
        stbf4(static_cast<bf16*>(g.out) + size_t(row) * g.ldo + oc, v);
    }
}

// rows are (token, slot < 32): the masked mean of relu over the 24 real slots, plus SE
template <int kld>
__device__ void epi_aggr(const Gemm& g, const float* st, int row0, int rows, int col0, int cols) {
    constexpr int kGroup = kGluHalf; // gathered rows per token
    const int c4s = cols / 4;
    for (int i = threadIdx.x; i < rows / kGroup * c4s; i += kThreads) {
        const int tk = i / c4s, c = col0 + 4 * (i % c4s), row = (row0 + tk * kGroup) / kGroup;
        if (row * kGroup >= g.m) {
            continue;
        }
        const int n = row % g.period;
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
        float wsum = 0.f;
        for (int j = 0; j < kSlots; j++) {
            const float w = g.rowmask[n * kSlots + j];
            const float4 v = ld4(st + (tk * kGroup + j) * kld + c - col0);
            sum.x += w * fmaxf(v.x, 0.f), sum.y += w * fmaxf(v.y, 0.f);
            sum.z += w * fmaxf(v.z, 0.f), sum.w += w * fmaxf(v.w, 0.f);
            wsum += w;
        }
        const float inv = 1.f / fmaxf(wsum, 1e-10f);
        const float4 se = ld4(g.resid + size_t(n) * g.n + c);
        st4(static_cast<float*>(g.out) + size_t(row) * g.ldo + c,
            make_float4(sum.x * inv + se.x, sum.y * inv + se.y, sum.z * inv + se.z,
                        sum.w * inv + se.w));
    }
}

// rows are pairs (q, k) in the swizzled table's own order, so row R is element R of every table;
// column c is table c of the ring group: the folded LN(PC) @ W + b, in base 2. A thread keeps two
// rows' terms and walks the columns; a warp's lanes store consecutive row pairs.
template <int kld>
__device__ void epi_pair(const Gemm& g, const float* st, int row0, int rows, int col0, int cols) {
    // rows kAdj rp.. are a lane's adjacent values of the table: side by side in memory
    const int pairs = rows / kAdj, rp = threadIdx.x % pairs, row = row0 + kAdj * rp;
    if (row >= g.m) {
        return;
    }
    float mean[kAdj], rstd[kAdj], extra[kAdj];
    bool masked[kAdj], pad[kAdj];
    for (int j = 0; j < kAdj; j++) {
        int q, kk;
        pair_of_row(row + j, g.np, q, kk);
        masked[j] = kk >= g.period || g.tmask[kk] == 0.f;
        pad[j] = q >= g.period;
        mean[j] = rstd[j] = extra[j] = 0.f;
        if (!masked[j] && !pad[j]) {
            const size_t pq = size_t(q) * g.ldn + kk;
            mean[j] = g.pcs[2 * pq], rstd[j] = g.pcs[2 * pq + 1];
            extra[j] = g.plx ? from_bf(g.plx[pq]) : 0.f;
        }
    }
    const size_t plane = size_t(g.np) * g.np;
    bf16* out = static_cast<bf16*>(g.out) + row;
    for (int c = threadIdx.x / pairs; c < cols && col0 + c < g.n; c += kThreads / pairs) {
        const int col = col0 + c;
        float y[kAdj];
        for (int j = 0; j < kAdj; j++) {
            const float x = st[(kAdj * rp + j) * kld + c];
            y[j] = kLog2e * (masked[j] ? -kMaskBias
                             : pad[j]  ? 0.f
                                       : rstd[j] * (x - mean[j] * g.plws[col]) + g.plb[col] + extra[j]);
        }
        adj(out + col * plane) = adj_bf(y);
    }
}

template <int kld, int kCols>
__device__ __forceinline__ void epilogue(const Gemm& g, const float* st, int row0, int rows,
                                         int col0) {
    switch (g.epi) {
    case EPI_GLU:
        epi_glu<kld>(g, st, row0, rows, col0, kCols);
        break;
    case EPI_AGGR:
        epi_aggr<kld>(g, st, row0, rows, col0, kCols);
        break;
    case EPI_PAIR:
        epi_pair<kld>(g, st, row0, rows, col0, kCols);
        break;
    case EPI_F32:
        epi_columns<EPI_F32, kld, kCols>(g, st, row0, rows, col0);
        break;
    case EPI_SCRES:
        epi_columns<EPI_SCRES, kld, kCols>(g, st, row0, rows, col0);
        break;
    case EPI_RESID:
        epi_columns<EPI_RESID, kld, kCols>(g, st, row0, rows, col0);
        break;
    case EPI_MODS:
        epi_columns<EPI_MODS, kld, kCols>(g, st, row0, rows, col0);
        break;
    default:
        epi_columns<EPI_QKVG, kld, kCols>(g, st, row0, rows, col0);
    }
}

// ---- portable tile

// the address of A's row `row`, or null for a zero row
__device__ __forceinline__ const bf16* a_row(const Gemm& g, int row) {
    switch (g.gather) {
    case ROWS_AGGR: {
        const int tok = row / kGluHalf, slot = row % kGluHalf;
        if (slot >= kSlots || tok >= g.m / kGluHalf) {
            return nullptr;
        }
        const int s = tok / g.period, n = tok % g.period;
        const int a = g.q2ta[n * kSlots + slot];
        return a < 0 ? nullptr : g.a + (size_t(s) * g.atoms + a) * g.lda;
    }
    case ROWS_PAIR: {
        int q, k;
        pair_of_row(row, g.np, q, k);
        return q < g.period && k < g.period ? g.a + (size_t(q) * g.ldn + k) * g.lda : nullptr;
    }
    default:
        return g.a + size_t(row) * g.lda;
    }
}

// a portable tile's pipeline slots and bf16 of padding per smem row (conflict-free fragment
// loads); AMD's within 16 KB, where only the 64-row tile has room to pad
struct PortableCfg {
    int stages, pad;
};
__host__ __device__ constexpr PortableCfg portable_cfg(int code) {
#if CCK_AMD
    return code == T64 ? PortableCfg{2, 8} : PortableCfg{2, 0};
#else
    return code == T64 ? PortableCfg{3, 8} : PortableCfg{2, 8};
#endif
}

template <int Code> struct Portable {
    static constexpr int BM = tile_of(Code).bm, BN = tile_of(Code).bn;
    static_assert(BM <= BN, "a thread copies W's rows alongside A's, then the rest of W's");
    static constexpr int STAGES = portable_cfg(Code).stages, BK = kBK;
    static constexpr int kLd = BK + portable_cfg(Code).pad;
    static constexpr int kCopiesA = BM * BK / 8 / kThreads; // 16-byte copies per thread
    static constexpr int kCopiesW = BN * BK / 8 / kThreads;
    static constexpr int kRowStep = kThreads * 8 / BK; // rows between a thread's copies
    static constexpr int MT = BM / 2 / kMma, NT = BN / 2 / kMma;
    static constexpr int kSld = BN + kStagePad;
    static constexpr size_t kPipe = STAGES * (BM + BN) * kLd * sizeof(bf16);
    static constexpr size_t kStage = kEpiRows * kSld * sizeof(float);
    static constexpr size_t kSmem =
        kPipe > kStage ? kPipe : kStage; // staging reuses the pipeline's
};

// a 2x2 grid of warps; each thread copies the same rows of A and of W, so square tiles share one
// smem offset
template <int Code>
__device__ __noinline__ void portable_tiles(const Gemm& gd, int t0, int stride) {
    using P = Portable<Code>;
    constexpr int STAGES = P::STAGES, BK = P::BK;
    constexpr int BM = P::BM;
    const Gemm g = gd;
    char* smem = smem_base();
    bf16* sa = reinterpret_cast<bf16*>(smem);
    bf16* sw = sa + STAGES * BM * P::kLd; // W's slots are BN rows
    float* st = reinterpret_cast<float*>(smem);
    const int warp = threadIdx.x / kWarp, wm = warp % 2, wn = warp / 2;
    const int crow = threadIdx.x / (BK / 8), ccol = threadIdx.x % (BK / 8) * 8;
    const int mt = cdiv(g.m, BM), tiles = mt * cdiv(g.n, P::BN), ktiles = g.k / BK;
    for (int t = t0; t < tiles; t += stride) {
        const int row0 = t % mt * BM, col0 = t / mt * P::BN;
        const bf16* ap[P::kCopiesA];
        const bf16* wp[P::kCopiesW];
        uint32_t aok = 0, wok = 0;
#pragma unroll
        for (int i = 0; i < P::kCopiesW; i++) {
            const int r = crow + i * P::kRowStep;
            if (i < P::kCopiesA) {
                const bf16* p = a_row(g, row0 + r);
                aok |= uint32_t(p != nullptr) << i;
                ap[i] = (p ? p : g.a) + ccol;
            }
            wok |= uint32_t(col0 + r < g.n) << i;
            wp[i] = g.w + size_t(col0 + r < g.n ? col0 + r : 0) * g.k + ccol;
        }
        Acc acc[P::MT][P::NT];
#pragma unroll
        for (int i = 0; i < P::MT; i++) {
#pragma unroll
            for (int j = 0; j < P::NT; j++) {
                acc_zero(acc[i][j]);
            }
        }
        Copy16 ca[P::kCopiesA], cw[P::kCopiesW];
        int kload = 0;
        bool landing = false;
#pragma unroll 1
        for (int kt = -(STAGES - 1); kt < ktiles; kt++) {
            if (landing) { // the previous iteration's copies; that slot's readers passed a barrier
                const int slot = (kload - 1) % STAGES;
#pragma unroll
                for (int i = 0; i < P::kCopiesW; i++) {
                    const int r = crow + i * P::kRowStep;
                    if (i < P::kCopiesA) {
                        copy_land(ca[i], sa + (slot * BM + r) * P::kLd + slab_col(r, ccol));
                    }
                    copy_land(cw[i], sw + (slot * P::BN + r) * P::kLd + slab_col(r, ccol));
                }
            }
            if (kt >= 0) {
                async_wait<STAGES - 2>();
                __syncthreads();
            }
            landing = kload < ktiles;
            if (landing) {
                const int slot = kload % STAGES, ko = kload * BK;
#pragma unroll
                for (int i = 0; i < P::kCopiesW; i++) {
                    const int r = crow + i * P::kRowStep;
                    if (i < P::kCopiesA) {
                        copy_start(ca[i], sa + (slot * BM + r) * P::kLd + slab_col(r, ccol),
                                   ap[i] + ko, aok >> i & 1);
                    }
                    copy_start(cw[i], sw + (slot * P::BN + r) * P::kLd + slab_col(r, ccol),
                               wp[i] + ko, wok >> i & 1);
                }
            }
            async_commit();
            kload++;
            if (kt < 0) {
                continue;
            }
            const bf16* as = sa + (kt % STAGES) * BM * P::kLd + wm * (BM / 2) * P::kLd;
            const bf16* ws = sw + (kt % STAGES) * P::BN * P::kLd + wn * (P::BN / 2) * P::kLd;
#pragma unroll
            for (int ks = 0; ks < BK; ks += kMma) {
                FragA fa[P::MT];
                FragB fb[P::NT];
#pragma unroll
                for (int i = 0; i < P::MT; i++) {
                    load_a_slab(fa[i], as + i * kMma * P::kLd + ks, P::kLd);
                }
#pragma unroll
                for (int j = 0; j < P::NT; j++) {
                    load_b_slab(fb[j], ws + j * kMma * P::kLd + ks, P::kLd);
                }
#pragma unroll
                for (int i = 0; i < P::MT; i++) {
#pragma unroll
                    for (int j = 0; j < P::NT; j++) {
                        mma(acc[i][j], fa[i], fb[j]);
                    }
                }
            }
        }
        async_wait<0>();
        __syncthreads();
#pragma unroll
        for (int r0 = 0; r0 < BM; r0 += kEpiRows) {
#pragma unroll
            for (int i = 0; i < P::MT; i++) {
#pragma unroll
                for (int j = 0; j < P::NT; j++) {
#pragma unroll
                    for (int v = 0; v < kAccVals; v++) {
                        const int r = wm * (BM / 2) + i * kMma + acc_row(v) - r0;
                        if (r >= 0 && r < kEpiRows) {
                            st[r * P::kSld + wn * (P::BN / 2) + j * kMma + acc_col(v)] =
                                acc[i][j].v[v];
                        }
                    }
                }
            }
            __syncthreads();
            epilogue<P::kSld, P::BN>(g, st, row0 + r0, kEpiRows, col0);
            __syncthreads();
        }
    }
}

// ---- CuTe tile
#if !CCK_AMD
constexpr int kCuteStages = 3;

template <int Code> struct CuteTile {
    static constexpr int BM = tile_of(Code).bm, BN = tile_of(Code).bn;
    using T = cute::bfloat16_t;
    // 8 x 32 atoms, 2-bit swizzle of the 16-byte chunks: conflict-free ldmatrix
    using Atom = decltype(cute::composition(
        cute::Swizzle<2, 3, 3>{}, cute::Layout<cute::Shape<cute::_8, cute::Int<kBK>>,
                                               cute::Stride<cute::Int<kBK>, cute::_1>>{}));
    using SA = decltype(cute::tile_to_shape(
        Atom{}, cute::Shape<cute::Int<BM>, cute::Int<kBK>, cute::Int<kCuteStages>>{}));
    using SB = decltype(cute::tile_to_shape(
        Atom{}, cute::Shape<cute::Int<BN>, cute::Int<kBK>, cute::Int<kCuteStages>>{}));
    static constexpr int kSld = BN + kStagePad;
    static constexpr size_t kSmem = (BM + BN) * kBK * kCuteStages * sizeof(bf16);
    static_assert(kSmem >= kEpiRows * kSld * sizeof(float), "staging must fit the pipeline smem");
};

template <int Code> __device__ __noinline__ void cute_tiles(const Gemm& gd, int t0, int stride) {
    using namespace cute;
    using C = CuteTile<Code>;
    constexpr int BM = C::BM, BN = C::BN;
    using T = typename C::T;
    const Gemm g = gd;
    char* smem = smem_base();
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem)), typename C::SA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem) + cosize(typename C::SA{})),
                            typename C::SB{});
    float* st = reinterpret_cast<float*>(smem);

    auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>{},
                               Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _8>>{});
    auto mma_t = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_2, _2, _1>>{},
                                Tile<_32, _32, _16>{});
    auto thr_g2s = g2s.get_slice(threadIdx.x);
    Tensor tAsA = thr_g2s.partition_D(sA);
    Tensor tBsB = thr_g2s.partition_D(sB);
    auto thr_mma = mma_t.get_slice(threadIdx.x);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    Tensor acc = partition_fragment_C(mma_t, Shape<Int<BM>, Int<BN>>{});
    auto s2r_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, mma_t);
    auto s2r_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, mma_t);
    auto thr_s2r_a = s2r_a.get_slice(threadIdx.x);
    auto thr_s2r_b = s2r_b.get_slice(threadIdx.x);
    Tensor tXsA = thr_s2r_a.partition_S(sA);
    Tensor tXrA = thr_s2r_a.retile_D(tCrA);
    Tensor tXsB = thr_s2r_b.partition_S(sB);
    Tensor tXrB = thr_s2r_b.retile_D(tCrB);
    Tensor tCcC = thr_mma.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BN>>{}));
    constexpr int kBlocks = size<2>(tCrA);

    const int mt = cdiv(g.m, BM), tiles = mt * cdiv(g.n, BN);
    for (int t = t0; t < tiles; t += stride) {
        const int row0 = t % mt * BM, col0 = t / mt * BN;
        Tensor mA =
            make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(g.a) + size_t(row0) * g.lda),
                        make_shape(Int<BM>{}, g.k), make_stride(g.lda, _1{}));
        Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(g.w) + size_t(col0) * g.k),
                                make_shape(Int<BN>{}, g.k), make_stride(g.k, _1{}));
        Tensor tAgA =
            thr_g2s.partition_S(local_tile(mA, Shape<Int<BM>, Int<kBK>>{}, make_coord(0, _)));
        Tensor tBgB =
            thr_g2s.partition_S(local_tile(mB, Shape<Int<BN>, Int<kBK>>{}, make_coord(0, _)));
        clear(acc);

        int ktiles = size<3>(tAgA), knext = 0;
#pragma unroll
        for (int s = 0; s < kCuteStages - 1; s++) {
            copy(g2s, tAgA(_, _, _, knext), tAsA(_, _, _, s));
            copy(g2s, tBgB(_, _, _, knext), tBsB(_, _, _, s));
            cp_async_fence();
            if (--ktiles > 0) {
                knext++;
            }
        }
        int read = 0, write = kCuteStages - 1;
        cp_async_wait<kCuteStages - 2>();
        __syncthreads();
        copy(s2r_a, tXsA(_, _, Int<0>{}, read), tXrA(_, _, Int<0>{}));
        copy(s2r_b, tXsB(_, _, Int<0>{}, read), tXrB(_, _, Int<0>{}));
#pragma unroll 1
        while (ktiles > -(kCuteStages - 1)) {
#pragma unroll
            for (int kb = 0; kb < kBlocks; kb++) {
                if (kb == kBlocks - 1) {
                    read = read == kCuteStages - 1 ? 0 : read + 1;
                    cp_async_wait<kCuteStages - 2>();
                    __syncthreads();
                }
                // the next k-block is in registers while this one multiplies
                const int kn = (kb + 1) % kBlocks;
                copy(s2r_a, tXsA(_, _, kn, read), tXrA(_, _, kn));
                copy(s2r_b, tXsB(_, _, kn, read), tXrB(_, _, kn));
                if (kb == 0) {
                    copy(g2s, tAgA(_, _, _, knext), tAsA(_, _, _, write));
                    copy(g2s, tBgB(_, _, _, knext), tBsB(_, _, _, write));
                    cp_async_fence();
                    if (--ktiles > 0) {
                        knext++;
                    }
                    write = write == kCuteStages - 1 ? 0 : write + 1;
                }
                gemm(mma_t, tCrA(_, _, kb), tCrB(_, _, kb), acc);
            }
        }
        cp_async_wait<0>();
        __syncthreads();
#pragma unroll
        for (int r0 = 0; r0 < BM; r0 += kEpiRows) {
            const int rows = BM - r0 < kEpiRows ? BM - r0 : kEpiRows;
#pragma unroll
            for (int i = 0; i < size(acc); i++) {
                const int r = get<0>(tCcC(i)) - r0;
                if (r >= 0 && r < rows) {
                    st[r * C::kSld + get<1>(tCcC(i))] = acc(i);
                }
            }
            __syncthreads();
            epilogue<C::kSld, BN>(g, st, row0 + r0, rows, col0);
            __syncthreads();
        }
    }
}
#endif

// one non-inlined tile loop per GEMM per CTA: a call per tile spilled callee-saved registers
__device__ inline void gemm_run(const Gemm& g, int t0, int stride) {
    switch (g.tile) {
    case T128:
#if !CCK_AMD
        if (g.gather == ROWS_PLAIN) {
            cute_tiles<T128>(g, t0, stride);
            break;
        }
#endif
        portable_tiles<T128>(g, t0, stride);
        break;
#if !CCK_AMD
    case T128x96:
        cute_tiles<T128x96>(g, t0, stride);
        break;
    case T96:
        cute_tiles<T96>(g, t0, stride);
        break;
#endif
    case T64x128:
        portable_tiles<T64x128>(g, t0, stride);
        break;
    default:
        portable_tiles<T64>(g, t0, stride);
    }
}

// deals the GEMMs' tiles over the CTAs, row tiles fastest: co-running CTAs share a weight slice
template <int N> __device__ void run_gemms(Deal& d, const Gemm* const (&list)[N]) {
#pragma unroll 1
    for (int i = 0; i < N; i++) {
        const int tiles = gemm_tiles(*list[i]);
        __syncthreads(); // whatever ran before may still read shared memory
        gemm_run(*list[i], d.first(), gridDim.x);
        d.advance(tiles);
    }
}

#if CCK_AMD
constexpr size_t kGemmSmem = Portable<T128>::kSmem;
#else
constexpr size_t kGemmSmem = CuteTile<T128>::kSmem;
#endif
constexpr int kMaxTileRows = tile_of(T128).bm;

} // namespace mk
