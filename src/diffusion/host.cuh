// Host side: the launch template and workspace layout, the configure kernel, the launch, the
// ctypes debug API and the XLA FFI handler.
#pragma once

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "diffusion/sampler.cuh"

#ifdef MK_FFI
#include "xla/ffi/api/ffi.h"
#endif

namespace mk {

constexpr size_t kAlign = 256;
constexpr int kSlackRows = kMaxTileRows; // a partial tile reads up to one tile past a buffer's end
constexpr int kExtent = 5;               // {N, NS, exact, seed_lo, seed_hi}

__host__ __device__ inline size_t align_up(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}
__host__ __device__ inline int pad_keys(int n) {
    return cdiv(n, kKeyPad) * kKeyPad;
}
__host__ __device__ inline size_t pair_table_bytes(int np) {
    return size_t(kPairHeads) * np * np * sizeof(bf16);
}

static const char* const kBufNames[] = {
#define MK_NAME(name) #name,
    MK_BUFFERS(MK_NAME)
#undef MK_NAME
};

// Everything that depends only on the bucket's shapes: model widths and the workspace layout
struct Template {
    int S, Nb, NSb, nsteps;
    uint32_t flags;
    int steps, cs, nb, tw1, aw1, nk, plc, plx, plg;
    size_t off[kNumBufs], bytes[kNumBufs];
    size_t table_off, zero_begin, zero_end, total;
};

static size_t pair_budget() {
    static size_t budget = 0;
    if (!budget) {
        const char* env = getenv("MK_PL_BUDGET_MB");
        if (env) {
            budget = size_t(atol(env)) << 20;
        } else {
            int dev;
            cudaDeviceProp prop;
            cudaGetDevice(&dev);
            cudaGetDeviceProperties(&prop, dev);
            budget = prop.totalGlobalMem / 4;
        }
    }
    return budget;
}

// widths come from the inputs' byte sizes, never from a model name
static Template make_template(int S, int Nb, int NSb, int nsteps, uint32_t flags,
                              const size_t* in) {
    Template t{};
    t.S = S, t.Nb = Nb, t.NSb = NSb, t.nsteps = nsteps, t.flags = flags;
    t.steps = int(in[SIGMA] / sizeof(float)) - 1;
    t.cs = int(in[SC_BASE] / (size_t(Nb) * sizeof(float)));
    t.nb = int(in[WQKVG] / (size_t(kQKVG) * kTokC * sizeof(bf16)));
    t.tw1 = int(in[W1] / (size_t(t.nb) * kTokC * sizeof(bf16)));
    t.aw1 = int(in[ENC_W1] / (size_t(kAtomBlocks) * kAtomC * sizeof(bf16)));
    t.nk = int(in[Q2K] / (size_t(NSb) * sizeof(int)));
    t.plc = int(in[PLW] / (size_t(t.nb) * kPairHeads * sizeof(bf16)));
    t.plx = in[PLX] == size_t(Nb) * Nb * sizeof(bf16);
    const int np = pad_keys(Nb);
    t.plg = int(std::min<size_t>(std::max<size_t>(pair_budget() / pair_table_bytes(np), 1), t.nb));

    const size_t rt = size_t(S) * Nb, ra = size_t(S) * NSb * kSubset;
    const size_t pos = size_t(S) * Nb * kSlots, chunks = size_t(S) * cdiv(Nb * kSlots, kChunk);
    const bool grad = flags & (HEUN | REALIGN);
    constexpr size_t kF32 = sizeof(float), kBf16 = sizeof(bf16);
    struct Shape {
        size_t cols, elem, rows;
    } shape[kNumBufs];
    const size_t cs = t.cs, nb = Nb;
    shape[B_POS] = shape[B_XN] = shape[B_OUT] = {kXyz, kF32, pos};
    shape[B_GRAD] = {kXyz, kF32, grad ? pos : 0};
    shape[B_CEN] = {kCenSums, kF32, chunks};
    shape[B_KAB] = {kMoments, kF32, chunks};
    shape[B_NTV] = {cs, kF32, 1};
    shape[B_SC] = {cs, kF32, nb};
    shape[B_SCT] = shape[B_SCLN0] = shape[B_SCBF] = {cs, kBf16, nb};
    shape[B_SCH] = {2 * cs, kBf16, nb};
    shape[B_MODS] = {kTokC, kBf16, size_t(t.nb) * kTokModSlots * Nb};
    shape[B_SE] = {kTokC, kF32, nb};
    shape[B_QA] = shape[B_SKIP] = {kAtomC, kF32, ra};
    shape[B_AX] = shape[B_AXK] = shape[B_AO] = shape[B_ABF] = {kAtomC, kBf16, ra};
    shape[B_AQG] = shape[B_AKV] = {2 * kAtomC, kBf16, ra};
    shape[B_ACT] = {kTokC, kF32, rt};
    shape[B_TX] = shape[B_TO] = {kTokC, kBf16, rt};
    shape[B_TX2] = {kTokC, kBf16, flags & PARALLEL ? rt : 0};
    shape[B_QKVG] = {kQKVG, kBf16, rt};
    shape[B_TC] = {kTokHidden, kBf16, rt};
    shape[B_TB] = {kAtomC, kF32, rt};
    shape[B_UPD] = {kXyz, kF32, ra};
    shape[B_PLG] = {pair_table_bytes(np) / kBf16, kBf16, size_t(t.plg)};
    size_t off = align_up(sizeof(KP), kAlign);
    t.table_off = off;
    off = align_up(off + sizeof(GemmTable), kAlign);
    t.zero_begin = off;
    for (int b = 0; b < kNumBufs; b++) {
        if (b == B_PLG) {
            t.zero_end = off; // the pair-logit ring is always written before it is read
        }
        t.off[b] = off;
        const size_t row = shape[b].cols * shape[b].elem;
        t.bytes[b] = row * shape[b].rows;
        const size_t slack = b == B_PLG ? 0 : row * kSlackRows;
        off = align_up(off + t.bytes[b] + slack, kAlign);
    }
    t.total = off;
    return t;
}

// ---- setup
__host__ __device__ inline Gemm make_gemm(const bf16* a, int lda, const bf16* w, int m, int n,
                                          int k, int epi, void* out, int ldo) {
    Gemm g{};
    g.a = a, g.lda = lda, g.w = w, g.m = m, g.n = n, g.k = k, g.epi = epi, g.out = out, g.ldo = ldo;
    g.gather = ROWS_PLAIN, g.sig_from = n, g.period = 1;
    return g;
}
__device__ inline void pick_tile(Gemm& g, int grid) {
    g.tile = choose_tile(g.m, g.n, g.epi == EPI_GLU ? g.unit : 0, g.gather != ROWS_PLAIN, grid);
}

// the GEMM table for the real extent
__device__ void build_gemms(const KP& p, GemmTable& t, int grid) {
    const Bufs& b = p.buf;
    const TokenW& w = p.tw;
    const int N = p.N, cs = p.cs, rt = p.S * N, ra = p.S * p.A;
    const bool idscale = p.flags & IDSCALE;
    const bf16* x = idscale ? b.scbf : b.scln0; // the adaLN modulations' input
    const int unit = p.flags & ATOB ? 3 * kGluHalf : 2 * kGluHalf;
    for (int i = 0; i < 2; i++) {
        Gemm& g1 = t.st_w1[i];
        g1 = make_gemm(b.sct, cs, w.st_w1[i], N, 4 * cs, cs, EPI_GLU, b.sch, 2 * cs);
        g1.unit = 2 * kGluHalf;
        t.st_w2[i] = make_gemm(b.sch, 2 * cs, w.st_w2[i], N, cs, 2 * cs, EPI_SCRES, b.sc, cs);
    }
    t.mods1 = make_gemm(x, cs, w.wm1, N, p.nb * kTokMods1 * kTokC, cs, EPI_MODS, b.mods, kTokC);
    t.mods1.bias = w.bm1, t.mods1.slots = kTokMods1, t.mods1.slot0 = 0, t.mods1.period = N;
    t.mods1.sig_slots = idscale ? 0 : (1 << TM_ATTN_SCALE) | (1 << TM_TRANS_SCALE);
    t.mods2 = make_gemm(b.scbf, cs, w.wm2, N, p.nb * (kTokModSlots - kTokMods1) * kTokC, cs,
                        EPI_MODS, b.mods, kTokC);
    t.mods2.bias = w.bm2, t.mods2.slots = kTokModSlots - kTokMods1, t.mods2.slot0 = kTokMods1;
    t.mods2.period = N, t.mods2.sig_slots = (1 << TM_ATTN_GATE) | (1 << TM_TRANS_GATE);
    t.embed = make_gemm(x, cs, w.wse, N, kTokC, cs, EPI_F32, b.se, kTokC);
    t.embed.bias = w.se_b;
    t.aggr =
        make_gemm(b.abf, kAtomC, w.waggr, rt * kGluHalf, kTokC, kAtomC, EPI_AGGR, b.act, kTokC);
    t.aggr.gather = ROWS_AGGR, t.aggr.q2ta = p.geo.q2ta, t.aggr.atoms = p.A, t.aggr.period = N;
    t.aggr.rowmask = p.geo.tam, t.aggr.resid = b.se;
    t.broadcast = make_gemm(b.tx, kTokC, w.wtb, rt, kAtomC, kTokC, EPI_F32, b.tb, kAtomC);
    t.broadcast.bias = w.tb_b;
    for (int k = 0; k < p.nb; k++) {
        const bf16* mods = b.mods + size_t(k) * kTokModSlots * N * kTokC;
        Gemm* g = t.tok[k];
        g[TG_QKVG] = make_gemm(b.tx, kTokC, w.wqkvg + size_t(k) * kQKVG * kTokC, rt, kQKVG, kTokC,
                               EPI_QKVG, b.qkvg, kQKVG);
        g[TG_QKVG].bias = w.bq + k * kTokC, g[TG_QKVG].bias_cols = kTokC;
        g[TG_QKVG].sig_from = 3 * kTokC;
        g[TG_WO] = make_gemm(b.to, kTokC, w.wo + size_t(k) * kTokC * kTokC, rt, kTokC, kTokC,
                             EPI_RESID, b.act, kTokC);
        g[TG_WO].gate = mods + size_t(TM_ATTN_GATE) * N * kTokC, g[TG_WO].period = N;
        g[TG_W1] =
            make_gemm(p.flags & PARALLEL ? b.tx2 : b.tx, kTokC, w.w1 + size_t(k) * p.tw1 * kTokC,
                      rt, p.tw1, kTokC, EPI_GLU, b.tc, kTokHidden);
        g[TG_W1].unit = unit;
        g[TG_W2] = make_gemm(b.tc, kTokHidden, w.w2 + size_t(k) * kTokC * kTokHidden, rt, kTokC,
                             kTokHidden, EPI_RESID, b.act, kTokC);
        g[TG_W2].gate = mods + size_t(TM_TRANS_GATE) * N * kTokC, g[TG_W2].period = N;
        g[TG_W2].rowmask = w.tmask;
    }
    for (int side = 0; side < 2; side++) {
        const AtomW& aw = side ? p.dec : p.enc;
        Gemm(*g)[2] = side ? t.dec : t.enc;
        for (int k = 0; k < kAtomBlocks; k++) {
            const size_t wo = size_t(k) * 2 * kAtomC * kAtomC;
            g[k][0] = make_gemm(b.ax, kAtomC, aw.wqg + wo, ra, 2 * kAtomC, kAtomC, EPI_QKVG, b.aqg,
                                2 * kAtomC);
            g[k][0].bias = aw.bq + k * kAtomC, g[k][0].bias_cols = kAtomC,
            g[k][0].sig_from = kAtomC;
            g[k][1] = make_gemm(b.axk, kAtomC, aw.wkv + wo, ra, 2 * kAtomC, kAtomC, EPI_STORE,
                                b.akv, 2 * kAtomC);
        }
    }
    for (int grp = 0; grp * p.plg < p.nb; grp++) {
        const int b0 = grp * p.plg, n = kPairHeads * min(p.plg, p.nb - b0);
        Gemm& g = t.pair[grp];
        g = make_gemm(w.pc, p.plc, w.plw + size_t(b0) * kPairHeads * p.plc, p.np * p.np, n, p.plc,
                      EPI_PAIR, b.plg, 0);
        g.gather = ROWS_PAIR, g.np = p.np, g.ldn = p.ldn, g.period = N;
        g.pcs = w.pcs, g.plws = w.plws + b0 * kPairHeads, g.plb = w.plb + b0 * kPairHeads;
        g.plx = p.plx ? w.plx : nullptr, g.tmask = w.tmask;
        g.tile = T128; // it gathers rows; a 64-wide tile is about 3x slower
    }
    // the table is an array of Gemm in all but name
    for (Gemm* g = &t.st_w1[0]; g < &t.pair[0]; g++) {
        if (g->a) {
            pick_tile(*g, grid);
        }
    }
}

// CTA 0 builds this launch's parameter block and GEMM table for the real extent while all CTAs
// zero the workspace, which XLA hands over uninitialised
__global__ void mk_configure(KP p, const int* ext, char* ws, size_t table_off, size_t zero_begin,
                             size_t zero_end, size_t plg_bytes, int grid) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int nb_tok = p.N, nb_sub = p.NS;
        if (ext && ext[2]) {
            p.N = ext[0], p.NS = ext[1];
        }
        p.N = max(1, min(p.N, nb_tok));
        p.NS = max(1, min(p.NS, nb_sub));
        if (ext) {
            p.seed_lo = uint32_t(ext[3]), p.seed_hi = uint32_t(ext[4]);
        }
        p.A = p.NS * kSubset;
        p.np = pad_keys(p.N);
        p.chunks = cdiv(p.N * kSlots, kChunk);
        p.plg = int(min(size_t(p.nb), plg_bytes / pair_table_bytes(p.np)));
        p.attn_wide = token_attention_wide(p.S, p.N, grid);
        GemmTable* t = reinterpret_cast<GemmTable*>(ws + table_off);
        p.gemms = t;
        build_gemms(p, *t, grid);
        *reinterpret_cast<KP*>(ws) = p;
    }
    uint4* z = reinterpret_cast<uint4*>(ws + zero_begin);
    const size_t n = (zero_end - zero_begin) / sizeof(uint4);
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x) {
        z[i] = make_uint4(0, 0, 0, 0);
    }
}

// ---- launch
constexpr size_t kSmem = std::max({kGemmSmem, kTokAttnSmem, kAtomAttnSmem}); // over all phases

static int default_grid() {
    static int grid = 0;
    if (!grid) {
        int dev, sms, occ;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaFuncSetAttribute(mk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem));
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mk_kernel, kThreads, kSmem);
        grid = occ * sms;
        if (const char* env = getenv("MK_GRID")) {
            grid = atoi(env);
        }
    }
    return grid;
}

// the inputs, typed
struct Inputs {
    const void* const* in;
    const float* f(int i) const {
        return static_cast<const float*>(in[i]);
    }
    const bf16* h(int i) const {
        return static_cast<const bf16*>(in[i]);
    }
    const int* n(int i) const {
        return static_cast<const int*>(in[i]);
    }
};

// the scratch buffers, typed
struct Space {
    char* ws;
    const size_t* off;
    float* f(int b) const {
        return reinterpret_cast<float*>(ws + off[b]);
    }
    bf16* h(int b) const {
        return reinterpret_cast<bf16*>(ws + off[b]);
    }
};

// the parameter block's bucket-sized part: widths, strides, pointers
static KP make_kp(const Template& t, const void* const* in, char* ws, float* out) {
    KP p{};
    p.S = t.S, p.N = t.Nb, p.NS = t.NSb, p.A = t.NSb * kSubset;
    p.steps = t.steps, p.nsteps = t.nsteps;
    p.ldn = t.Nb, p.as = t.NSb * kSubset, p.nss = t.NSb;
    p.cs = t.cs, p.nb = t.nb, p.tw1 = t.tw1, p.aw1 = t.aw1;
    p.nk = t.nk, p.plc = t.plc, p.plx = t.plx;
    p.flags = t.flags;
    p.ln_eps = t.flags & IDSCALE ? kIdScaleEps : kLnEps;
    p.atom_eps = t.flags & ESM ? FLT_EPSILON : p.ln_eps;
    p.out = out, p.out_tokens = t.Nb;
    const Inputs x{in};
    TokenW& w = p.tw;
    w.sigma = x.f(SIGMA), w.sc_base = x.f(SC_BASE), w.wn = x.f(WN);
    w.fourier_w = x.f(FOURIER_W), w.fourier_b = x.f(FOURIER_B);
    w.noise_ln = x.f(NOISE_LN), w.noise_lno = x.f(NOISE_LNO);
    for (int i = 0; i < 2; i++) {
        const int st = i ? ST1_S : ST0_S; // S, O, W1, W2
        w.st_s[i] = x.f(st), w.st_o[i] = x.f(st + 1);
        w.st_w1[i] = x.h(st + 2), w.st_w2[i] = x.h(st + 3);
    }
    w.wm1 = x.h(WM1), w.bm1 = x.f(BM1), w.wm2 = x.h(WM2), w.bm2 = x.f(BM2);
    w.wse = x.h(WSE), w.se_b = x.f(SE_B), w.wtb = x.h(WTB), w.tb_b = x.f(TB_B);
    w.wqkvg = x.h(WQKVG), w.bq = x.f(BQ), w.wo = x.h(WO), w.w1 = x.h(W1), w.w2 = x.h(W2);
    w.waggr = x.h(WAGGR), w.tmask = x.f(TMASK), w.scf = x.f(SCF), w.tqkn = x.f(TQKN);
    w.pc = x.h(PC), w.pcs = x.f(PCS), w.plx = x.h(PLX);
    w.plw = x.h(PLW), w.plb = x.f(PLB), w.plws = x.f(PLWS);
    for (int side = 0; side < 2; side++) {
        AtomW& a = side ? p.dec : p.enc;
        const int in0 = side ? DEC_AMODS : ENC_AMODS; // AMODS, APL, WQG, BQ, WKV, WO, W1, W2
        a.amods = x.h(in0), a.apl = x.h(in0 + 1), a.wqg = x.h(in0 + 2), a.bq = x.f(in0 + 3);
        a.wkv = x.h(in0 + 4), a.wo = x.h(in0 + 5), a.w1 = x.h(in0 + 6), a.w2 = x.h(in0 + 7);
        a.qkn = x.f(side ? DQKN : EQKN);
    }
    Geo& g = p.geo;
    g.wpos = x.f(WPOS), g.scq = x.f(SCQ), g.wupd = x.f(WUPD), g.upd_b = x.f(UPD_B);
    g.qmask = x.f(QMASK), g.kmask = x.f(KMASK), g.atom_mask = x.f(ATOM_MASK), g.tam = x.f(TAM);
    g.q2k = x.n(Q2K), g.tok2q = x.n(TOK2Q), g.q2ta = x.n(Q2TA);
    g.rope = x.f(ROPE), g.misc = x.f(MISC);
    g.chir_w = x.f(CHIR_W), g.chir_c = x.n(CHIR_C), g.chir_a = x.f(CHIR_A);
    g.chir_ptr = x.n(CHIR_PTR), g.chir_idx = x.n(CHIR_IDX);
    const Space sp{ws, t.off};
    Bufs& b = p.buf;
    b.pos = sp.f(B_POS), b.xn = sp.f(B_XN), b.out = sp.f(B_OUT), b.grad = sp.f(B_GRAD);
    b.cen = sp.f(B_CEN), b.kab = sp.f(B_KAB), b.ntv = sp.f(B_NTV), b.sc = sp.f(B_SC);
    b.se = sp.f(B_SE), b.qa = sp.f(B_QA), b.skip = sp.f(B_SKIP), b.act = sp.f(B_ACT);
    b.tb = sp.f(B_TB), b.upd = sp.f(B_UPD);
    b.sct = sp.h(B_SCT), b.scln0 = sp.h(B_SCLN0), b.scbf = sp.h(B_SCBF), b.sch = sp.h(B_SCH);
    b.mods = sp.h(B_MODS), b.ax = sp.h(B_AX), b.axk = sp.h(B_AXK), b.ao = sp.h(B_AO);
    b.abf = sp.h(B_ABF), b.aqg = sp.h(B_AQG), b.akv = sp.h(B_AKV), b.tx = sp.h(B_TX);
    b.tx2 = sp.h(B_TX2), b.qkvg = sp.h(B_QKVG), b.to = sp.h(B_TO), b.tc = sp.h(B_TC);
    b.plg = sp.h(B_PLG);
    return p;
}

// configure | parameter block to __constant__ | the sampler: three stream operations, no host sync
static cudaError_t launch(const Template& t, KP p, const int* ext, char* ws, int grid,
                          cudaStream_t st) {
    mk_configure<<<grid, kThreads, 0, st>>>(p, ext, ws, t.table_off, t.zero_begin, t.zero_end,
                                            t.bytes[B_PLG], grid);
    cudaMemcpyToSymbolAsync(c_kp, ws, sizeof(KP), 0, cudaMemcpyDeviceToDevice, st);
    return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(mk_kernel), grid, kThreads, nullptr,
                                       kSmem, st);
}

// ---- ctypes
struct State {
    Template t;
    std::vector<void*> in;
    char* ws = nullptr;
    int* ext = nullptr;
    unsigned long long* timing = nullptr;
    int grid = 0;
};
constexpr int kTimingWords = 3 * kKinds;
static State g_state;

} // namespace mk

using namespace mk;

extern "C" size_t mk_workspace_bytes(int S, int Nb, int NSb, int nsteps, int flags,
                                     const size_t* bytes) {
    return make_template(S, Nb, NSb, nsteps, uint32_t(flags), bytes).total;
}

extern "C" void mk_teardown() {
    for (void* p : g_state.in) {
        cudaFree(p);
    }
    cudaFree(g_state.ws), cudaFree(g_state.ext), cudaFree(g_state.timing);
    g_state = State{};
}

extern "C" int mk_setup(int S, int N, int A, int NS, int steps, const void** host,
                        const size_t* bytes) {
    mk_teardown();
    (void)A;
    const uint32_t flags = uint32_t(static_cast<const float*>(host[MISC])[MISC_FLAGS]);
    g_state.t = make_template(S, N, NS, steps, flags, bytes);
    g_state.in.resize(kNumInputs);
    for (int i = 0; i < kNumInputs; i++) {
        cudaMalloc(&g_state.in[i], bytes[i]);
        cudaMemcpy(g_state.in[i], host[i], bytes[i], cudaMemcpyHostToDevice);
    }
    cudaMalloc(&g_state.ws, g_state.t.total);
    cudaMalloc(&g_state.ext, kExtent * sizeof(int));
    cudaMalloc(&g_state.timing, kTimingWords * sizeof(unsigned long long));
    g_state.grid = default_grid();
    if (getenv("MK_VERBOSE")) {
        printf("mk: %d CTAs, %zu B shared memory per CTA; pair logits for %d of %d blocks; "
               "workspace %zu MB\n",
               g_state.grid, kSmem, g_state.t.plg, g_state.t.nb, g_state.t.total >> 20);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int mk_set_grid(int grid) {
    if (grid > 0) {
        g_state.grid = grid;
    }
    return g_state.grid ? g_state.grid : default_grid();
}

extern "C" int mk_run(int mode, int nsteps, int dbg_stop, uint64_t seed, const float* xn, float t,
                      float* ms, unsigned long long* timing) {
    State& s = g_state;
    const int ext[kExtent] = {s.t.Nb, s.t.NSb, 1, int(uint32_t(seed)), int(uint32_t(seed >> 32))};
    cudaMemcpy(s.ext, ext, sizeof ext, cudaMemcpyHostToDevice);
    KP p = make_kp(s.t, s.in.data(), s.ws, nullptr);
    p.mode = mode, p.nsteps = mode == 0 ? nsteps : 0, p.t = t, p.dbg_stop = dbg_stop;
    p.timing = s.timing;
    cudaMemset(s.timing, 0, kTimingWords * sizeof(unsigned long long));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    if (mode == 1) {
        // the host's noisy positions go in after configure has zeroed the workspace
        mk_configure<<<s.grid, kThreads>>>(p, s.ext, s.ws, s.t.table_off, s.t.zero_begin,
                                           s.t.zero_end, s.t.bytes[B_PLG], s.grid);
        cudaMemcpy(s.ws + s.t.off[B_XN], xn, s.t.bytes[B_XN], cudaMemcpyHostToDevice);
        cudaEventRecord(e0);
        cudaMemcpyToSymbolAsync(c_kp, s.ws, sizeof(KP), 0, cudaMemcpyDeviceToDevice, 0);
        cudaLaunchCooperativeKernel(reinterpret_cast<void*>(mk_kernel), s.grid, kThreads, nullptr,
                                    kSmem, 0);
    } else {
        cudaEventRecord(e0);
        launch(s.t, p, s.ext, s.ws, s.grid, 0);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(ms, e0, e1);
    cudaEventDestroy(e0), cudaEventDestroy(e1);
#ifdef CCK_TIMING
    if (timing) {
        cudaMemcpy(timing, s.timing, kTimingWords * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        for (int k = 0; k < kKinds; k++) {
            timing[2 * kKinds + k] /= s.grid; // busy: the mean over CTAs
        }
    }
#else
    (void)timing;
#endif
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "mk_run: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

// the phase kinds of mk_run's timing record, comma-separated in its order
extern "C" const char* mk_timing_names() {
#define MK_NAME(name) #name ","
    static const char names[] = MK_KINDS(MK_NAME);
#undef MK_NAME
    return names;
}

extern "C" int mk_buf_id(const char* name) {
    for (int b = 0; b < kNumBufs; b++) {
        if (!strcmp(name, kBufNames[b])) {
            return b;
        }
    }
    return -1;
}

extern "C" size_t mk_buf_bytes(int id) {
    return id >= 0 && id < kNumBufs ? g_state.t.bytes[id] : 0;
}

extern "C" int mk_read(int id, void* host, size_t bytes) {
    if (id < 0 || id >= kNumBufs || bytes > g_state.t.bytes[id]) {
        return 1;
    }
    return cudaMemcpy(host, g_state.ws + g_state.t.off[id], bytes, cudaMemcpyDeviceToHost) !=
           cudaSuccess;
}

// ---- FFI
#ifdef MK_FFI
namespace ffi = xla::ffi;

static ffi::Error mk_sample(cudaStream_t stream, ffi::RemainingArgs args,
                            ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> ws,
                            int32_t samples, int32_t nsteps, int32_t flags) {
    if (args.size() != kNumInputs + 1) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MkSample: wrong number of inputs");
    }
    const void* in[kNumInputs];
    size_t bytes[kNumInputs];
    for (int i = 0; i < kNumInputs; i++) {
        auto b = args.get<ffi::AnyBuffer>(i + 1);
        if (b.has_error()) {
            return b.error();
        }
        in[i] = b->untyped_data(), bytes[i] = b->size_bytes();
    }
    auto ext = args.get<ffi::AnyBuffer>(0);
    if (ext.has_error()) {
        return ext.error();
    }
    const int nb = int(out->dimensions()[1]);
    const int nsb = int(bytes[QMASK] / sizeof(float) / kSubset);
    const Template t = make_template(samples, nb, nsb, nsteps, uint32_t(flags), bytes);
    if (ws->size_bytes() < t.total) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MkSample: workspace too small");
    }
    char* w = static_cast<char*>(ws->untyped_data());
    KP p = make_kp(t, in, w, static_cast<float*>(out->untyped_data()));
    const cudaError_t err =
        launch(t, p, static_cast<const int*>(ext->untyped_data()), w, default_grid(), stream);
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(MkSample, mk_sample,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .RemainingArgs()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int32_t>("samples")
                                  .Attr<int32_t>("nsteps")
                                  .Attr<int32_t>("flags"),
                              {ffi::Traits::kCmdBufferCompatible});
#endif
