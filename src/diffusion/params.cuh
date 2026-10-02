// The launch's parameter block (KP) and the GEMM descriptors, as plain data.
#pragma once

#include "diffusion/dims.cuh"

namespace mk {

enum Epi : int {
    EPI_STORE, // bf16(C + bias), sigmoid from sig_from
    EPI_QKVG,  // bias on the first bias_cols, sigmoid from sig_from; bf16
    EPI_F32,   // C + bias as f32
    EPI_MODS,  // token adaLN modulations into MODS, block-major
    EPI_RESID, // out = (resid or out) + gate ⊙ C · rowmask, f32
    EPI_GLU,   // silu(gate)·proj(·a_to_b) per unit; bf16
    EPI_AGGR,  // masked mean of relu over a token's 24 gathered slots, + SE
    EPI_SCRES, // out += C, f32
    EPI_PAIR,  // token pair logits into the swizzled PLG layout
};

enum Gather : int {
    ROWS_PLAIN,
    ROWS_AGGR, // row (s, n, slot < 32): the atom row Q2TA[n][slot] of sample s, or zero
    ROWS_PAIR, // row (q, k) of NP x NP: PC[q][k], or zero past the real tokens
};

// C = A Wᵀ, m x n over k, A row-major with lda, W [n][k]
struct Gemm {
    const bf16* a;
    const bf16* w;
    void* out;
    const float* bias;
    int lda, ldo, m, n, k;
    int epi, gather, tile;
    int sig_from;         // STORE, QKVG: sigmoid from this column on
    int bias_cols;        // QKVG: the bias covers these first columns
    int unit;             // GLU: unit width (64, or 96 with the a_to_b gate)
    int period;           // rows per sample of the gate, rowmask and MODS rows; tokens for gathers
    const bf16* gate;     // RESID: [period][n]
    const float* rowmask; // RESID: [period] (TMASK); AGGR: TAM
    const float* resid;   // RESID: the input residual, else out; AGGR: SE
    int slots, slot0, sig_slots;           // MODS: slots per block, first slot, sigmoid slot mask
    const int* q2ta;                       // AGGR
    int atoms;                             // AGGR: atom rows per sample
    int np, ldn;                           // PAIR: padded tokens, PC's row stride
    const float *pcs, *plws, *plb, *tmask; // PAIR
    const bf16* plx;                       // PAIR: opendde's extra bias, or null
};

struct GemmTable {
    Gemm st_w1[2], st_w2[2], mods1, mods2, embed, aggr, broadcast;
    Gemm tok[kNbMax][4];                           // QKVG, WO, W1, W2
    Gemm enc[kAtomBlocks][2], dec[kAtomBlocks][2]; // [q|g], [k|v]
    Gemm pair[kNbMax];                             // one per ring group
};
enum TokGemm { TG_QKVG, TG_WO, TG_W1, TG_W2 };

struct TokenW {
    const float *sigma, *sc_base, *fourier_w, *fourier_b, *noise_ln, *noise_lno, *wn;
    const float *st_s[2], *st_o[2];
    const bf16 *st_w1[2], *st_w2[2];
    const bf16 *wm1, *wm2, *wse, *wqkvg, *wo, *w1, *w2, *pc, *wtb, *waggr, *plw, *plx;
    const float *bm1, *bm2, *bq, *tmask, *se_b, *tb_b, *scf, *pcs, *plb, *plws, *tqkn;
};

struct AtomW {
    const bf16 *amods, *apl, *wqg, *wkv, *wo, *w1, *w2;
    const float *bq, *qkn;
};

struct Geo {
    const float *wpos, *scq, *qmask, *kmask, *atom_mask, *tam, *wupd, *upd_b, *rope, *chir_w,
        *chir_a, *misc;
    const int *q2k, *tok2q, *q2ta, *chir_c, *chir_ptr, *chir_idx;
};

struct Bufs {
    float *pos, *xn, *out, *grad, *cen, *kab, *ntv, *sc, *se, *qa, *skip, *act, *tb, *upd;
    bf16 *sct, *scln0, *scbf, *sch, *mods, *ax, *axk, *ao, *abf, *aqg, *akv, *tx, *tx2, *qkvg, *to,
        *tc, *plg;
};

struct KP {
    int S, N, A, NS, steps, nsteps, mode;
    int ldn, as, nss; // bucket strides: PC rows, AMODS atoms per block, APL subsets per block
    int np, chunks;   // padded tokens, reduction chunks per sample
    uint32_t seed_lo, seed_hi;
    int cs, nb, tw1, aw1, nk, plc, plx;
    uint32_t flags;
    float ln_eps, atom_eps;
    int plg;      // pair-logit tables in the ring
    float t;      // mode 1: the noise level
    int dbg_stop; // stop after this many phases (0: never)
    TokenW tw;
    AtomW enc, dec;
    Geo geo;
    Bufs buf;
    const GemmTable* gemms;
    float* out;                 // FFI: [S][Nb][24][3]
    int out_tokens;             // Nb
    unsigned long long* timing; // CCK_TIMING: per phase kind [wall ns, phases, CTA-busy ns]
};

} // namespace mk

#ifdef __CUDACC__
namespace mk {
__constant__ KP c_kp; // copied from the workspace each launch: reading it from global cost 7-9%
} // namespace mk
#endif
