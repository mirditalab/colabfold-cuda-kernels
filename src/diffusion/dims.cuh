// Fixed dimensions, flags and the names of the inputs and scratch buffers.
#pragma once

#include <cstddef>
#include <cstdint>

namespace mk {

typedef uint16_t bf16; // raw bits: conversions are explicit (frag.cuh)

constexpr int kTokC = 768;
constexpr int kHeads = 16;
constexpr int kHeadD = kTokC / kHeads;
constexpr int kQKVG = 4 * kTokC;
constexpr int kTokHidden = 2 * kTokC;
constexpr int kTokModSlots = 6;
constexpr int kAtomC = 128;
constexpr int kAtomHeads = 4;
constexpr int kAtomHeadD = kAtomC / kAtomHeads;
constexpr int kSubset = 32;
constexpr int kSubsetKeys = 128;    // NK
constexpr int kSubsetKeysEsm = 192; // NK of esmfold2's wider window
constexpr int kAtomModSlots = 8;
constexpr int kAtomBlocks = 3;
constexpr int kSlots = 24;
constexpr int kFourier = 256;
constexpr float kSigmaData = 16.0f;
constexpr int kKeyPad = 64;
constexpr float kMaskBias = 1e9f;
constexpr int kThreads = 128;

constexpr int kWarp = 32;
constexpr int kWarps = kThreads / kWarp;
constexpr int kXyz = 3;
constexpr int kNbMax = 24;           // token blocks of the largest model
constexpr int kGluHalf = 32;         // rows per part of a GLU unit
constexpr int kPairHeads = kHeads;   // one pair logit per token head
constexpr int kTokMods1 = 4;         // MODS slots from WM1; the rest come from WM2
constexpr int kChunk = kThreads * 8; // slots per deterministic reduction chunk
constexpr int kMoments = 16;         // REALIGN partial moments: w, w·a, w·d, w·a·dᵀ
constexpr int kCenSums = 4;          // centring partial sums: w·x, w·y, w·z, w
constexpr float kLnEps = 1e-5f;
constexpr float kIdScaleEps = 0.1f;
constexpr float kLog2e = 1.4426950408889634f;

enum Flag : uint32_t {
    CHAIN_K = 1,
    ATOB = 2,
    QKNORM = 4,
    PARALLEL = 8,
    CHIRAL = 16,
    IDSCALE = 32,
    SCFINAL = 64,
    HEUN = 128,
    ESM = 256,
    REALIGN = 512,
};

// Inputs in the contract's order (§3.3)
// clang-format off
enum In {
    SIGMA, SC_BASE, FOURIER_W, FOURIER_B, NOISE_LN, WN,
    ST0_S, ST0_O, ST0_W1, ST0_W2, ST1_S, ST1_O, ST1_W1, ST1_W2,
    WM1, BM1, WM2, BM2, WSE,
    WQKVG, BQ, WO, W1, W2, PC, TMASK, WTB,
    WPOS, SCQ, QMASK, KMASK, Q2K, TOK2Q, Q2TA, ATOM_MASK, TAM, WAGGR, WUPD,
    ENC_AMODS, ENC_APL, ENC_WQG, ENC_BQ, ENC_WKV, ENC_WO, ENC_W1, ENC_W2,
    DEC_AMODS, DEC_APL, DEC_WQG, DEC_BQ, DEC_WKV, DEC_WO, DEC_W1, DEC_W2,
    NOISE_LNO, SE_B, TB_B, UPD_B, SCF, ROPE,
    PCS, PLW, PLB, PLWS, PLX,
    TQKN, EQKN, DQKN, CHIR_W, CHIR_C, CHIR_A, CHIR_PTR, CHIR_IDX,
    MISC,
    kNumInputs
};
// clang-format on
constexpr int kAtomInputs = DEC_AMODS - ENC_AMODS;

// Scratch buffers (§5.4); the order is the workspace layout
// clang-format off
#define MK_BUFFERS(X)                                                                              \
    X(POS) X(XN) X(OUT) X(GRAD) X(CEN) X(KAB) X(NTV) X(SC) X(SCT) X(SCLN0) X(SCBF) X(SCH) X(MODS)   \
    X(SE) X(QA) X(SKIP) X(AX) X(AXK) X(AO) X(ABF) X(AQG) X(AKV) X(ACT) X(TX) X(TX2) X(QKVG) X(TO)  \
    X(TC) X(TB) X(UPD) X(PLG)
// clang-format on
#define MK_ENUM(name) B_##name,
enum Buf { MK_BUFFERS(MK_ENUM) kNumBufs };
#undef MK_ENUM

enum Misc { GAMMA0, GAMMA_MIN, NOISE_SCALE, STEP_SCALE, MISC_FLAGS, CHURN };

// Token adaLN slots of MODS (§4.3)
enum TokMod {
    TM_ATTN_SCALE,
    TM_ATTN_BIAS,
    TM_TRANS_SCALE,
    TM_TRANS_BIAS,
    TM_ATTN_GATE,
    TM_TRANS_GATE
};
// Atom adaLN slots of AMODS (§4.4)
enum AtomMod {
    AM_Q_SCALE,
    AM_Q_BIAS,
    AM_K_SCALE,
    AM_K_BIAS,
    AM_T_SCALE,
    AM_T_BIAS,
    AM_ATTN_GATE,
    AM_T_GATE
};
// rosettafold3's q/k LayerNorm rows per block (TQKN, EQKN, DQKN)
enum QkNorm { QKN_Q_SCALE, QKN_Q_OFFSET, QKN_K_SCALE, QKN_K_OFFSET, kQkNormRows };

} // namespace mk
