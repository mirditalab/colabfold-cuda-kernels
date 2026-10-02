// The phase schedule: one denoiser call, the step loop, the kernel. Work listed between two
// next() calls (which grid-sync) runs in the same phase.
#pragma once

#include "diffusion/atom_mlp.cuh"
#include "diffusion/attention.cuh"
#include "diffusion/positions.cuh"

#ifdef CCK_TIMING
#include <cuda/std/chrono>
#endif

namespace mk {

// CCK_TIMING phase kinds in the harness's order (mk.py KINDS); A_ADALN, A_O and A_W2 stay empty:
// the fused passes A_QKV and A_GLU do the reference's separate phases
// clang-format off
#define MK_KINDS(X)                                                                                \
    X(AUG) X(SC) X(SC_GEMM) X(MODS) X(ATOM_INIT) X(A_ADALN) X(A_QKV) X(A_ATTN) X(A_O) X(A_GLU)      \
    X(A_W2) X(AGG) X(T_ADALN) X(T_QKVG) X(T_ATTN) X(T_O) X(T_GLU) X(T_W2) X(T_OUT) X(DEC_GLUE)     \
    X(POS) X(QKN)
// clang-format on
#define MK_ENUM(name) K_##name,
enum Kind { MK_KINDS(MK_ENUM) kKinds };
#undef MK_ENUM

struct Run {
    int phase = 0;
    bool live = true; // false once a debug run reached dbg_stop
#ifdef CCK_TIMING
    uint64_t t0 = 0;
#endif
};

__device__ __forceinline__ Deal deal(const Run& r) {
    Deal d;
    d.live = r.live;
    return d;
}

#ifdef CCK_TIMING
__device__ inline uint64_t now_ns() {
    return cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(
               cuda::std::chrono::system_clock::now().time_since_epoch())
        .count();
}
#endif

// ends a phase: [wall ns, phases, CTA-busy ns summed over CTAs] per kind under CCK_TIMING
__device__ Deal next(Run& r, Kind kind) {
    if (!r.live) {
        return deal(r);
    }
#ifdef CCK_TIMING
    unsigned long long* tm = c_kp.timing;
    const uint64_t arrive = now_ns();
    if (threadIdx.x == 0 && r.t0) {
        atomicAdd(tm + 2 * kKinds + kind, arrive - r.t0);
    }
#endif
    __syncthreads(); // a missing CTA barrier once made samples nondeterministic
    cg::this_grid().sync();
#ifdef CCK_TIMING
    const uint64_t t = now_ns();
    if (blockIdx.x == 0 && threadIdx.x == 0 && r.t0) {
        tm[kind] += t - r.t0;
        tm[kKinds + kind] += 1;
    }
    r.t0 = t;
#else
    (void)kind;
#endif
    r.live = !(c_kp.dbg_stop > 0 && ++r.phase >= c_kp.dbg_stop);
    return deal(r);
}

__device__ __forceinline__ const GemmTable& gemms() {
    return *c_kp.gemms;
}
__device__ __forceinline__ bool resident_pairs() {
    return c_kp.plg >= c_kp.nb;
}

// fused adaLN, projections and q/k norms | attention | fused MLP
__device__ void atom_block(Run& r, const AtomW& w, int b, MlpTail tail) {
    Deal d = deal(r);
    atom_proj(d, w, b);
    d = next(r, K_A_QKV);
    atom_attention(d, w, b);
    d = next(r, K_A_ATTN);
    atom_mlp(d, w, b, tail);
    next(r, K_A_GLU);
}

__device__ void token_block(Run& r, int b) {
    const Gemm* g = gemms().tok[b];
    const bool parallel = c_kp.flags & PARALLEL;
    Deal d = deal(r);
    if (!resident_pairs() && b % c_kp.plg == 0) {
        run_gemms(d, {&gemms().pair[b / c_kp.plg]}); // ‖ this ring group's pair logits
    }
    token_adaln(d, b, TM_ATTN_SCALE, c_kp.buf.tx, parallel ? c_kp.buf.tx2 : nullptr);
    d = next(r, K_T_ADALN);
    if (parallel) {
        run_gemms(d, {&g[TG_QKVG], &g[TG_W1]});
    } else {
        run_gemms(d, {&g[TG_QKVG]});
    }
    d = next(r, K_T_QKVG);
    if (c_kp.flags & QKNORM) {
        token_qknorm(d, b);
        d = next(r, K_QKN);
    }
    if (parallel) {
        run_gemms(d, {&g[TG_W2]}); // both residuals read the block input
        __syncthreads();
    }
    token_attention(d, b);
    d = next(r, K_T_ATTN);
    run_gemms(d, {&g[TG_WO]});
    d = next(r, K_T_O);
    if (!parallel) {
        token_adaln(d, b, TM_TRANS_SCALE, c_kp.buf.tx, nullptr);
        d = next(r, K_T_ADALN);
        run_gemms(d, {&g[TG_W1]});
        d = next(r, K_T_GLU);
        run_gemms(d, {&g[TG_W2]});
        next(r, K_T_W2);
    }
}

// One denoiser call at noise level t on XN, ending with UPD
__device__ void denoise(Run& r, float t) {
    const GemmTable& g = gemms();
    const AtomW& enc = c_kp.enc;
    // the single conditioning, overlapped with the encoder
    Deal d = deal(r);
    noise_vector(d, t);
    atom_init(d, t);
    d = next(r, K_ATOM_INIT);
    for (int b = 0; b < kAtomBlocks; b++) {
        if (b < 2) {
            sc_layernorm(d, b, b == 0);
        } else {
            sc_prep(d);
        }
        atom_proj(d, enc, b);
        d = next(r, K_A_QKV);
        if (b < 2) {
            run_gemms(d, {&g.st_w1[b]});
        } else {
            run_gemms(d, {&g.mods1, &g.mods2, &g.embed});
        }
        atom_attention(d, enc, b);
        d = next(r, b < 2 ? K_A_ATTN : K_MODS);
        if (b < 2) {
            run_gemms(d, {&g.st_w2[b]});
        }
        atom_mlp(d, enc, b, b + 1 < kAtomBlocks ? TAIL_NONE : TAIL_SKIP);
        d = next(r, K_A_GLU);
    }
    // atoms to tokens, the token transformer, tokens back to atoms
    run_gemms(d, {&g.aggr});
    next(r, K_AGG);
    for (int b = 0; b < c_kp.nb; b++) {
        token_block(r, b);
    }
    d = deal(r);
    token_final_ln(d);
    d = next(r, K_T_OUT);
    run_gemms(d, {&g.broadcast});
    d = next(r, K_T_OUT);
    decoder_init(d);
    next(r, K_DEC_GLUE);
    for (int b = 0; b < kAtomBlocks; b++) {
        atom_block(r, c_kp.dec, b, b + 1 < kAtomBlocks ? TAIL_NONE : TAIL_UPDATE);
    }
}

__device__ void sample(Run& r) {
    Deal d = deal(r);
    positions_init(d);
    if (resident_pairs()) {
        run_gemms(d, {&gemms().pair[0]}); // ‖ the resident pair-logit build
    }
    next(r, K_AUG);
    for (int step = 1; step <= c_kp.nsteps && r.live; step++) {
        const StepNoise sn = step_noise(step);
        d = deal(r);
        positions_centre(d);
        d = next(r, K_AUG);
        positions_augment(d, step);
        next(r, K_AUG);
        denoise(r, sn.t_hat);
        d = deal(r);
        if (c_kp.flags & REALIGN) {
            positions_update(d, step, UPDATE_MOMENTS);
            d = next(r, K_POS);
            positions_realign(d, step);
        } else if (c_kp.flags & HEUN) {
            positions_update(d, step, UPDATE_HEUN1);
            next(r, K_POS);
            denoise(r, sn.sigma);
            d = deal(r);
            positions_update(d, step, UPDATE_HEUN2);
        } else {
            positions_update(d, step, UPDATE_EULER);
        }
        next(r, K_POS);
    }
    d = deal(r);
    positions_output(d);
}

// mode 1: one denoiser call on the host's XN at t, denoised positions to OUT
__device__ void denoise_once(Run& r) {
    if (resident_pairs()) {
        Deal d = deal(r);
        run_gemms(d, {&gemms().pair[0]});
        next(r, K_T_ADALN);
    }
    denoise(r, c_kp.t);
    Deal d = deal(r);
    positions_update(d, 0, UPDATE_OUT);
    next(r, K_POS);
}

} // namespace mk

extern "C" __global__ void __launch_bounds__(mk::kThreads) mk_kernel() {
    mk::Run r;
    if (mk::c_kp.mode == 0) {
        mk::sample(r);
    } else {
        mk::denoise_once(r);
    }
}
