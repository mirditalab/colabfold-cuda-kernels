// Positions: noise, centring, augmentation, step updates and the atom input features.
// Cross-CTA reductions use fixed chunks of kChunk slots, summed in a fixed order.
#pragma once

#include "diffusion/rng.cuh"
#include "diffusion/rows.cuh"

namespace mk {

constexpr int kSlotsPerThread = kChunk / kThreads;
constexpr float kChurnMin = 4e-4f, kChurnMax = 80.f; // chai1's churn window on σ_p (S_tmin, S_tmax)
// chai1 floors t̂² - σ_p², so its no-churn steps still add 1e-3 Å of noise
constexpr float kHeunFloor = 1e-6f;
constexpr float kWeightFloor = 1e-6f; // of the centring weight sum
constexpr float kChiralEps = 1e-6f;   // oracle.chiral_grads' eps
constexpr float kTiny = 1e-30f;
constexpr int kJacobiSweeps = 16;

// K sums over the CTA through red[kWarps * K] of shared memory, in a fixed order, for all threads
template <int K> __device__ void block_reduce(float (&v)[K], float* red) {
#pragma unroll
    for (int k = 0; k < K; k++) {
        v[k] = warp_sum(v[k]);
    }
    __syncthreads();
    if (lane_id() == 0) {
        for (int k = 0; k < K; k++) {
            red[warp_id() * K + k] = v[k];
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; k++) {
        v[k] = 0.f;
        for (int w = 0; w < kWarps; w++) {
            v[k] += red[w * K + k];
        }
    }
}

// the noise levels of one step
struct StepNoise {
    float sigma, t_hat, noise;
};
__device__ inline StepNoise step_noise(int step) {
    const float* misc = c_kp.geo.misc;
    const float sigma = c_kp.tw.sigma[step], prev = c_kp.tw.sigma[step - 1];
    float gamma;
    if (c_kp.flags & HEUN) {
        gamma = prev >= kChurnMin && prev <= kChurnMax ? misc[CHURN] : 0.f;
    } else {
        gamma = sigma > misc[GAMMA_MIN] ? misc[GAMMA0] : 0.f;
    }
    const float t_hat = prev * (1.f + gamma);
    const float floor = c_kp.flags & HEUN ? kHeunFloor : 0.f;
    return {sigma, t_hat, misc[NOISE_SCALE] * sqrtf(fmaxf(floor, t_hat * t_hat - prev * prev))};
}

__device__ __forceinline__ int slots() {
    return c_kp.N * kSlots;
}
__device__ __forceinline__ size_t pos_index(int s, int slot) {
    return (size_t(s) * slots() + slot) * kXyz;
}

// pos = σ_0 z
__device__ void positions_init(Deal& d) {
    const int n = c_kp.S * slots(), items = cdiv(n, kThreads);
    for (int it = d.first(); it < items; it += gridDim.x) {
        const int i = it * kThreads + threadIdx.x;
        if (i < n) {
            const int s = i / slots(), slot = i % slots();
            const Normals z = normals(slot, 0, s, STREAM_INIT, c_kp.seed_lo, c_kp.seed_hi);
            for (int j = 0; j < kXyz; j++) {
                c_kp.buf.pos[pos_index(s, slot) + j] = c_kp.tw.sigma[0] * z.z[j];
            }
        }
    }
    d.advance(items);
}

// per (sample, chunk): Σ w·pos and Σ w into CEN
__device__ void positions_centre(Deal& d) {
    const int items = c_kp.S * c_kp.chunks;
    float* red = reinterpret_cast<float*>(smem_base());
    for (int it = d.first(); it < items; it += gridDim.x) {
        const int s = it / c_kp.chunks, c = it % c_kp.chunks;
        float v[kCenSums] = {};
        for (int i = 0; i < kSlotsPerThread; i++) {
            const int slot = c * kChunk + i * kThreads + threadIdx.x;
            if (slot < slots()) {
                const float w = c_kp.geo.atom_mask[slot];
                const float* p = c_kp.buf.pos + pos_index(s, slot);
                v[0] += w * p[0], v[1] += w * p[1], v[2] += w * p[2], v[3] += w;
            }
        }
        block_reduce(v, red);
        if (threadIdx.x < kCenSums) {
            c_kp.buf.cen[size_t(it) * kCenSums + threadIdx.x] = v[threadIdx.x];
        }
    }
    d.advance(items);
}

// xn = ((pos - c_s) R_s + τ_s)·mask + noise·z
__device__ void positions_augment(Deal& d, int step) {
    const int items = c_kp.S * c_kp.chunks;
    const StepNoise sn = step_noise(step);
    for (int it = d.first(); it < items; it += gridDim.x) {
        const int s = it / c_kp.chunks, c = it % c_kp.chunks;
        float sum[kCenSums] = {};
        for (int k = 0; k < c_kp.chunks; k++) {
            for (int j = 0; j < kCenSums; j++) {
                sum[j] += c_kp.buf.cen[(size_t(s) * c_kp.chunks + k) * kCenSums + j];
            }
        }
        const float wsum = fmaxf(sum[3], kWeightFloor);
        const float centre[kXyz] = {sum[0] / wsum, sum[1] / wsum, sum[2] / wsum};
        const Rigid g = random_rigid(step, s, c_kp.seed_lo, c_kp.seed_hi);
        for (int i = 0; i < kSlotsPerThread; i++) {
            const int slot = c * kChunk + i * kThreads + threadIdx.x;
            if (slot >= slots()) {
                continue;
            }
            const float w = c_kp.geo.atom_mask[slot];
            const float* p = c_kp.buf.pos + pos_index(s, slot);
            const float dp[kXyz] = {p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]};
            const Normals z = normals(slot, step, s, STREAM_CHURN, c_kp.seed_lo, c_kp.seed_hi);
            float* x = c_kp.buf.xn + pos_index(s, slot);
            for (int j = 0; j < kXyz; j++) {
                const float rot = dp[0] * g.r[0][j] + dp[1] * g.r[1][j] + dp[2] * g.r[2][j];
                x[j] = (rot + g.t[j]) * w + sn.noise * z.z[j];
            }
        }
    }
    d.advance(items);
}

// the denoised position of one slot: (c_skip·xn + c_out·upd)·mask
__device__ __forceinline__ void denoised(int s, int slot, float t, float (&den)[kXyz]) {
    const float sd2 = kSigmaData * kSigmaData;
    const float c_skip = sd2 / (t * t + sd2), c_out = t * kSigmaData / sqrtf(t * t + sd2);
    const int a = c_kp.geo.q2ta[slot];
    const float w = c_kp.geo.atom_mask[slot];
    const float* x = c_kp.buf.xn + pos_index(s, slot);
    for (int j = 0; j < kXyz; j++) {
        const float u = a >= 0 ? c_kp.buf.upd[(size_t(s) * c_kp.A + a) * kXyz + j] : 0.f;
        den[j] = (c_skip * x[j] + c_out * u) * w;
    }
}

enum Update {
    UPDATE_EULER,   // pos = xn + step_scale (σ - t̂)(xn - den)/t̂
    UPDATE_HEUN1,   // d1 = (xn - den)/t̂ into GRAD, x1 = xn + (σ - t̂) d1 into XN
    UPDATE_HEUN2,   // pos = x1 + (σ - t̂)((x1 - den2)/σ + d1)/2
    UPDATE_MOMENTS, // REALIGN: den into GRAD, the alignment's partial moments into KAB
    UPDATE_OUT,     // mode 1: den into OUT
};

__device__ void positions_update(Deal& d, int step, Update kind) {
    const int items = c_kp.S * c_kp.chunks;
    float* red = reinterpret_cast<float*>(smem_base());
    const StepNoise sn = kind == UPDATE_OUT ? StepNoise{0.f, c_kp.t, 0.f} : step_noise(step);
    const float dt = sn.sigma - sn.t_hat;
    for (int it = d.first(); it < items; it += gridDim.x) {
        const int s = it / c_kp.chunks, c = it % c_kp.chunks;
        float mom[kMoments] = {};
        for (int i = 0; i < kSlotsPerThread; i++) {
            const int slot = c * kChunk + i * kThreads + threadIdx.x;
            if (slot >= slots()) {
                continue;
            }
            const size_t pi = pos_index(s, slot);
            float* x = c_kp.buf.xn + pi;
            float den[kXyz];
            denoised(s, slot, kind == UPDATE_HEUN2 ? sn.sigma : sn.t_hat, den);
            for (int j = 0; j < kXyz; j++) {
                switch (kind) {
                case UPDATE_EULER:
                    c_kp.buf.pos[pi + j] =
                        x[j] + c_kp.geo.misc[STEP_SCALE] * dt * (x[j] - den[j]) / sn.t_hat;
                    break;
                case UPDATE_HEUN1: {
                    const float d1 = (x[j] - den[j]) / sn.t_hat;
                    c_kp.buf.grad[pi + j] = d1;
                    x[j] += dt * d1;
                    break;
                }
                case UPDATE_HEUN2:
                    c_kp.buf.pos[pi + j] =
                        x[j] + dt * ((x[j] - den[j]) / sn.sigma + c_kp.buf.grad[pi + j]) / 2.f;
                    break;
                case UPDATE_MOMENTS:
                    c_kp.buf.grad[pi + j] = den[j];
                    break;
                case UPDATE_OUT:
                    c_kp.buf.out[pi + j] = den[j];
                    break;
                }
            }
            if (kind == UPDATE_MOMENTS) {
                const float w = c_kp.geo.atom_mask[slot];
                mom[0] += w;
                for (int j = 0; j < kXyz; j++) {
                    mom[1 + j] += w * x[j];
                    mom[4 + j] += w * den[j];
                    for (int k = 0; k < kXyz; k++) {
                        mom[7 + 3 * j + k] += w * x[j] * den[k];
                    }
                }
            }
        }
        if (kind == UPDATE_MOMENTS) {
            block_reduce(mom, red);
            if (threadIdx.x < kMoments) {
                c_kp.buf.kab[size_t(it) * kMoments + threadIdx.x] = mom[threadIdx.x];
            }
        }
    }
    d.advance(items);
}

// the top eigenvector of a symmetric 4x4 matrix by cyclic Jacobi sweeps
__device__ inline void top_eigenvector(float (&a)[4][4], float (&q)[4]) {
    float v[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    for (int sweep = 0; sweep < kJacobiSweeps; sweep++) {
        float off = 0.f;
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                off += a[i][j] * a[i][j];
            }
        }
        if (off < kTiny) {
            break;
        }
        for (int p = 0; p < 4; p++) {
            for (int r = p + 1; r < 4; r++) {
                if (fabsf(a[p][r]) < kTiny) {
                    continue;
                }
                const float theta = (a[r][r] - a[p][p]) / (2.f * a[p][r]);
                const float t = copysignf(1.f, theta) / (fabsf(theta) + sqrtf(theta * theta + 1.f));
                const float c = 1.f / sqrtf(t * t + 1.f), s = t * c;
                for (int k = 0; k < 4; k++) {
                    const float akp = a[k][p], akr = a[k][r];
                    a[k][p] = c * akp - s * akr, a[k][r] = s * akp + c * akr;
                }
                for (int k = 0; k < 4; k++) {
                    const float apk = a[p][k], ark = a[r][k];
                    a[p][k] = c * apk - s * ark, a[r][k] = s * apk + c * ark;
                }
                for (int k = 0; k < 4; k++) {
                    const float vkp = v[k][p], vkr = v[k][r];
                    v[k][p] = c * vkp - s * vkr, v[k][r] = s * vkp + c * vkr;
                }
            }
        }
    }
    int best = 0;
    for (int i = 1; i < 4; i++) {
        best = a[i][i] > a[best][best] ? i : best;
    }
    for (int k = 0; k < 4; k++) {
        q[k] = v[k][best];
    }
}

// REALIGN: align xn rigidly onto den (Horn's quaternions), then Euler from the aligned xn
__device__ void positions_realign(Deal& d, int step) {
    const int items = c_kp.S * c_kp.chunks;
    float* rt = reinterpret_cast<float*>(smem_base()); // R[3][3], then x̄ and d̄
    const StepNoise sn = step_noise(step);
    const float dt = sn.sigma - sn.t_hat;
    for (int it = d.first(); it < items; it += gridDim.x) {
        const int s = it / c_kp.chunks, c = it % c_kp.chunks;
        __syncthreads();
        if (threadIdx.x == 0) {
            float m[kMoments] = {};
            for (int k = 0; k < c_kp.chunks; k++) {
                for (int j = 0; j < kMoments; j++) {
                    m[j] += c_kp.buf.kab[(size_t(s) * c_kp.chunks + k) * kMoments + j];
                }
            }
            const float w = m[0];
            float h[kXyz][kXyz]; // Σ w (x - x̄)(d - d̄)ᵀ
            for (int i = 0; i < kXyz; i++) {
                for (int j = 0; j < kXyz; j++) {
                    h[i][j] = m[7 + 3 * i + j] - m[1 + i] * m[4 + j] / w;
                }
            }
            float n[4][4] = {{h[0][0] + h[1][1] + h[2][2], h[1][2] - h[2][1], h[2][0] - h[0][2],
                              h[0][1] - h[1][0]},
                             {h[1][2] - h[2][1], h[0][0] - h[1][1] - h[2][2], h[0][1] + h[1][0],
                              h[2][0] + h[0][2]},
                             {h[2][0] - h[0][2], h[0][1] + h[1][0], -h[0][0] + h[1][1] - h[2][2],
                              h[1][2] + h[2][1]},
                             {h[0][1] - h[1][0], h[2][0] + h[0][2], h[1][2] + h[2][1],
                              -h[0][0] - h[1][1] + h[2][2]}};
            float q[4];
            top_eigenvector(n, q);
            const float q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
            // R maps x - x̄ onto d - d̄ as column vectors; stored transposed for row vectors
            const float r[kXyz][kXyz] = {{q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
                                          2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)},
                                         {2 * (q1 * q2 + q0 * q3),
                                          q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
                                          2 * (q2 * q3 - q0 * q1)},
                                         {2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1),
                                          q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3}};
            for (int i = 0; i < kXyz; i++) {
                for (int j = 0; j < kXyz; j++) {
                    rt[i * kXyz + j] = r[j][i];
                }
                rt[9 + i] = m[1 + i] / w;
                rt[12 + i] = m[4 + i] / w;
            }
        }
        __syncthreads();
        for (int i = 0; i < kSlotsPerThread; i++) {
            const int slot = c * kChunk + i * kThreads + threadIdx.x;
            if (slot >= slots()) {
                continue;
            }
            const size_t pi = pos_index(s, slot);
            const float* x = c_kp.buf.xn + pi;
            const float dx[kXyz] = {x[0] - rt[9], x[1] - rt[10], x[2] - rt[11]};
            for (int j = 0; j < kXyz; j++) {
                const float al =
                    dx[0] * rt[j] + dx[1] * rt[kXyz + j] + dx[2] * rt[2 * kXyz + j] + rt[12 + j];
                const float den = c_kp.buf.grad[pi + j];
                c_kp.buf.pos[pi + j] = al + c_kp.geo.misc[STEP_SCALE] * dt * (al - den) / sn.t_hat;
            }
        }
    }
    d.advance(items);
}

// ---- atom features

__device__ __forceinline__ void sub3(const float* a, const float* b, float* o) {
    o[0] = a[0] - b[0], o[1] = a[1] - b[1], o[2] = a[2] - b[2];
}
__device__ __forceinline__ float dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
__device__ __forceinline__ void cross3(const float* a, const float* b, float* o) {
    o[0] = a[1] * b[2] - a[2] * b[1], o[1] = a[2] * b[0] - a[0] * b[2],
    o[2] = a[0] * b[1] - a[1] * b[0];
}

// d/d(role's atom) of (improper dihedral - ideal)², oracle.chiral_grads' closed form
__device__ void chiral_grad(const float (&p)[4][kXyz], float ideal, int role, float* out) {
    const float eps = kChiralEps;
    float b0[3], b1[3], b2[3], u[3], v[3], w[3], t[3];
    sub3(p[0], p[1], b0), sub3(p[2], p[1], b1), sub3(p[3], p[2], b2);
    const float n1 = sqrtf(dot3(b1, b1)), nb = n1 + eps;
    for (int i = 0; i < 3; i++) {
        u[i] = b1[i] / nb;
    }
    const float b0u = dot3(b0, u), b2u = dot3(b2, u);
    for (int i = 0; i < 3; i++) {
        v[i] = b0[i] - b0u * u[i], w[i] = b2[i] - b2u * u[i];
    }
    cross3(u, v, t);
    const float y = dot3(t, w) + eps, x = dot3(v, w) + eps;
    const float g = 2.f * (atan2f(y, x) - ideal), den = x * x + y * y;
    const float gy = g * x / den, gx = -g * y / den;
    float gv[3], gw[3], gu[3], wu[3], uv[3], vw[3];
    cross3(w, u, wu), cross3(u, v, uv), cross3(v, w, vw);
    for (int i = 0; i < 3; i++) {
        gv[i] = gy * wu[i] + gx * w[i];
        gw[i] = gy * uv[i] + gx * v[i];
        gu[i] = gy * vw[i];
    }
    const float gvu = dot3(gv, u), gwu = dot3(gw, u);
    float gb0[3], gb1[3], gb2[3];
    for (int i = 0; i < 3; i++) {
        gb0[i] = gv[i] - gvu * u[i];
        gu[i] -= gvu * b0[i] + b0u * gv[i];
        gb2[i] = gw[i] - gwu * u[i];
        gu[i] -= gwu * b2[i] + b2u * gw[i];
    }
    const float gub1 = dot3(gu, b1);
    for (int i = 0; i < 3; i++) {
        gb1[i] = gu[i] / nb - b1[i] * gub1 / (nb * nb * fmaxf(n1, kTiny));
    }
    for (int i = 0; i < 3; i++) {
        const float r[4] = {gb0[i], -gb0[i] - gb1[i], gb1[i] - gb2[i], gb2[i]};
        out[i] = r[role];
    }
}

// qa = (xn·mask/√(t² + σ_d²) @ WPOS (+ chirality gradient @ CHIR_W))·QMASK + SCQ
__device__ void atom_init(Deal& d, float t) {
    const int rows = c_kp.S * c_kp.A;
    const float scale = 1.f / sqrtf(t * t + kSigmaData * kSigmaData);
    const Geo& g = c_kp.geo;
    for (int r = row_first(d); r < rows; r += row_step()) {
        const int s = r / c_kp.A, a = r % c_kp.A, slot = g.tok2q[a];
        float x[kXyz] = {}, gr[kXyz] = {};
        if (slot >= 0) {
            const float* p = c_kp.buf.xn + pos_index(s, slot);
            for (int j = 0; j < kXyz; j++) {
                x[j] = p[j] * g.atom_mask[slot] * scale;
            }
            if (c_kp.flags & CHIRAL) {
                for (int e = g.chir_ptr[slot]; e < g.chir_ptr[slot + 1]; e++) {
                    const int centre = g.chir_idx[e] / 4, role = g.chir_idx[e] % 4;
                    const float ideal = g.chir_a[centre];
                    if (ideal == 0.f) {
                        continue;
                    }
                    float p4[4][kXyz], gi[kXyz];
                    for (int k = 0; k < 4; k++) {
                        const int sk = g.chir_c[centre * 4 + k];
                        const float* pk = c_kp.buf.xn + pos_index(s, sk);
                        for (int j = 0; j < kXyz; j++) {
                            p4[k][j] = pk[j] * g.atom_mask[sk] * scale;
                        }
                    }
                    chiral_grad(p4, ideal, role, gi);
                    for (int j = 0; j < kXyz; j++) {
                        gr[j] += gi[j];
                    }
                }
                for (int j = 0; j < kXyz; j++) {
                    gr[j] = isfinite(gr[j]) ? gr[j] : 0.f;
                }
            }
        }
        const int c = 4 * lane_id();
        float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int j = 0; j < kXyz; j++) {
            const float4 wp = ld4(g.wpos + j * kAtomC + c);
            f = make_float4(f.x + x[j] * wp.x, f.y + x[j] * wp.y, f.z + x[j] * wp.z,
                            f.w + x[j] * wp.w);
            if (c_kp.flags & CHIRAL) {
                const float4 wc = ld4(g.chir_w + j * kAtomC + c);
                f = make_float4(f.x + gr[j] * wc.x, f.y + gr[j] * wc.y, f.z + gr[j] * wc.z,
                                f.w + gr[j] * wc.w);
            }
        }
        const float m = g.qmask[a];
        const float4 q = ld4(g.scq + size_t(a) * kAtomC + c);
        st4(c_kp.buf.qa + size_t(r) * kAtomC + c,
            make_float4(f.x * m + q.x, f.y * m + q.y, f.z * m + q.z, f.w * m + q.w));
    }
    row_done(d, rows);
}

// the FFI's output: positions, zero past the real tokens
__device__ void positions_output(Deal& d) {
    if (!c_kp.out) {
        return;
    }
    const int per = c_kp.out_tokens * kSlots * kXyz, n = c_kp.S * per, items = cdiv(n, kThreads);
    for (int it = d.first(); it < items; it += gridDim.x) {
        const int i = it * kThreads + threadIdx.x;
        if (i < n) {
            const int s = i / per, j = i % per;
            c_kp.out[i] = j < slots() * kXyz ? c_kp.buf.pos[size_t(s) * slots() * kXyz + j] : 0.f;
        }
    }
    d.advance(items);
}

} // namespace mk
