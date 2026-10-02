// Philox4x32-10 normals and the random rigid augmentation; bit-exact with the reference.
#pragma once

#include "diffusion/dims.cuh"

namespace mk {

enum Stream : uint32_t { STREAM_CHURN, STREAM_AUGMENT, STREAM_INIT };

struct Normals {
    float z[4];
};

// counter (idx, step, sample, stream), key (seed_lo, seed_hi)
__device__ inline Normals normals(uint32_t idx, uint32_t step, uint32_t sample, uint32_t stream,
                                  uint32_t k0, uint32_t k1) {
    uint32_t c[4] = {idx, step, sample, stream};
#pragma unroll
    for (int r = 0; r < 10; r++) {
        const uint32_t p0 = 0xD2511F53u * c[0], h0 = __umulhi(0xD2511F53u, c[0]);
        const uint32_t p1 = 0xCD9E8D57u * c[2], h1 = __umulhi(0xCD9E8D57u, c[2]);
        c[0] = h1 ^ c[1] ^ k0, c[1] = p1, c[2] = h0 ^ c[3] ^ k1, c[3] = p0;
        k0 += 0x9E3779B9u, k1 += 0xBB67AE85u;
    }
    constexpr float kU = 1.0f / (1 << 24);
    constexpr float kTwoPi = 6.2831853071795864f;
    Normals n;
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const float u1 = float((c[2 * j] >> 8) + 1) * kU, u2 = float(c[2 * j + 1] >> 8) * kU;
        const float r = sqrtf(-2.f * logf(u1)), th = kTwoPi * u2;
        n.z[2 * j] = r * cosf(th);
        n.z[2 * j + 1] = r * sinf(th);
    }
    return n;
}

struct Rigid {
    float r[kXyz][kXyz]; // rows e0, e1, e2: out_j = Σ_i d_i r[i][j]
    float t[kXyz];
};

__device__ inline float norm3(const float* v) {
    return fmaxf(sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 1e-10f);
}

// a random rotation by Gram-Schmidt on two normal vectors, and a normal translation
__device__ inline Rigid random_rigid(uint32_t step, uint32_t sample, uint32_t k0, uint32_t k1) {
    float z[12];
    for (int i = 0; i < 3; i++) {
        const Normals n = normals(i, step, sample, STREAM_AUGMENT, k0, k1);
        for (int j = 0; j < 4; j++) {
            z[4 * i + j] = n.z[j];
        }
    }
    Rigid g;
    float* e0 = g.r[0];
    float* e1 = g.r[1];
    float* e2 = g.r[2];
    const float n0 = norm3(z);
    for (int i = 0; i < kXyz; i++) {
        e0[i] = z[i] / n0;
    }
    const float d = z[3] * e0[0] + z[4] * e0[1] + z[5] * e0[2];
    float v1[kXyz];
    for (int i = 0; i < kXyz; i++) {
        v1[i] = z[3 + i] - d * e0[i];
    }
    const float n1 = norm3(v1);
    for (int i = 0; i < kXyz; i++) {
        e1[i] = v1[i] / n1;
    }
    e2[0] = e0[1] * e1[2] - e0[2] * e1[1];
    e2[1] = e0[2] * e1[0] - e0[0] * e1[2];
    e2[2] = e0[0] * e1[1] - e0[1] * e1[0];
    for (int i = 0; i < kXyz; i++) {
        g.t[i] = z[6 + i];
    }
    return g;
}

} // namespace mk
