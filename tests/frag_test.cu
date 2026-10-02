// Unit test of the fragment layer: nvcc -std=c++17 -arch=sm_120 -I src -I $CUTLASS/include
// tests/frag_test.cu, or hipcc -std=c++17 --offload-arch=gfx1151 -I src -x hip tests/frag_test.cu
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "diffusion/frag.cuh"

using namespace mk;

constexpr int T = kMma;

// C = A Bᵀ and (A Bᵀ) as bf16 times V, both through global and shared-memory loads
__global__ void frag_kernel(const bf16* a, const bf16* b, const bf16* v, float* c, float* c_lds,
                            float* pv, int* rc, float* red) {
    __shared__ __align__(16) bf16 sa[T * T], sb[T * T], sv[T * T], tile[kAccTile];
    for (int i = threadIdx.x; i < T * T; i += kWarp) {
        sa[i] = a[i], sb[i] = b[i], sv[i] = v[i];
    }
    __syncwarp();
    FragA fa;
    FragB fb;
    Acc acc;
    acc_zero(acc);
    load_a(fa, a, T);
    load_b(fb, b, T);
    mma(acc, fa, fb);
    Acc acc2;
    acc_zero(acc2);
    load_a_lds(fa, sa, T);
    load_b_lds(fb, sb, T);
    mma(acc2, fa, fb);
    FragB fv;
    load_b_lds_t(fv, sv, T);
    Acc o;
    acc_zero(o);
    mma(o, acc_to_a(acc, tile), fv);
    float mx[kRowSlots] = {}, sm[kRowSlots] = {};
    for (int i = 0; i < kAccVals; i++) {
        c[acc_row(i) * T + acc_col(i)] = acc.v[i];
        c_lds[acc_row(i) * T + acc_col(i)] = acc2.v[i];
        pv[acc_row(i) * T + acc_col(i)] = o.v[i];
        rc[(lane_id() * kAccVals + i) * 2] = acc_row(i);
        rc[(lane_id() * kAccVals + i) * 2 + 1] = acc_col(i);
        if (acc_lane(acc_row(i), acc_col(i)) != lane_id() ||
            acc_slot(acc_row(i), acc_col(i)) != i) {
            rc[0] = -1;
        }
        mx[row_slot(i)] = fmaxf(mx[row_slot(i)], acc.v[i]);
        sm[row_slot(i)] += acc.v[i];
    }
    row_reduce_max(mx);
    row_reduce_sum(sm);
    for (int i = 0; i < kAccVals; i++) {
        red[acc_row(i) * 2] = mx[row_slot(i)];
        red[acc_row(i) * 2 + 1] = sm[row_slot(i)];
    }
}

static float bfr(float x) {
    return from_bf(to_bf(x));
}

int main() {
    std::vector<bf16> a(T * T), b(T * T), v(T * T);
    for (int i = 0; i < T * T; i++) {
        a[i] = to_bf(sinf(i * 0.37f));
        b[i] = to_bf(cosf(i * 0.11f));
        v[i] = to_bf(sinf(i * 0.05f + 1));
    }
    bf16 *da, *db, *dv;
    float *dc, *dcl, *dpv, *dred;
    int* drc;
    cudaMalloc(&da, T * T * 2), cudaMalloc(&db, T * T * 2), cudaMalloc(&dv, T * T * 2);
    cudaMalloc(&dc, T * T * 4), cudaMalloc(&dcl, T * T * 4), cudaMalloc(&dpv, T * T * 4);
    cudaMalloc(&drc, kWarp * kAccVals * 2 * 4), cudaMalloc(&dred, T * 2 * 4);
    cudaMemcpy(da, a.data(), T * T * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), T * T * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), T * T * 2, cudaMemcpyHostToDevice);
    frag_kernel<<<1, kWarp>>>(da, db, dv, dc, dcl, dpv, drc, dred);
    std::vector<float> c(T * T), cl(T * T), pv(T * T), red(T * 2);
    std::vector<int> rc(kWarp * kAccVals * 2);
    cudaMemcpy(c.data(), dc, T * T * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(cl.data(), dcl, T * T * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(pv.data(), dpv, T * T * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(rc.data(), drc, rc.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(red.data(), dred, T * 2 * 4, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) {
        printf("cuda error\n");
        return 1;
    }
    double err = 0, err_lds = 0, err_pv = 0, err_red = 0;
    int bad_map = rc[0] < 0;
    for (int r = 0; r < T; r++) {
        float mx = -1e30f, sm = 0;
        for (int n = 0; n < T; n++) {
            float ref = 0, rpv = 0;
            for (int k = 0; k < T; k++) {
                ref += from_bf(a[r * T + k]) * from_bf(b[n * T + k]);
            }
            for (int k = 0; k < T; k++) {
                float p = 0;
                for (int j = 0; j < T; j++) {
                    p += from_bf(a[r * T + j]) * from_bf(b[k * T + j]);
                }
                rpv += bfr(p) * from_bf(v[k * T + n]);
            }
            err = fmax(err, fabs(c[r * T + n] - ref));
            err_lds = fmax(err_lds, fabs(cl[r * T + n] - ref));
            err_pv = fmax(err_pv, fabs(pv[r * T + n] - rpv));
            mx = fmaxf(mx, fmaxf(c[r * T + n], 0.f));
            sm += c[r * T + n];
        }
        err_red = fmax(err_red, fabs(red[r * 2] - mx) + fabs(red[r * 2 + 1] - sm));
    }
    // _prep's swizzle: NVIDIA row (lane>>2) + 8((i>>1)&1), column 8(i>>2) + 2(lane&3) + (i&1);
    // AMD row 2i + (lane>>4), column lane&15
    for (int l = 0; l < kWarp; l++) {
        for (int i = 0; i < kAccVals; i++) {
#if CCK_AMD
            int r = 2 * i + (l >> 4), col = l & 15;
#else
            int r = (l >> 2) + 8 * ((i >> 1) & 1), col = 8 * (i >> 2) + 2 * (l & 3) + (i & 1);
#endif
            bad_map |= rc[(l * kAccVals + i) * 2] != r || rc[(l * kAccVals + i) * 2 + 1] != col;
        }
    }
    printf("mma %.2e  lds %.2e  acc_to_a/ldsm_t %.2e  row reductions %.2e  layout %s\n", err,
           err_lds, err_pv, err_red, bad_map ? "BAD" : "ok");
    bool ok = err < 1e-4 && err_lds < 1e-4 && err_pv < 1e-3 && err_red < 1e-4 && !bad_map;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
