// The AF3-family diffusion sampler: all steps and samples of a fold in one persistent launch.
#include "diffusion/host.cuh"

__global__ void __launch_bounds__(kThreads) mk_gemm_test_kernel(const Gemm g) {
    gemm_run(g, blockIdx.x, gridDim.x);
}

extern "C" int mk_gemm_test(const uint16_t* A, const uint16_t* W, float* C, int M, int N, int K,
                            int tile) {
    const int pad = kSlackRows; // a partial tile reads past the operands' ends
    const int np = cdiv(N, pad) * pad;
    bf16 *da, *dw;
    float* dc;
    cudaMalloc(&da, size_t(M + pad) * K * sizeof(bf16));
    cudaMalloc(&dw, size_t(np) * K * sizeof(bf16));
    cudaMalloc(&dc, size_t(M) * N * sizeof(float));
    cudaMemset(da, 0, size_t(M + pad) * K * sizeof(bf16));
    cudaMemset(dw, 0, size_t(np) * K * sizeof(bf16));
    cudaMemcpy(da, A, size_t(M) * K * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, W, size_t(N) * K * sizeof(bf16), cudaMemcpyHostToDevice);
    Gemm g{};
    g.a = da, g.w = dw, g.out = dc, g.lda = K, g.ldo = N, g.m = M, g.n = N, g.k = K;
    g.epi = EPI_F32, g.gather = ROWS_PLAIN, g.tile = tile;
    cudaFuncSetAttribute(mk_gemm_test_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(kGemmSmem));
    mk_gemm_test_kernel<<<gemm_tiles(g), kThreads, kGemmSmem>>>(g);
    cudaMemcpy(C, dc, size_t(M) * N * sizeof(float), cudaMemcpyDeviceToHost);
    const cudaError_t err = cudaGetLastError();
    cudaFree(da), cudaFree(dw), cudaFree(dc);
    return err == cudaSuccess ? 0 : 1;
}
