# colabfold-cuda-kernels

Prebuilt CUDA kernels to make ColabFold fast on Ampere and newer NVIDIA GPUs.
This package has one kernel so far: the AF3-family diffusion sampler, which
runs every step and every sample of a fold in a single launch.

```python
import colabfold_cuda_kernels as cck
cck.available(cc=120)                  # True if the wheel has kernels for sm_120
cck.library_path("diffusion", cc=120)  # path of the shared library
cck.symbol("diffusion", cc=120)        # XLA FFI target name to register
```

The library holds SASS for sm_80, 86, 89, 90, 100 and 120 and PTX for sm_120.
colabfold-kernels loads it and registers the FFI target.

To build the wheel, run pip. It needs nvcc 12.8 or newer and the XLA FFI
headers, which come from the installed jax unless `CCK_FFI_INCLUDE_DIR` is set,
and fetches CUTLASS itself.

```bash
pip wheel . -w dist                                                      # all architectures
SKBUILD_CMAKE_DEFINE="CMAKE_CUDA_ARCHITECTURES=120" pip wheel . -w dist  # one architecture
```
