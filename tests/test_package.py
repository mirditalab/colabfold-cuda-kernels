"""The package's kernel table, and the built library's exports when it is there."""
import ctypes
import os

import pytest

import colabfold_cuda_kernels as cck


def test_diffusion_needs_ampere():
    assert cck.symbol("diffusion", 80) == "MkSample"
    with pytest.raises(FileNotFoundError):
        cck.symbol("diffusion", 75)
    assert not cck.available(75)


def test_unknown_kernel():
    with pytest.raises(KeyError):
        cck.library_path("nope", 120)
    assert not cck.available(120, "nope")


@pytest.mark.skipif(not cck.available(120), reason="library not built")
@pytest.mark.parametrize("cc", [80, 86, 87, 89, 90, 100, 103, 120, 121])
def test_library_exports_every_symbol(cc):
    path = cck.library_path("diffusion", cc)
    assert os.path.basename(path) == "libcolabfold_cuda_kernels.so"
    lib = ctypes.cdll.LoadLibrary(path)
    for kernel in cck.KERNELS:
        getattr(lib, cck.symbol(kernel, cc))
