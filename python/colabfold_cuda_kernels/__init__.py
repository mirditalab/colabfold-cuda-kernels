"""ColabFold CUDA kernels for sm_80 and newer."""

from pathlib import Path

__all__ = [
    "KERNELS",
    "available",
    "library_path",
    "symbol",
]

__version__ = "0.1.0"

# One library for every architecture: SASS per built one, PTX for later GPUs.
_LIBRARY = Path(__file__).resolve().parent / "libcolabfold_cuda_kernels.so"

# kernel: (minimum compute capability, XLA FFI target name)
_TABLE = {
    "diffusion": (80, "MkSample"),
    "fp8_contract": (89, "CckFp8Contract"),
}

KERNELS = tuple(_TABLE)


def _resolve(kernel, cc):
    if kernel not in _TABLE:
        raise KeyError(
            "unknown kernel %r, expected one of %s" % (kernel, ", ".join(KERNELS)))
    min_cc, sym = _TABLE[kernel]
    if int(cc) < min_cc:
        raise FileNotFoundError(
            "no %s kernel for compute capability %d (needs %d+)" % (kernel, int(cc), min_cc))
    return sym


def library_path(kernel, cc):
    """Give the path of the library that supplies one kernel."""
    _resolve(kernel, cc)
    if not _LIBRARY.is_file():
        raise FileNotFoundError("%s is not in this package" % _LIBRARY)
    return str(_LIBRARY)


def symbol(kernel, cc):
    """Give the XLA FFI target name that the library exports."""
    return _resolve(kernel, cc)


def available(cc, kernel=None):
    """Tell if the kernels are present. With kernel=None, test all of them."""
    names = KERNELS if kernel is None else (kernel,)
    for name in names:
        try:
            library_path(name, cc)
        except (FileNotFoundError, KeyError):
            return False
    return True
