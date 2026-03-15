import warnings
from contextlib import contextmanager

import mlx.core as mx

_DTYPE_EPS = {
    mx.float16: 9.765625e-4,
    mx.bfloat16: 7.8125e-3,
    mx.float32: 1.1920929e-7,
    mx.float64: 2.220446049250313e-16,
}


def _warn_if_float64(P: mx.array, Q: mx.array, stacklevel: int = 3) -> None:
    """Emit a UserWarning if either P or Q is float64.

    MLX does not support float64 on GPU. When float64 inputs are detected,
    operations are temporarily routed to CPU via _float64_device_guard.
    stacklevel=3 is correct for direct callers (kabsch, horn, etc.).
    For rmsd wrappers (kabsch_rmsd -> kabsch -> here), stacklevel points
    one level into the wrapper rather than the user call site; this is an
    accepted limitation.
    """
    if P.dtype == mx.float64 or Q.dtype == mx.float64:
        warnings.warn(
            "MLX does not support float64 on GPU; operations will temporarily "
            "run on CPU for this call.",
            UserWarning,
            stacklevel=stacklevel,
        )


@contextmanager
def _float64_device_guard(*tensors):
    """Temporarily set the default device to CPU if any tensor is float64.

    Restores the original default device on exit.
    """
    needs_cpu = any(t.dtype == mx.float64 for t in tensors)
    if needs_cpu:
        original = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            yield
        finally:
            mx.set_default_device(original)
    else:
        yield
