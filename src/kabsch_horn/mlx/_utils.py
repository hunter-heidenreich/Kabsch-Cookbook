import warnings

import mlx.core as mx


def _warn_if_float64(P: mx.array, Q: mx.array, stacklevel: int = 3) -> None:
    """Emit a UserWarning if either P or Q is float64.

    MLX does not support float64 on GPU. Passing float64 inputs sets the
    process-wide default device to CPU (mx.set_default_device(mx.cpu)).
    This is a persistent side effect: all subsequent MLX operations in the
    process will also run on CPU until the default device is changed again.
    stacklevel=3 is correct for direct callers (kabsch, horn, etc.).
    For rmsd wrappers (kabsch_rmsd -> kabsch -> here), stacklevel points
    one level into the wrapper rather than the user call site; this is an
    accepted limitation.
    """
    if P.dtype == mx.float64 or Q.dtype == mx.float64:
        warnings.warn(
            "MLX does not support float64 on GPU; falling back to CPU. "
            "This sets the process-wide default device to CPU "
            "(mx.set_default_device(mx.cpu)) and will affect all subsequent "
            "MLX operations in this process.",
            UserWarning,
            stacklevel=stacklevel,
        )
        mx.set_default_device(mx.cpu)
