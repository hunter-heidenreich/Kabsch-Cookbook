# JAX

Custom VJP support with safe backward passes. Compatible with `jax.jit`.

Float64 requires `JAX_ENABLE_X64=True` set before importing JAX.

::: kabsch_horn.jax
    options:
      members:
        - kabsch
        - kabsch_umeyama
        - horn
        - horn_with_scale
        - kabsch_rmsd
        - kabsch_umeyama_rmsd
