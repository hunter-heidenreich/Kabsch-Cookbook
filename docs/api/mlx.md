# MLX

Metal-accelerated on Apple Silicon. Kabsch is restricted to 3D inputs (`dim == 3`).

Float64 operations run on CPU (Apple Silicon GPUs do not support true float64).

::: kabsch_horn.mlx
    options:
      members:
        - kabsch
        - kabsch_umeyama
        - horn
        - horn_with_scale
        - kabsch_rmsd
        - kabsch_umeyama_rmsd
