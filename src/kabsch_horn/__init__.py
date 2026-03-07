"""Kabsch-Umeyama Algorithm Implementation across Frameworks."""

# Attempt to load backends, fallback silently if not present
try:
    from .numpy.horn_quat_3d import horn as horn_numpy
    from .numpy.horn_quat_3d import horn_with_scale as horn_with_scale_numpy
    from .numpy.kabsch_svd_nd import kabsch as kabsch_numpy
    from .numpy.kabsch_svd_nd import kabsch_umeyama as kabsch_umeyama_numpy
except ImportError:
    pass
try:
    from .pytorch.horn_quat_3d import horn as horn_torch
    from .pytorch.horn_quat_3d import horn_with_scale as horn_with_scale_torch
    from .pytorch.kabsch_svd_nd import kabsch as kabsch_torch
    from .pytorch.kabsch_svd_nd import kabsch_rmsd as kabsch_rmsd_torch
    from .pytorch.kabsch_svd_nd import kabsch_umeyama as kabsch_umeyama_torch
    from .pytorch.kabsch_svd_nd import (
        kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_torch,
    )
except ImportError:
    pass

try:
    from .jax.horn_quat_3d import horn as horn_jax
    from .jax.horn_quat_3d import horn_with_scale as horn_with_scale_jax
    from .jax.kabsch_svd_nd import kabsch as kabsch_jax
    from .jax.kabsch_svd_nd import kabsch_rmsd as kabsch_rmsd_jax
    from .jax.kabsch_svd_nd import kabsch_umeyama as kabsch_umeyama_jax
    from .jax.kabsch_svd_nd import (
        kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_jax,
    )
except ImportError:
    pass

try:
    from .tensorflow.horn_quat_3d import horn as horn_tf
    from .tensorflow.horn_quat_3d import horn_with_scale as horn_with_scale_tf
    from .tensorflow.kabsch_svd_nd import kabsch as kabsch_tf
    from .tensorflow.kabsch_svd_nd import kabsch_rmsd as kabsch_rmsd_tf
    from .tensorflow.kabsch_svd_nd import kabsch_umeyama as kabsch_umeyama_tf
    from .tensorflow.kabsch_svd_nd import (
        kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_tf,
    )
except ImportError:
    pass

try:
    from .mlx.horn_quat_3d import horn as horn_mlx
    from .mlx.horn_quat_3d import horn_with_scale as horn_with_scale_mlx
    from .mlx.kabsch_svd_nd import kabsch as kabsch_mlx
    from .mlx.kabsch_svd_nd import kabsch_rmsd as kabsch_rmsd_mlx
    from .mlx.kabsch_svd_nd import kabsch_umeyama as kabsch_umeyama_mlx
    from .mlx.kabsch_svd_nd import (
        kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_mlx,
    )
except ImportError:
    pass

__all__ = [
    "horn_jax",
    "horn_mlx",
    "horn_numpy",
    "horn_tf",
    "horn_torch",
    "horn_with_scale_jax",
    "horn_with_scale_mlx",
    "horn_with_scale_numpy",
    "horn_with_scale_tf",
    "horn_with_scale_torch",
    "kabsch_jax",
    "kabsch_mlx",
    "kabsch_numpy",
    "kabsch_rmsd_jax",
    "kabsch_rmsd_mlx",
    "kabsch_rmsd_tf",
    "kabsch_rmsd_torch",
    "kabsch_tf",
    "kabsch_torch",
    "kabsch_umeyama_jax",
    "kabsch_umeyama_mlx",
    "kabsch_umeyama_numpy",
    "kabsch_umeyama_rmsd_jax",
    "kabsch_umeyama_rmsd_mlx",
    "kabsch_umeyama_rmsd_tf",
    "kabsch_umeyama_rmsd_torch",
    "kabsch_umeyama_tf",
    "kabsch_umeyama_torch",
]
