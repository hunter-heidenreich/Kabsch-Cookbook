"""Kabsch-Umeyama Algorithm Implementation across Frameworks."""

from .numpy.horn_quat_3d import horn as horn_numpy
from .numpy.horn_quat_3d import horn_with_scale as horn_with_scale_numpy
from .numpy.kabsch_svd_nd import kabsch as kabsch_numpy
from .numpy.kabsch_svd_nd import kabsch_umeyama as kabsch_umeyama_numpy

# Attempt to load backends, fallback silently if not present
try:
    from .pytorch.horn_quat_3d import horn as horn_torch
    from .pytorch.horn_quat_3d import horn_with_scale as horn_with_scale_torch
    from .pytorch.kabsch_svd_nd import kabsch as kabsch_torch
    from .pytorch.kabsch_svd_nd import kabsch_umeyama as kabsch_umeyama_torch

    try:
        from .pytorch.kabsch_svd_nd import (
            kabsch_rmsd as kabsch_rmsd_torch,  # noqa: F401  # noqa: F401
        )
        from .pytorch.kabsch_svd_nd import (
            kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_torch,  # noqa: F401  # noqa: F401
        )
    except ImportError:
        pass
except ImportError:
    pass

try:
    from .jax.horn_quat_3d import horn as horn_jax
    from .jax.horn_quat_3d import horn_with_scale as horn_with_scale_jax
    from .jax.kabsch_svd_nd import kabsch as kabsch_jax
    from .jax.kabsch_svd_nd import kabsch_umeyama as kabsch_umeyama_jax

    try:
        from .jax.kabsch_svd_nd import (
            kabsch_rmsd as kabsch_rmsd_jax,  # noqa: F401  # noqa: F401
        )
        from .jax.kabsch_svd_nd import (
            kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_jax,  # noqa: F401  # noqa: F401
        )
    except ImportError:
        pass
except ImportError:
    pass

try:
    from .tensorflow.horn_quat_3d import horn as horn_tf
    from .tensorflow.horn_quat_3d import horn_with_scale as horn_with_scale_tf
    from .tensorflow.kabsch_svd_nd import kabsch as kabsch_tf
    from .tensorflow.kabsch_svd_nd import kabsch_umeyama as kabsch_umeyama_tf

    try:
        from .tensorflow.kabsch_svd_nd import (
            kabsch_rmsd as kabsch_rmsd_tf,  # noqa: F401  # noqa: F401
        )
        from .tensorflow.kabsch_svd_nd import (
            kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_tf,  # noqa: F401  # noqa: F401
        )
    except ImportError:
        pass
except ImportError:
    pass

try:
    from .mlx.horn_quat_3d import horn as horn_mlx
    from .mlx.horn_quat_3d import horn_with_scale as horn_with_scale_mlx
    from .mlx.kabsch_svd_nd import kabsch as kabsch_mlx
    from .mlx.kabsch_svd_nd import kabsch_umeyama as kabsch_umeyama_mlx

    try:
        from .mlx.kabsch_svd_nd import (
            kabsch_rmsd as kabsch_rmsd_mlx,  # noqa: F401  # noqa: F401
        )
        from .mlx.kabsch_svd_nd import (
            kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_mlx,  # noqa: F401  # noqa: F401
        )
    except ImportError:
        pass
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
    "kabsch_tf",
    "kabsch_torch",
    "kabsch_umeyama_jax",
    "kabsch_umeyama_mlx",
    "kabsch_umeyama_numpy",
    "kabsch_umeyama_tf",
    "kabsch_umeyama_torch",
]

# Add conditional exports dynamically to __all__
add_dyn = []
for name in [
    "kabsch_rmsd_torch",
    "kabsch_umeyama_rmsd_torch",
    "kabsch_rmsd_jax",
    "kabsch_umeyama_rmsd_jax",
    "kabsch_rmsd_tf",
    "kabsch_umeyama_rmsd_tf",
    "kabsch_rmsd_mlx",
    "kabsch_umeyama_rmsd_mlx",
]:
    if name in locals():
        add_dyn.append(name)
__all__.extend(add_dyn)
