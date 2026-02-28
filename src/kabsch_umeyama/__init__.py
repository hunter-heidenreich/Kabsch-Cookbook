"""Kabsch-Umeyama Algorithm Implementation across Frameworks."""

from .numpy import kabsch as kabsch_numpy
from .numpy import kabsch_umeyama as kabsch_umeyama_numpy

# Attempt to load backends, fallback silently if not present
try:
    from .pytorch import kabsch as kabsch_torch
    from .pytorch import kabsch_rmsd as kabsch_rmsd_torch
    from .pytorch import kabsch_umeyama as kabsch_umeyama_torch
    from .pytorch import kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_torch
except ImportError:
    pass

try:
    from .jax import kabsch as kabsch_jax
    from .jax import kabsch_rmsd as kabsch_rmsd_jax
    from .jax import kabsch_umeyama as kabsch_umeyama_jax
    from .jax import kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_jax
except ImportError:
    pass

try:
    from .tensorflow import kabsch as kabsch_tf
    from .tensorflow import kabsch_rmsd as kabsch_rmsd_tf
    from .tensorflow import kabsch_umeyama as kabsch_umeyama_tf
    from .tensorflow import kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_tf
except ImportError:
    pass

try:
    from .mlx import kabsch as kabsch_mlx
    from .mlx import kabsch_rmsd as kabsch_rmsd_mlx
    from .mlx import kabsch_umeyama as kabsch_umeyama_mlx
    from .mlx import kabsch_umeyama_rmsd as kabsch_umeyama_rmsd_mlx
except ImportError:
    pass

__all__ = [
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
