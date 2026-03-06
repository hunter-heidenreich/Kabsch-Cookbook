"""
External reference validation.

Validates our implementations against two independent external libraries:
- `rmsd` package (Kabsch algorithm, charnley/rmsd)
- `scipy.spatial.transform` (Horn-equivalent via Rotation.align_vectors)
"""

import numpy as np
import pytest
import rmsd as rmsd_lib
from adapters import FrameworkAdapter, frameworks
from scipy.spatial.transform import Rotation

from kabsch_horn import numpy as kabsch_np


def _reference_kabsch_3d(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Kabsch rotation via rmsd package (charnley/rmsd).

    rmsd.kabsch(P, Q) returns R s.t. P @ R aligns to Q (column-vector convention).
    Our convention: P @ R.T + t = Q, so our R == rmsd_R.T.
    We return our R directly by taking rmsd_R.T.
    """
    P_c = P - P.mean(0)
    Q_c = Q - Q.mean(0)
    R_rmsd = rmsd_lib.kabsch(P_c, Q_c)
    return R_rmsd.T


def _reference_horn_3d(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Horn-equivalent rotation via scipy.

    Rotation.align_vectors(a, b) finds R s.t. R @ b[i] ~= a[i].
    Our convention: P @ R.T + t = Q  =>  R @ P[i] ~= Q[i] - t.
    So align_vectors(Q_c, P_c) gives our R.
    """
    P_c = P - P.mean(0)
    Q_c = Q - Q.mean(0)
    result = Rotation.align_vectors(Q_c, P_c)
    return result[0].as_matrix()


class TestReferenceValidation:
    @pytest.mark.parametrize("adapter", frameworks)
    def test_kabsch_matches_rmsd_package(self, adapter: FrameworkAdapter) -> None:
        """Our kabsch rotation matches the rmsd package (seed=42, 3D)."""
        rng = np.random.default_rng(42)
        P_np = rng.random((20, 3))
        Q_np = rng.random((20, 3))

        R_ref = _reference_kabsch_3d(P_np, Q_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        res = adapter.kabsch(P, Q)
        R_ours = adapter.convert_out(res[0])

        np.testing.assert_allclose(R_ours, R_ref, atol=adapter.atol * 10)

    @pytest.mark.parametrize("adapter", frameworks)
    def test_horn_matches_scipy(self, adapter: FrameworkAdapter) -> None:
        """Our horn rotation matches scipy Rotation.align_vectors (seed=42, 3D)."""
        rng = np.random.default_rng(42)
        P_np = rng.random((20, 3))
        Q_np = rng.random((20, 3))

        R_ref = _reference_horn_3d(P_np, Q_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        res = adapter.horn(P, Q)
        R_ours = adapter.convert_out(res[0])

        np.testing.assert_allclose(R_ours, R_ref, atol=adapter.atol * 10)

    def test_umeyama_rotation_matches_rmsd_reference(self) -> None:
        """Umeyama rotation component matches rmsd kabsch rotation (numpy, seed=123)."""
        rng = np.random.default_rng(123)
        P_np = rng.random((20, 3))
        Q_np = rng.random((20, 3))

        R_ref = _reference_kabsch_3d(P_np, Q_np)
        R_ours, _, _, _ = kabsch_np.kabsch_umeyama(P_np, Q_np)

        np.testing.assert_allclose(R_ours, R_ref, atol=1e-5)

    def test_rmsd_value_matches_rmsd_package(self) -> None:
        """Our kabsch RMSD scalar matches rmsd.kabsch_rmsd (numpy, seed=42)."""
        rng = np.random.default_rng(42)
        P_np = rng.random((20, 3))
        Q_np = rng.random((20, 3))

        P_c = P_np - P_np.mean(0)
        Q_c = Q_np - Q_np.mean(0)
        rmsd_ref = rmsd_lib.kabsch_rmsd(P_c, Q_c)

        _, _, rmsd_ours = kabsch_np.kabsch(P_np, Q_np)

        assert float(rmsd_ours) == pytest.approx(float(rmsd_ref), abs=1e-5)
