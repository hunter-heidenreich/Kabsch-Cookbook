import os

import numpy as np
import pytest

os.environ["JAX_ENABLE_X64"] = "True"


@pytest.fixture(params=[2, 3, 4, 10, 100], ids=lambda x: f"{x}D")
def dim(request) -> int:
    return request.param


def _get_random_rotation(rng, dim: int) -> np.ndarray:
    # Generate random orthogonal matrix (Haar measure) using QR decomposition
    A = rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    ph = d / np.abs(d)
    rot = Q * ph
    # Ensure it's a proper rotation (determinant +1)
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    return rot


@pytest.fixture
def identity_points(dim) -> np.ndarray:
    rng = np.random.default_rng(42)
    # Give enough points so we don't under-constrain Kabsch in higher dimensions
    n_points = max(10, dim * 2)
    return rng.random((n_points, dim))


@pytest.fixture
def known_transform_points(
    dim,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    n_points = max(10, dim * 2)
    P = rng.random((n_points, dim))

    R_true = _get_random_rotation(rng, dim)
    t_true = rng.random((dim,)) * 5.0 - 2.5
    c_true = 2.5

    Q_kabsch = P @ R_true.T + t_true
    Q_umeyama = c_true * (P @ R_true.T) + t_true

    return P, Q_kabsch, Q_umeyama, R_true, t_true, c_true


@pytest.fixture
def coplanar_points(dim) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_points = max(10, dim * 2)
    P = rng.random((n_points, dim))
    # Make points lie on a dim-1 dimensional subspace
    P[:, -1] = 0.0

    R = _get_random_rotation(rng, dim)
    t = rng.random((dim,))

    Q = P @ R.T + t
    return P, Q


@pytest.fixture
def collinear_points(dim) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    # Ensure we have enough points, though they are collinear
    n_points = max(10, dim * 2)
    P = np.zeros((n_points, dim))
    # All points lie on x-axis
    P[:, 0] = rng.random(n_points) * 10.0

    R = _get_random_rotation(rng, dim)
    t = rng.random((dim,))

    Q = P @ R.T + t
    return P, Q


@pytest.fixture
def perfect_cube(dim) -> tuple[np.ndarray, np.ndarray]:
    # In n-D, generating 2^n vertices of a hypercube grows too fast
    # (2^100 is impossible).
    # So we'll limit the "hypercube" to just points representing bounding box corners,
    # or just use 2 * dim orthogonal vertices + something else.
    # Actually, a generalized simplex or cross-polytope is easier to construct linearly.
    # Let's just create points at +/- 1 along each axis, which gives 2*dim points.

    points = []
    for i in range(dim):
        p1, p2 = np.zeros(dim), np.zeros(dim)
        p1[i] = -1.0
        p2[i] = 1.0
        points.append(p1)
        points.append(p2)
    P = np.array(points, dtype=np.float64)
    Q = P + 0.5
    return P, Q


@pytest.fixture
def reflected_points(dim) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_points = max(10, dim * 2)
    P = rng.random((n_points, dim))
    # Target is mirrored across first dimension, det = -1
    Q = np.copy(P)
    Q[:, 0] = -Q[:, 0]
    return P, Q


@pytest.fixture
def batch_points(dim) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_points = max(10, dim * 2)
    P = rng.random((5, n_points, dim))  # Batch size 5
    Q = P + rng.random((5, 1, dim))
    return P, Q


@pytest.fixture
def nd_batch_points(dim) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_points = max(10, dim * 2)
    P = rng.random((2, 3, n_points, dim))  # Batch size 2x3
    Q = P + rng.random((2, 3, 1, dim))
    return P, Q


@pytest.fixture(autouse=True)
def skip_unsupported_dims(request: pytest.FixtureRequest) -> None:
    """
    Automatically skips tests where the requested framework adapter
    does not support the requested spatial dimension.
    """
    if "dim" in request.fixturenames and "adapter" in request.fixturenames:
        dim = request.getfixturevalue("dim")
        adapter = request.getfixturevalue("adapter")
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
