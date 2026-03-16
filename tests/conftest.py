import os

import numpy as np
import pytest

os.environ["JAX_ENABLE_X64"] = "True"

# Algorithm constants -- imported by test modules for parametrization
ALGORITHMS = ["kabsch", "umeyama", "horn", "horn_with_scale"]
ALGORITHMS_WITH_SCALE = {"umeyama", "horn_with_scale"}
ALGORITHMS_3D_ONLY = {"horn", "horn_with_scale"}


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="Run full test suite (all precisions, full Hypothesis examples)",
    )


def pytest_configure(config):
    if not config.getoption("--full", default=False):
        os.environ["KABSCH_TEST_FAST"] = "1"


@pytest.fixture(params=[2, 3, 4], ids=lambda x: f"{x}D")
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
    c_true = float(rng.uniform(0.5, 5.0))

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
    Q = np.empty_like(P)
    for b in range(5):
        R_b = _get_random_rotation(rng, dim)
        t_b = rng.random((dim,))
        Q[b] = P[b] @ R_b.T + t_b
    return P, Q


@pytest.fixture
def nd_batch_points(dim) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_points = max(10, dim * 2)
    P = rng.random((2, 3, n_points, dim))  # Batch size 2x3
    Q = np.empty_like(P)
    for i in range(2):
        for j in range(3):
            R_b = _get_random_rotation(rng, dim)
            t_b = rng.random((dim,))
            Q[i, j] = P[i, j] @ R_b.T + t_b
    return P, Q


def pytest_collection_modifyitems(session, config, items) -> None:
    """
    Filters out tests where the requested framework adapter
    does not support the requested spatial dimension.

    Also drops gradient-only test modules for forward-only adapters
    (e.g. NumPy), and when ``--full`` is not passed, skips
    float16/bfloat16 adapter tests except for dtype-preservation tests
    (which exist specifically to verify the upcast path).

    Note: Hypothesis tests parametrised by `adapter` but with `dim` drawn
    inside `@given` are not filtered here -- they guard themselves with
    `assume(adapter.supports_dim(dim))` inside the test body.
    """
    full = config.getoption("--full", default=False)
    kept = []
    for item in items:
        # Check if the test has a callspec (i.e. is parametrized)
        if hasattr(item, "callspec"):
            params = item.callspec.params
            # Skip gradient-only test modules for forward-only adapters
            if "adapter" in params and not params["adapter"].supports_grad:
                module_name = item.module.__name__
                if module_name in {
                    "test_differentiability_traps",
                    "test_gradient_verification",
                    "test_rmsd_wrappers",
                }:
                    continue
            # Skip MLX on unsupported dims,
            # unless the test explicitly checks rejection behaviour.
            if "dim" in params and "adapter" in params:
                dim = params["dim"]
                adapter = params["adapter"]
                # Convention: tests that check rejection behaviour for non-3D
                # inputs must include "non_3d" in their test name so they
                # bypass this skip and actually run with the unsupported dim.
                if not adapter.supports_dim(dim) and "non_3d" not in item.name:
                    continue
            # Skip 3D-only algorithms (Horn) for non-3D dims,
            # unless the test explicitly checks that rejection behaviour.
            if "algo" in params and "dim" in params:
                if params["algo"] in ("horn", "horn_with_scale") and params["dim"] != 3:
                    if "non_3d" not in item.name:
                        continue
            # Skip float16/bfloat16 except dtype-preservation tests
            if not full and "adapter" in params:
                adapter = params["adapter"]
                if (
                    hasattr(adapter, "precision")
                    and adapter.precision in ("float16", "bfloat16")
                    and "preserves_input_dtype" not in item.name
                    and "preserves_dtype" not in item.name
                    and "float16" not in item.name.split("[")[0].lower()
                    and "dtype" not in item.name.split("[")[0].lower()
                ):
                    continue
        kept.append(item)

    items[:] = kept
