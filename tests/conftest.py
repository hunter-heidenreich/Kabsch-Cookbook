import os

import numpy as np
import pytest

os.environ["JAX_ENABLE_X64"] = "True"


@pytest.fixture
def identity_points() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((10, 3))


@pytest.fixture
def known_transform_points() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    rng = np.random.default_rng(42)
    P = rng.random((10, 3))

    # 90 degrees around Z axis
    theta = np.pi / 2
    R_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    t_true = np.array([1.0, -2.0, 3.0])
    c_true = 2.5

    Q_kabsch = P @ R_true.T + t_true
    Q_umeyama = c_true * (P @ R_true.T) + t_true

    return P, Q_kabsch, Q_umeyama, R_true, t_true, c_true


@pytest.fixture
def coplanar_points() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    P = rng.random((10, 3))
    P[:, 2] = 0.0  # Z is zero

    # Random gentle turn around Z
    theta = np.pi / 4
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    Q = P @ R.T + np.array([1.0, 1.0, 0.0])
    return P, Q


@pytest.fixture
def collinear_points() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    P = rng.random((10, 3))
    P[:, 1] = 0.0  # Y is zero
    P[:, 2] = 0.0  # Z is zero
    Q = P + np.array([2.0, 0.0, 0.0])
    return P, Q


@pytest.fixture
def perfect_cube() -> tuple[np.ndarray, np.ndarray]:
    # Symmetric shape
    P = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    Q = P + np.array([0.5, 0.5, 0.5])
    return P, Q


@pytest.fixture
def reflected_points() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    P = rng.random((10, 3))
    # Target is mirrored across X-axis, det = -1
    Q = np.copy(P)
    Q[:, 0] = -Q[:, 0]
    return P, Q


@pytest.fixture
def batch_points() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    P = rng.random((5, 10, 3))  # Batch size 5
    Q = P + rng.random((5, 1, 3))
    return P, Q
