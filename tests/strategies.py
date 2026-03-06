import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@st.composite
def point_clouds_nd(draw, dim=None, n_points=None):
    """Random N-D point cloud (float64, bounded, no NaN/inf)."""
    if dim is None:
        dim = draw(st.integers(2, 6))
    if n_points is None:
        n_points = draw(st.integers(dim + 2, dim * 4 + 4))
    return draw(
        arrays(
            dtype=np.float64,
            shape=(n_points, dim),
            elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
        )
    )


@st.composite
def point_clouds_3d(draw, n_points=None):
    """Random 3-D point cloud (float64, bounded, no NaN/inf)."""
    return draw(point_clouds_nd(dim=3, n_points=n_points))


@st.composite
def aligned_pair_3d(draw):
    """P + known proper rotation + translation = Q."""
    P = draw(point_clouds_3d())
    A = draw(
        arrays(
            np.float64,
            (3, 3),
            elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
        )
    )
    # QR decomposition gives a Haar-uniform rotation; offset avoids singular A
    Q_mat, _ = np.linalg.qr(A + np.eye(3) * 0.1)
    if np.linalg.det(Q_mat) < 0:
        Q_mat[:, 0] *= -1
    t = draw(
        arrays(
            np.float64,
            (3,),
            elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
        )
    )
    Q = P @ Q_mat.T + t
    return P, Q_mat, t, Q


@st.composite
def aligned_pair_nd(draw):
    """N-D version for kabsch/umeyama."""
    dim = draw(st.integers(2, 6))
    P = draw(point_clouds_nd(dim=dim))
    A = draw(
        arrays(
            np.float64,
            (dim, dim),
            elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
        )
    )
    Q_mat, _ = np.linalg.qr(A + np.eye(dim) * 0.1)
    if np.linalg.det(Q_mat) < 0:
        Q_mat[:, 0] *= -1
    t = draw(
        arrays(
            np.float64,
            (dim,),
            elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
        )
    )
    Q = P @ Q_mat.T + t
    return P, Q_mat, t, Q, dim
