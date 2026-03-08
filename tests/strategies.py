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
    seed = draw(st.integers(0, 2**31 - 1))
    A = np.random.default_rng(seed).standard_normal((3, 3))
    # QR of standard-normal matrix gives Haar-uniform rotation
    Q_mat, _ = np.linalg.qr(A)
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
    seed = draw(st.integers(0, 2**31 - 1))
    A = np.random.default_rng(seed).standard_normal((dim, dim))
    # QR of standard-normal matrix gives Haar-uniform rotation
    Q_mat, _ = np.linalg.qr(A)
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


@st.composite
def nearly_collinear_3d(draw):
    """3-D point cloud lying near a single line (small perpendicular noise)."""
    n = draw(st.integers(5, 20))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction) + 1e-8
    t_vals = rng.uniform(-10, 10, size=n)
    noise_scale = draw(st.floats(1e-4, 1e-2))
    noise = rng.standard_normal((n, 3)) * noise_scale
    return np.outer(t_vals, direction) + noise


@st.composite
def nearly_coplanar_nd(draw, dim=None):
    """N-D point cloud lying near a (dim-1) hyperplane (small normal noise)."""
    if dim is None:
        dim = draw(st.integers(2, 6))
    n = draw(st.integers(dim + 2, dim * 4 + 4))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    P = rng.uniform(-10, 10, size=(n, dim))
    noise_scale = draw(st.floats(1e-4, 1e-2))
    P[:, -1] = rng.standard_normal(n) * noise_scale
    return P


@st.composite
def near_duplicate_cloud(draw):
    """Pair (P, Q) where most points nearly coincide (small displacement)."""
    dim = draw(st.integers(2, 6))
    n = draw(st.integers(dim + 2, dim * 4 + 4))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    P = rng.uniform(-10, 10, size=(n, dim))
    noise_scale = draw(st.floats(1e-4, 1e-2))
    Q = P + rng.standard_normal((n, dim)) * noise_scale
    return P, Q


@st.composite
def extreme_scale_cloud(draw):
    """Pair (P, Q) with very large or very small coordinate magnitudes."""
    dim = draw(st.integers(2, 6))
    n = draw(st.integers(dim + 2, dim * 4 + 4))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    scale = draw(st.sampled_from([1e-6, 1e-3, 1e3, 1e6]))
    P = rng.standard_normal((n, dim)) * scale
    Q = rng.standard_normal((n, dim)) * scale
    return P, Q


@st.composite
def batched_point_clouds(draw, batch_dims=None, dim=None):
    """Batch of point clouds with arbitrary leading dims, shape [..., N, D]."""
    if dim is None:
        dim = draw(st.integers(2, 4))
    if batch_dims is None:
        n_batch = draw(st.integers(1, 2))
        batch_dims = tuple(draw(st.integers(2, 4)) for _ in range(n_batch))
    n = draw(st.integers(dim + 2, 12))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    shape = (*batch_dims, n, dim)
    return rng.uniform(-10, 10, size=shape)
