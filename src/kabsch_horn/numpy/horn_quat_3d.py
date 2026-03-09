import numpy as np


def _build_horn_matrix(H: np.ndarray) -> np.ndarray:
    """Build the 4x4 symmetric N matrix from cross-covariance H.

    Args:
        H: Cross-covariance matrices, shape (B, 3, 3).

    Returns:
        N matrix, shape (B, 4, 4).
    """
    S = H + H.transpose(0, 2, 1)
    tr = H.diagonal(axis1=-2, axis2=-1).sum(-1)  # (B,)

    Delta = np.stack(
        [
            H[..., 1, 2] - H[..., 2, 1],
            H[..., 2, 0] - H[..., 0, 2],
            H[..., 0, 1] - H[..., 1, 0],
        ],
        axis=-1,
    )

    B = H.shape[0]
    I3 = np.broadcast_to(np.eye(3), (B, 3, 3))

    top_row = np.concatenate([tr[..., np.newaxis], Delta], axis=-1)[:, np.newaxis, :]
    bottom_block = np.concatenate(
        [Delta[:, :, np.newaxis], S - tr[:, np.newaxis, np.newaxis] * I3], axis=-1
    )

    return np.concatenate([top_row, bottom_block], axis=-2)


def _quat_to_rotation(q_opt: np.ndarray) -> np.ndarray:
    """Convert unit quaternions (B, 4) to rotation matrices (B, 3, 3)."""
    qw = q_opt[..., 0]
    qx = q_opt[..., 1]
    qy = q_opt[..., 2]
    qz = q_opt[..., 3]

    R11 = 1 - 2 * (qy**2 + qz**2)
    R12 = 2 * (qx * qy - qw * qz)
    R13 = 2 * (qx * qz + qw * qy)
    R21 = 2 * (qx * qy + qw * qz)
    R22 = 1 - 2 * (qx**2 + qz**2)
    R23 = 2 * (qy * qz - qw * qx)
    R31 = 2 * (qx * qz - qw * qy)
    R32 = 2 * (qy * qz + qw * qx)
    R33 = 1 - 2 * (qx**2 + qy**2)

    return np.stack(
        [
            np.stack([R11, R12, R13], axis=-1),
            np.stack([R21, R22, R23], axis=-1),
            np.stack([R31, R32, R33], axis=-1),
        ],
        axis=-2,
    )


def horn(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    if P.shape[-1] != 3:
        raise ValueError("Horn's method is strictly for 3D point clouds")
    if P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")

    is_single = P.ndim == 2
    if is_single:
        P = P[np.newaxis, ...]
        Q = Q[np.newaxis, ...]

    orig_shape = P.shape
    batch_dims = orig_shape[:-2]
    N, D = orig_shape[-2:]

    P = np.reshape(P, (-1, N, D))
    Q = np.reshape(Q, (-1, N, D))

    centroid_P = np.mean(P, axis=1, keepdims=True)
    centroid_Q = np.mean(Q, axis=1, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = np.matmul(p.transpose(0, 2, 1), q)

    N_mat = _build_horn_matrix(H)

    # eigh returns eigenvalues in ascending order
    _L, V = np.linalg.eigh(N_mat)
    q_opt = V[..., -1]

    R = _quat_to_rotation(q_opt)

    t = np.squeeze(centroid_Q, axis=1) - np.squeeze(
        np.matmul(centroid_P, R.transpose(0, 2, 1)), axis=1
    )

    aligned = np.matmul(p, R.transpose(0, 2, 1))
    rmsd = np.sqrt(
        np.clip(
            np.sum(np.square(aligned - q), axis=(1, 2)) / N,
            a_min=0.0,
            a_max=None,
        )
    )

    if is_single:
        return R[0], t[0], rmsd[0]
    return (
        R.reshape(*batch_dims, D, D),
        t.reshape(*batch_dims, D),
        rmsd.reshape(*batch_dims),
    )


def horn_with_scale(
    P: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    if P.shape[-1] != 3:
        raise ValueError("Horn's method is strictly for 3D point clouds")
    if P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")

    is_single = P.ndim == 2
    if is_single:
        P = P[np.newaxis, ...]
        Q = Q[np.newaxis, ...]

    orig_shape = P.shape
    batch_dims = orig_shape[:-2]
    N, D = orig_shape[-2:]

    P = np.reshape(P, (-1, N, D))
    Q = np.reshape(Q, (-1, N, D))

    centroid_P = np.mean(P, axis=1, keepdims=True)
    centroid_Q = np.mean(Q, axis=1, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    var_P = np.sum(np.square(p), axis=(1, 2)) / N

    H = np.matmul(p.transpose(0, 2, 1), q) / N

    N_mat = _build_horn_matrix(H)

    _L, V = np.linalg.eigh(N_mat)
    q_opt = V[..., -1]

    R = _quat_to_rotation(q_opt)

    RH = np.sum(R * H.transpose(0, 2, 1), axis=(1, 2))
    c = RH / np.clip(var_P, a_min=1e-12, a_max=None)

    t = np.squeeze(centroid_Q, axis=1) - c[:, np.newaxis] * np.squeeze(
        np.matmul(centroid_P, R.transpose(0, 2, 1)), axis=1
    )

    aligned_P = (
        c[:, np.newaxis, np.newaxis] * np.matmul(P, R.transpose(0, 2, 1))
        + t[:, np.newaxis, :]
    )
    diff = aligned_P - Q
    rmsd = np.sqrt(
        np.clip(np.sum(np.square(diff), axis=(1, 2)) / N, a_min=0.0, a_max=None)
    )

    if is_single:
        return R[0], t[0], c[0], rmsd[0]
    return (
        R.reshape(*batch_dims, D, D),
        t.reshape(*batch_dims, D),
        c.reshape(*batch_dims),
        rmsd.reshape(*batch_dims),
    )
