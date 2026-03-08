import numpy as np


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the optimal rotation and translation to align P to Q.

    Args:
        P: Source points, shape [..., N, D].
        Q: Target points, shape [..., N, D].

    Returns:
        (R, t, rmsd): Rotation [..., D, D], translation [..., D], RMSD [...].
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )

    # Auto-batch single elements
    is_single = P.ndim == 2
    if is_single:
        P = P[np.newaxis, ...]
        Q = Q[np.newaxis, ...]

    orig_shape = P.shape
    batch_dims = orig_shape[:-2]
    N, D = orig_shape[-2:]

    P = np.reshape(P, (-1, N, D))
    Q = np.reshape(Q, (-1, N, D))

    # Compute centroids
    centroid_P = np.mean(P, axis=1, keepdims=True)  # Bx1x3
    centroid_Q = np.mean(Q, axis=1, keepdims=True)  # Bx1x3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Compute the covariance matrix
    H = np.matmul(p.transpose(0, 2, 1), q)  # Bx3x3

    # SVD
    U, _, Vt = np.linalg.svd(H)  # Bx3x3

    # Validate right-handed coordinate system
    d = np.linalg.det(np.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1)))

    # Correction array
    d_sign = np.sign(d)
    # If d is exactly 0, sign is 0, but we want 1 for non-reflection
    d_sign[d_sign == 0] = 1.0

    # Optimal rotation
    R = np.matmul(
        Vt.transpose(0, 2, 1)
        * np.stack([np.ones_like(d_sign)] * (D - 1) + [d_sign], axis=-1)[
            :, np.newaxis, :
        ],
        U.transpose(0, 2, 1),
    )

    # Optimal translation
    t = centroid_Q.squeeze(1) - np.squeeze(
        np.matmul(centroid_P, R.transpose(0, 2, 1)), 1
    )

    # RMSD
    aligned_P = np.matmul(P, R.transpose(0, 2, 1)) + t[:, np.newaxis, :]
    rmsd = np.sqrt(np.sum(np.square(aligned_P - Q), axis=(1, 2)) / P.shape[1])

    if is_single:
        return R[0], t[0], rmsd[0]
    return (
        R.reshape(*batch_dims, D, D),
        t.reshape(*batch_dims, D),
        rmsd.reshape(*batch_dims),
    )


def kabsch_umeyama(
    P: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the optimal rotation, translation, and scale to align P to Q
    (Q ~ c * R @ P + t).

    Args:
        P: Source points, shape [..., N, D].
        Q: Target points, shape [..., N, D].

    Returns:
        (R, t, c, rmsd): Rotation [..., D, D], translation [..., D], scale [...],
        RMSD [...].
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )

    is_single = P.ndim == 2
    if is_single:
        P = P[np.newaxis, ...]
        Q = Q[np.newaxis, ...]

    orig_shape = P.shape
    batch_dims = orig_shape[:-2]
    N, D = orig_shape[-2:]

    P = np.reshape(P, (-1, N, D))
    Q = np.reshape(Q, (-1, N, D))

    # Compute centroids
    centroid_P = np.mean(P, axis=1, keepdims=True)  # Bx1xD
    centroid_Q = np.mean(Q, axis=1, keepdims=True)  # Bx1xD

    # Center the points
    p = P - centroid_P  # BxNxD
    q = Q - centroid_Q  # BxNxD

    # Cross-covariance matrix
    H = np.matmul(p.transpose(0, 2, 1), q) / N  # BxDxD

    # Variances
    var_P = np.sum(np.square(p), axis=(1, 2)) / N  # B

    # SVD
    U, S, Vt = np.linalg.svd(H)  # BxDxD, BxD, BxDxD

    # Validate right-handed coordinate system
    V = Vt.transpose(0, 2, 1)
    d = np.linalg.det(np.matmul(V, U.transpose(0, 2, 1)))

    # Correction array
    d_sign = np.sign(d)
    d_sign[d_sign == 0] = 1.0

    # S factor
    S_corr = np.stack([np.ones_like(d_sign)] * (D - 1) + [d_sign], axis=-1)  # BxD

    # Scale
    c = np.sum(S * S_corr, axis=-1) / np.clip(var_P, a_min=1e-12, a_max=None)  # B

    # Optimal rotation
    R = np.matmul(V * S_corr[:, np.newaxis, :], U.transpose(0, 2, 1))  # BxDxD

    # Optimal translation
    t = centroid_Q.squeeze(1) - c[:, np.newaxis] * np.squeeze(
        np.matmul(centroid_P, R.transpose(0, 2, 1)), 1
    )

    # RMSD
    aligned_P = (
        c[:, np.newaxis, np.newaxis] * np.matmul(P, R.transpose(0, 2, 1))
        + t[:, np.newaxis, :]
    )
    rmsd = np.sqrt(np.sum(np.square(aligned_P - Q), axis=(1, 2)) / N)

    if is_single:
        return R[0], t[0], c[0], rmsd[0]
    return (
        R.reshape(*batch_dims, D, D),
        t.reshape(*batch_dims, D),
        c.reshape(*batch_dims),
        rmsd.reshape(*batch_dims),
    )
