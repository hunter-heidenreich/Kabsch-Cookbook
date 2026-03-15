import numpy as np


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the optimal rotation and translation to align P to Q.

    Args:
        P: Source points, shape [..., N, D].
        Q: Target points, shape [..., N, D].

    Returns:
        (R, t, rmsd): Rotation [..., D, D], translation [..., D], RMSD [...].

    Note:
        R is only stable under global translation when the cross-covariance matrix
        H = P_c.T @ Q_c is well-conditioned. When the smallest singular value of H
        is near zero, U and V from the SVD are not unique, and a small perturbation
        can select a different rotation. Check the singular values of H if rotation
        stability matters for your use case.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    if P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")

    orig_dtype = P.dtype
    if P.dtype != Q.dtype:
        # Mixed dtypes: promote to higher precision
        target = np.float64 if np.float64 in (P.dtype, Q.dtype) else np.float32
        P = P.astype(target)
        Q = Q.astype(target)
        orig_dtype = target
    elif orig_dtype in (np.float16,):
        P = P.astype(np.float32)
        Q = Q.astype(np.float32)

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
    centroid_P = np.mean(P, axis=1, keepdims=True)  # Bx1xD
    centroid_Q = np.mean(Q, axis=1, keepdims=True)  # Bx1xD

    # Center the points
    p = P - centroid_P  # BxNxD
    q = Q - centroid_Q  # BxNxD

    # Compute the covariance matrix
    H = np.matmul(p.transpose(0, 2, 1), q)  # BxDxD

    # SVD
    U, _, Vt = np.linalg.svd(H)  # BxDxD

    # Validate right-handed coordinate system
    d = np.linalg.det(np.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1)))

    # Correction array; treat d==0 as positive (non-reflection)
    d_sign = np.where(d == 0, 1.0, np.sign(d))

    # Optimal rotation
    B_diag = np.ones((*d_sign.shape, D), dtype=P.dtype)
    B_diag[..., -1] = d_sign
    R = np.matmul(
        Vt.transpose(0, 2, 1) * B_diag[:, np.newaxis, :],
        U.transpose(0, 2, 1),
    )

    # Optimal translation
    t = centroid_Q.squeeze(1) - np.squeeze(
        np.matmul(centroid_P, R.transpose(0, 2, 1)), 1
    )

    # RMSD
    aligned_P = np.matmul(P, R.transpose(0, 2, 1)) + t[:, np.newaxis, :]
    rmsd = np.sqrt(
        np.clip(
            np.sum(np.square(aligned_P - Q), axis=(1, 2)) / N,
            a_min=0.0,
            a_max=None,
        )
    )

    if is_single:
        R, t, rmsd = R[0], t[0], rmsd[0]
    else:
        R = R.reshape(*batch_dims, D, D)
        t = t.reshape(*batch_dims, D)
        rmsd = rmsd.reshape(*batch_dims)
    if orig_dtype in (np.float16,):
        R = R.astype(orig_dtype)
        t = t.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)
    return R, t, rmsd


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

    Note:
        Unlike kabsch, the cross-covariance H is divided by N here. This per-point
        normalization is required by the Umeyama scale estimator
        (c = trace(S * D) / var_P) and does not affect the rotation or translation.

        R is only stable under global translation and uniform scaling when the
        cross-covariance matrix H = P_c.T @ Q_c is well-conditioned. When the
        smallest singular value of H is near zero, U and V from the SVD are not
        unique, and a small perturbation can select a different rotation. Check
        the singular values of H if rotation stability matters for your use case.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    if P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")

    orig_dtype = P.dtype
    if P.dtype != Q.dtype:
        # Mixed dtypes: promote to higher precision
        target = np.float64 if np.float64 in (P.dtype, Q.dtype) else np.float32
        P = P.astype(target)
        Q = Q.astype(target)
        orig_dtype = target
    elif orig_dtype in (np.float16,):
        P = P.astype(np.float32)
        Q = Q.astype(np.float32)

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

    # Cross-covariance matrix (divided by N for Umeyama scale estimation)
    H = np.matmul(p.transpose(0, 2, 1), q) / N  # BxDxD

    # Variances
    var_P = np.sum(np.square(p), axis=(1, 2)) / N  # B

    # SVD
    U, S, Vt = np.linalg.svd(H)  # BxDxD, BxD, BxDxD

    # Validate right-handed coordinate system
    V = Vt.transpose(0, 2, 1)
    d = np.linalg.det(np.matmul(V, U.transpose(0, 2, 1)))

    # Correction array; treat d==0 as positive (non-reflection)
    d_sign = np.where(d == 0, 1.0, np.sign(d))

    # S factor
    S_corr = np.ones((*d_sign.shape, D), dtype=P.dtype)
    S_corr[..., -1] = d_sign

    # Scale
    _eps = np.finfo(P.dtype).eps
    c = np.sum(S * S_corr, axis=-1) / np.clip(var_P, a_min=_eps, a_max=None)  # B

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
    rmsd = np.sqrt(
        np.clip(
            np.sum(np.square(aligned_P - Q), axis=(1, 2)) / N,
            a_min=0.0,
            a_max=None,
        )
    )

    if is_single:
        R, t, c, rmsd = R[0], t[0], c[0], rmsd[0]
    else:
        R = R.reshape(*batch_dims, D, D)
        t = t.reshape(*batch_dims, D)
        c = c.reshape(*batch_dims)
        rmsd = rmsd.reshape(*batch_dims)
    if orig_dtype in (np.float16,):
        R = R.astype(orig_dtype)
        t = t.astype(orig_dtype)
        c = c.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)
    return R, t, c, rmsd
