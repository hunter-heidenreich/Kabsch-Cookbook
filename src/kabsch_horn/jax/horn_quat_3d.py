import jax
import jax.numpy as jnp


@jax.custom_vjp
def safe_eigh(A: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gradient-safe eigendecomposition for symmetric matrices. Masks near-zero
    eigenvalue differences (< eps) in the backward pass to prevent NaN gradients."""
    return jnp.linalg.eigh(A)


def _eigh_fwd(
    A: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple]:
    L, V = jnp.linalg.eigh(A)
    return (L, V), (L, V)


def _eigh_bwd(res: tuple, g: tuple) -> tuple:
    L, V = res
    grad_L, grad_V = g

    grad_L = (
        jnp.zeros_like(L)
        if isinstance(grad_L, jax.custom_derivatives.SymbolicZero)
        else grad_L
    )
    grad_V = (
        jnp.zeros_like(V)
        if isinstance(grad_V, jax.custom_derivatives.SymbolicZero)
        else grad_V
    )

    def mH(x):
        return jnp.swapaxes(jnp.conj(x), -1, -2)

    D = L[..., jnp.newaxis, :] - L[..., jnp.newaxis]

    eps = jnp.finfo(L.dtype).eps
    mask = jnp.abs(D) < eps
    safe_D = jnp.where(mask, jnp.where(D >= 0, eps, -eps), D)
    diag_mask = jnp.eye(D.shape[-1], dtype=bool)
    safe_D = jnp.where(diag_mask, 1.0, safe_D)
    F = jnp.where(diag_mask, 0.0, 1.0 / safe_D)

    Vt_dV = jnp.matmul(mH(V), grad_V)

    L_diag = grad_L[..., jnp.newaxis] * diag_mask.astype(grad_L.dtype)

    term = L_diag + F * (Vt_dV - mH(Vt_dV)) / 2
    grad_A = jnp.matmul(V, jnp.matmul(term, mH(V)))

    return (grad_A,)


safe_eigh.defvjp(_eigh_fwd, _eigh_bwd)


def _horn_core(
    P: jnp.ndarray, Q: jnp.ndarray, weights: jnp.ndarray | None = None
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Core Horn computation on batched [B, N, 3] float32 arrays.

    Returns (R, p, q, centroid_P, centroid_Q, H) where q = centered Q points
    and H = (w*p).T @ q (weighted, unscaled) or p.T @ q (unweighted, unscaled).
    """
    if weights is not None:
        w = weights[:, :, jnp.newaxis]  # BxNx1
        w_sum = jnp.sum(weights, axis=-1)  # B
        centroid_P = (
            jnp.sum(w * P, axis=1, keepdims=True) / w_sum[:, jnp.newaxis, jnp.newaxis]
        )
        centroid_Q = (
            jnp.sum(w * Q, axis=1, keepdims=True) / w_sum[:, jnp.newaxis, jnp.newaxis]
        )
    else:
        centroid_P = jnp.mean(P, axis=1, keepdims=True)
        centroid_Q = jnp.mean(Q, axis=1, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    if weights is not None:
        H = jnp.matmul(jnp.swapaxes(w * p, 1, 2), q)  # [B, 3, 3], weighted unscaled
    else:
        H = jnp.matmul(jnp.swapaxes(p, 1, 2), q)  # [B, 3, 3], unscaled

    S = H + jnp.swapaxes(H, 1, 2)
    tr = jnp.trace(H, axis1=1, axis2=2)
    Delta = jnp.stack(
        [
            H[..., 1, 2] - H[..., 2, 1],
            H[..., 2, 0] - H[..., 0, 2],
            H[..., 0, 1] - H[..., 1, 0],
        ],
        axis=-1,
    )

    B = H.shape[0]
    I3 = jnp.broadcast_to(jnp.eye(3, dtype=H.dtype), (B, 3, 3))
    top_row = jnp.concatenate([tr[..., jnp.newaxis], Delta], axis=-1)[:, jnp.newaxis, :]
    bottom_block = jnp.concatenate(
        [Delta[:, :, jnp.newaxis], S - tr[:, jnp.newaxis, jnp.newaxis] * I3], axis=-1
    )
    N_mat = jnp.concatenate([top_row, bottom_block], axis=-2)

    _L, V = safe_eigh(N_mat)
    q_opt = V[..., -1]
    qw, qx, qy, qz = q_opt[..., 0], q_opt[..., 1], q_opt[..., 2], q_opt[..., 3]

    row0 = jnp.stack(
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        axis=-1,
    )
    row1 = jnp.stack(
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
        axis=-1,
    )
    row2 = jnp.stack(
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        axis=-1,
    )
    R = jnp.stack([row0, row1, row2], axis=-2)

    return R, p, q, centroid_P, centroid_Q, H


def horn(
    P: jnp.ndarray, Q: jnp.ndarray, weights: jnp.ndarray | None = None
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes optimal rotation and translation to align P to Q using Horn's
    quaternion method.

    Strictly 3D only. Uses gradient-safe eigendecomposition (safe_eigh) to avoid
    NaN gradients when point clouds are symmetric or degenerate.

    Args:
        P: Source points, shape [..., N, 3].
        Q: Target points, shape [..., N, 3].
        weights: Per-point weights, shape [..., N]. Non-negative, must sum to > 0.
            When None, all points are weighted equally.

    Returns:
        (R, t, rmsd): Rotation [..., 3, 3], translation [..., 3], and RMSD [...].
        float16/bfloat16 inputs are upcast to float32 internally and downcast on output.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    if P.ndim < 2:
        raise ValueError(
            f"Input must be at least 2D with shape [..., N, D], got shape {P.shape}"
        )
    if P.shape[-1] != 3:
        raise ValueError("Horn's method is strictly for 3D point clouds")
    if P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")

    if weights is not None:
        if weights.shape != P.shape[:-1]:
            raise ValueError(
                f"weights shape {weights.shape} does not match "
                f"P.shape[:-1] {P.shape[:-1]}"
            )
        if jnp.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if not jnp.all(jnp.sum(weights, axis=-1) > 0):
            raise ValueError("weights must sum to a positive value")

    orig_dtype = P.dtype
    if P.dtype != Q.dtype:
        # Mixed dtypes: promote to higher precision
        target = jnp.float64 if jnp.float64 in (P.dtype, Q.dtype) else jnp.float32
        P = P.astype(target)
        Q = Q.astype(target)
        orig_dtype = target
    elif orig_dtype in (jnp.float16, jnp.bfloat16):
        P = P.astype(jnp.float32)
        Q = Q.astype(jnp.float32)

    if weights is not None:
        weights = weights.astype(P.dtype)

    is_single = P.ndim == 2
    if is_single:
        P = P[jnp.newaxis, ...]
        Q = Q[jnp.newaxis, ...]
        if weights is not None:
            weights = weights[jnp.newaxis, :]

    orig_shape = P.shape
    N_pts = orig_shape[-2]
    batch_dims = orig_shape[:-2]
    P = P.reshape(-1, N_pts, 3)
    Q = Q.reshape(-1, N_pts, 3)
    if weights is not None:
        weights = weights.reshape(-1, N_pts)

    R, p, q, centroid_P, centroid_Q, _H = _horn_core(P, Q, weights)

    t = jnp.squeeze(centroid_Q, axis=1) - jnp.squeeze(
        jnp.matmul(centroid_P, jnp.swapaxes(R, 1, 2)), axis=1
    )

    aligned = jnp.matmul(p, jnp.swapaxes(R, 1, 2))
    _eps = jnp.finfo(P.dtype).eps
    if weights is not None:
        w_sum = jnp.sum(weights, axis=-1)  # B
        residual_sq = jnp.sum(jnp.square(aligned - q), axis=-1)  # BxN
        mse = jnp.sum(weights * residual_sq, axis=-1) / w_sum  # B
    else:
        mse = jnp.sum(jnp.square(aligned - q), axis=(1, 2)) / N_pts
    rmsd = jnp.sqrt(mse + _eps)

    if is_single:
        R, t, rmsd = R[0], t[0], rmsd[0]
        if orig_dtype in (jnp.float16, jnp.bfloat16):
            R = R.astype(orig_dtype)
            t = t.astype(orig_dtype)
            rmsd = rmsd.astype(orig_dtype)
        return R, t, rmsd

    R = R.reshape(*batch_dims, 3, 3)
    t = t.reshape(*batch_dims, 3)
    rmsd = rmsd.reshape(*batch_dims)
    if orig_dtype in (jnp.float16, jnp.bfloat16):
        R = R.astype(orig_dtype)
        t = t.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)
    return R, t, rmsd


def horn_with_scale(
    P: jnp.ndarray, Q: jnp.ndarray, weights: jnp.ndarray | None = None
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes optimal rotation, translation, and scale to align P to Q
    (Q ~ c * R @ P + t).

    Strictly 3D only. Uses gradient-safe eigendecomposition (safe_eigh).

    Args:
        P: Source points, shape [..., N, 3].
        Q: Target points, shape [..., N, 3].
        weights: Per-point weights, shape [..., N]. Non-negative, must sum to > 0.
            When None, all points are weighted equally.

    Returns:
        (R, t, c, rmsd): Rotation [..., 3, 3], translation [..., 3],
        scale [...], RMSD [...].
        float16/bfloat16 inputs are upcast to float32 and downcast on output.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    if P.ndim < 2:
        raise ValueError(
            f"Input must be at least 2D with shape [..., N, D], got shape {P.shape}"
        )
    if P.shape[-1] != 3:
        raise ValueError("Horn's method is strictly for 3D point clouds")
    if P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")

    if weights is not None:
        if weights.shape != P.shape[:-1]:
            raise ValueError(
                f"weights shape {weights.shape} does not match "
                f"P.shape[:-1] {P.shape[:-1]}"
            )
        if jnp.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if not jnp.all(jnp.sum(weights, axis=-1) > 0):
            raise ValueError("weights must sum to a positive value")

    orig_dtype = P.dtype
    if P.dtype != Q.dtype:
        # Mixed dtypes: promote to higher precision
        target = jnp.float64 if jnp.float64 in (P.dtype, Q.dtype) else jnp.float32
        P = P.astype(target)
        Q = Q.astype(target)
        orig_dtype = target
    elif orig_dtype in (jnp.float16, jnp.bfloat16):
        P = P.astype(jnp.float32)
        Q = Q.astype(jnp.float32)

    if weights is not None:
        weights = weights.astype(P.dtype)

    is_single = P.ndim == 2
    if is_single:
        P = P[jnp.newaxis, ...]
        Q = Q[jnp.newaxis, ...]
        if weights is not None:
            weights = weights[jnp.newaxis, :]

    orig_shape = P.shape
    N_pts = orig_shape[-2]
    batch_dims = orig_shape[:-2]
    P = P.reshape(-1, N_pts, 3)
    Q = Q.reshape(-1, N_pts, 3)
    if weights is not None:
        weights = weights.reshape(-1, N_pts)

    R, p, _q, centroid_P, centroid_Q, H = _horn_core(P, Q, weights)

    # H is unscaled (weighted or not); compute scale
    _eps = jnp.finfo(P.dtype).eps
    if weights is not None:
        w_sum = jnp.sum(weights, axis=-1)  # B
        var_P = jnp.sum(weights * jnp.sum(jnp.square(p), axis=-1), axis=-1) / w_sum
        # c = trace(R^T @ H) / (var_P * w_sum)
        c = jnp.sum(R * jnp.swapaxes(H, 1, 2), axis=(1, 2)) / (
            jnp.clip(var_P, min=_eps) * w_sum
        )
    else:
        var_P = jnp.sum(jnp.square(p), axis=(1, 2)) / N_pts
        c = jnp.sum(R * jnp.swapaxes(H, 1, 2), axis=(1, 2)) / (
            jnp.clip(var_P, min=_eps) * N_pts
        )

    t = jnp.squeeze(centroid_Q, axis=1) - c[:, jnp.newaxis] * jnp.squeeze(
        jnp.matmul(centroid_P, jnp.swapaxes(R, 1, 2)), axis=1
    )

    aligned_P = (
        c[:, jnp.newaxis, jnp.newaxis] * jnp.matmul(P, jnp.swapaxes(R, 1, 2))
        + t[:, jnp.newaxis, :]
    )
    diff = aligned_P - Q
    if weights is not None:
        residual_sq = jnp.sum(jnp.square(diff), axis=-1)  # BxN
        mse = jnp.sum(weights * residual_sq, axis=-1) / w_sum  # B
    else:
        mse = jnp.sum(jnp.square(diff), axis=(1, 2)) / N_pts
    rmsd = jnp.sqrt(mse + _eps)

    if is_single:
        R, t, c, rmsd = R[0], t[0], c[0], rmsd[0]
        if orig_dtype in (jnp.float16, jnp.bfloat16):
            R = R.astype(orig_dtype)
            t = t.astype(orig_dtype)
            c = jnp.clip(c, max=jnp.finfo(orig_dtype).max)
            c = c.astype(orig_dtype)
            rmsd = rmsd.astype(orig_dtype)
        return R, t, c, rmsd

    R = R.reshape(*batch_dims, 3, 3)
    t = t.reshape(*batch_dims, 3)
    c = c.reshape(*batch_dims)
    rmsd = rmsd.reshape(*batch_dims)
    if orig_dtype in (jnp.float16, jnp.bfloat16):
        R = R.astype(orig_dtype)
        t = t.astype(orig_dtype)
        c = jnp.clip(c, max=jnp.finfo(orig_dtype).max)
        c = c.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)
    return R, t, c, rmsd
