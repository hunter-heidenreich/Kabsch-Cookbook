import jax
import jax.numpy as jnp


@jax.custom_vjp
def safe_eigh(A: jnp.ndarray, eps: float = 1e-12) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gradient-safe eigendecomposition for symmetric matrices. Masks near-zero
    eigenvalue differences (< eps) in the backward pass to prevent NaN gradients."""
    return jnp.linalg.eigh(A)


def _eigh_fwd(
    A: jnp.ndarray, eps: float
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple]:
    L, V = jnp.linalg.eigh(A)
    return (L, V), (L, V, eps)


def _eigh_bwd(res: tuple, g: tuple) -> tuple:
    L, V, eps = res
    grad_L, grad_V = g

    grad_L = (
        jnp.zeros_like(L)
        if type(grad_L) is jax.custom_derivatives.SymbolicZero
        else grad_L
    )
    grad_V = (
        jnp.zeros_like(V)
        if type(grad_V) is jax.custom_derivatives.SymbolicZero
        else grad_V
    )

    def mH(x):
        return jnp.swapaxes(jnp.conj(x), -1, -2)

    D = L[..., jnp.newaxis, :] - L[..., jnp.newaxis]

    mask = jnp.abs(D) < eps
    safe_D = jnp.where(mask, eps * jnp.sign(D + eps), D)
    diag_mask = jnp.eye(D.shape[-1], dtype=bool)
    safe_D = jnp.where(diag_mask, 1.0, safe_D)
    F = jnp.where(diag_mask, 0.0, 1.0 / safe_D)

    Vt_dV = jnp.matmul(mH(V), grad_V)

    L_diag = grad_L[..., jnp.newaxis] * diag_mask.astype(grad_L.dtype)

    term = L_diag + F * (Vt_dV - mH(Vt_dV)) / 2
    grad_A = jnp.matmul(V, jnp.matmul(term, mH(V)))

    return (grad_A, None)


safe_eigh.defvjp(_eigh_fwd, _eigh_bwd)


def _horn_core(
    P: jnp.ndarray, Q: jnp.ndarray
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
    and H = p.T @ q (unscaled).
    """
    centroid_P = jnp.mean(P, axis=1, keepdims=True)
    centroid_Q = jnp.mean(Q, axis=1, keepdims=True)
    p = P - centroid_P
    q = Q - centroid_Q

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
    P: jnp.ndarray, Q: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes optimal rotation and translation to align P to Q using Horn's
    quaternion method.

    Strictly 3D only. Uses gradient-safe eigendecomposition (safe_eigh) to avoid
    NaN gradients when point clouds are symmetric or degenerate.

    Args:
        P: Source points, shape [..., N, 3].
        Q: Target points, shape [..., N, 3].

    Returns:
        (R, t, rmsd): Rotation [..., 3, 3], translation [..., 3], and RMSD [...].
        float16/bfloat16 inputs are upcast to float32 internally and downcast on output.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )

    orig_dtype = P.dtype
    if orig_dtype in (jnp.float16, jnp.bfloat16):
        P = P.astype(jnp.float32)
        Q = Q.astype(jnp.float32)

    is_single = P.ndim == 2
    if is_single:
        P = P[jnp.newaxis, ...]
        Q = Q[jnp.newaxis, ...]

    orig_shape = P.shape
    N_pts = orig_shape[-2]
    batch_dims = orig_shape[:-2]
    P = P.reshape(-1, N_pts, 3)
    Q = Q.reshape(-1, N_pts, 3)

    R, p, q, centroid_P, centroid_Q, _H = _horn_core(P, Q)

    t = jnp.squeeze(centroid_Q, axis=1) - jnp.squeeze(
        jnp.matmul(centroid_P, jnp.swapaxes(R, 1, 2)), axis=1
    )

    aligned = jnp.matmul(p, jnp.swapaxes(R, 1, 2))
    rmsd = jnp.sqrt(
        jnp.clip(
            jnp.sum(jnp.square(aligned - q), axis=(1, 2)) / N_pts,
            min=1e-12,
            max=None,
        )
    )

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
    P: jnp.ndarray, Q: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes optimal rotation, translation, and scale to align P to Q
    (Q ~ c * R @ P + t).

    Strictly 3D only. Uses gradient-safe eigendecomposition (safe_eigh).

    Args:
        P: Source points, shape [..., N, 3].
        Q: Target points, shape [..., N, 3].

    Returns:
        (R, t, c, rmsd): Rotation [..., 3, 3], translation [..., 3],
        scale [...], RMSD [...].
        float16/bfloat16 inputs are upcast to float32 and downcast on output.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )

    orig_dtype = P.dtype
    if orig_dtype in (jnp.float16, jnp.bfloat16):
        P = P.astype(jnp.float32)
        Q = Q.astype(jnp.float32)

    is_single = P.ndim == 2
    if is_single:
        P = P[jnp.newaxis, ...]
        Q = Q[jnp.newaxis, ...]

    orig_shape = P.shape
    N_pts = orig_shape[-2]
    batch_dims = orig_shape[:-2]
    P = P.reshape(-1, N_pts, 3)
    Q = Q.reshape(-1, N_pts, 3)

    R, p, _q, centroid_P, centroid_Q, H = _horn_core(P, Q)

    # H is unscaled (p.T @ q); var_P * N_pts = sum(sq(p)), so c = tr(R^T H) / sum(sq(p))
    var_P = jnp.sum(jnp.square(p), axis=(1, 2)) / N_pts
    c = jnp.sum(R * jnp.swapaxes(H, 1, 2), axis=(1, 2)) / (
        jnp.clip(var_P, min=1e-12) * N_pts
    )

    t = jnp.squeeze(centroid_Q, axis=1) - c[:, jnp.newaxis] * jnp.squeeze(
        jnp.matmul(centroid_P, jnp.swapaxes(R, 1, 2)), axis=1
    )

    aligned_P = (
        c[:, jnp.newaxis, jnp.newaxis] * jnp.matmul(P, jnp.swapaxes(R, 1, 2))
        + t[:, jnp.newaxis, :]
    )
    diff = aligned_P - Q
    rmsd = jnp.sqrt(
        jnp.clip(jnp.sum(jnp.square(diff), axis=(1, 2)) / N_pts, min=1e-12, max=None)
    )

    if is_single:
        R, t, c, rmsd = R[0], t[0], c[0], rmsd[0]
        if orig_dtype in (jnp.float16, jnp.bfloat16):
            R = R.astype(orig_dtype)
            t = t.astype(orig_dtype)
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
        c = c.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)
    return R, t, c, rmsd
