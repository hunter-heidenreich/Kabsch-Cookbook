import jax
import jax.numpy as jnp


@jax.custom_vjp
def safe_svd(
    A: jnp.ndarray, eps: float = 1e-12
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Gradient-safe SVD. Masks near-zero singular value differences (< eps) in the
    backward pass to prevent NaN gradients for symmetric or degenerate inputs."""
    U, S, Vh = jnp.linalg.svd(A)
    # Like PyTorch logic, we must return V to remain coherent with algorithm
    # JAX returns U, S, Vh (where Vh is V^T)
    return U, S, jnp.swapaxes(jnp.conj(Vh), -1, -2)


def _fwd(
    A: jnp.ndarray, eps: float
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple]:
    U, S, Vh = jnp.linalg.svd(A)
    # return primal outputs and residuals for backward pass
    V = jnp.swapaxes(jnp.conj(Vh), -1, -2)
    return (U, S, V), (U, S, Vh, eps)


def _bwd(res, g):
    # Retrieve
    U, S, Vh, eps = res
    grad_U, grad_S, grad_V = g

    # Check if None or SymbolicZero
    grad_U = (
        jnp.zeros_like(U)
        if type(grad_U) is jax.custom_derivatives.SymbolicZero
        else grad_U
    )
    grad_S = (
        jnp.zeros_like(S)
        if type(grad_S) is jax.custom_derivatives.SymbolicZero
        else grad_S
    )
    grad_V = (
        jnp.zeros_like(Vh)
        if type(grad_V) is jax.custom_derivatives.SymbolicZero
        else grad_V
    )

    def mH(x):
        return jnp.swapaxes(jnp.conj(x), -1, -2)

    grad_Vh = mH(grad_V)

    # 1. Square of S
    S_sq = jnp.square(S)  # BxD

    # 2. D = S_sq - S_sq^T
    D = S_sq[..., jnp.newaxis] - S_sq[..., jnp.newaxis, :]  # BxDxD

    # 3. Safe F
    safe_D = jnp.where(jnp.abs(D) < eps, eps * jnp.sign(D + eps), D)
    safe_D = jnp.where(jnp.eye(D.shape[-1], dtype=bool), 1.0, safe_D)

    F = 1.0 / safe_D
    F = jnp.where(jnp.eye(F.shape[-1], dtype=bool), 0.0, F)

    # 4. J and K
    Ut_dU = jnp.matmul(mH(U), grad_U)
    J = F * (Ut_dU - mH(Ut_dU))

    Vht_dVh = jnp.matmul(Vh, mH(grad_Vh))
    K = F * (Vht_dVh - mH(Vht_dVh))

    # 5. Build term
    eye = jnp.eye(S.shape[-1], dtype=S.dtype)
    S_diag = grad_S[..., jnp.newaxis] * eye
    S_mat = S[..., jnp.newaxis] * eye

    term = S_diag - jnp.matmul(J, S_mat) - jnp.matmul(S_mat, K)

    grad_A = jnp.matmul(U, jnp.matmul(term, Vh))

    return grad_A, None


safe_svd.defvjp(_fwd, _bwd)


def kabsch(
    P: jnp.ndarray, Q: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the optimal rotation and translation to align P to Q using
    gradient-safe SVD.

    Args:
        P: Source points, shape [..., N, D].
        Q: Target points, shape [..., N, D].

    Returns:
        (R, t, rmsd): Rotation [..., D, D], translation [..., D], and RMSD [...].
        float16/bfloat16 inputs are upcast to float32 internally and downcast on output.

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

    orig_dtype = P.dtype
    if orig_dtype in (jnp.float16, jnp.bfloat16):
        P = P.astype(jnp.float32)
        Q = Q.astype(jnp.float32)

    is_single = P.ndim == 2
    if is_single:
        P = P[jnp.newaxis, ...]
        Q = Q[jnp.newaxis, ...]

    orig_shape = P.shape
    D = orig_shape[-1]
    N = orig_shape[-2]
    batch_dims = orig_shape[:-2]

    P = P.reshape(-1, N, D)
    Q = Q.reshape(-1, N, D)

    centroid_P = jnp.mean(P, axis=1, keepdims=True)
    centroid_Q = jnp.mean(Q, axis=1, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = jnp.matmul(jnp.swapaxes(p, 1, 2), q)

    U, _S, V = safe_svd(H)

    d = jnp.linalg.det(jnp.matmul(V, jnp.swapaxes(U, 1, 2)))
    d_sign = jnp.sign(d + 1e-12)

    ones = jnp.ones_like(d_sign)
    B_diag = jnp.stack([ones] * (D - 1) + [d_sign], axis=-1)

    R = jnp.matmul(V * B_diag[:, jnp.newaxis, :], jnp.swapaxes(U, 1, 2))

    t = jnp.squeeze(centroid_Q, axis=1) - jnp.squeeze(
        jnp.matmul(centroid_P, jnp.swapaxes(R, 1, 2)), axis=1
    )

    aligned = jnp.matmul(P, jnp.swapaxes(R, 1, 2)) + t[:, jnp.newaxis, :]
    diff_sq = jnp.sum(jnp.square(aligned - Q), axis=(1, 2)) / N
    rmsd = jnp.sqrt(jnp.clip(diff_sq, min=1e-12, max=None))

    if is_single:
        R, t, rmsd = R[0], t[0], rmsd[0]
        if orig_dtype in (jnp.float16, jnp.bfloat16):
            R = R.astype(orig_dtype)
            t = t.astype(orig_dtype)
            rmsd = rmsd.astype(orig_dtype)
        return R, t, rmsd

    R = R.reshape(*batch_dims, D, D)
    t = t.reshape(*batch_dims, D)
    rmsd = rmsd.reshape(*batch_dims)

    if orig_dtype in (jnp.float16, jnp.bfloat16):
        R = R.astype(orig_dtype)
        t = t.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)

    return R, t, rmsd


def kabsch_umeyama(
    P: jnp.ndarray, Q: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the optimal rotation, translation, and scale to align P to Q
    (Q ~ c * R @ P + t).

    Args:
        P: Source points, shape [..., N, D].
        Q: Target points, shape [..., N, D].

    Returns:
        (R, t, c, rmsd): Rotation [..., D, D], translation [..., D],
        scale [...], RMSD [...].
        float16/bfloat16 inputs are upcast to float32 and downcast on output.

    Note:
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

    orig_dtype = P.dtype
    if orig_dtype in (jnp.float16, jnp.bfloat16):
        P = P.astype(jnp.float32)
        Q = Q.astype(jnp.float32)

    is_single = P.ndim == 2
    if is_single:
        P = P[jnp.newaxis, ...]
        Q = Q[jnp.newaxis, ...]

    orig_shape = P.shape
    D = orig_shape[-1]
    N = orig_shape[-2]
    batch_dims = orig_shape[:-2]

    P = P.reshape(-1, N, D)
    Q = Q.reshape(-1, N, D)

    centroid_P = jnp.mean(P, axis=1, keepdims=True)
    centroid_Q = jnp.mean(Q, axis=1, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = jnp.matmul(jnp.swapaxes(p, 1, 2), q) / N

    var_P = jnp.sum(jnp.square(p), axis=(1, 2)) / N

    U, S, V = safe_svd(H)

    d = jnp.linalg.det(jnp.matmul(V, jnp.swapaxes(U, 1, 2)))
    d_sign = jnp.sign(d + 1e-12)

    ones = jnp.ones_like(d_sign)
    S_corr = jnp.stack([ones] * (D - 1) + [d_sign], axis=-1)

    c = jnp.sum(S * S_corr, axis=-1) / jnp.clip(var_P, min=1e-12, max=None)

    R = jnp.matmul(V * S_corr[:, jnp.newaxis, :], jnp.swapaxes(U, 1, 2))

    t = jnp.squeeze(centroid_Q, axis=1) - c[:, jnp.newaxis] * jnp.squeeze(
        jnp.matmul(centroid_P, jnp.swapaxes(R, 1, 2)), axis=1
    )

    aligned_P = (
        c[:, jnp.newaxis, jnp.newaxis] * jnp.matmul(P, jnp.swapaxes(R, 1, 2))
        + t[:, jnp.newaxis, :]
    )
    rmsd = jnp.sqrt(
        jnp.clip(
            jnp.sum(jnp.square(aligned_P - Q), axis=(1, 2)) / N, min=1e-12, max=None
        )
    )

    if is_single:
        R, t, c, rmsd = R[0], t[0], c[0], rmsd[0]
        if orig_dtype in (jnp.float16, jnp.bfloat16):
            R = R.astype(orig_dtype)
            t = t.astype(orig_dtype)
            c = c.astype(orig_dtype)
            rmsd = rmsd.astype(orig_dtype)

        return R, t, c, rmsd

    R = R.reshape(*batch_dims, D, D)
    t = t.reshape(*batch_dims, D)
    c = c.reshape(*batch_dims)
    rmsd = rmsd.reshape(*batch_dims)

    if orig_dtype in (jnp.float16, jnp.bfloat16):
        R = R.astype(orig_dtype)
        t = t.astype(orig_dtype)
        c = c.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)

    return R, t, c, rmsd


def kabsch_rmsd(P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Computes RMSD after Kabsch alignment. Gradient-safe training loss."""
    _R, _t, rmsd = kabsch(P, Q)
    return rmsd


def kabsch_umeyama_rmsd(P: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Computes RMSD after Kabsch-Umeyama alignment. Gradient-safe training loss."""
    _R, _t, _c, rmsd = kabsch_umeyama(P, Q)
    return rmsd
