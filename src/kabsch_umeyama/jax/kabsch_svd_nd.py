import jax
import jax.numpy as jnp


def _safe_svd_fwd(A, eps=1e-12):
    U, S, V = jnp.linalg.svd(A)
    # Return (U, S, V) and save for backward
    return (U, S, V), (U, S, V, eps)


def _safe_svd_bwd(res, g):
    U, S, V, eps = res
    grad_U, grad_S, grad_V = g

    # Replace None/zeros if gradient isn't provided
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
        jnp.zeros_like(V)
        if type(grad_V) is jax.custom_derivatives.SymbolicZero
        else grad_V
    )

    # V in JAX SVD is already V^T (like PyTorch Vh) if full_matrices is false, wait!
    # jnp.linalg.svd returns U, S, Vh. The original math was derived assuming
    # A = U @ S @ Vh.
    # The gradient of A with respect to Vh (where Vh = V^T) is grad_Vh.
    grad_Vh = grad_V
    Vh = V

    # 1. Square of S
    S_sq = jnp.square(S)  # BxD

    # 2. D = S_sq - S_sq^T
    D = S_sq[..., jnp.newaxis] - S_sq[..., jnp.newaxis, :]  # BxDxD

    # 3. Safe denominator
    mask = jnp.abs(D) < eps
    safe_D = jnp.where(mask, eps * jnp.sign(D + eps), D)

    # Set diagonal to exactly 1.0 before inversion
    safe_D = jnp.where(jnp.eye(D.shape[-1], dtype=bool), 1.0, safe_D)

    F = 1.0 / safe_D
    # Zero diagonal
    F = jnp.where(jnp.eye(F.shape[-1], dtype=bool), 0.0, F)

    # 4. J and K
    def mH(x):
        return jnp.swapaxes(jnp.conj(x), -1, -2)

    Ut_dU = jnp.matmul(mH(U), grad_U)
    J = F * (Ut_dU - mH(Ut_dU))

    Vht_dVh = jnp.matmul(Vh, mH(grad_Vh))
    K = F * (Vht_dVh - mH(Vht_dVh))

    # 5. Gradient term
    S_diag = jax.vmap(jnp.diag)(grad_S) if grad_S.ndim > 1 else jnp.diag(grad_S)
    S_mat = jax.vmap(jnp.diag)(S) if S.ndim > 1 else jnp.diag(S)

    term = S_diag - jnp.matmul(J, S_mat) - jnp.matmul(S_mat, K)

    grad_A = jnp.matmul(U, jnp.matmul(term, Vh))

    return (grad_A, None)


@jax.custom_vjp
def safe_svd(A, eps=1e-12):
    U, S, Vh = jnp.linalg.svd(A)
    # Like PyTorch logic, we must return V to remain coherent with algorithm
    # JAX returns U, S, Vh (where Vh is V^T)
    return U, S, jnp.swapaxes(jnp.conj(Vh), -1, -2)


def _fwd(A, eps):
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
    vmap_diag = jax.vmap(jnp.diag) if S.ndim > 1 else jnp.diag
    S_diag = vmap_diag(grad_S)
    S_mat = vmap_diag(S)

    term = S_diag - jnp.matmul(J, S_mat) - jnp.matmul(S_mat, K)

    grad_A = jnp.matmul(U, jnp.matmul(term, Vh))

    return grad_A, None


safe_svd.defvjp(_fwd, _bwd)


def kabsch(
    P: jnp.ndarray, Q: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    _B = P.shape[0]

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
        return R[0], t[0], rmsd[0]

    R = R.reshape(*batch_dims, D, D)
    t = t.reshape(*batch_dims, D)
    rmsd = rmsd.reshape(*batch_dims)
    return R, t, rmsd


def kabsch_umeyama(
    P: jnp.ndarray, Q: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    _B = P.shape[0]

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
        return R[0], t[0], c[0], rmsd[0]

    R = R.reshape(*batch_dims, D, D)
    t = t.reshape(*batch_dims, D)
    c = c.reshape(*batch_dims)
    rmsd = rmsd.reshape(*batch_dims)
    return R, t, c, rmsd
