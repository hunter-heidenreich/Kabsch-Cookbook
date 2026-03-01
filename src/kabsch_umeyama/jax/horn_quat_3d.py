import jax
import jax.numpy as jnp


@jax.custom_vjp
def safe_eigh(A, eps=1e-12):
    return jnp.linalg.eigh(A)


def _eigh_fwd(A, eps):
    L, V = jnp.linalg.eigh(A)
    return (L, V), (L, V, eps)


def _eigh_bwd(res, g):
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
    safe_D = jnp.where(jnp.eye(D.shape[-1], dtype=bool), 1.0, safe_D)

    F = 1.0 / safe_D
    F = jnp.where(jnp.eye(F.shape[-1], dtype=bool), 0.0, F)

    Vt_dV = jnp.matmul(mH(V), grad_V)

    vmap_diag = jax.vmap(jnp.diag) if L.ndim > 1 else jnp.diag
    L_diag = vmap_diag(grad_L)

    term = L_diag + F * (Vt_dV - mH(Vt_dV))
    grad_A = jnp.matmul(V, jnp.matmul(term, mH(V)))

    return (grad_A, None)


safe_eigh.defvjp(_eigh_fwd, _eigh_bwd)


def horn(
    P: jnp.ndarray, Q: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    is_single = P.ndim == 2
    if is_single:
        P = P[jnp.newaxis, ...]
        Q = Q[jnp.newaxis, ...]

    centroid_P = jnp.mean(P, axis=1, keepdims=True)
    centroid_Q = jnp.mean(Q, axis=1, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = jnp.matmul(jnp.swapaxes(p, 1, 2), q)

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
    I3 = jnp.broadcast_to(jnp.eye(3), (B, 3, 3))

    top_row = jnp.concatenate([tr[..., jnp.newaxis], Delta], axis=-1)[:, jnp.newaxis, :]
    bottom_block = jnp.concatenate(
        [Delta[:, :, jnp.newaxis], S - tr[:, jnp.newaxis, jnp.newaxis] * I3], axis=-1
    )

    N_mat = jnp.concatenate([top_row, bottom_block], axis=-2)

    _L, V = safe_eigh(N_mat)
    q_opt = V[..., -1]

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

    R = jnp.stack(
        [
            jnp.stack([R11, R12, R13], axis=-1),
            jnp.stack([R21, R22, R23], axis=-1),
            jnp.stack([R31, R32, R33], axis=-1),
        ],
        axis=-2,
    )

    t = jnp.squeeze(centroid_Q, axis=1) - jnp.squeeze(
        jnp.matmul(centroid_P, jnp.swapaxes(R, 1, 2)), axis=1
    )

    aligned = jnp.matmul(p, jnp.swapaxes(R, 1, 2))
    rmsd = jnp.sqrt(
        jnp.clip(
            jnp.sum(jnp.square(aligned - q), axis=(1, 2)) / P.shape[1],
            min=1e-12,
            max=None,
        )
    )

    if is_single:
        return R[0], t[0], rmsd[0]
    return R, t, rmsd


def horn_with_scale(
    P: jnp.ndarray, Q: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    is_single = P.ndim == 2
    if is_single:
        P = P[jnp.newaxis, ...]
        Q = Q[jnp.newaxis, ...]

    _B, N_pts, _D = P.shape

    centroid_P = jnp.mean(P, axis=1, keepdims=True)
    centroid_Q = jnp.mean(Q, axis=1, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    var_P = jnp.sum(jnp.square(p), axis=(1, 2)) / N_pts

    H = jnp.matmul(jnp.swapaxes(p, 1, 2), q) / N_pts

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
    I3 = jnp.broadcast_to(jnp.eye(3), (B, 3, 3))

    top_row = jnp.concatenate([tr[..., jnp.newaxis], Delta], axis=-1)[:, jnp.newaxis, :]
    bottom_block = jnp.concatenate(
        [Delta[:, :, jnp.newaxis], S - tr[:, jnp.newaxis, jnp.newaxis] * I3], axis=-1
    )

    N_mat = jnp.concatenate([top_row, bottom_block], axis=-2)

    _L, V = safe_eigh(N_mat)
    q_opt = V[..., -1]

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

    R = jnp.stack(
        [
            jnp.stack([R11, R12, R13], axis=-1),
            jnp.stack([R21, R22, R23], axis=-1),
            jnp.stack([R31, R32, R33], axis=-1),
        ],
        axis=-2,
    )

    RH = jnp.sum(R * jnp.swapaxes(H, 1, 2), axis=(1, 2))
    c = RH / jnp.clip(var_P, min=1e-12, max=None)

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
        return R[0], t[0], c[0], rmsd[0]
    return R, t, c, rmsd
