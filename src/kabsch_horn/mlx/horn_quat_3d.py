import warnings

import mlx.core as mx


def _warn_if_float64(arr: mx.array) -> None:
    if arr.dtype == mx.float64:
        warnings.warn(
            "MLX does not support float64 on GPU; falling back to CPU.",
            UserWarning,
            stacklevel=3,
        )
        mx.set_default_device(mx.cpu)


@mx.custom_function
def safe_eigh_fwd(A: mx.array) -> tuple[mx.array, mx.array]:
    L, V = mx.linalg.eigh(A, stream=mx.cpu)
    return L, V


@safe_eigh_fwd.vjp
def safe_eigh_bwd(primals, cotangents, outputs):
    _A = primals[0]
    # unpacking tuple fails in MLX if cotangents elements can be None ?
    dL = cotangents[0] if cotangents[0] is not None else mx.zeros_like(outputs[0])
    dV = cotangents[1] if cotangents[1] is not None else mx.zeros_like(outputs[1])

    L, V = outputs

    D = mx.expand_dims(L, -2) - mx.expand_dims(L, -1)

    eye = mx.eye(D.shape[-1], dtype=D.dtype)
    eps = 1e-12
    mask = mx.abs(D) < eps
    safe_D = mx.where(mask, eps * mx.sign(D + eps), D)
    # Set diagonal to 1.0 so 1/safe_D is defined everywhere; zero it out after
    safe_D = mx.where(eye == 1, mx.ones_like(safe_D), safe_D)

    F = 1.0 / safe_D
    F = mx.where(eye == 1, mx.zeros_like(F), F)

    Vt_dV = mx.matmul(V.swapaxes(-1, -2), dV)

    dL_mat = mx.expand_dims(dL, -1) * eye

    term = dL_mat + F * (Vt_dV - Vt_dV.swapaxes(-1, -2)) / 2

    dA = mx.matmul(V, mx.matmul(term, V.swapaxes(-1, -2)))
    dA = mx.where(mx.isnan(dA), mx.zeros_like(dA), dA)

    return (dA,)


def horn(P: mx.array, Q: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """
    Computes optimal rotation and translation to align P to Q using Horn's
    quaternion method.

    Strictly 3D only. Uses gradient-safe eigendecomposition (safe_eigh_fwd) to
    avoid NaN gradients when point clouds are symmetric or degenerate.

    Args:
        P: Source points, shape [..., N, 3].
        Q: Target points, shape [..., N, 3].

    Returns:
        (R, t, rmsd): Rotation [..., 3, 3], translation [..., 3], and RMSD [...].
        float16/bfloat16 inputs are upcast to float32 internally and downcast on output.
    """
    P = mx.array(P)
    Q = mx.array(Q)
    _warn_if_float64(P)
    orig_dtype = P.dtype
    if orig_dtype in (mx.float16, mx.bfloat16):
        P = P.astype(mx.float32)
        Q = Q.astype(mx.float32)

    centroid_P = mx.mean(P, axis=-2, keepdims=True)
    centroid_Q = mx.mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = mx.matmul(p.swapaxes(-1, -2), q)

    S = H + H.swapaxes(-1, -2)
    tr = H[..., 0, 0] + H[..., 1, 1] + H[..., 2, 2]

    Delta = mx.stack(
        [
            H[..., 1, 2] - H[..., 2, 1],
            H[..., 2, 0] - H[..., 0, 2],
            H[..., 0, 1] - H[..., 1, 0],
        ],
        axis=-1,
    )

    D_dim = 3
    I_shape = (*H.shape[:-2], D_dim, D_dim)
    I3 = mx.expand_dims(mx.eye(D_dim, dtype=H.dtype), list(range(len(I_shape) - 2)))
    I3 = mx.broadcast_to(I3, I_shape)

    tr_exp = mx.expand_dims(tr, -1)
    top_row = mx.expand_dims(mx.concatenate([tr_exp, Delta], axis=-1), -2)

    bottom_block = mx.concatenate(
        [mx.expand_dims(Delta, -1), S - mx.expand_dims(tr_exp, -1) * I3], axis=-1
    )

    N_mat = mx.concatenate([top_row, bottom_block], axis=-2)

    _L, V = safe_eigh_fwd(N_mat)
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

    R = mx.stack(
        [
            mx.stack([R11, R12, R13], axis=-1),
            mx.stack([R21, R22, R23], axis=-1),
            mx.stack([R31, R32, R33], axis=-1),
        ],
        axis=-2,
    )

    t = mx.squeeze(centroid_Q, -2) - mx.squeeze(
        mx.matmul(centroid_P, R.swapaxes(-1, -2)), -2
    )

    aligned = mx.matmul(p, R.swapaxes(-1, -2))

    diff = aligned - q
    mse = mx.mean(mx.sum(mx.square(diff), axis=-1), axis=-1)
    rmsd = mx.sqrt(mx.maximum(mse, 1e-12))

    if orig_dtype in (mx.float16, mx.bfloat16):
        R = R.astype(orig_dtype)
        t = t.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)
    return R, t, rmsd


def horn_with_scale(
    P: mx.array, Q: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Computes optimal rotation, translation, and scale to align P to Q
    (Q ~ c * R @ P + t).

    Strictly 3D only. Uses gradient-safe eigendecomposition (safe_eigh_fwd).

    Args:
        P: Source points, shape [..., N, 3].
        Q: Target points, shape [..., N, 3].

    Returns:
        (R, t, c, rmsd): Rotation [..., 3, 3], translation [..., 3],
        scale [...], RMSD [...].
        float16/bfloat16 inputs are upcast to float32 and downcast on output.
    """
    P = mx.array(P)
    Q = mx.array(Q)
    _warn_if_float64(P)
    orig_dtype = P.dtype
    if orig_dtype in (mx.float16, mx.bfloat16):
        P = P.astype(mx.float32)
        Q = Q.astype(mx.float32)
    N_pts_f = mx.array(P.shape[-2], dtype=P.dtype)

    centroid_P = mx.mean(P, axis=-2, keepdims=True)
    centroid_Q = mx.mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    var_P = mx.sum(mx.square(p), axis=(-2, -1)) / N_pts_f

    H = mx.matmul(p.swapaxes(-1, -2), q) / N_pts_f

    S = H + H.swapaxes(-1, -2)
    tr = H[..., 0, 0] + H[..., 1, 1] + H[..., 2, 2]

    Delta = mx.stack(
        [
            H[..., 1, 2] - H[..., 2, 1],
            H[..., 2, 0] - H[..., 0, 2],
            H[..., 0, 1] - H[..., 1, 0],
        ],
        axis=-1,
    )

    D_dim = 3
    I_shape = (*H.shape[:-2], D_dim, D_dim)
    I3 = mx.expand_dims(mx.eye(D_dim, dtype=H.dtype), list(range(len(I_shape) - 2)))
    I3 = mx.broadcast_to(I3, I_shape)

    tr_exp = mx.expand_dims(tr, -1)
    top_row = mx.expand_dims(mx.concatenate([tr_exp, Delta], axis=-1), -2)

    bottom_block = mx.concatenate(
        [mx.expand_dims(Delta, -1), S - mx.expand_dims(tr_exp, -1) * I3], axis=-1
    )

    N_mat = mx.concatenate([top_row, bottom_block], axis=-2)

    _L, V = safe_eigh_fwd(N_mat)
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

    R = mx.stack(
        [
            mx.stack([R11, R12, R13], axis=-1),
            mx.stack([R21, R22, R23], axis=-1),
            mx.stack([R31, R32, R33], axis=-1),
        ],
        axis=-2,
    )

    RH = mx.sum(R * H.swapaxes(-1, -2), axis=(-1, -2))
    c = RH / mx.maximum(var_P, 1e-12)

    t = mx.squeeze(centroid_Q, -2) - mx.expand_dims(c, -1) * mx.squeeze(
        mx.matmul(centroid_P, R.swapaxes(-1, -2)), -2
    )

    aligned_P = mx.expand_dims(mx.expand_dims(c, -1), -1) * mx.matmul(
        P, R.swapaxes(-1, -2)
    ) + mx.expand_dims(t, -2)
    diff = aligned_P - Q
    mse = mx.mean(mx.sum(mx.square(diff), axis=-1), axis=-1)
    rmsd = mx.sqrt(mx.maximum(mse, 1e-12))

    if orig_dtype in (mx.float16, mx.bfloat16):
        R = R.astype(orig_dtype)
        t = t.astype(orig_dtype)
        c = c.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)
    return R, t, c, rmsd
