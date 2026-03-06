import mlx.core as mx


@mx.custom_function
def safe_svd(A: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """
    Computes SVD with a custom gradient to handle degenerate cases (zero singular values
    and coincident singular values) robustly for MLX.
    Uses mlx.core.linalg.svd
    """
    U, S, Vt = mx.linalg.svd(A, stream=mx.cpu)
    return U, S, Vt


@safe_svd.vjp
def safe_svd_bwd(primals, cotangents, outputs):
    # Depending on how mlx passes primals, A is the first element
    _A = primals[0]
    dU, dS, dVt = cotangents
    U, S, Vt = outputs

    # We replace None with zeros
    if dU is None:
        dU = mx.zeros_like(U)
    if dS is None:
        dS = mx.zeros_like(S)
    if dVt is None:
        dVt = mx.zeros_like(Vt)

    # mlx.diag expects 1D or 2D. S is shape (..., D).
    # Use broadcasting for S_mat.
    eye_D = mx.eye(S.shape[-1], dtype=S.dtype)
    S_mat = mx.expand_dims(S, -1) * eye_D
    # J = U^T @ dU
    Ut_dU = mx.matmul(U.swapaxes(-1, -2), dU)

    # K = V^T @ dV
    # We have dVt, dV = dVt^T. V = Vt^T
    # K = V^T @ dV = Vt @ dVt.T
    dV = dVt.swapaxes(-1, -2)
    Vt_dV = mx.matmul(Vt, dV)

    S_sq = mx.square(S)
    S_sq_diff = mx.expand_dims(S_sq, -2) - mx.expand_dims(S_sq, -1)

    # Add epsilon to diagonal before reciprocal
    eye = mx.eye(S_sq_diff.shape[-1], dtype=S_sq_diff.dtype)
    S_sq_diff_safe = mx.where(mx.abs(S_sq_diff) < 1e-12, eye * 1e-12, S_sq_diff)

    F = 1.0 / S_sq_diff_safe
    F = mx.where(eye == 1, mx.zeros_like(F), F)

    J = F * (Ut_dU - Ut_dU.swapaxes(-1, -2))
    K = F * (Vt_dV - Vt_dV.swapaxes(-1, -2))

    dS_mat = mx.expand_dims(dS, -1) * mx.eye(dS.shape[-1], dtype=dS.dtype)

    # For PyTorch and tf we found term = dS + J@S + S@K or similar.
    # We will test +/-
    term = dS_mat + mx.matmul(J, S_mat) + mx.matmul(S_mat, K)

    dA = mx.matmul(U, mx.matmul(term, Vt))
    dA = mx.where(mx.isnan(dA), mx.zeros_like(dA), dA)

    return (dA,)


def kabsch(P: mx.array, Q: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """
    Computes the optimal rotation and translation to align P to Q.

    MLX only supports 3D inputs (dim=3) due to the hardcoded 3x3 determinant correction.

    Args:
        P: Source points, shape [..., N, 3].
        Q: Target points, shape [..., N, 3].

    Returns:
        (R, t, rmsd): Rotation [..., 3, 3], translation [..., 3], and RMSD [...].
        float16/bfloat16 inputs are upcast to float32 internally and downcast on output.

    Raises:
        ValueError: If inputs are not 3-dimensional (D != 3).
    """
    if P.shape[-1] != 3:
        raise ValueError(
            f"MLX Kabsch only supports dim=3, got dim={P.shape[-1]}. "
            "Use the JAX, PyTorch, or TensorFlow implementations for N-D alignment."
        )
    orig_dtype = P.dtype
    if orig_dtype in (mx.float16, mx.bfloat16):
        P = P.astype(mx.float32)
        Q = Q.astype(mx.float32)

    centroid_P = mx.mean(P, axis=-2, keepdims=True)
    centroid_Q = mx.mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    # mlx doesn't have transpose_b kwarg for all functions, we manually transpose
    H = mx.matmul(p.swapaxes(-1, -2), q)

    U, _S, Vt = safe_svd(H)

    R = mx.matmul(Vt.swapaxes(-1, -2), U.swapaxes(-1, -2))

    # MLX does not have det, compute det of 3x3 manually
    # R is batched 3x3 or just 3x3
    # A B C
    # D E F
    # G H I
    # Det = A(EI - FH) - B(DI - FG) + C(DH - EG)
    R00 = R[..., 0, 0]
    R01 = R[..., 0, 1]
    R02 = R[..., 0, 2]
    R10 = R[..., 1, 0]
    R11 = R[..., 1, 1]
    R12 = R[..., 1, 2]
    R20 = R[..., 2, 0]
    R21 = R[..., 2, 1]
    R22 = R[..., 2, 2]

    d = (
        R00 * (R11 * R22 - R12 * R21)
        - R01 * (R10 * R22 - R12 * R20)
        + R02 * (R10 * R21 - R11 * R20)
    )

    # Correction
    d_sign = mx.where(
        d < 0, mx.array(-1.0, dtype=P.dtype), mx.array(1.0, dtype=P.dtype)
    )

    D = P.shape[-1]
    ones = mx.ones((*P.shape[:-2], D - 1), dtype=P.dtype)
    diag = mx.concatenate([ones, mx.expand_dims(d_sign, -1)], axis=-1)

    # MLX diag only works for 1D. We need to construct batched diag
    # Best way is broadcasting multiplication using expanded dims
    I_reflect_V = Vt.swapaxes(-1, -2) * mx.expand_dims(diag, -2)

    R = mx.matmul(I_reflect_V, U.swapaxes(-1, -2))

    t = mx.squeeze(centroid_Q, -2) - mx.squeeze(
        mx.matmul(centroid_P, R.swapaxes(-1, -2)), -2
    )

    P_aligned = mx.matmul(P, R.swapaxes(-1, -2)) + mx.expand_dims(t, -2)
    diff = P_aligned - Q
    mse = mx.mean(mx.sum(mx.square(diff), axis=-1), axis=-1)
    rmsd = mx.sqrt(mx.maximum(mse, 1e-12))

    if orig_dtype in (mx.float16, mx.bfloat16):
        R = R.astype(orig_dtype)
        t = t.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)

    return R, t, rmsd


def kabsch_umeyama(
    P: mx.array, Q: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Computes the optimal rotation, translation, and scale to align P to Q
    (Q ~ c * R @ P + t).

    MLX only supports 3D inputs (dim=3) due to the hardcoded 3x3 determinant
    correction.

    Args:
        P: Source points, shape [..., N, 3].
        Q: Target points, shape [..., N, 3].

    Returns:
        (R, t, c, rmsd): Rotation [..., 3, 3], translation [..., 3],
        scale [...], RMSD [...].
        float16/bfloat16 inputs are upcast to float32 and downcast on output.

    Raises:
        ValueError: If inputs are not 3-dimensional (D != 3).
    """
    if P.shape[-1] != 3:
        raise ValueError(
            f"MLX Kabsch only supports dim=3, got dim={P.shape[-1]}. "
            "Use the JAX, PyTorch, or TensorFlow implementations for N-D alignment."
        )
    orig_dtype = P.dtype
    if orig_dtype in (mx.float16, mx.bfloat16):
        P = P.astype(mx.float32)
        Q = Q.astype(mx.float32)

    N = mx.array(P.shape[-2], dtype=P.dtype)

    centroid_P = mx.mean(P, axis=-2, keepdims=True)
    centroid_Q = mx.mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    var_P = mx.sum(mx.square(p), axis=(-2, -1)) / N
    H = mx.matmul(p.swapaxes(-1, -2), q) / N

    U, S, Vt = safe_svd(H)

    R = mx.matmul(Vt.swapaxes(-1, -2), U.swapaxes(-1, -2))

    # Compute det of 3x3 manually
    R00 = R[..., 0, 0]
    R01 = R[..., 0, 1]
    R02 = R[..., 0, 2]
    R10 = R[..., 1, 0]
    R11 = R[..., 1, 1]
    R12 = R[..., 1, 2]
    R20 = R[..., 2, 0]
    R21 = R[..., 2, 1]
    R22 = R[..., 2, 2]

    d = (
        R00 * (R11 * R22 - R12 * R21)
        - R01 * (R10 * R22 - R12 * R20)
        + R02 * (R10 * R21 - R11 * R20)
    )
    d_sign = mx.where(
        d < 0, mx.array(-1.0, dtype=P.dtype), mx.array(1.0, dtype=P.dtype)
    )

    D = P.shape[-1]
    ones = mx.ones((*P.shape[:-2], D - 1), dtype=P.dtype)
    diag = mx.concatenate([ones, mx.expand_dims(d_sign, -1)], axis=-1)

    # MLX diag only works for 1D. We need to construct batched diag
    # Best way is broadcasting multiplication using expanded dims
    I_reflect_V = Vt.swapaxes(-1, -2) * mx.expand_dims(diag, -2)

    R = mx.matmul(I_reflect_V, U.swapaxes(-1, -2))

    S_corr = mx.concatenate(
        [mx.ones((*S.shape[:-1], D - 1), dtype=P.dtype), mx.expand_dims(d_sign, -1)],
        axis=-1,
    )

    c = mx.sum(S * S_corr, axis=-1) / mx.maximum(var_P, 1e-12)

    centroid_P_rot = mx.matmul(centroid_P, R.swapaxes(-1, -2))
    t = mx.squeeze(centroid_Q, -2) - mx.expand_dims(c, -1) * mx.squeeze(
        centroid_P_rot, -2
    )

    c_exp = mx.expand_dims(mx.expand_dims(c, -1), -1)
    P_aligned = c_exp * mx.matmul(P, R.swapaxes(-1, -2)) + mx.expand_dims(t, -2)
    diff = P_aligned - Q
    mse = mx.mean(mx.sum(mx.square(diff), axis=-1), axis=-1)
    rmsd = mx.sqrt(mx.maximum(mse, 1e-12))

    if orig_dtype in (mx.float16, mx.bfloat16):
        R = R.astype(orig_dtype)
        t = t.astype(orig_dtype)
        c = c.astype(orig_dtype)
        rmsd = rmsd.astype(orig_dtype)

    return R, t, c, rmsd


def kabsch_rmsd(P: mx.array, Q: mx.array) -> mx.array:
    """Computes RMSD after Kabsch alignment. Gradient-safe training loss."""
    _R, _t, rmsd = kabsch(P, Q)
    return rmsd


def kabsch_umeyama_rmsd(P: mx.array, Q: mx.array) -> mx.array:
    """Computes RMSD after Kabsch-Umeyama alignment. Gradient-safe training loss."""
    _R, _t, _c, rmsd = kabsch_umeyama(P, Q)
    return rmsd
