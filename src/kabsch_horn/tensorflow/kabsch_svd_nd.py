import tensorflow as tf


@tf.custom_gradient
def safe_svd(A: tf.Tensor) -> tuple[tf.Tensor, ...]:
    """
    Computes SVD with a custom gradient to handle degenerate cases (zero singular values
    and coincident singular values) robustly.
    """
    S, U, V = tf.linalg.svd(A, full_matrices=False)

    def grad(dS: tf.Tensor, dU: tf.Tensor, dV: tf.Tensor) -> tf.Tensor:
        # Note: tf.linalg.svd returns V, not V.H. So V in tf.svd is V_python
        # The reconstruction is A = U * S * V.T

        # S_mat is a diagonal matrix containing the singular values
        S_mat = tf.linalg.diag(S)

        # Transpose U, V
        Ut = tf.linalg.matrix_transpose(U)
        Vt = tf.linalg.matrix_transpose(V)

        # dS needs to be expanded or diagonalized safely
        if dS is not None:
            dS_mat = tf.linalg.diag(dS)
        else:
            dS_mat = tf.zeros_like(S_mat)

        # Create F matrix: F_ij = 1 / (S_j^2 - S_i^2)
        S_sq = tf.square(S)
        S_sq_diff = tf.expand_dims(S_sq, -2) - tf.expand_dims(S_sq, -1)

        # Safe denominator: replace near-zero differences with eps * sign
        # to prevent 1/0 = inf on off-diagonal entries where S_i ≈ S_j
        eps = tf.cast(1e-12, S.dtype)
        mask = tf.abs(S_sq_diff) < eps
        safe_D = tf.where(mask, eps * tf.sign(S_sq_diff + eps), S_sq_diff)
        safe_D = tf.linalg.set_diag(safe_D, tf.ones_like(tf.linalg.diag_part(safe_D)))
        F = 1.0 / safe_D
        # Zero out the diagonal of F
        F = tf.linalg.set_diag(F, tf.zeros_like(tf.linalg.diag_part(F)))

        if dU is not None:
            Ut_dU = tf.matmul(Ut, dU)
            J = F * (Ut_dU - tf.linalg.matrix_transpose(Ut_dU))
        else:
            J = tf.zeros_like(S_mat)

        if dV is not None:
            # Note for PyTorch we did Vht_dVh = Vh @ dVh.mH -> V^T @ dV
            # TF returns V. So Vt_dV = V^T @ dV
            Vt_dV = tf.matmul(Vt, dV)
            K = F * (Vt_dV - tf.linalg.matrix_transpose(Vt_dV))
        else:
            K = tf.zeros_like(S_mat)

        term = dS_mat + tf.matmul(J, S_mat) + tf.matmul(S_mat, K)

        # This worked perfectly for the exact same formulation
        # dA = U @ term @ V.T
        dA = tf.matmul(U, tf.matmul(term, Vt))

        # Safe-guard NaNs
        dA = tf.where(tf.math.is_nan(dA), tf.zeros_like(dA), dA)

        return dA

    # TF svd returns (S, U, V), so returning those. Output of svd is S, U, V
    return (S, U, V), grad


def kabsch(P: tf.Tensor, Q: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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
    if orig_dtype in (tf.float16, tf.bfloat16):
        P = tf.cast(P, tf.float32)
        Q = tf.cast(Q, tf.float32)

    # Centering
    centroid_P = tf.reduce_mean(P, axis=-2, keepdims=True)
    centroid_Q = tf.reduce_mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    N = tf.cast(tf.shape(P)[-2], P.dtype)
    H = tf.matmul(p, q, transpose_a=True) / N

    # SVD
    _S, U, V = safe_svd(H)

    # Calculate rotation
    R = tf.matmul(V, tf.linalg.matrix_transpose(U))

    # Reflection check
    det = tf.linalg.det(R)
    # If det < 0, reflect the last column of V
    # Create the reflection matrix
    dim = tf.shape(P)[-1]

    # Sign of determinant
    d_sign = tf.sign(det)
    d_sign = tf.where(d_sign == 0, tf.ones_like(d_sign), d_sign)

    # Process batched tensors differently if P is batched or unbatched
    # We construct a diagonal array
    ones = tf.ones_like(d_sign)
    diag_vals = tf.concat(
        [
            tf.repeat(tf.expand_dims(ones, -1), dim - 1, axis=-1),
            tf.expand_dims(d_sign, -1),
        ],
        axis=-1,
    )

    I_reflect = tf.linalg.diag(diag_vals)

    R = tf.matmul(V, tf.matmul(I_reflect, tf.linalg.matrix_transpose(U)))

    # Translation
    t = tf.squeeze(centroid_Q, -2) - tf.squeeze(
        tf.matmul(centroid_P, R, transpose_b=True), -2
    )

    # RMSD
    P_aligned = tf.matmul(P, R, transpose_b=True) + tf.expand_dims(t, -2)
    diff = P_aligned - Q
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=-1), axis=-1)
    rmsd = tf.sqrt(tf.maximum(mse, 1e-12))

    if orig_dtype in (tf.float16, tf.bfloat16):
        R = tf.cast(R, orig_dtype)
        t = tf.cast(t, orig_dtype)
        rmsd = tf.cast(rmsd, orig_dtype)

    return R, t, rmsd


def kabsch_umeyama(
    P: tf.Tensor, Q: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes the optimal rotation, translation, and scale (Q ~ c * R @ P + t).

    Args:
        P: Source points, shape [..., N, D].
        Q: Target points, shape [..., N, D].

    Returns:
        (R, t, c, rmsd): Rotation [..., D, D], translation [..., D], scale [...],
        RMSD [...].

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
    if P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")

    orig_dtype = P.dtype
    if orig_dtype in (tf.float16, tf.bfloat16):
        P = tf.cast(P, tf.float32)
        Q = tf.cast(Q, tf.float32)

    # Centering
    centroid_P = tf.reduce_mean(P, axis=-2, keepdims=True)
    centroid_Q = tf.reduce_mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    # Variance of P
    var_P = tf.reduce_sum(tf.square(p), axis=[-2, -1]) / tf.cast(
        tf.shape(P)[-2], P.dtype
    )

    # Covariance
    N = tf.cast(tf.shape(P)[-2], P.dtype)
    H = tf.matmul(p, q, transpose_a=True) / N

    # SVD
    S, U, V_m = safe_svd(H)

    R = tf.matmul(V_m, tf.linalg.matrix_transpose(U))

    # Reflection check
    det = tf.linalg.det(R)

    dim = tf.shape(P)[-1]

    d_sign = tf.sign(det)
    d_sign = tf.where(d_sign == 0, tf.ones_like(d_sign), d_sign)

    ones = tf.ones_like(d_sign)
    diag_vals = tf.concat(
        [
            tf.repeat(tf.expand_dims(ones, -1), dim - 1, axis=-1),
            tf.expand_dims(d_sign, -1),
        ],
        axis=-1,
    )

    I_reflect = tf.linalg.diag(diag_vals)

    R = tf.matmul(V_m, tf.matmul(I_reflect, tf.linalg.matrix_transpose(U)))

    # Scale
    S_corr = tf.linalg.diag_part(I_reflect)

    # Var_P is batched, S is batched. S * S_corr sum over last dim
    c = tf.reduce_sum(S * S_corr, axis=-1) / tf.maximum(var_P, 1e-12)

    # Translation
    # t = mean_Q - c * R @ mean_P
    centroid_P_rot = tf.matmul(centroid_P, R, transpose_b=True)
    t = tf.squeeze(centroid_Q, -2) - tf.expand_dims(c, -1) * tf.squeeze(
        centroid_P_rot, -2
    )

    # RMSD
    c_exp = tf.expand_dims(tf.expand_dims(c, -1), -1)
    P_aligned = c_exp * tf.matmul(P, R, transpose_b=True) + tf.expand_dims(t, -2)
    diff = P_aligned - Q
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=-1), axis=-1)
    rmsd = tf.sqrt(tf.maximum(mse, 1e-12))

    if orig_dtype in (tf.float16, tf.bfloat16):
        R = tf.cast(R, orig_dtype)
        t = tf.cast(t, orig_dtype)
        c = tf.cast(c, orig_dtype)
        rmsd = tf.cast(rmsd, orig_dtype)

    return R, t, c, rmsd


def kabsch_rmsd(P: tf.Tensor, Q: tf.Tensor) -> tf.Tensor:
    """Computes RMSD after Kabsch alignment."""
    _R, _t, rmsd = kabsch(P, Q)
    return rmsd


def kabsch_umeyama_rmsd(P: tf.Tensor, Q: tf.Tensor) -> tf.Tensor:
    """Computes RMSD after Kabsch-Umeyama alignment."""
    _R, _t, _c, rmsd = kabsch_umeyama(P, Q)
    return rmsd
