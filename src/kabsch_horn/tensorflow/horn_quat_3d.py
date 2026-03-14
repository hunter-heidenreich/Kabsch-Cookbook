import numpy as np
import tensorflow as tf


def safe_eigh(A: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Eigendecomposition of a symmetric matrix. Used by call_safe_eigh."""
    return tf.linalg.eigh(A)


@tf.custom_gradient
def call_safe_eigh(A: tf.Tensor) -> tuple[tuple[tf.Tensor, tf.Tensor], ...]:
    """Gradient-safe eigendecomposition for symmetric matrices. Masks near-zero
    eigenvalue differences (< eps) in the backward pass to prevent NaN gradients."""
    L, V = safe_eigh(A)
    eps = tf.cast(np.finfo(L.dtype.as_numpy_dtype).eps, L.dtype)

    def grad(grad_L, grad_V):
        if grad_L is None:
            grad_L = tf.zeros_like(L)
        if grad_V is None:
            grad_V = tf.zeros_like(V)

        D = tf.expand_dims(L, -2) - tf.expand_dims(L, -1)

        mask = tf.abs(D) < eps
        safe_D = tf.where(mask, tf.where(D >= 0, eps, -eps), D)
        safe_D = tf.linalg.set_diag(safe_D, tf.ones_like(tf.linalg.diag_part(safe_D)))

        F = 1.0 / safe_D
        F = tf.linalg.set_diag(F, tf.zeros_like(tf.linalg.diag_part(F)))

        Vt_dV = tf.matmul(V, grad_V, transpose_a=True)

        sym = (Vt_dV - tf.linalg.matrix_transpose(Vt_dV)) / 2
        term = tf.linalg.diag(grad_L) + F * sym
        grad_A = tf.matmul(V, tf.matmul(term, tf.linalg.matrix_transpose(V)))

        return grad_A

    return (L, V), grad


def horn(P: tf.Tensor, Q: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes optimal rotation and translation to align P to Q using Horn's
    quaternion method.

    Strictly 3D only. Uses gradient-safe eigendecomposition (call_safe_eigh) to
    avoid NaN gradients when point clouds are symmetric or degenerate.

    Args:
        P: Source points, shape [..., N, 3].
        Q: Target points, shape [..., N, 3].

    Returns:
        (R, t, rmsd): Rotation [..., 3, 3], translation [..., 3], and RMSD [...].
        float16/bfloat16 inputs are upcast to float32 internally and downcast on output.
    """
    P = tf.convert_to_tensor(P)
    Q = tf.convert_to_tensor(Q)

    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    tf.debugging.assert_equal(
        tf.shape(P), tf.shape(Q), message="P and Q must have the same shape"
    )
    if P.shape[-1] is not None and P.shape[-1] != 3:
        raise ValueError("Horn's method is strictly for 3D point clouds")
    tf.debugging.assert_equal(
        tf.shape(P)[-1], 3, message="Horn's method is strictly for 3D point clouds"
    )
    if P.shape[-2] is not None and P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")
    tf.debugging.assert_greater_equal(
        tf.shape(P)[-2], 2, message="At least 2 points are required for alignment"
    )

    orig_dtype = P.dtype
    if orig_dtype in (tf.float16, tf.bfloat16):
        P = tf.cast(P, tf.float32)
        Q = tf.cast(Q, tf.float32)

    is_single = len(P.shape) == 2
    if is_single:
        P = tf.expand_dims(P, 0)
        Q = tf.expand_dims(Q, 0)

    centroid_P = tf.reduce_mean(P, axis=-2, keepdims=True)
    centroid_Q = tf.reduce_mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = tf.matmul(p, q, transpose_a=True)

    S = H + tf.linalg.matrix_transpose(H)
    tr = tf.linalg.trace(H)

    Delta = tf.stack(
        [
            H[..., 1, 2] - H[..., 2, 1],
            H[..., 2, 0] - H[..., 0, 2],
            H[..., 0, 1] - H[..., 1, 0],
        ],
        axis=-1,
    )

    I3 = tf.eye(3, dtype=H.dtype)

    top_row = tf.expand_dims(tf.concat([tf.expand_dims(tr, -1), Delta], axis=-1), -2)
    bottom_block = tf.concat(
        [
            tf.expand_dims(Delta, -1),
            S - tf.expand_dims(tf.expand_dims(tr, -1), -1) * I3,
        ],
        axis=-1,
    )

    N_mat = tf.concat([top_row, bottom_block], axis=-2)

    _L, V = call_safe_eigh(N_mat)
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

    R = tf.stack(
        [
            tf.stack([R11, R12, R13], axis=-1),
            tf.stack([R21, R22, R23], axis=-1),
            tf.stack([R31, R32, R33], axis=-1),
        ],
        axis=-2,
    )

    t = tf.squeeze(centroid_Q, axis=-2) - tf.squeeze(
        tf.matmul(centroid_P, R, transpose_b=True), axis=-2
    )

    aligned = tf.matmul(p, R, transpose_b=True)
    N_pts_f = tf.cast(tf.shape(P)[-2], P.dtype)
    _eps = np.finfo(P.dtype.as_numpy_dtype).eps
    mse = tf.reduce_sum(tf.square(aligned - q), axis=[-2, -1]) / N_pts_f
    rmsd = tf.sqrt(mse + _eps)

    if is_single:
        R, t, rmsd = R[0], t[0], rmsd[0]
    if orig_dtype in (tf.float16, tf.bfloat16):
        R = tf.cast(R, orig_dtype)
        t = tf.cast(t, orig_dtype)
        rmsd = tf.cast(rmsd, orig_dtype)
    return R, t, rmsd


def horn_with_scale(
    P: tf.Tensor, Q: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes optimal rotation, translation, and scale to align P to Q
    (Q ~ c * R @ P + t).

    Strictly 3D only. Uses gradient-safe eigendecomposition (call_safe_eigh).

    Args:
        P: Source points, shape [..., N, 3].
        Q: Target points, shape [..., N, 3].

    Returns:
        (R, t, c, rmsd): Rotation [..., 3, 3], translation [..., 3],
        scale [...], RMSD [...].
        float16/bfloat16 inputs are upcast to float32 and downcast on output.
    """
    P = tf.convert_to_tensor(P)
    Q = tf.convert_to_tensor(Q)

    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    tf.debugging.assert_equal(
        tf.shape(P), tf.shape(Q), message="P and Q must have the same shape"
    )
    if P.shape[-1] is not None and P.shape[-1] != 3:
        raise ValueError("Horn's method is strictly for 3D point clouds")
    tf.debugging.assert_equal(
        tf.shape(P)[-1], 3, message="Horn's method is strictly for 3D point clouds"
    )
    if P.shape[-2] is not None and P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")
    tf.debugging.assert_greater_equal(
        tf.shape(P)[-2], 2, message="At least 2 points are required for alignment"
    )

    orig_dtype = P.dtype
    if orig_dtype in (tf.float16, tf.bfloat16):
        P = tf.cast(P, tf.float32)
        Q = tf.cast(Q, tf.float32)

    is_single = len(P.shape) == 2
    if is_single:
        P = tf.expand_dims(P, 0)
        Q = tf.expand_dims(Q, 0)

    centroid_P = tf.reduce_mean(P, axis=-2, keepdims=True)
    centroid_Q = tf.reduce_mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    N_pts_f = tf.cast(tf.shape(P)[-2], P.dtype)
    var_P = tf.reduce_sum(tf.square(p), axis=[-2, -1]) / N_pts_f

    H = tf.matmul(p, q, transpose_a=True) / N_pts_f

    S = H + tf.linalg.matrix_transpose(H)
    tr = tf.linalg.trace(H)

    Delta = tf.stack(
        [
            H[..., 1, 2] - H[..., 2, 1],
            H[..., 2, 0] - H[..., 0, 2],
            H[..., 0, 1] - H[..., 1, 0],
        ],
        axis=-1,
    )

    I3 = tf.eye(3, dtype=H.dtype)

    top_row = tf.expand_dims(tf.concat([tf.expand_dims(tr, -1), Delta], axis=-1), -2)
    bottom_block = tf.concat(
        [
            tf.expand_dims(Delta, -1),
            S - tf.expand_dims(tf.expand_dims(tr, -1), -1) * I3,
        ],
        axis=-1,
    )

    N_mat = tf.concat([top_row, bottom_block], axis=-2)

    _L, V = call_safe_eigh(N_mat)
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

    R = tf.stack(
        [
            tf.stack([R11, R12, R13], axis=-1),
            tf.stack([R21, R22, R23], axis=-1),
            tf.stack([R31, R32, R33], axis=-1),
        ],
        axis=-2,
    )

    _eps = np.finfo(P.dtype.as_numpy_dtype).eps
    RH = tf.reduce_sum(R * tf.linalg.matrix_transpose(H), axis=[-2, -1])
    c = RH / tf.maximum(var_P, _eps)

    t = tf.squeeze(centroid_Q, axis=-2) - tf.expand_dims(c, -1) * tf.squeeze(
        tf.matmul(centroid_P, R, transpose_b=True), axis=-2
    )

    aligned_P = tf.expand_dims(tf.expand_dims(c, -1), -1) * tf.matmul(
        P, R, transpose_b=True
    ) + tf.expand_dims(t, -2)
    diff = aligned_P - Q
    mse = tf.reduce_sum(tf.square(diff), axis=[-2, -1]) / N_pts_f
    rmsd = tf.sqrt(mse + _eps)

    if is_single:
        R, t, c, rmsd = R[0], t[0], c[0], rmsd[0]
    if orig_dtype in (tf.float16, tf.bfloat16):
        R = tf.cast(R, orig_dtype)
        t = tf.cast(t, orig_dtype)
        c = tf.cast(c, orig_dtype)
        rmsd = tf.cast(rmsd, orig_dtype)
    return R, t, c, rmsd
