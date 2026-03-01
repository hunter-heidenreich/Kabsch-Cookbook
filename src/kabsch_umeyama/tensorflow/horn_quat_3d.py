import tensorflow as tf


def safe_eigh(A, eps=1e-12):
    return tf.linalg.eigh(A)


@tf.custom_gradient
def call_safe_eigh(A):
    L, V = safe_eigh(A)
    eps = tf.cast(1e-12, L.dtype)

    def grad(grad_L, grad_V):
        if grad_L is None:
            grad_L = tf.zeros_like(L)
        if grad_V is None:
            grad_V = tf.zeros_like(V)

        D = tf.expand_dims(L, -1) - tf.expand_dims(L, -2)

        mask = tf.abs(D) < eps
        safe_D = tf.where(mask, eps * tf.sign(D + eps), D)
        safe_D = tf.linalg.set_diag(safe_D, tf.ones_like(tf.linalg.diag_part(safe_D)))

        F = 1.0 / safe_D
        F = tf.linalg.set_diag(F, tf.zeros_like(tf.linalg.diag_part(F)))

        Vt_dV = tf.matmul(V, grad_V, transpose_a=True)

        term = tf.linalg.diag(grad_L) + F * (Vt_dV - tf.linalg.matrix_transpose(Vt_dV))
        grad_A = tf.matmul(V, tf.matmul(term, tf.linalg.matrix_transpose(V)))

        return grad_A

    return (L, V), grad


def horn(P: tf.Tensor, Q: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # Assume 3D
    P = tf.convert_to_tensor(P)
    Q = tf.convert_to_tensor(Q)

    is_single = tf.rank(P) == 2
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

    B_sz = tf.shape(H)[0]
    I3 = tf.eye(3, batch_shape=[B_sz], dtype=H.dtype)

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
    N_pts_f = tf.cast(tf.shape(P)[1], P.dtype)
    rmsd = tf.sqrt(
        tf.maximum(tf.reduce_sum(tf.square(aligned - q), axis=[1, 2]) / N_pts_f, 1e-12)
    )

    if is_single:
        return R[0], t[0], rmsd[0]
    return R, t, rmsd


def horn_with_scale(
    P: tf.Tensor, Q: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    P = tf.convert_to_tensor(P)
    Q = tf.convert_to_tensor(Q)

    is_single = tf.rank(P) == 2
    if is_single:
        P = tf.expand_dims(P, 0)
        Q = tf.expand_dims(Q, 0)

    centroid_P = tf.reduce_mean(P, axis=-2, keepdims=True)
    centroid_Q = tf.reduce_mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    N_pts_f = tf.cast(tf.shape(P)[1], P.dtype)
    var_P = tf.reduce_sum(tf.square(p), axis=[1, 2]) / N_pts_f

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

    B_sz = tf.shape(H)[0]
    I3 = tf.eye(3, batch_shape=[B_sz], dtype=H.dtype)

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

    RH = tf.reduce_sum(R * tf.linalg.matrix_transpose(H), axis=[1, 2])
    c = RH / tf.maximum(var_P, 1e-12)

    t = tf.squeeze(centroid_Q, axis=-2) - tf.expand_dims(c, -1) * tf.squeeze(
        tf.matmul(centroid_P, R, transpose_b=True), axis=-2
    )

    aligned_P = tf.expand_dims(tf.expand_dims(c, -1), -1) * tf.matmul(
        P, R, transpose_b=True
    ) + tf.expand_dims(t, -2)
    diff = aligned_P - Q
    rmsd = tf.sqrt(
        tf.maximum(tf.reduce_sum(tf.square(diff), axis=[1, 2]) / N_pts_f, 1e-12)
    )

    if is_single:
        return R[0], t[0], c[0], rmsd[0]
    return R, t, c, rmsd
