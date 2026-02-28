import tensorflow as tf
from typing import Tuple

@tf.custom_gradient
def safe_svd(A: tf.Tensor) -> Tuple[tf.Tensor, ...]:
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
        
        # Add epsilon to diagonal before reciprocal to avoid Division by Zero problems
        eye = tf.eye(tf.shape(S)[-1], dtype=S.dtype)
        S_sq_diff_safe = tf.where(tf.abs(S_sq_diff) < 1e-12, eye * 1e-12, S_sq_diff)
        
        F = 1.0 / S_sq_diff_safe
        # Zero out the diagonal of F
        F = tf.linalg.set_diag(F, tf.zeros_like(S))
        
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


def kabsch(P: tf.Tensor, Q: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Computes the optimal rotation and translation to align P and Q.
    """
    # Centering
    centroid_P = tf.reduce_mean(P, axis=-2, keepdims=True)
    centroid_Q = tf.reduce_mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    N = tf.cast(tf.shape(P)[-2], P.dtype)
    H = tf.matmul(p, q, transpose_a=True) / N

    # SVD
    S, U, V = safe_svd(H)

    # Calculate rotation
    Vt = tf.linalg.matrix_transpose(V)
    R = tf.matmul(V, tf.linalg.matrix_transpose(U))

    # Reflection check
    det = tf.linalg.det(R)
    # If det < 0, reflect the last column of V
    # Create the reflection matrix
    dim = tf.shape(P)[-1]
    
    # Create ones except the last element which is det
    ones = tf.ones([tf.shape(P)[0], dim - 1], dtype=P.dtype)
    diag = tf.concat([ones, tf.expand_dims(det, -1)], axis=-1)
    
    # We need to broadcast the eye
    I_reflect = tf.linalg.diag(diag)

    R = tf.matmul(V, tf.matmul(I_reflect, tf.linalg.matrix_transpose(U)))

    # Translation
    t = tf.squeeze(centroid_Q, -2) - tf.squeeze(tf.matmul(centroid_P, R, transpose_b=True), -2)

    # RMSD
    P_aligned = tf.matmul(P, R, transpose_b=True) + tf.expand_dims(t, -2)
    diff = P_aligned - Q
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=-1), axis=-1)
    rmsd = tf.sqrt(tf.maximum(mse, 1e-12))

    return R, t, rmsd


def kabsch_umeyama(P: tf.Tensor, Q: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes the optimal rotation, translation, and scale.
    """
    # Centering
    centroid_P = tf.reduce_mean(P, axis=-2, keepdims=True)
    centroid_Q = tf.reduce_mean(Q, axis=-2, keepdims=True)

    p = P - centroid_P
    q = Q - centroid_Q

    # Variance of P
    var_P = tf.reduce_sum(tf.square(p), axis=[-2, -1]) / tf.cast(tf.shape(P)[-2], P.dtype)

    # Covariance
    N = tf.cast(tf.shape(P)[-2], P.dtype)
    H = tf.matmul(p, q, transpose_a=True) / N

    # SVD
    S, U, V_m = safe_svd(H)
    
    R = tf.matmul(V_m, tf.linalg.matrix_transpose(U))

    # Reflection check
    det = tf.linalg.det(R)
    
    dim = tf.shape(P)[-1]
    ones = tf.ones([tf.shape(P)[0], dim - 1], dtype=P.dtype)
    diag = tf.concat([ones, tf.expand_dims(det, -1)], axis=-1)
    I_reflect = tf.linalg.diag(diag)

    R = tf.matmul(V_m, tf.matmul(I_reflect, tf.linalg.matrix_transpose(U)))

    # Scale
    S_corr = tf.linalg.diag_part(I_reflect)
    
    # Var_P is batched, S is batched. S * S_corr sum over last dim
    c = tf.reduce_sum(S * S_corr, axis=-1) / tf.maximum(var_P, 1e-12)

    # Translation
    # t = mean_Q - c * R @ mean_P
    centroid_P_rot = tf.matmul(centroid_P, R, transpose_b=True)
    t = tf.squeeze(centroid_Q, -2) - tf.expand_dims(c, -1) * tf.squeeze(centroid_P_rot, -2)
    
    # RMSD
    c_exp = tf.expand_dims(tf.expand_dims(c, -1), -1)
    P_aligned = c_exp * tf.matmul(P, R, transpose_b=True) + tf.expand_dims(t, -2)
    diff = P_aligned - Q
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=-1), axis=-1)
    rmsd = tf.sqrt(tf.maximum(mse, 1e-12))
    
    return R, t, c, rmsd


def kabsch_rmsd(P: tf.Tensor, Q: tf.Tensor) -> tf.Tensor:
    """Computes RMSD after Kabsch alignment."""
    R, t, rmsd = kabsch(P, Q)
    return rmsd

def kabsch_umeyama_rmsd(P: tf.Tensor, Q: tf.Tensor) -> tf.Tensor:
    """Computes RMSD after Kabsch-Umeyama alignment."""
    R, t, c, rmsd = kabsch_umeyama(P, Q)
    return rmsd
