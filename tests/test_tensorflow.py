import pytest
import numpy as np
import tensorflow as tf

from kabsch_umeyama.numpy import kabsch_umeyama as kabsch_umeyama_np
from kabsch_umeyama.tensorflow import kabsch_umeyama, kabsch, safe_svd

def test_tf_kabsch_numpy_parity():
    np.random.seed(42)
    P = np.random.randn(3, 10, 3)
    Q = np.random.randn(3, 10, 3)

    R_np, t_np, c_np, rmsd_np = kabsch_umeyama_np(P, Q)

    P_tf = tf.convert_to_tensor(P, dtype=tf.float64)
    Q_tf = tf.convert_to_tensor(Q, dtype=tf.float64)

    R_tf, t_tf, c_tf, rmsd_tf = kabsch_umeyama(P_tf, Q_tf)

    np.testing.assert_allclose(R_tf.numpy(), R_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t_tf.numpy(), t_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(c_tf.numpy(), c_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rmsd_tf.numpy(), rmsd_np, rtol=1e-5, atol=1e-5)


def test_tf_differentiability_trap_cube():
    # A perfect cube is highly degenerate (identical singular values)
    pts = np.array([[
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1]
    ]], dtype=np.float64)

    P = tf.Variable(pts)
    Q = tf.constant(pts)

    with tf.GradientTape() as tape:
        R, t, c, rmsd = kabsch_umeyama(P, Q)
        loss = tf.reduce_sum(R) + tf.reduce_sum(t) + tf.reduce_sum(c) + tf.reduce_sum(rmsd)

    grads = tape.gradient(loss, P)
    assert not tf.reduce_any(tf.math.is_nan(grads))
    assert not tf.reduce_any(tf.math.is_inf(grads))


def test_tf_differentiability_trap_collinear():
    # Collinear points (one non-zero singular value, two zero singular values)
    P_collinear = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]])
    Q_collinear = np.array([[[1.1, 0.9, 1.0], [2.1, 1.9, 2.0], [3.1, 2.9, 3.0]]])

    P = tf.Variable(P_collinear, dtype=tf.float64)
    Q = tf.constant(Q_collinear, dtype=tf.float64)

    with tf.GradientTape() as tape:
        R, t, c, rmsd = kabsch_umeyama(P, Q)
        loss = tf.reduce_sum(R) + tf.reduce_sum(t) + tf.reduce_sum(c) + tf.reduce_sum(rmsd)

    grads = tape.gradient(loss, P)
    assert not tf.reduce_any(tf.math.is_nan(grads))
    assert not tf.reduce_any(tf.math.is_inf(grads))
def test_tf_gradcheck():
    np.random.seed(42)
    P = np.random.randn(1, 4, 3)
    Q = np.random.randn(1, 4, 3)
    
    P_tf = tf.convert_to_tensor(P, dtype=tf.float64)
    Q_tf = tf.convert_to_tensor(Q, dtype=tf.float64)
    
    # We can check gradients using tf.test.compute_gradient
    def wrapped_kabsch(x):
        R, t, rmsd = kabsch(x, Q_tf)
        return R

    # Only run on small shape due to compute_gradient overhead
    theoretical, numerical = tf.test.compute_gradient(wrapped_kabsch, [P_tf])
    # err is max diff
    np.testing.assert_allclose(theoretical[0], numerical[0], rtol=1e-4, atol=1e-4)

