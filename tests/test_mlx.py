import pytest
import numpy as np
import mlx.core as mx

from kabsch_umeyama.numpy import kabsch_umeyama as kabsch_umeyama_np
from kabsch_umeyama.mlx import kabsch_umeyama, kabsch, safe_svd

def test_mlx_kabsch_numpy_parity():
    np.random.seed(42)
    P_np = np.random.randn(3, 10, 3)
    Q_np = np.random.randn(3, 10, 3)

    R_np, t_np, c_np, rmsd_np = kabsch_umeyama_np(P_np, Q_np)

    P_mx = mx.array(P_np, mx.float32)
    Q_mx = mx.array(Q_np, mx.float32)

    R_mx, t_mx, c_mx = kabsch_umeyama(P_mx, Q_mx)

    np.testing.assert_allclose(np.array(R_mx), R_np, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(t_mx), t_np, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(c_mx), c_np, rtol=1e-4, atol=1e-4)


def test_mlx_differentiability_trap_cube():
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
    ]], dtype=np.float32)

    def compute_loss(P_mx, Q_mx):
        R, t, c = kabsch_umeyama(P_mx, Q_mx)
        return mx.sum(R) + mx.sum(t) + mx.sum(c)

    grad_fn = mx.grad(compute_loss, argnums=0)
    P_mx = mx.array(pts)
    Q_mx = mx.array(pts)
    
    grads = grad_fn(P_mx, Q_mx)
    
    assert not np.any(np.isnan(grads))
    assert not np.any(np.isinf(grads))


def test_mlx_differentiability_trap_collinear():
    # Collinear points (one non-zero singular value, two zero singular values)
    P_collinear = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]], dtype=np.float32)
    Q_collinear = np.array([[[1.1, 0.9, 1.0], [2.1, 1.9, 2.0], [3.1, 2.9, 3.0]]], dtype=np.float32)

    def compute_loss(P_mx, Q_mx):
        R, t, c = kabsch_umeyama(P_mx, Q_mx)
        return mx.sum(R) + mx.sum(t) + mx.sum(c)

    grad_fn = mx.grad(compute_loss, argnums=0)
    P_mx = mx.array(P_collinear)
    Q_mx = mx.array(Q_collinear)
    
    grads = grad_fn(P_mx, Q_mx)
    
    assert not np.any(np.isnan(grads))
    assert not np.any(np.isinf(grads))
