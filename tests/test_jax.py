import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.test_util import check_vjp
from kabsch_umeyama.numpy import kabsch as kabsch_np, kabsch_umeyama as umeyama_np
from kabsch_umeyama.jax import kabsch as kabsch_jax, kabsch_umeyama as umeyama_jax

jax.config.update("jax_enable_x64", True)

@pytest.fixture
def identity_points():
    np.random.seed(42)
    P = np.random.randn(2, 10, 3)
    return P, P.copy()

@pytest.fixture
def known_transform_points():
    np.random.seed(42)
    B, N, D = 2, 10, 3
    P = np.random.randn(B, N, D)
    
    theta = np.pi / 2
    R = np.array([
        [[np.cos(theta), -np.sin(theta), 0],
         [np.sin(theta), np.cos(theta),  0],
         [0,             0,              1]],
        [[1, 0, 0],
         [0, np.cos(theta), -np.sin(theta)],
         [0, np.sin(theta), np.cos(theta)]]
    ])
    
    t = np.array([[1.0, -2.0, 3.0], 
                  [-1.0, 0.5, 2.0]])
    c = np.array([2.0, 0.5])
    
    Q = c[:, np.newaxis, np.newaxis] * np.matmul(P, R.transpose(0, 2, 1)) + t[:, np.newaxis, :]
    return P, Q, R, t, c

@pytest.fixture
def degenerate_points():
    np.random.seed(42)
    # Perfect cube
    P_cube = np.array([
        [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1],
        [1,-1,-1],  [1,-1,1],  [1,1,-1],  [1,1,1]
    ], dtype=np.float32)
    
    P_collinear = P_cube.copy()
    P_collinear[:, 1] = 0
    P_collinear[:, 2] = 0
    
    return P_cube[np.newaxis, ...], P_collinear[np.newaxis, ...]


def test_jax_kabsch_numpy_parity(known_transform_points):
    P_np, Q_np, R_true, t_true, c_true = known_transform_points
    Q_np_unscaled = np.matmul(P_np, R_true.transpose(0, 2, 1)) + t_true[:, np.newaxis, :]
    
    R_np, t_np, rmsd_np = kabsch_np(P_np, Q_np_unscaled)
    
    P_jax = jnp.array(P_np, dtype=jnp.float64)
    Q_jax = jnp.array(Q_np_unscaled, dtype=jnp.float64)
    
    R_jax, t_jax, rmsd_jax = kabsch_jax(P_jax, Q_jax)
    
    np.testing.assert_allclose(np.array(R_jax), R_np, atol=1e-5)
    np.testing.assert_allclose(np.array(t_jax), t_np, atol=1e-5)
    np.testing.assert_allclose(np.array(rmsd_jax), rmsd_np, atol=1e-5)


def test_jax_umeyama_numpy_parity(known_transform_points):
    P_np, Q_np, R_true, t_true, c_true = known_transform_points
    
    R_np, t_np, c_np, rmsd_np = umeyama_np(P_np, Q_np)
    
    P_jax = jnp.array(P_np, dtype=jnp.float64)
    Q_jax = jnp.array(Q_np, dtype=jnp.float64)
    
    R_jax, t_jax, c_jax, rmsd_jax = umeyama_jax(P_jax, Q_jax)
    
    np.testing.assert_allclose(np.array(R_jax), R_np, atol=1e-5)
    np.testing.assert_allclose(np.array(t_jax), t_np, atol=1e-5)
    np.testing.assert_allclose(np.array(c_jax), c_np, atol=1e-5)
    np.testing.assert_allclose(np.array(rmsd_jax), rmsd_np, atol=1e-5)


def test_jax_differentiability_trap_cube(degenerate_points):
    P_cube, P_collinear = degenerate_points
    
    def loss_fn(P):
        Q = P + 0.1
        R, t, rmsd = kabsch_jax(P, Q)
        return jnp.sum(rmsd)
        
    grad_fn = jax.grad(loss_fn)
    P_jax = jnp.array(P_cube, dtype=jnp.float64)
    
    grads = grad_fn(P_jax)
    assert not jnp.isnan(grads).any()
    assert not jnp.isinf(grads).any()


def test_jax_differentiability_trap_collinear(degenerate_points):
    P_cube, P_collinear = degenerate_points
    
    def loss_fn(P):
        Q = P + 0.1
        R, t, c, rmsd = umeyama_jax(P, Q)
        return jnp.sum(rmsd) + jnp.sum(R) + jnp.sum(t) + jnp.sum(c)
        
    grad_fn = jax.grad(loss_fn)
    P_jax = jnp.array(P_collinear, dtype=jnp.float64)
    
    grads = grad_fn(P_jax)
    assert not jnp.isnan(grads).any()
    assert not jnp.isinf(grads).any()


def test_jax_gradcheck():
    np.random.seed(42)
    P = jnp.array(np.random.randn(1, 4, 3), dtype=jnp.float64)
    Q = jnp.array(np.random.randn(1, 4, 3), dtype=jnp.float64)
    
    p = P - jnp.mean(P, axis=1, keepdims=True)
    H = jnp.matmul(jnp.swapaxes(p, 1, 2), p)
    
    from kabsch_umeyama.jax import safe_svd
    from jax.test_util import check_grads
    
    # Check custom VJP natively
    check_grads(safe_svd, args=(H,), order=1, modes=('rev',), rtol=1e-4, atol=1e-4)

    def wrapped_kabsch(x):
        from kabsch_umeyama.jax import kabsch as kabsch_jax
        return kabsch_jax(x, Q)[0]
        
    check_grads(wrapped_kabsch, args=(P,), order=1, modes=('rev',), rtol=1e-4, atol=1e-4)
