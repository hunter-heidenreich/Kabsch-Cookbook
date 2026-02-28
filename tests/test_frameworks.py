import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest
import tensorflow as tf
import torch

from kabsch_umeyama import jax as kabsch_jax
from kabsch_umeyama import mlx as kabsch_mlx
from kabsch_umeyama import numpy as kabsch_np
from kabsch_umeyama import pytorch as kabsch_torch
from kabsch_umeyama import tensorflow as kabsch_tf


@pytest.fixture
def identity_points():
    np.random.seed(42)
    return np.random.rand(10, 3)


@pytest.fixture
def known_transform_points():
    np.random.seed(42)
    P = np.random.rand(10, 3)

    # 90 degrees around Z axis
    theta = np.pi / 2
    R_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    t_true = np.array([1.0, -2.0, 3.0])
    c_true = 2.5

    Q_kabsch = P @ R_true.T + t_true
    Q_umeyama = c_true * (P @ R_true.T) + t_true

    return P, Q_kabsch, Q_umeyama, R_true, t_true, c_true


# --- Framework Adapters ---
class FrameworkAdapter:
    def convert_in(self, arr: np.ndarray):
        raise NotImplementedError

    def convert_out(self, obj) -> np.ndarray:
        raise NotImplementedError

    def kabsch(self, P, Q):
        raise NotImplementedError

    def kabsch_umeyama(self, P, Q):
        raise NotImplementedError

    def is_nan(self, tensor) -> bool:
        raise NotImplementedError


class PyTorchAdapter(FrameworkAdapter):
    def convert_in(self, arr: np.ndarray):
        return torch.tensor(arr, dtype=torch.float64, requires_grad=True)

    def convert_out(self, obj) -> np.ndarray:
        return obj.detach().numpy() if isinstance(obj, torch.Tensor) else obj

    def kabsch(self, P, Q):
        return kabsch_torch.kabsch(P, Q)

    def kabsch_umeyama(self, P, Q):
        return kabsch_torch.kabsch_umeyama(P, Q)

    def is_nan(self, tensor) -> bool:
        return torch.isnan(tensor).any().item()

    def get_grad(self, P, Q, func):
        res = func(P, Q)
        loss = res[-1].sum()  # RMSD is last
        loss.backward()
        return P.grad.numpy()


class JAXAdapter(FrameworkAdapter):
    def convert_in(self, arr: np.ndarray):
        return jnp.array(arr, dtype=jnp.float64)

    def convert_out(self, obj) -> np.ndarray:
        return np.array(obj)

    def kabsch(self, P, Q):
        return kabsch_jax.kabsch(P, Q)

    def kabsch_umeyama(self, P, Q):
        return kabsch_jax.kabsch_umeyama(P, Q)

    def is_nan(self, tensor) -> bool:
        return jnp.isnan(tensor).any()

    def get_grad(self, P, Q, func):
        def loss_fn(P_inner):
            res = func(P_inner, Q)
            return jnp.sum(res[-1])

        grad_fn = jax.grad(loss_fn)
        return np.array(grad_fn(P))


class TFAdapter(FrameworkAdapter):
    def convert_in(self, arr: np.ndarray):
        return tf.Variable(arr, dtype=tf.float64)

    def convert_out(self, obj) -> np.ndarray:
        return obj.numpy()

    def kabsch(self, P, Q):
        return kabsch_tf.kabsch(P, Q)

    def kabsch_umeyama(self, P, Q):
        return kabsch_tf.kabsch_umeyama(P, Q)

    def is_nan(self, tensor) -> bool:
        return tf.math.is_nan(tensor).numpy().any()

    def get_grad(self, P, Q, func):
        with tf.GradientTape() as tape:
            res = func(P, Q)
            loss = tf.reduce_sum(res[-1])
        return tape.gradient(loss, P).numpy()


class MLXAdapter(FrameworkAdapter):
    def convert_in(self, arr: np.ndarray):
        return mx.array(arr)

    def convert_out(self, obj) -> np.ndarray:
        return np.array(obj)

    def kabsch(self, P, Q):
        return kabsch_mlx.kabsch(P, Q)

    def kabsch_umeyama(self, P, Q):
        return kabsch_mlx.kabsch_umeyama(P, Q)

    def is_nan(self, tensor) -> bool:
        return mx.any(mx.isnan(tensor)).item()

    def get_grad(self, P, Q, func):
        def loss_fn(P_inner):
            res = func(P_inner, Q)
            return mx.sum(res[-1])

        grad_fn = mx.grad(loss_fn)
        return np.array(grad_fn(P))


frameworks = [
    ("PyTorch", PyTorchAdapter()),
    ("JAX", JAXAdapter()),
    ("TensorFlow", TFAdapter()),
    ("MLX", MLXAdapter()),
]


# --- Category A: Forward Pass Equivalence ---
@pytest.mark.parametrize("fw_name, adapter", frameworks)
def test_identity(identity_points, fw_name, adapter):
    P_np = identity_points
    Q_np = np.copy(P_np)

    P = adapter.convert_in(P_np)
    Q = adapter.convert_in(Q_np)

    # Kabsch
    R, t, rmsd = adapter.kabsch(P, Q)
    np.testing.assert_allclose(adapter.convert_out(R), np.eye(3), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        adapter.convert_out(t), np.zeros(3), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(adapter.convert_out(rmsd), 0.0, atol=1e-5, rtol=1e-5)

    # Umeyama
    R, t, c, rmsd = adapter.kabsch_umeyama(P, Q)
    np.testing.assert_allclose(adapter.convert_out(R), np.eye(3), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        adapter.convert_out(t), np.zeros(3), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(adapter.convert_out(c), 1.0, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(adapter.convert_out(rmsd), 0.0, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fw_name, adapter", frameworks)
def test_known_transform(known_transform_points, fw_name, adapter):
    P_np, Q_kabsch_np, Q_umeyama_np, R_true, t_true, c_true = known_transform_points

    P = adapter.convert_in(P_np)

    # Kabsch
    Q_k = adapter.convert_in(Q_kabsch_np)
    R, t, _rmsd = adapter.kabsch(P, Q_k)
    np.testing.assert_allclose(adapter.convert_out(R), R_true, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(adapter.convert_out(t), t_true, atol=1e-5, rtol=1e-5)

    # Umeyama
    Q_u = adapter.convert_in(Q_umeyama_np)
    R, t, c, _rmsd = adapter.kabsch_umeyama(P, Q_u)
    np.testing.assert_allclose(adapter.convert_out(R), R_true, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(adapter.convert_out(t), t_true, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(adapter.convert_out(c), c_true, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fw_name, adapter", frameworks)
def test_numpy_parity(known_transform_points, fw_name, adapter):
    P_np, Q_kabsch_np, Q_umeyama_np, _, _, _ = known_transform_points

    # NumPy ground truth
    R_np, t_np, rmsd_np = kabsch_np.kabsch(P_np, Q_kabsch_np)
    R_u_np, t_u_np, c_u_np, rmsd_u_np = kabsch_np.kabsch_umeyama(P_np, Q_umeyama_np)

    P = adapter.convert_in(P_np)

    # Framework Kabsch
    Q_k = adapter.convert_in(Q_kabsch_np)
    R, t, rmsd = adapter.kabsch(P, Q_k)
    np.testing.assert_allclose(adapter.convert_out(R), R_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(adapter.convert_out(t), t_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(adapter.convert_out(rmsd), rmsd_np, atol=1e-4, rtol=1e-4)

    # Framework Umeyama
    Q_u = adapter.convert_in(Q_umeyama_np)
    R_u, t_u, c_u, rmsd_u = adapter.kabsch_umeyama(P, Q_u)
    np.testing.assert_allclose(adapter.convert_out(R_u), R_u_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(adapter.convert_out(t_u), t_u_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(adapter.convert_out(c_u), c_u_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        adapter.convert_out(rmsd_u), rmsd_u_np, atol=1e-4, rtol=1e-4
    )


# --- Category B: Differentiability Traps ---
@pytest.fixture
def coplanar_points():
    np.random.seed(42)
    P = np.random.rand(10, 3)
    P[:, 2] = 0.0  # Z is zero

    # Random gentle turn around Z
    theta = np.pi / 4
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    Q = P @ R.T + np.array([1.0, 1.0, 0.0])
    return P, Q


@pytest.fixture
def collinear_points():
    np.random.seed(42)
    P = np.random.rand(10, 3)
    P[:, 1] = 0.0  # Y is zero
    P[:, 2] = 0.0  # Z is zero
    Q = P + np.array([2.0, 0.0, 0.0])
    return P, Q


@pytest.fixture
def perfect_cube():
    # Symmetric shape
    P = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    Q = P + np.array([0.5, 0.5, 0.5])
    return P, Q


@pytest.fixture
def reflected_points():
    np.random.seed(42)
    P = np.random.rand(10, 3)
    # Target is mirrored across X-axis, det = -1
    Q = np.copy(P)
    Q[:, 0] = -Q[:, 0]
    return P, Q


@pytest.mark.parametrize("fw_name, adapter", frameworks)
def test_trap_coplanar(coplanar_points, fw_name, adapter):
    P_np, Q_np = coplanar_points
    P = adapter.convert_in(P_np)
    Q = adapter.convert_in(Q_np)

    grad_kabsch = adapter.get_grad(P, Q, adapter.kabsch)
    assert not np.isnan(grad_kabsch).any(), (
        f"Coplanar Trap Failed for {fw_name} (Kabsch)"
    )

    # Reset gradient if TF/PyTorch doesn't automatically
    P = adapter.convert_in(P_np)
    Q = adapter.convert_in(Q_np)

    grad_umeyama = adapter.get_grad(P, Q, adapter.kabsch_umeyama)
    assert not np.isnan(grad_umeyama).any(), (
        f"Coplanar Trap Failed for {fw_name} (Umeyama)"
    )


@pytest.mark.parametrize("fw_name, adapter", frameworks)
def test_trap_collinear(collinear_points, fw_name, adapter):
    P_np, Q_np = collinear_points
    P = adapter.convert_in(P_np)
    Q = adapter.convert_in(Q_np)

    grad = adapter.get_grad(P, Q, adapter.kabsch)
    assert not np.isnan(grad).any(), f"Collinear Trap Failed for {fw_name}"


@pytest.mark.parametrize("fw_name, adapter", frameworks)
def test_trap_perfect_cube(perfect_cube, fw_name, adapter):
    P_np, Q_np = perfect_cube
    P = adapter.convert_in(P_np)
    Q = adapter.convert_in(Q_np)

    grad = adapter.get_grad(P, Q, adapter.kabsch)
    assert not np.isnan(grad).any(), f"Perfect Cube Trap Failed for {fw_name}"


@pytest.mark.parametrize("fw_name, adapter", frameworks)
def test_trap_reflection(reflected_points, fw_name, adapter):
    P_np, Q_np = reflected_points
    P = adapter.convert_in(P_np)
    Q = adapter.convert_in(Q_np)

    # First assert forward pass has DET=1 for Rotation
    R, _t, _rmsd = adapter.kabsch(P, Q)
    R_np = adapter.convert_out(R)
    det = np.linalg.det(R_np)
    # The determinent should be strictly 1, not -1
    assert np.isclose(det, 1.0, atol=1e-3), (
        f"Reflection Trap: Det is not 1 for {fw_name}"
    )

    # Assert gradient
    P = adapter.convert_in(P_np)
    Q = adapter.convert_in(Q_np)
    grad = adapter.get_grad(P, Q, adapter.kabsch)
    assert not np.isnan(grad).any(), f"Reflection Trap Failed for {fw_name}"


# --- Category C: Gradient Verification ---


@pytest.fixture
def batch_points():
    np.random.seed(42)
    P = np.random.rand(5, 10, 3)  # Batch size 5
    Q = P + np.random.rand(5, 1, 3)
    return P, Q


@pytest.mark.parametrize("fw_name, adapter", frameworks)
def test_batching_consistency(batch_points, fw_name, adapter):
    P_np, Q_np = batch_points

    # Batched compute
    P_batch = adapter.convert_in(P_np)
    Q_batch = adapter.convert_in(Q_np)
    grad_batch = adapter.get_grad(P_batch, Q_batch, adapter.kabsch)

    # Sequential compute
    grads_seq = []
    for i in range(5):
        P_seq = adapter.convert_in(P_np[i])
        Q_seq = adapter.convert_in(Q_np[i])
        g = adapter.get_grad(P_seq, Q_seq, adapter.kabsch)
        grads_seq.append(g)

    grad_seq_stacked = np.stack(grads_seq)

    # Tolerances are slightly higher because JAX/TF reduction graphs differ slightly
    np.testing.assert_allclose(
        grad_batch,
        grad_seq_stacked,
        atol=1e-3,
        rtol=1e-3,
        err_msg=f"Batching consistency failed for {fw_name}",
    )
