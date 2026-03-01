from collections.abc import Callable
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest
import tensorflow as tf
import torch

from kabsch_umeyama import jax as kabsch_jax
from kabsch_umeyama import mlx as kabsch_mlx
from kabsch_umeyama import pytorch as kabsch_torch
from kabsch_umeyama import tensorflow as kabsch_tf

T = TypeVar("T")


class FrameworkAdapter(Generic[T]):
    @property
    def atol(self) -> float:
        return 1e-5

    @property
    def rtol(self) -> float:
        return 1e-5

    def convert_in(self, arr: np.ndarray) -> T:
        raise NotImplementedError

    def convert_out(self, obj: T) -> np.ndarray:
        raise NotImplementedError

    def kabsch(self, P: T, Q: T) -> tuple[T, ...]:
        raise NotImplementedError

    def kabsch_umeyama(self, P: T, Q: T) -> tuple[T, ...]:
        raise NotImplementedError

    def horn(self, P: T, Q: T) -> tuple[T, ...]:
        raise NotImplementedError

    def horn_with_scale(self, P: T, Q: T) -> tuple[T, ...]:
        raise NotImplementedError

    def is_nan(self, tensor: T) -> bool:
        raise NotImplementedError

    def get_grad(self, P: T, Q: T, func: Callable[[T, T], tuple[T, ...]]) -> np.ndarray:
        raise NotImplementedError


class PyTorchAdapter(FrameworkAdapter[torch.Tensor]):
    def convert_in(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float64, requires_grad=True)

    def convert_out(self, obj: torch.Tensor) -> np.ndarray:
        return obj.detach().numpy() if isinstance(obj, torch.Tensor) else obj

    def kabsch(self, P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return kabsch_torch.kabsch(P, Q)

    def kabsch_umeyama(
        self, P: torch.Tensor, Q: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return kabsch_torch.kabsch_umeyama(P, Q)

    def horn(self, P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return kabsch_torch.horn(P, Q)

    def horn_with_scale(
        self, P: torch.Tensor, Q: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return kabsch_torch.horn_with_scale(P, Q)

    def is_nan(self, tensor: torch.Tensor) -> bool:
        return torch.isnan(tensor).any().item()

    def get_grad(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
        func: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]],
    ) -> np.ndarray:
        res = func(P, Q)
        loss = res[-1].sum()
        loss.backward()
        return P.grad.numpy()


class JAXAdapter(FrameworkAdapter[jax.Array]):
    def convert_in(self, arr: np.ndarray) -> jax.Array:
        return jnp.array(arr, dtype=jnp.float64)

    def convert_out(self, obj: jax.Array) -> np.ndarray:
        return np.array(obj)

    def kabsch(self, P: jax.Array, Q: jax.Array) -> tuple[jax.Array, ...]:
        return kabsch_jax.kabsch(P, Q)

    def kabsch_umeyama(self, P: jax.Array, Q: jax.Array) -> tuple[jax.Array, ...]:
        return kabsch_jax.kabsch_umeyama(P, Q)

    def horn(self, P: jax.Array, Q: jax.Array) -> tuple[jax.Array, ...]:
        return kabsch_jax.horn(P, Q)

    def horn_with_scale(self, P: jax.Array, Q: jax.Array) -> tuple[jax.Array, ...]:
        return kabsch_jax.horn_with_scale(P, Q)

    def is_nan(self, tensor: jax.Array) -> bool:
        return jnp.isnan(tensor).any()

    def get_grad(
        self,
        P: jax.Array,
        Q: jax.Array,
        func: Callable[[jax.Array, jax.Array], tuple[jax.Array, ...]],
    ) -> np.ndarray:
        def loss_fn(P_inner):
            res = func(P_inner, Q)
            return jnp.sum(res[-1])

        grad_fn = jax.grad(loss_fn)
        return np.array(grad_fn(P))


class TFAdapter(FrameworkAdapter[tf.Tensor | tf.Variable]):
    def convert_in(self, arr: np.ndarray) -> tf.Tensor | tf.Variable:
        return tf.Variable(arr, dtype=tf.float64)

    def convert_out(self, obj: tf.Tensor | tf.Variable) -> np.ndarray:
        return obj.numpy()

    def kabsch(
        self, P: tf.Tensor | tf.Variable, Q: tf.Tensor | tf.Variable
    ) -> tuple[tf.Tensor | tf.Variable, ...]:
        return kabsch_tf.kabsch(P, Q)

    def kabsch_umeyama(
        self, P: tf.Tensor | tf.Variable, Q: tf.Tensor | tf.Variable
    ) -> tuple[tf.Tensor | tf.Variable, ...]:
        return kabsch_tf.kabsch_umeyama(P, Q)

    def horn(
        self, P: tf.Tensor | tf.Variable, Q: tf.Tensor | tf.Variable
    ) -> tuple[tf.Tensor | tf.Variable, ...]:
        return kabsch_tf.horn(P, Q)

    def horn_with_scale(
        self, P: tf.Tensor | tf.Variable, Q: tf.Tensor | tf.Variable
    ) -> tuple[tf.Tensor | tf.Variable, ...]:
        return kabsch_tf.horn_with_scale(P, Q)

    def is_nan(self, tensor: tf.Tensor | tf.Variable) -> bool:
        return tf.math.is_nan(tensor).numpy().any()

    def get_grad(
        self,
        P: tf.Tensor | tf.Variable,
        Q: tf.Tensor | tf.Variable,
        func: Callable[
            [tf.Tensor | tf.Variable, tf.Tensor | tf.Variable],
            tuple[tf.Tensor | tf.Variable, ...],
        ],
    ) -> np.ndarray:
        with tf.GradientTape() as tape:
            res = func(P, Q)
            loss = tf.reduce_sum(res[-1])
        return tape.gradient(loss, P).numpy()


class MLXFloat64Adapter(FrameworkAdapter[mx.array]):
    """
    MLX Float64 Adapter (CPU only)
    Apple Silicon GPUs don't support true float64, so MLX forces these ops onto the CPU.
    """

    def convert_in(self, arr: np.ndarray) -> mx.array:
        mx.set_default_device(mx.cpu)
        return mx.array(arr, dtype=mx.float64)

    def convert_out(self, obj: mx.array) -> np.ndarray:
        return np.array(obj)

    def kabsch(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
        mx.set_default_device(mx.cpu)
        return kabsch_mlx.kabsch(P, Q)

    def kabsch_umeyama(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
        mx.set_default_device(mx.cpu)
        return kabsch_mlx.kabsch_umeyama(P, Q)

    def horn(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
        mx.set_default_device(mx.cpu)
        return kabsch_mlx.horn(P, Q)

    def horn_with_scale(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
        mx.set_default_device(mx.cpu)
        return kabsch_mlx.horn_with_scale(P, Q)

    def is_nan(self, tensor: mx.array) -> bool:
        mx.set_default_device(mx.cpu)
        return mx.any(mx.isnan(tensor)).item()

    def get_grad(
        self,
        P: mx.array,
        Q: mx.array,
        func: Callable[[mx.array, mx.array], tuple[mx.array, ...]],
    ) -> np.ndarray:
        mx.set_default_device(mx.cpu)

        def loss_fn(P_inner):
            res = func(P_inner, Q)
            return mx.sum(res[-1])

        grad_fn = mx.grad(loss_fn)
        return np.array(grad_fn(P))


class MLXFloat32Adapter(FrameworkAdapter[mx.array]):
    """
    MLX Float32 Adapter (GPU accelerated)
    Typical usecase for MLX. Lower precision forces us to loosen atol/rtol
    for assertions.
    """

    @property
    def atol(self) -> float:
        return 1e-2

    @property
    def rtol(self) -> float:
        return 1e-2

    def convert_in(self, arr: np.ndarray) -> mx.array:
        mx.set_default_device(mx.gpu)
        # Default mx.array downcasts standard numpy float64 to float32
        return mx.array(arr, dtype=mx.float32)

    def convert_out(self, obj: mx.array) -> np.ndarray:
        return np.array(obj)

    def kabsch(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
        mx.set_default_device(mx.gpu)
        return kabsch_mlx.kabsch(P, Q)

    def kabsch_umeyama(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
        mx.set_default_device(mx.gpu)
        return kabsch_mlx.kabsch_umeyama(P, Q)

    def horn(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
        mx.set_default_device(mx.gpu)
        return kabsch_mlx.horn(P, Q)

    def horn_with_scale(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
        mx.set_default_device(mx.gpu)
        return kabsch_mlx.horn_with_scale(P, Q)

    def is_nan(self, tensor: mx.array) -> bool:
        mx.set_default_device(mx.gpu)
        return mx.any(mx.isnan(tensor)).item()

    def get_grad(
        self,
        P: mx.array,
        Q: mx.array,
        func: Callable[[mx.array, mx.array], tuple[mx.array, ...]],
    ) -> np.ndarray:
        mx.set_default_device(mx.gpu)

        def loss_fn(P_inner):
            res = func(P_inner, Q)
            return mx.sum(res[-1])

        grad_fn = mx.grad(loss_fn)
        return np.array(grad_fn(P))


frameworks = [
    pytest.param(PyTorchAdapter(), id="PyTorch"),
    pytest.param(JAXAdapter(), id="JAX"),
    pytest.param(TFAdapter(), id="TensorFlow"),
    pytest.param(MLXFloat64Adapter(), id="MLX-Float64-CPU"),
    pytest.param(MLXFloat32Adapter(), id="MLX-Float32-GPU"),
]
