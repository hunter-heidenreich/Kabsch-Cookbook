from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch

from kabsch_horn import jax as kabsch_jax
from kabsch_horn import pytorch as kabsch_torch
from kabsch_horn import tensorflow as kabsch_tf

try:
    import mlx.core as mx

    from kabsch_horn import mlx as kabsch_mlx

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

T = TypeVar("T")


class FrameworkAdapter(Generic[T]):
    _TOLERANCES: ClassVar[dict[str, dict[str, float]]] = {
        "float16": {"eps": 1e-2, "atol": 1e-1, "rtol": 1e-1},
        "bfloat16": {"eps": 1e-2, "atol": 1e-1, "rtol": 1e-1},
        "float32": {"eps": 1e-3, "atol": 5e-2, "rtol": 5e-2},
        "float64": {"eps": 1e-5, "atol": 1e-5, "rtol": 1e-5},
    }

    def __init__(self, precision: str = "float64"):
        if precision not in self._TOLERANCES:
            raise ValueError(f"Unknown precision: {precision}")
        self.precision = precision

    @property
    def eps(self) -> float:
        return self._TOLERANCES[self.precision]["eps"]

    @property
    def atol(self) -> float:
        return self._TOLERANCES[self.precision]["atol"]

    @property
    def rtol(self) -> float:
        return self._TOLERANCES[self.precision]["rtol"]

    def supports_dim(self, dim: int) -> bool:
        """Indicates whether this adapter supports N-D inputs."""
        return True

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

    def get_transform_func(self, algo: str) -> Callable[[T, T], tuple[T, ...]]:
        """Returns the corresponding transformation function for the given algorithm."""
        if algo == "kabsch":
            return self.kabsch
        if algo == "umeyama":
            return self.kabsch_umeyama
        if algo == "horn":
            return self.horn
        if algo == "horn_with_scale":
            return self.horn_with_scale
        raise ValueError(f"Unknown algorithm: {algo}")

    def is_nan(self, tensor: T) -> bool:
        raise NotImplementedError

    def get_grad(
        self,
        P: T,
        Q: T,
        func: Callable[[T, T], tuple[T, ...]],
        seed: int | None = 42,
        wrt: str = "P",
    ) -> np.ndarray:
        raise NotImplementedError

    @property
    def mismatch_exception_type(self) -> type[Exception] | tuple[type[Exception], ...]:
        """The exception(s) raised when P and Q have different point counts."""
        return Exception


class PyTorchAdapter(FrameworkAdapter[torch.Tensor]):
    _DTYPE_MAP: ClassVar[dict[str, torch.dtype]] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    def convert_in(self, arr: np.ndarray) -> torch.Tensor:
        dtype = self._DTYPE_MAP[self.precision]
        return torch.tensor(arr, dtype=dtype, requires_grad=True)

    def convert_out(self, obj: torch.Tensor) -> np.ndarray:
        if isinstance(obj, torch.Tensor):
            if obj.dtype in (torch.bfloat16, torch.float16):
                obj = obj.float()
            return obj.detach().numpy()
        return obj

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
        seed: int | None = 42,
        wrt: str = "P",
    ) -> np.ndarray:
        res = func(P, Q)
        if seed is not None:
            rng = np.random.RandomState(seed)
            dtype = self._DTYPE_MAP[self.precision]
            weights = [
                torch.tensor(rng.normal(size=tensor.shape), dtype=dtype)
                for tensor in res
            ]
            loss = sum(
                (tensor * weight).sum()
                for tensor, weight in zip(res, weights, strict=False)
            )
        else:
            loss = sum([tensor.sum() for tensor in res])
        loss.backward()

        grad = P.grad if wrt == "P" else Q.grad
        if grad is not None and grad.dtype in (torch.bfloat16, torch.float16):
            grad = grad.float()

        return grad.numpy()

    @property
    def mismatch_exception_type(self) -> type[Exception] | tuple[type[Exception], ...]:
        return (AssertionError, ValueError)


class JAXAdapter(FrameworkAdapter[jax.Array]):
    _DTYPE_MAP: ClassVar[dict[str, jnp.dtype]] = {
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
        "float64": jnp.float64,
    }

    def convert_in(self, arr: np.ndarray) -> jax.Array:
        dtype = self._DTYPE_MAP[self.precision]
        return jnp.array(arr, dtype=dtype)

    def convert_out(self, obj: jax.Array) -> np.ndarray:
        ret = np.array(obj)
        if ret.dtype in (np.float16,):
            ret = ret.astype(np.float32)
        return ret

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
        seed: int | None = 42,
        wrt: str = "P",
    ) -> np.ndarray:
        def loss_fn(P_inner, Q_inner):
            res = func(P_inner, Q_inner)
            if seed is not None:
                rng = np.random.RandomState(seed)
                dtype = self._DTYPE_MAP[self.precision]
                weights = [
                    jnp.array(rng.normal(size=tensor.shape), dtype=dtype)
                    for tensor in res
                ]
                return sum(
                    [
                        jnp.sum(tensor * weight)
                        for tensor, weight in zip(res, weights, strict=False)
                    ]
                )
            else:
                return sum([jnp.sum(tensor) for tensor in res])

        arg_idx = 0 if wrt == "P" else 1
        grad_fn = jax.grad(loss_fn, argnums=arg_idx)
        return np.array(grad_fn(P, Q))

    @property
    def mismatch_exception_type(self) -> type[Exception] | tuple[type[Exception], ...]:
        return (TypeError, ValueError)


class TFAdapter(FrameworkAdapter[tf.Tensor | tf.Variable]):
    _DTYPE_MAP: ClassVar[dict[str, tf.DType]] = {
        "float16": tf.float16,
        "bfloat16": tf.bfloat16,
        "float32": tf.float32,
        "float64": tf.float64,
    }

    def convert_in(self, arr: np.ndarray) -> tf.Tensor | tf.Variable:
        dtype = self._DTYPE_MAP[self.precision]
        return tf.Variable(arr, dtype=dtype)

    def convert_out(self, obj: tf.Tensor | tf.Variable) -> np.ndarray:
        ret = obj.numpy()
        if ret.dtype in (np.float16,):
            ret = ret.astype(np.float32)
        return ret

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
        seed: int | None = 42,
        wrt: str = "P",
    ) -> np.ndarray:
        with tf.GradientTape() as tape:
            tape.watch([P, Q])
            res = func(P, Q)
            if seed is not None:
                rng = np.random.RandomState(seed)
                dtype = self._DTYPE_MAP[self.precision]
                weights = [
                    tf.constant(rng.normal(size=tensor.shape), dtype=dtype)
                    for tensor in res
                ]
                loss = sum(
                    [
                        tf.reduce_sum(tensor * weight)
                        for tensor, weight in zip(res, weights, strict=False)
                    ]
                )
            else:
                loss = sum([tf.reduce_sum(tensor) for tensor in res])
        return tape.gradient(loss, P if wrt == "P" else Q).numpy()

    @property
    def mismatch_exception_type(self) -> type[Exception] | tuple[type[Exception], ...]:
        return tf.errors.InvalidArgumentError


if _MLX_AVAILABLE:

    class MLXAdapter(FrameworkAdapter[mx.array]):
        """
        MLX Adapter.
        For float64, forces ops onto the CPU because
        Apple Silicon GPUs don't support true float64.
        For float32, uses GPU acceleration.
        """

        _DTYPE_MAP: ClassVar[dict[str, mx.Dtype]] = {
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
            "float64": mx.float64,
        }

        def _set_device(self) -> None:
            if self.precision == "float64":
                mx.set_default_device(mx.cpu)
            else:
                mx.set_default_device(mx.gpu)

        def supports_dim(self, dim: int) -> bool:
            # MLX implementation hardcodes 3x3 determinant correction
            return dim == 3

        def convert_in(self, arr: np.ndarray) -> mx.array:
            self._set_device()
            dtype = self._DTYPE_MAP[self.precision]
            return mx.array(arr, dtype=dtype)

        def convert_out(self, obj: mx.array) -> np.ndarray:
            if obj.dtype in (mx.bfloat16, mx.float16):
                obj = obj.astype(mx.float32)
            ret = np.array(obj)
            # bfloat16 numpy arrays come out as mysterious void types sometimes
            # upcasting helps determinant test stay happy
            if self.precision in ("float16", "bfloat16") or ret.dtype in (np.float16,):
                ret = ret.astype(np.float32)
            return ret

        def kabsch(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
            self._set_device()
            return kabsch_mlx.kabsch(P, Q)

        def kabsch_umeyama(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
            self._set_device()
            return kabsch_mlx.kabsch_umeyama(P, Q)

        def horn(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
            self._set_device()
            return kabsch_mlx.horn(P, Q)

        def horn_with_scale(self, P: mx.array, Q: mx.array) -> tuple[mx.array, ...]:
            self._set_device()
            return kabsch_mlx.horn_with_scale(P, Q)

        def is_nan(self, tensor: mx.array) -> bool:
            self._set_device()
            return mx.any(mx.isnan(tensor)).item()

        def get_grad(
            self,
            P: mx.array,
            Q: mx.array,
            func: Callable[[mx.array, mx.array], tuple[mx.array, ...]],
            seed: int | None = 42,
            wrt: str = "P",
        ) -> np.ndarray:
            self._set_device()

            def loss_fn(P_inner, Q_inner):
                res = func(P_inner, Q_inner)
                if seed is not None:
                    rng = np.random.RandomState(seed)
                    dtype = self._DTYPE_MAP[self.precision]
                    weights = [
                        mx.array(rng.normal(size=tensor.shape), dtype=dtype)
                        for tensor in res
                    ]
                    return sum(
                        [
                            mx.sum(tensor * weight)
                            for tensor, weight in zip(res, weights, strict=False)
                        ]
                    )
                else:
                    return sum([mx.sum(tensor) for tensor in res])

            arg_idx = 0 if wrt == "P" else 1
            grad_fn = mx.grad(loss_fn, argnums=arg_idx)
            grad = grad_fn(P, Q)
            if grad.dtype in (mx.bfloat16, mx.float16):
                grad = grad.astype(mx.float32)
            return np.array(grad)

        @property
        def mismatch_exception_type(
            self,
        ) -> type[Exception] | tuple[type[Exception], ...]:
            return ValueError


frameworks = [
    pytest.param(PyTorchAdapter("bfloat16"), id="PyTorch-BFloat16"),
    pytest.param(PyTorchAdapter("float16"), id="PyTorch-Float16"),
    pytest.param(PyTorchAdapter("float32"), id="PyTorch-Float32"),
    pytest.param(PyTorchAdapter("float64"), id="PyTorch-Float64"),
    pytest.param(JAXAdapter("bfloat16"), id="JAX-BFloat16"),
    pytest.param(JAXAdapter("float16"), id="JAX-Float16"),
    pytest.param(JAXAdapter("float32"), id="JAX-Float32"),
    pytest.param(JAXAdapter("float64"), id="JAX-Float64"),
    pytest.param(TFAdapter("bfloat16"), id="TensorFlow-BFloat16"),
    pytest.param(TFAdapter("float16"), id="TensorFlow-Float16"),
    pytest.param(TFAdapter("float32"), id="TensorFlow-Float32"),
    pytest.param(TFAdapter("float64"), id="TensorFlow-Float64"),
    *(
        [
            pytest.param(MLXAdapter("bfloat16"), id="MLX-BFloat16-GPU"),
            pytest.param(MLXAdapter("float16"), id="MLX-Float16-GPU"),
            pytest.param(MLXAdapter("float64"), id="MLX-Float64-CPU"),
            pytest.param(MLXAdapter("float32"), id="MLX-Float32-GPU"),
        ]
        if _MLX_AVAILABLE
        else []
    ),
]
