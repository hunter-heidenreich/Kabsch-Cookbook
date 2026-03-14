"""Tests for TensorFlow dynamic-shape validation under tf.function.

When tf.function traces with dynamic shapes (None dims), static shape checks
like P.shape[-2] return None and are skipped. These tests verify that
tf.debugging.assert_* runtime checks catch invalid inputs that static checks
miss.
"""

import numpy as np
import pytest
import tensorflow as tf

from kabsch_horn.tensorflow import horn, horn_with_scale, kabsch, kabsch_umeyama


class TestDynamicShapeNTooSmall:
    """N=1 inputs must be caught at runtime under tf.function with dynamic N."""

    @pytest.mark.parametrize(
        "func,extra_returns",
        [
            (kabsch, 2),
            (kabsch_umeyama, 3),
            (horn, 2),
            (horn_with_scale, 3),
        ],
    )
    def test_n1_rejected_dynamic(self, func, extra_returns):
        @tf.function(
            input_signature=[
                tf.TensorSpec([None, 3], tf.float64),
                tf.TensorSpec([None, 3], tf.float64),
            ]
        )
        def wrapped(P, Q):
            return func(P, Q)

        P = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float64)
        Q = tf.constant([[4.0, 5.0, 6.0]], dtype=tf.float64)

        with pytest.raises(tf.errors.InvalidArgumentError, match="2 points"):
            wrapped(P, Q)


class TestDynamicShapeHornAcceptsDim3:
    """Horn with fully-dynamic shapes must accept valid (N, 3) inputs."""

    @pytest.mark.parametrize("func", [horn, horn_with_scale])
    def test_horn_dynamic_dim_accepts_3d(self, func):
        @tf.function(
            input_signature=[
                tf.TensorSpec([None, None], tf.float64),
                tf.TensorSpec([None, None], tf.float64),
            ]
        )
        def wrapped(P, Q):
            return func(P, Q)

        rng = np.random.default_rng(42)
        P = tf.constant(rng.random((5, 3)), dtype=tf.float64)
        Q = tf.constant(rng.random((5, 3)), dtype=tf.float64)

        result = wrapped(P, Q)
        # Should succeed without error
        assert result[0].shape == (3, 3)  # R is 3x3


class TestDynamicShapeMismatch:
    """Shape mismatches must be caught at runtime under tf.function."""

    @pytest.mark.parametrize(
        "func",
        [kabsch, kabsch_umeyama, horn, horn_with_scale],
    )
    def test_shape_mismatch_dynamic(self, func):
        @tf.function(
            input_signature=[
                tf.TensorSpec([None, 3], tf.float64),
                tf.TensorSpec([None, 3], tf.float64),
            ]
        )
        def wrapped(P, Q):
            return func(P, Q)

        P = tf.constant(np.random.default_rng(0).random((5, 3)), dtype=tf.float64)
        Q = tf.constant(np.random.default_rng(1).random((4, 3)), dtype=tf.float64)

        with pytest.raises(tf.errors.InvalidArgumentError):
            wrapped(P, Q)


class TestEagerModeRegression:
    """Static validation still works in eager mode (no regressions)."""

    @pytest.mark.parametrize(
        "func",
        [kabsch, kabsch_umeyama, horn, horn_with_scale],
    )
    def test_n1_rejected_eager(self, func):
        P = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float64)
        Q = tf.constant([[4.0, 5.0, 6.0]], dtype=tf.float64)

        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            func(P, Q)

    @pytest.mark.parametrize("func", [horn, horn_with_scale])
    def test_horn_dim_not_3_rejected_eager(self, func):
        rng = np.random.default_rng(0)
        P = tf.constant(rng.random((5, 4)), dtype=tf.float64)
        Q = tf.constant(rng.random((5, 4)), dtype=tf.float64)

        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            func(P, Q)

    @pytest.mark.parametrize(
        "func",
        [kabsch, kabsch_umeyama, horn, horn_with_scale],
    )
    def test_valid_inputs_succeed_eager(self, func):
        rng = np.random.default_rng(42)
        P = tf.constant(rng.random((5, 3)), dtype=tf.float64)
        Q = tf.constant(rng.random((5, 3)), dtype=tf.float64)

        result = func(P, Q)
        assert result[0].shape == (3, 3)
