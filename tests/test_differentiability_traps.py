import os

import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from conftest import ALGORITHMS, ALGORITHMS_3D_ONLY, ALGORITHMS_WITH_SCALE
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from strategies import extreme_scale_cloud, nearly_collinear_3d, nearly_coplanar_nd

_FAST = os.environ.get("KABSCH_TEST_FAST") == "1"
_MAX_EXAMPLES = 15 if _FAST else 60


class TestDifferentiabilityTraps:
    """Gradient stability tests for all algorithms (Kabsch, Umeyama, Horn,
    Horn-with-scale).

    Horn tests are automatically skipped for dim != 3 via the collection hook.
    """

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_are_coplanar(
        self,
        coplanar_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain numerically stable when the input points
        are coplanar.
        """
        P_np, Q_np = coplanar_points
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_are_collinear(
        self,
        collinear_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain numerically stable when the input points
        are collinear.
        """
        P_np, Q_np = collinear_points
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_form_perfect_cube(
        self,
        perfect_cube: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain stable when the input points form a perfect
        cube.
        """
        P_np, Q_np = perfect_cube
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_positive_determinant_when_points_are_reflected(
        self,
        reflected_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Ensures the calculated rotation matrix has a determinant of +1 for
        reflections.
        """
        P_np, Q_np = reflected_points
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        res = func(P_fw, Q_fw)
        R = adapter.convert_out(res[0])

        assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=adapter.atol)

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_are_reflected(
        self,
        reflected_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain numerically stable when inputs require
        reflection.
        """
        P_np, Q_np = reflected_points
        P_grad_in = adapter.convert_in(P_np)
        Q_grad_in = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P_grad_in, Q_grad_in, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_are_identical(
        self,
        identity_points: np.ndarray,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain numerically stable when the input points
        are identical (tests for RMSD gradient NaN trap).
        """
        P_np = identity_points
        Q_np = np.copy(P_np)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_system_is_underdetermined(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
        dim: int,
    ) -> None:
        """
        Checks that gradients remain numerically stable when the system is
        underdetermined (2 points in dim-D space).
        """
        rng = np.random.default_rng(42)
        # 2 points in dim-D (underdetermined)
        P_np = rng.random((2, dim)).astype(np.float64)
        # Random rotation + translation
        A = rng.standard_normal((dim, dim))
        R_true, _ = np.linalg.qr(A)
        if np.linalg.det(R_true) < 0:
            R_true[:, 0] *= -1
        t_true = rng.random(dim).astype(np.float64) * 5
        Q_np = (P_np @ R_true.T) + t_true

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("collapse_target", ["P", "Q", "Both"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_collapse_to_origin(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
        collapse_target: str,
        dim: int,
    ) -> None:
        """
        Checks that gradients remain numerically stable when the inputs collapse
        to the origin.
        """
        if algo in ALGORITHMS_WITH_SCALE and getattr(
            adapter, "precision", "float64"
        ) in ("float16", "bfloat16"):
            pytest.skip(
                "Scale algorithms require division by variance, which overflows "
                "float16 on origin collapse."
            )

        rng = np.random.default_rng(42)
        P_np = rng.random((5, dim)).astype(np.float64)
        Q_np = rng.random((5, dim)).astype(np.float64)

        if collapse_target in ["P", "Both"]:
            P_np = np.zeros_like(P_np)
        if collapse_target in ["Q", "Both"]:
            Q_np = np.zeros_like(Q_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize(
        "fixture_name",
        [
            pytest.param("coplanar_points", id="coplanar"),
            pytest.param("collinear_points", id="collinear"),
            pytest.param("perfect_cube", id="perfect_cube"),
            pytest.param("reflected_points", id="reflected"),
            pytest.param("identity_points", id="identity"),
        ],
    )
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_point_in_descent_direction_at_singularities(
        self,
        request,
        fixture_name: str,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
        dim: int,
    ) -> None:
        """
        Validates that the computed gradients at singularities actually point in a
        direction that decreases the RMSD.
        """
        points = request.getfixturevalue(fixture_name)
        if isinstance(points, tuple):
            P_np, Q_np = points
        else:
            P_np = points
            Q_np = np.copy(P_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        def rmsd_func(P_inner, Q_inner):
            res = func(P_inner, Q_inner)
            return (res[-1],)

        grad = adapter.get_grad(P, Q, rmsd_func, seed=None, wrt=wrt)

        # Gradient norm below atol means we are at or near a minimum; skip descent check
        if np.linalg.norm(grad) < adapter.atol:
            assert np.isfinite(grad).all()
            return

        alpha = max(1e-4, adapter.eps)
        if wrt == "P":
            P_new_np = P_np - alpha * grad
            P_new = adapter.convert_in(P_new_np)
            Q_new = Q
        else:
            Q_new_np = Q_np - alpha * grad
            Q_new = adapter.convert_in(Q_new_np)
            P_new = P

        rmsd_orig = float(adapter.convert_out(rmsd_func(P, Q)[0]))
        rmsd_new = float(adapter.convert_out(rmsd_func(P_new, Q_new)[0]))

        # Relative bound: gradient step should not increase RMSD by more than 1%
        # plus eps for near-zero RMSD (matches the Hypothesis near-degenerate test)
        assert rmsd_new <= rmsd_orig * 1.01 + adapter.eps

    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_gradients_stable_nearly_collinear_hypothesis(
        self,
        adapter: FrameworkAdapter,
        wrt: str,
        algo: str,
    ) -> None:
        """Gradients remain finite for near-collinear point clouds (Hypothesis)."""
        if algo in ALGORITHMS_WITH_SCALE and getattr(
            adapter, "precision", "float64"
        ) in ("float16", "bfloat16"):
            pytest.skip(
                "Scale algorithms require division by variance, which can overflow "
                "float16 on near-collinear inputs."
            )

        @settings(
            max_examples=_MAX_EXAMPLES,
            suppress_health_check=[HealthCheck.too_slow],
            deadline=None,
        )
        @given(nearly_collinear_3d())
        def _inner(P_np: np.ndarray) -> None:
            Q_np = P_np.copy()
            P = adapter.convert_in(P_np.astype(np.float64))
            Q = adapter.convert_in(Q_np.astype(np.float64))
            func = adapter.get_transform_func(algo)
            grad = adapter.get_grad(P, Q, func, wrt=wrt)
            assert np.all(np.isfinite(grad))

        _inner()

    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_gradients_stable_nearly_collinear_different_clouds(
        self,
        adapter: FrameworkAdapter,
        wrt: str,
        algo: str,
    ) -> None:
        """Gradients remain finite when P != Q are both near-collinear (Hypothesis).

        Compared to test_gradients_stable_nearly_collinear_hypothesis (which sets
        Q = P), this test draws P and Q independently so that the cross-covariance H
        is rank-1 but non-zero -- the realistic hard case SafeSVD/SafeEigh is
        designed for.
        """
        if getattr(adapter, "precision", "float64") in ("float16", "bfloat16"):
            pytest.skip(
                "Near-collinear independent clouds produce large gradients "
                "that overflow float16 range."
            )

        @settings(
            max_examples=_MAX_EXAMPLES,
            suppress_health_check=[HealthCheck.too_slow],
            deadline=None,
        )
        @given(nearly_collinear_3d(), nearly_collinear_3d())
        def _inner(P_np: np.ndarray, Q_np: np.ndarray) -> None:
            # Truncate to the shorter cloud so P and Q have the same number of points.
            n = min(P_np.shape[0], Q_np.shape[0])
            P_np_t = P_np[:n]
            Q_np_t = Q_np[:n]
            P = adapter.convert_in(P_np_t.astype(np.float64))
            Q = adapter.convert_in(Q_np_t.astype(np.float64))
            func = adapter.get_transform_func(algo)
            grad = adapter.get_grad(P, Q, func, wrt=wrt)
            assert np.all(np.isfinite(grad))

        _inner()

    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    def test_gradients_stable_nearly_coplanar_hypothesis(
        self,
        adapter: FrameworkAdapter,
        wrt: str,
        algo: str,
    ) -> None:
        """Gradients remain finite for near-coplanar point clouds (Hypothesis)."""
        if algo in ALGORITHMS_WITH_SCALE and getattr(
            adapter, "precision", "float64"
        ) in ("float16", "bfloat16"):
            pytest.skip(
                "Umeyama variance division overflows float16 on near-coplanar inputs."
            )
        if adapter.name == "MLXAdapter" and adapter.precision == "float32":
            pytest.skip(
                "MLX float32 SafeSVD backward is unstable for near-coplanar inputs"
            )

        dims = [d for d in adapter.supported_dims() if d >= 3]

        @settings(
            max_examples=_MAX_EXAMPLES,
            suppress_health_check=[HealthCheck.too_slow],
            deadline=None,
        )
        @given(data=st.data())
        def _inner(data) -> None:
            P_np = data.draw(nearly_coplanar_nd(dims=dims))
            Q_np = P_np.copy()
            P = adapter.convert_in(P_np.astype(np.float64))
            Q = adapter.convert_in(Q_np.astype(np.float64))
            func = adapter.get_transform_func(algo)
            grad = adapter.get_grad(P, Q, func, wrt=wrt)
            assert np.all(np.isfinite(grad))

        _inner()

    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_gradients_stable_extreme_scale_hypothesis(
        self,
        adapter: FrameworkAdapter,
        wrt: str,
        algo: str,
    ) -> None:
        """Gradients remain finite for extreme-scale point clouds (Hypothesis)."""
        # Precision check is parametric -- skip before Hypothesis generates examples.
        if getattr(adapter, "precision", "float64") not in ("float32", "float64"):
            pytest.skip("extreme scale unsafe for float16/bfloat16")
        if adapter.name == "MLXAdapter" and adapter.precision == "float64":
            pytest.skip(
                "MLX float64 SafeSVD backward is unstable at extreme scales on CPU"
            )

        if algo in ALGORITHMS_3D_ONLY:
            cloud_dims = [3]
        else:
            cloud_dims = adapter.supported_dims()

        @settings(
            max_examples=_MAX_EXAMPLES,
            suppress_health_check=[HealthCheck.too_slow],
            deadline=None,
        )
        @given(data=st.data())
        def _inner(data) -> None:
            PQ = data.draw(extreme_scale_cloud(dims=cloud_dims))
            P_np, Q_np = PQ
            P = adapter.convert_in(P_np.astype(np.float64))
            Q = adapter.convert_in(Q_np.astype(np.float64))
            func = adapter.get_transform_func(algo)
            grad = adapter.get_grad(P, Q, func, wrt=wrt)
            assert np.all(np.isfinite(grad))

        _inner()
