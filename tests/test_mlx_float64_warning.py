import warnings

import numpy as np
import pytest

try:
    import mlx.core as mx
    from adapters import MLXAdapter

    from kabsch_horn import mlx as kabsch_mlx

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX not available")

_RNG = np.random.default_rng(0)
_P_NP = _RNG.random((8, 3)).astype(np.float64)
_Q_NP = _RNG.random((8, 3)).astype(np.float64)


@pytest.fixture
def P():
    return mx.array(_P_NP, dtype=mx.float64)  # type: ignore[name-defined]


@pytest.fixture
def Q():
    return mx.array(_Q_NP, dtype=mx.float64)  # type: ignore[name-defined]


_WARN_FNS = (
    [
        pytest.param(kabsch_mlx.kabsch, id="kabsch"),
        pytest.param(kabsch_mlx.kabsch_umeyama, id="kabsch_umeyama"),
        pytest.param(kabsch_mlx.kabsch_rmsd, id="kabsch_rmsd"),
        pytest.param(kabsch_mlx.kabsch_umeyama_rmsd, id="kabsch_umeyama_rmsd"),
        pytest.param(kabsch_mlx.horn, id="horn"),
        pytest.param(kabsch_mlx.horn_with_scale, id="horn_with_scale"),
    ]
    if _MLX_AVAILABLE
    else []
)

_NO_WARN_FNS = (
    [
        pytest.param(kabsch_mlx.kabsch, id="kabsch"),
        pytest.param(kabsch_mlx.kabsch_umeyama, id="kabsch_umeyama"),
        pytest.param(kabsch_mlx.horn, id="horn"),
        pytest.param(kabsch_mlx.horn_with_scale, id="horn_with_scale"),
    ]
    if _MLX_AVAILABLE
    else []
)


@pytest.mark.parametrize("fn", _WARN_FNS)
def test_float64_emits_user_warning(fn, P, Q):
    """float64 MLX inputs must emit a UserWarning about CPU fallback."""
    with pytest.warns(UserWarning, match="float64"):
        fn(P, Q)


@pytest.mark.parametrize("fn", _NO_WARN_FNS)
def test_float32_no_warning(fn):
    """float32 MLX inputs must not emit a float64 warning."""
    P32 = mx.array(_P_NP.astype(np.float32))  # type: ignore[name-defined]
    Q32 = mx.array(_Q_NP.astype(np.float32))  # type: ignore[name-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fn(P32, Q32)


def test_mlx_adapter_float64_emits_warning():
    """MLXAdapter._set_device warns on float64."""
    adapter = MLXAdapter("float64")  # type: ignore[name-defined]
    with pytest.warns(UserWarning, match="float64"):
        adapter._set_device()


def test_mlx_adapter_float32_no_warning():
    """MLXAdapter._set_device does not warn on float32."""
    adapter = MLXAdapter("float32")  # type: ignore[name-defined]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        adapter._set_device()
