import numpy as np
import pytest
from adapters import FrameworkAdapter


def compute_numeric_grad(
    P_np: np.ndarray,
    Q_np: np.ndarray,
    adapter: FrameworkAdapter,
    func,
    seed: int = 42,
) -> np.ndarray:
    eps = adapter.eps
    flat_P = P_np.flatten()
    grad_num = np.zeros_like(flat_P)

    for i in range(len(flat_P)):
        val_orig = flat_P[i]

        flat_P[i] = val_orig + eps
        P_plus = flat_P.reshape(P_np.shape)
        P_plus_fw = adapter.convert_in(P_plus)
        Q_fw = adapter.convert_in(Q_np)
        res_plus = func(P_plus_fw, Q_fw)
        
        # Consistent random projection
        rng_plus = np.random.RandomState(seed)
        loss_plus = sum(
            np.sum(adapter.convert_out(tensor) * rng_plus.normal(size=tensor.shape)) 
            for tensor in res_plus
        )

        flat_P[i] = val_orig - eps
        P_minus = flat_P.reshape(P_np.shape)
        P_minus_fw = adapter.convert_in(P_minus)
        Q_fw = adapter.convert_in(Q_np)
        res_minus = func(P_minus_fw, Q_fw)
        
        # Consistent random projection
        rng_minus = np.random.RandomState(seed)
        loss_minus = sum(
            np.sum(adapter.convert_out(tensor) * rng_minus.normal(size=tensor.shape)) 
            for tensor in res_minus
        )

        flat_P[i] = val_orig
        grad_num[i] = (loss_plus - loss_minus) / (2.0 * eps)

    return grad_num.reshape(P_np.shape)


def check_transform_close(
    adapter: FrameworkAdapter,
    res: tuple,
    R_expected: np.ndarray,
    t_expected: np.ndarray,
    c_expected: float = 1.0,
    rmsd_expected: float | None = None,
    algo: str = "kabsch",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    __tracebackhide__ = True

    if algo == "umeyama":
        R, t, c, rmsd = res
        c_out = adapter.convert_out(c)
    elif algo == "kabsch":
        R, t, rmsd = res
        c_out = 1.0
    else:
        raise ValueError(f"Unknown algorithm {algo}")

    assert adapter.convert_out(R) == pytest.approx(R_expected, rel=rtol, abs=atol)
    assert adapter.convert_out(t) == pytest.approx(t_expected, rel=rtol, abs=atol)
    assert float(c_out) == pytest.approx(c_expected, rel=rtol, abs=atol)

    if rmsd_expected is not None:
        assert float(adapter.convert_out(rmsd)) == pytest.approx(
            rmsd_expected, rel=rtol, abs=atol
        )
