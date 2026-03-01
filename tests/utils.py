import numpy as np
import pytest
from adapters import FrameworkAdapter


def compute_numeric_grad(
    P_np: np.ndarray,
    Q_np: np.ndarray,
    adapter: FrameworkAdapter,
    func,
    seed: int = 42,
    wrt: str = "P",
) -> np.ndarray:
    eps = adapter.eps

    if wrt == "P":
        X_np = P_np
        flat_X = X_np.flatten()
        grad_num = np.zeros_like(flat_X)

        for i in range(len(flat_X)):
            val_orig = flat_X[i]

            flat_X[i] = val_orig + eps
            X_plus = flat_X.reshape(X_np.shape)
            P_plus_fw = adapter.convert_in(X_plus)
            Q_fw = adapter.convert_in(Q_np)
            res_plus = func(P_plus_fw, Q_fw)

            rng_plus = np.random.RandomState(seed)
            loss_plus = sum(
                np.sum(adapter.convert_out(tensor) * rng_plus.normal(size=tensor.shape))
                for tensor in res_plus
            )

            flat_X[i] = val_orig - eps
            X_minus = flat_X.reshape(X_np.shape)
            P_minus_fw = adapter.convert_in(X_minus)
            Q_fw = adapter.convert_in(Q_np)
            res_minus = func(P_minus_fw, Q_fw)

            rng_minus = np.random.RandomState(seed)
            loss_minus = sum(
                np.sum(
                    adapter.convert_out(tensor) * rng_minus.normal(size=tensor.shape)
                )
                for tensor in res_minus
            )

            flat_X[i] = val_orig
            grad_num[i] = (loss_plus - loss_minus) / (2.0 * eps)

        return grad_num.reshape(X_np.shape)
    else:
        Y_np = Q_np
        flat_Y = Y_np.flatten()
        grad_num = np.zeros_like(flat_Y)

        for i in range(len(flat_Y)):
            val_orig = flat_Y[i]

            flat_Y[i] = val_orig + eps
            Y_plus = flat_Y.reshape(Y_np.shape)
            P_fw = adapter.convert_in(P_np)
            Q_plus_fw = adapter.convert_in(Y_plus)
            res_plus = func(P_fw, Q_plus_fw)

            rng_plus = np.random.RandomState(seed)
            loss_plus = sum(
                np.sum(adapter.convert_out(tensor) * rng_plus.normal(size=tensor.shape))
                for tensor in res_plus
            )

            flat_Y[i] = val_orig - eps
            Y_minus = flat_Y.reshape(Y_np.shape)
            P_fw = adapter.convert_in(P_np)
            Q_minus_fw = adapter.convert_in(Y_minus)
            res_minus = func(P_fw, Q_minus_fw)

            rng_minus = np.random.RandomState(seed)
            loss_minus = sum(
                np.sum(
                    adapter.convert_out(tensor) * rng_minus.normal(size=tensor.shape)
                )
                for tensor in res_minus
            )

            flat_Y[i] = val_orig
            grad_num[i] = (loss_plus - loss_minus) / (2.0 * eps)

        return grad_num.reshape(Y_np.shape)


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
