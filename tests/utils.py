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
    weight_adapter: FrameworkAdapter | None = None,
) -> np.ndarray:
    if weight_adapter is None:
        weight_adapter = adapter

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
                np.sum(
                    adapter.convert_out(tensor)
                    * weight_adapter.convert_out(
                        weight_adapter.convert_in(rng_plus.normal(size=tensor.shape))
                    )
                )
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
                    adapter.convert_out(tensor)
                    * weight_adapter.convert_out(
                        weight_adapter.convert_in(rng_minus.normal(size=tensor.shape))
                    )
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
                np.sum(
                    adapter.convert_out(tensor)
                    * weight_adapter.convert_out(
                        weight_adapter.convert_in(rng_plus.normal(size=tensor.shape))
                    )
                )
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
                    adapter.convert_out(tensor)
                    * weight_adapter.convert_out(
                        weight_adapter.convert_in(rng_minus.normal(size=tensor.shape))
                    )
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
    c_expected: float,
    rmsd_expected: float | None,
    algo: str,
    *,
    atol: float,
    rtol: float,
) -> None:
    __tracebackhide__ = True

    if algo in ("umeyama", "horn_with_scale"):
        R, t, c, rmsd = res
        c_out = adapter.convert_out(c)
    elif algo in ("kabsch", "horn"):
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


def compute_sequential_expected_tensors(
    P_np: np.ndarray,
    Q_np: np.ndarray,
    adapter: FrameworkAdapter,
    algo: str,
) -> list[np.ndarray]:
    """Computes expected batched outputs by running sequential computations.

    Args:
        P_np: Input N-D point cloud array with arbitrary leading batch dims.
        Q_np: Target N-D point cloud array with arbitrary leading batch dims.
        adapter: The framework adapter to use.
        algo: Algorithm to use ('kabsch', 'umeyama', 'horn', or 'horn_with_scale').

    Returns:
        List of numpy arrays matching batched output structure.
    """
    func = adapter.get_transform_func(algo)
    batch_shape = P_np.shape[:-2]  # arbitrary leading dims

    results = {}
    for idx in np.ndindex(*batch_shape):
        P_seq = adapter.convert_in(P_np[idx])
        Q_seq = adapter.convert_in(Q_np[idx])
        results[idx] = [adapter.convert_out(t) for t in func(P_seq, Q_seq)]

    num_tensors = len(next(iter(results.values())))
    first_key = next(iter(results.keys()))
    expected = []
    for t_idx in range(num_tensors):
        sample = results[first_key][t_idx]
        arr = np.empty(batch_shape + sample.shape)
        for idx in np.ndindex(*batch_shape):
            arr[idx] = results[idx][t_idx]
        expected.append(arr)

    return expected
