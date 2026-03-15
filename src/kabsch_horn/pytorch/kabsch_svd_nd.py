import torch


class SafeSVD(torch.autograd.Function):
    """
    Computes a safe Singular Value Decomposition (SVD) for covariance matrices.
    Returns (U, S, V) where A = U @ diag(S) @ V^T.

    The backward pass masks near-zero singular-value differences with
    dtype-aware eps, preventing NaN gradients for symmetric or degenerate
    inputs. Higher-order gradients (create_graph=True) are supported since
    the backward uses standard differentiable torch operations.
    """

    @staticmethod
    def forward(
        ctx, A: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            U, S, Vh = torch.linalg.svd(A)
        except torch.linalg.LinAlgError:
            # NaN or degenerate input: propagate NaN without crashing.
            # torch.linalg.svd raises rather than returning NaN, so we handle
            # it explicitly to satisfy the NaN-propagation contract.
            # U is [..., M, M], S is [..., min(M,N)], Vh is [..., N, N]
            nan_u = A.new_full((*A.shape[:-1], A.shape[-2]), float("nan"))
            M, N = A.shape[-2], A.shape[-1]
            nan_s = A.new_full((*A.shape[:-2], min(M, N)), float("nan"))
            nan_vh = A.new_full((*A.shape[:-2], A.shape[-1], A.shape[-1]), float("nan"))
            ctx.save_for_backward(nan_u, nan_s, nan_vh)
            return nan_u, nan_s, nan_vh
        # In PyTorch 1.11+, linalg.svd returns Vh (V^T or V^H).
        # We want V for the standard Kabsch logic, so V = Vh.transpose(-2, -1)
        V = Vh.mH  # Conjugate transpose / standard transpose for real.

        ctx.save_for_backward(U, S, Vh)
        return U, S, V

    @staticmethod
    def backward(
        ctx, grad_U: torch.Tensor, grad_S: torch.Tensor, grad_V: torch.Tensor
    ) -> tuple[torch.Tensor]:
        if not ctx.needs_input_grad[0]:
            return (None,)

        U, S, Vh = ctx.saved_tensors
        eps = torch.finfo(S.dtype).eps

        # Backward pass of SVD for real matrices:
        # A = U S V^T
        # dA = U (diag(dS) - J S - S K) V^T

        # Replace None gradients with zeros
        grad_U = torch.zeros_like(U) if grad_U is None else grad_U
        grad_S = torch.zeros_like(S) if grad_S is None else grad_S
        grad_V = torch.zeros_like(Vh.mH) if grad_V is None else grad_V

        # Given grad_V is dV, and Vh = V^T
        # Then dVh = dV^T
        grad_Vh = grad_V.mH

        # 1. Compute square of singular values
        S_sq = torch.square(S)  # BxD

        # 2. Compute difference in squares
        D = S_sq.unsqueeze(-1) - S_sq.unsqueeze(-2)  # BxDxD

        # 3. Safe F matrix computation
        # eps = finfo(dtype).eps masks singular value differences at the dtype's
        # machine precision, preventing division-by-zero NaN gradients without
        # distorting the backward signal.
        D_abs = torch.abs(D)
        mask = D_abs < eps

        # Safe denominator preserving sign for sub-eps values
        safe_D = torch.where(mask, torch.where(D >= 0, eps, -eps), D)

        # Protect diagonal from 1/0 (out-of-place for torch.compile)
        diag_mask = torch.eye(safe_D.shape[-1], dtype=torch.bool, device=safe_D.device)
        safe_D = torch.where(diag_mask, torch.ones_like(safe_D), safe_D)

        F = 1.0 / safe_D
        # Set diagonal to exactly 0 to satisfy Hadamard condition
        F = torch.where(diag_mask, torch.zeros_like(F), F)

        # 4. Compute J and K
        Ut_dU = torch.matmul(U.mH, grad_U)
        J = F * (Ut_dU - Ut_dU.mH)

        # K = F * (V^T dV - dV^T V)
        # Using Vh instead:
        Vht_dVh = torch.matmul(Vh, grad_Vh.mH)
        K = F * (Vht_dVh - Vht_dVh.mH)

        # 5. Core matrix gradient
        S_diag = torch.diag_embed(grad_S)
        S_mat = torch.diag_embed(S)

        term = S_diag - torch.matmul(J, S_mat) - torch.matmul(S_mat, K)

        grad_A = torch.matmul(U, torch.matmul(term, Vh))

        return (grad_A,)


def safe_svd(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable wrapper for SVD avoiding NaNs.
    Args:
        A: (..., D, D) tensor (ndim >= 3)
    Returns:
        U, S, V  (V, NOT Vh)
    """
    return SafeSVD.apply(A)


def kabsch(
    P: torch.Tensor, Q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the optimal rotation and translation to align P to Q using Safe SVD.

    Args:
        P: Source points, shape [..., N, D].
        Q: Target points, shape [..., N, D].

    Returns:
        (R, t, rmsd): Rotation [..., D, D], translation [..., D], RMSD [...].

    Note:
        R is only stable under global translation when the cross-covariance matrix
        H = P_c.T @ Q_c is well-conditioned. When the smallest singular value of H
        is near zero, U and V from the SVD are not unique, and a small perturbation
        can select a different rotation. Check the singular values of H if rotation
        stability matters for your use case.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    if P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")

    orig_dtype = P.dtype
    if P.dtype != Q.dtype:
        # Mixed dtypes: promote to higher precision
        target = torch.float64 if torch.float64 in (P.dtype, Q.dtype) else torch.float32
        P = P.to(target)
        Q = Q.to(target)
        orig_dtype = target
    elif orig_dtype in (torch.float16, torch.bfloat16):
        P = P.to(torch.float32)
        Q = Q.to(torch.float32)

    is_single = P.ndim == 2
    if is_single:
        P = P.unsqueeze(0)
        Q = Q.unsqueeze(0)

    orig_shape = P.shape
    D = orig_shape[-1]
    _N = orig_shape[-2]
    batch_dims = orig_shape[:-2]

    P = P.view(-1, _N, D)
    Q = Q.view(-1, _N, D)

    # Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdim=True)  # Bx1x3
    centroid_Q = torch.mean(Q, dim=1, keepdim=True)  # Bx1x3

    # Center points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Cross-covariance matrix
    H = torch.matmul(p.transpose(1, 2), q)  # Bx3x3

    # Safe SVD
    U, _S, V = safe_svd(H)  # Bx3x3, Bx3, Bx3x3

    # 1. Determinant validation for right-handed coordinate system
    # (Checking for reflections)
    d = torch.linalg.det(torch.matmul(V, U.transpose(1, 2)))  # B

    # 2. Build B_diag
    # Sign safely mapping 0 determinant to 1.0 instead of 0.0
    d_sign = torch.sign(d + torch.finfo(d.dtype).eps)

    B_diag = torch.ones(*d_sign.shape, D, dtype=P.dtype, device=P.device)
    B_diag[..., -1] = d_sign

    # 3. Optimal Rotation: R = V * B_diag * U^T
    R = torch.matmul(V * B_diag.unsqueeze(1), U.transpose(1, 2))  # Bx3x3

    # RMSD (mse + eps for smooth sqrt gradient near zero)
    aligned = torch.matmul(p, R.transpose(1, 2))
    _eps = torch.finfo(P.dtype).eps
    mse = torch.sum(torch.square(aligned - q), dim=(1, 2)) / P.shape[1]
    rmsd = torch.sqrt(mse + _eps)

    # Fast Translation
    t = centroid_Q.squeeze(1) - torch.squeeze(
        torch.matmul(centroid_P, R.transpose(1, 2)), 1
    )

    if is_single:
        R, t, rmsd = R[0], t[0], rmsd[0]
        if orig_dtype in (torch.float16, torch.bfloat16):
            R = R.to(orig_dtype)
            t = t.to(orig_dtype)
            rmsd = rmsd.to(orig_dtype)
        return R, t, rmsd

    R = R.view(*batch_dims, D, D)
    t = t.view(*batch_dims, D)
    rmsd = rmsd.view(*batch_dims)

    if orig_dtype in (torch.float16, torch.bfloat16):
        R = R.to(orig_dtype)
        t = t.to(orig_dtype)
        rmsd = rmsd.to(orig_dtype)

    return R, t, rmsd


def kabsch_umeyama(
    P: torch.Tensor, Q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes optimal rotation, translation, and scale (Q ~ c * R @ P + t).

    Args:
        P: Source points, shape [..., N, D].
        Q: Target points, shape [..., N, D].

    Returns:
        (R, t, c, rmsd): Rotation [..., D, D], translation [..., D], scale [...],
        RMSD [...].

    Note:
        Unlike kabsch, the cross-covariance H is divided by N here. This per-point
        normalization is required by the Umeyama scale estimator
        (c = trace(S * D) / var_P) and does not affect the rotation or translation.

        R is only stable under global translation and uniform scaling when the
        cross-covariance matrix H = P_c.T @ Q_c is well-conditioned. When the
        smallest singular value of H is near zero, U and V from the SVD are not
        unique, and a small perturbation can select a different rotation. Check
        the singular values of H if rotation stability matters for your use case.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    if P.shape[-2] < 2:
        raise ValueError("At least 2 points are required for alignment")

    orig_dtype = P.dtype
    if P.dtype != Q.dtype:
        # Mixed dtypes: promote to higher precision
        target = torch.float64 if torch.float64 in (P.dtype, Q.dtype) else torch.float32
        P = P.to(target)
        Q = Q.to(target)
        orig_dtype = target
    elif orig_dtype in (torch.float16, torch.bfloat16):
        P = P.to(torch.float32)
        Q = Q.to(torch.float32)

    is_single = P.ndim == 2
    if is_single:
        P = P.unsqueeze(0)
        Q = Q.unsqueeze(0)

    orig_shape = P.shape
    D = orig_shape[-1]
    N = orig_shape[-2]
    batch_dims = orig_shape[:-2]

    P = P.view(-1, N, D)
    Q = Q.view(-1, N, D)

    # Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdim=True)
    centroid_Q = torch.mean(Q, dim=1, keepdim=True)

    # Center points
    p = P - centroid_P
    q = Q - centroid_Q

    # Cross-covariance matrix (divided by N for Umeyama scale estimation)
    H = torch.matmul(p.transpose(1, 2), q) / N

    # Variances of P
    var_P = torch.sum(torch.square(p), dim=(1, 2)) / N

    # Safe SVD
    U, S, V = safe_svd(H)

    # Right-hand coordinate system
    d = torch.linalg.det(torch.matmul(V, U.transpose(1, 2)))
    d_sign = torch.sign(d + torch.finfo(d.dtype).eps)

    S_corr = torch.ones(*d_sign.shape, D, dtype=P.dtype, device=P.device)
    S_corr[..., -1] = d_sign

    # Scale
    _eps = torch.finfo(P.dtype).eps
    c = torch.sum(S * S_corr, dim=-1) / torch.clamp(var_P, min=_eps)

    # Rotation
    R = torch.matmul(V * S_corr.unsqueeze(1), U.transpose(1, 2))

    # Translation
    t = centroid_Q.squeeze(1) - c.unsqueeze(-1) * torch.matmul(
        centroid_P, R.transpose(1, 2)
    ).squeeze(1)

    # RMSD
    aligned_P = c.unsqueeze(-1).unsqueeze(-1) * torch.matmul(
        P, R.transpose(1, 2)
    ) + t.unsqueeze(1)
    mse = torch.sum(torch.square(aligned_P - Q), dim=(1, 2)) / N
    rmsd = torch.sqrt(mse + _eps)

    if is_single:
        R, t, c, rmsd = R[0], t[0], c[0], rmsd[0]
        if orig_dtype in (torch.float16, torch.bfloat16):
            R = R.to(orig_dtype)
            t = t.to(orig_dtype)
            c = c.to(orig_dtype)
            rmsd = rmsd.to(orig_dtype)
        return R, t, c, rmsd

    R = R.view(*batch_dims, D, D)
    t = t.view(*batch_dims, D)
    c = c.view(*batch_dims)
    rmsd = rmsd.view(*batch_dims)

    if orig_dtype in (torch.float16, torch.bfloat16):
        R = R.to(orig_dtype)
        t = t.to(orig_dtype)
        c = c.to(orig_dtype)
        rmsd = rmsd.to(orig_dtype)

    return R, t, c, rmsd


def kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Computes RMSD after Kabsch alignment. Gradient-safe training loss."""
    _R, _t, rmsd = kabsch(P, Q)
    return rmsd


def kabsch_umeyama_rmsd(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Computes RMSD after Kabsch-Umeyama alignment. Gradient-safe training loss."""
    _R, _t, _c, rmsd = kabsch_umeyama(P, Q)
    return rmsd
