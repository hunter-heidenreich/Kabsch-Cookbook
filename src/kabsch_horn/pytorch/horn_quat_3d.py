import torch


class SafeEigh(torch.autograd.Function):
    """
    Computes a safe Eigendecomposition for symmetric matrices.
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, eps: float = 1e-12):
        L, V = torch.linalg.eigh(A)
        ctx.save_for_backward(L, V)
        ctx.eps = eps
        return L, V

    @staticmethod
    def backward(ctx, grad_L, grad_V):
        L, V = ctx.saved_tensors
        eps = ctx.eps

        grad_L = torch.zeros_like(L) if grad_L is None else grad_L
        grad_V = torch.zeros_like(V) if grad_V is None else grad_V

        # 1. Compute pairwise difference in eigenvalues
        # D_ij = L_j - L_i
        D = L.unsqueeze(-2) - L.unsqueeze(-1)

        # 2. Mask unstable divides directly
        mask = torch.abs(D) < eps
        safe_D = torch.where(mask, torch.where(D >= 0, eps, -eps), D)

        # 3. Prevent diagonal inversion problems (out-of-place for torch.compile)
        diag_mask = torch.eye(safe_D.shape[-1], dtype=torch.bool, device=safe_D.device)
        safe_D = torch.where(diag_mask, torch.ones_like(safe_D), safe_D)
        F = 1.0 / safe_D
        F = torch.where(diag_mask, torch.zeros_like(F), F)

        # 4. Standard backprop algebra using safe denominators
        Vt_dV = torch.matmul(V.mH, grad_V)
        term = torch.diag_embed(grad_L) + F * (Vt_dV - Vt_dV.mH) / 2

        grad_A = torch.matmul(V, torch.matmul(term, V.mH))

        return grad_A, None


def horn(
    P: torch.Tensor, Q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes optimal rotation and translation to align P to Q using Horn's quaternion
    method.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    if P.shape[-1] != 3:
        raise ValueError("Horn's method is strictly for 3D point clouds")
    orig_dtype = P.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        P = P.to(torch.float32)
        Q = Q.to(torch.float32)

    is_single = P.ndim == 2
    if is_single:
        P = P.unsqueeze(0)
        Q = Q.unsqueeze(0)

    orig_shape = P.shape
    N_pts = orig_shape[-2]
    batch_dims = orig_shape[:-2]
    P = P.reshape(-1, N_pts, 3)
    Q = Q.reshape(-1, N_pts, 3)

    # 1. Compute Centers and 3x3 Cross-Covariance
    centroid_P = P.mean(dim=1, keepdim=True)
    centroid_Q = Q.mean(dim=1, keepdim=True)

    p = P - centroid_P
    q = Q - centroid_Q

    H = torch.matmul(p.transpose(1, 2), q)

    # 2. Construct the 4x4 Symmetric Matrix N
    S = H + H.transpose(-1, -2)
    # Trace of H
    tr = H.diagonal(dim1=-2, dim2=-1).sum(-1)

    # Delta vector
    Delta = torch.stack(
        [
            H[..., 1, 2] - H[..., 2, 1],
            H[..., 2, 0] - H[..., 0, 2],
            H[..., 0, 1] - H[..., 1, 0],
        ],
        dim=-1,
    )

    # Construct N
    B_sz = H.shape[0]
    I3 = torch.eye(3, dtype=H.dtype, device=H.device).expand(B_sz, 3, 3)

    top_row = torch.cat([tr.unsqueeze(-1), Delta], dim=-1).unsqueeze(-2)  # Bx1x4
    bottom_block = torch.cat(
        [Delta.unsqueeze(-1), S - tr.unsqueeze(-1).unsqueeze(-1) * I3], dim=-1
    )  # Bx3x4

    N = torch.cat([top_row, bottom_block], dim=-2)  # Bx4x4

    # 3. Extract the quaternion
    _L, V = SafeEigh.apply(N, 1e-12)
    q_opt = V[..., -1]  # Bx4

    # 4. Convert to Rotation Matrix
    qw = q_opt[..., 0]
    qx = q_opt[..., 1]
    qy = q_opt[..., 2]
    qz = q_opt[..., 3]

    R11 = 1 - 2 * (qy**2 + qz**2)
    R12 = 2 * (qx * qy - qw * qz)
    R13 = 2 * (qx * qz + qw * qy)
    R21 = 2 * (qx * qy + qw * qz)
    R22 = 1 - 2 * (qx**2 + qz**2)
    R23 = 2 * (qy * qz - qw * qx)
    R31 = 2 * (qx * qz - qw * qy)
    R32 = 2 * (qy * qz + qw * qx)
    R33 = 1 - 2 * (qx**2 + qy**2)

    R = torch.stack(
        [
            torch.stack([R11, R12, R13], dim=-1),
            torch.stack([R21, R22, R23], dim=-1),
            torch.stack([R31, R32, R33], dim=-1),
        ],
        dim=-2,
    )

    # Translation
    t = centroid_Q.squeeze(1) - torch.squeeze(
        torch.matmul(centroid_P, R.transpose(1, 2)), 1
    )

    # RMSD
    aligned = torch.matmul(p, R.transpose(1, 2))
    rmsd = torch.sqrt(
        torch.clamp(torch.sum(torch.square(aligned - q), dim=(1, 2)) / N_pts, min=1e-12)
    )

    if is_single:
        R, t, rmsd = R[0], t[0], rmsd[0]
        if orig_dtype in (torch.float16, torch.bfloat16):
            R = R.to(orig_dtype)
            t = t.to(orig_dtype)
            rmsd = rmsd.to(orig_dtype)
        return R, t, rmsd

    R = R.reshape(*batch_dims, 3, 3)
    t = t.reshape(*batch_dims, 3)
    rmsd = rmsd.reshape(*batch_dims)
    if orig_dtype in (torch.float16, torch.bfloat16):
        R = R.to(orig_dtype)
        t = t.to(orig_dtype)
        rmsd = rmsd.to(orig_dtype)
    return R, t, rmsd


def horn_with_scale(
    P: torch.Tensor, Q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes optimal rotation, translation, and scale using Horn's method.
    """
    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P.shape} vs {Q.shape}"
        )
    if P.shape[-1] != 3:
        raise ValueError("Horn's method is strictly for 3D point clouds")
    orig_dtype = P.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        P = P.to(torch.float32)
        Q = Q.to(torch.float32)

    is_single = P.ndim == 2
    if is_single:
        P = P.unsqueeze(0)
        Q = Q.unsqueeze(0)

    orig_shape = P.shape
    N_pts = orig_shape[-2]
    batch_dims = orig_shape[:-2]
    P = P.reshape(-1, N_pts, 3)
    Q = Q.reshape(-1, N_pts, 3)

    centroid_P = P.mean(dim=1, keepdim=True)
    centroid_Q = Q.mean(dim=1, keepdim=True)

    p = P - centroid_P
    q = Q - centroid_Q

    var_P = torch.sum(torch.square(p), dim=(1, 2)) / N_pts

    # Cross-variance matrix
    H = torch.matmul(p.transpose(1, 2), q) / N_pts

    # S
    S = H + H.transpose(-1, -2)
    tr = H.diagonal(dim1=-2, dim2=-1).sum(-1)

    Delta = torch.stack(
        [
            H[..., 1, 2] - H[..., 2, 1],
            H[..., 2, 0] - H[..., 0, 2],
            H[..., 0, 1] - H[..., 1, 0],
        ],
        dim=-1,
    )

    B_sz = H.shape[0]
    I3 = torch.eye(3, dtype=H.dtype, device=H.device).expand(B_sz, 3, 3)

    top_row = torch.cat([tr.unsqueeze(-1), Delta], dim=-1).unsqueeze(-2)
    bottom_block = torch.cat(
        [Delta.unsqueeze(-1), S - tr.unsqueeze(-1).unsqueeze(-1) * I3], dim=-1
    )

    N = torch.cat([top_row, bottom_block], dim=-2)

    _L, V = SafeEigh.apply(N, 1e-12)
    q_opt = V[..., -1]

    qw = q_opt[..., 0]
    qx = q_opt[..., 1]
    qy = q_opt[..., 2]
    qz = q_opt[..., 3]

    R11 = 1 - 2 * (qy**2 + qz**2)
    R12 = 2 * (qx * qy - qw * qz)
    R13 = 2 * (qx * qz + qw * qy)
    R21 = 2 * (qx * qy + qw * qz)
    R22 = 1 - 2 * (qx**2 + qz**2)
    R23 = 2 * (qy * qz - qw * qx)
    R31 = 2 * (qx * qz - qw * qy)
    R32 = 2 * (qy * qz + qw * qx)
    R33 = 1 - 2 * (qx**2 + qy**2)

    R = torch.stack(
        [
            torch.stack([R11, R12, R13], dim=-1),
            torch.stack([R21, R22, R23], dim=-1),
            torch.stack([R31, R32, R33], dim=-1),
        ],
        dim=-2,
    )

    RH = torch.sum(R * H.transpose(-1, -2), dim=(1, 2))
    c = RH / torch.clamp(var_P, min=1e-12)

    t = centroid_Q.squeeze(1) - c.unsqueeze(-1) * torch.squeeze(
        torch.matmul(centroid_P, R.transpose(1, 2)), 1
    )

    aligned_P = c.unsqueeze(-1).unsqueeze(-1) * torch.matmul(
        P, R.transpose(1, 2)
    ) + t.unsqueeze(1)
    diff = aligned_P - Q
    rmsd = torch.sqrt(
        torch.clamp(torch.sum(torch.square(diff), dim=(1, 2)) / N_pts, min=1e-12)
    )

    if is_single:
        R, t, c, rmsd = R[0], t[0], c[0], rmsd[0]
        if orig_dtype in (torch.float16, torch.bfloat16):
            R = R.to(orig_dtype)
            t = t.to(orig_dtype)
            c = c.to(orig_dtype)
            rmsd = rmsd.to(orig_dtype)
        return R, t, c, rmsd

    R = R.reshape(*batch_dims, 3, 3)
    t = t.reshape(*batch_dims, 3)
    c = c.reshape(*batch_dims)
    rmsd = rmsd.reshape(*batch_dims)
    if orig_dtype in (torch.float16, torch.bfloat16):
        R = R.to(orig_dtype)
        t = t.to(orig_dtype)
        c = c.to(orig_dtype)
        rmsd = rmsd.to(orig_dtype)
    return R, t, c, rmsd
