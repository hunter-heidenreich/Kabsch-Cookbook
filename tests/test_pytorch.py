import numpy as np
import pytest
import torch
from kabsch_umeyama.numpy import kabsch as kabsch_np, kabsch_umeyama as umeyama_np
from kabsch_umeyama.pytorch import kabsch as kabsch_pt, kabsch_umeyama as umeyama_pt

# Generate synthetic point clouds
@pytest.fixture
def identity_points():
    np.random.seed(42)
    P = np.random.randn(2, 10, 3)
    return P, P.copy()

@pytest.fixture
def known_transform_points():
    np.random.seed(42)
    B, N, D = 2, 10, 3
    P = np.random.randn(B, N, D)
    
    # 90 degrees around Z
    theta = np.pi / 2
    R = np.array([
        [[np.cos(theta), -np.sin(theta), 0],
         [np.sin(theta), np.cos(theta),  0],
         [0,             0,              1]],
        [[1, 0, 0],
         [0, np.cos(theta), -np.sin(theta)],
         [0, np.sin(theta), np.cos(theta)]]
    ])
    
    t = np.array([[1.0, -2.0, 3.0], 
                  [-1.0, 0.5, 2.0]])
    c = np.array([2.0, 0.5])
    
    # Q = c * R @ P + t
    Q = c[:, np.newaxis, np.newaxis] * np.matmul(P, R.transpose(0, 2, 1)) + t[:, np.newaxis, :]
    
    return P, Q, R, t, c

@pytest.fixture
def degenerate_points():
    np.random.seed(42)
    # Perfect cube
    P_cube = np.array([
        [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1],
        [1,-1,-1],  [1,-1,1],  [1,1,-1],  [1,1,1]
    ], dtype=np.float32)
    
    # Collinear map (only X varies)
    P_collinear = P_cube.copy()
    P_collinear[:, 1] = 0
    P_collinear[:, 2] = 0
    
    # Add batch dim
    P_cube = P_cube[np.newaxis, ...]
    P_collinear = P_collinear[np.newaxis, ...]
    
    return P_cube, P_collinear


def test_pytorch_kabsch_numpy_parity(known_transform_points):
    P_np, Q_np, R_true, t_true, c_true = known_transform_points
    
    R_np, t_np, _ = kabsch_np(P_np, Q_np)
    
    # NOTE: Kabsch without umeyama does NOT handle scale properly!
    # Let's re-generate a non-scaled Q for just plain kabsch test
    Q_np_unscaled = np.matmul(P_np, R_true.transpose(0, 2, 1)) + t_true[:, np.newaxis, :]
    
    R_np, t_np, rmsd_np = kabsch_np(P_np, Q_np_unscaled)
    
    P_pt = torch.tensor(P_np, dtype=torch.float64)
    Q_pt = torch.tensor(Q_np_unscaled, dtype=torch.float64)
    
    R_pt, t_pt, rmsd_pt = kabsch_pt(P_pt, Q_pt)
    
    np.testing.assert_allclose(R_pt.numpy(), R_np, atol=1e-5)
    np.testing.assert_allclose(t_pt.numpy(), t_np, atol=1e-5)
    np.testing.assert_allclose(rmsd_pt.numpy(), rmsd_np, atol=1e-5)


def test_pytorch_umeyama_numpy_parity(known_transform_points):
    P_np, Q_np, R_true, t_true, c_true = known_transform_points
    
    R_np, t_np, c_np, rmsd_np = umeyama_np(P_np, Q_np)
    
    P_pt = torch.tensor(P_np, dtype=torch.float64)
    Q_pt = torch.tensor(Q_np, dtype=torch.float64)
    
    R_pt, t_pt, c_pt, rmsd_pt = umeyama_pt(P_pt, Q_pt)
    
    np.testing.assert_allclose(R_pt.numpy(), R_np, atol=1e-5)
    np.testing.assert_allclose(t_pt.numpy(), t_np, atol=1e-5)
    np.testing.assert_allclose(c_pt.numpy(), c_np, atol=1e-5)
    np.testing.assert_allclose(rmsd_pt.numpy(), rmsd_np, atol=1e-5)


def test_pytorch_differentiability_trap_cube(degenerate_points):
    P_cube, P_collinear = degenerate_points
    
    # 1. Cube has degenerate singular values! This usually breaks backprop.
    P_pt = torch.tensor(P_cube, dtype=torch.float64, requires_grad=True)
    # Give a small rotation
    Q_pt = P_pt + 0.1
    
    R, t, rmsd = kabsch_pt(P_pt, Q_pt)
    
    # Calculate a mock loss
    loss = rmsd.sum()
    
    # Run backprop
    loss.backward()
    
    # Ensure gradients are finite (NO NaNs!)
    assert not torch.isnan(P_pt.grad).any()
    assert not torch.isinf(P_pt.grad).any()


def test_pytorch_differentiability_trap_collinear(degenerate_points):
    P_cube, P_collinear = degenerate_points
    
    P_pt = torch.tensor(P_collinear, dtype=torch.float64, requires_grad=True)
    Q_pt = P_pt + 0.1
    
    # Standard PyTorch SVD usually throws an error/NaN on backward for this
    R, t, c, rmsd = umeyama_pt(P_pt, Q_pt)
    
    loss = rmsd.sum() + R.sum() + t.sum() + c.sum()
    loss.backward()
    
    assert not torch.isnan(P_pt.grad).any()
    assert not torch.isinf(P_pt.grad).any()


def test_pytorch_gradcheck():
    # Requires double precision
    np.random.seed(42)
    # Gradcheck needs non-degenerate points because finite-difference 
    # jumps around zero can still be numerically tough, but we will test standard mode
    P = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
    Q = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=False)
    
    # We will test our safe_svd wrapper first
    # Construct a positive definite symmetric matrix to easily check
    p = P - P.mean(dim=1, keepdim=True)
    H = torch.matmul(p.transpose(1, 2), p)
    
    from kabsch_umeyama.pytorch import safe_svd
    # torch.autograd.gradcheck takes (func, inputs)
    # We must ensure fast_check is false to do the full Jacobian check
    res = torch.autograd.gradcheck(safe_svd, (H,), eps=1e-6, atol=1e-4)
    assert res, "SafeSVD Gradcheck failed"

    # Now check full Kabsch
    def wrapped_kabsch(x):
        return kabsch_pt(x, Q)[0] # Test rotation matrix gradient
        
    res = torch.autograd.gradcheck(wrapped_kabsch, (P,), eps=1e-6, atol=1e-4)
    assert res, "Kabsch Gradcheck failed"

