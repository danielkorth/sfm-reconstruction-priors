from abc import ABC, abstractmethod

import torch


class LinearSolver(ABC):
    """Abstract base class for linear solvers."""

    @abstractmethod
    def solve(self, A, b, device=None):
        """Solve the linear system Ax = b."""
        pass


class TorchLinearSolver(LinearSolver):
    """Direct linear solver using torch.linalg.solve."""

    def solve(self, A, b, device=None):
        try:
            return torch.linalg.solve(A, b)
        except torch.linalg.LinAlgError as e:
            print(f"TorchLinearSolver failed: {e}")
            raise


class ConjugateGradientSolver(LinearSolver):
    """Iterative linear solver using preconditioned conjugate gradient method."""

    def __init__(self, max_iterations=100, tolerance=1e-10, use_preconditioner=True):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_preconditioner = use_preconditioner

    def solve(self, A, b, device=None):
        if device is None:
            device = A.device

        n = A.shape[0]
        x = torch.zeros_like(b, device=device)

        # Initial residual
        r = b - torch.mm(A, x.unsqueeze(1)).squeeze(1)

        # Jacobi preconditioner: M = diag(A)
        if self.use_preconditioner:
            M_inv = 1.0 / (torch.diag(A) + 1e-10)  # Add small constant for stability
            z = M_inv * r
        else:
            z = r

        p = z.clone()
        r_norm_sq = r @ z
        initial_residual = r_norm_sq

        # Improved convergence check
        if torch.sqrt(r_norm_sq) < self.tolerance:
            return x

        for i in range(self.max_iterations):
            Ap = torch.mm(A, p.unsqueeze(1)).squeeze(1)
            alpha = r_norm_sq / (p @ Ap + 1e-10)  # Add small constant for stability

            # Update solution and residual
            x = x + alpha * p
            r = r - alpha * Ap

            # Apply preconditioner
            if self.use_preconditioner:
                z = M_inv * r
            else:
                z = r

            r_norm_sq_new = r @ z

            # Improved convergence check using relative residual
            relative_residual = torch.sqrt(r_norm_sq_new) / (torch.sqrt(initial_residual) + 1e-10)
            if relative_residual < self.tolerance:
                break

            beta = r_norm_sq_new / (r_norm_sq + 1e-10)
            r_norm_sq = r_norm_sq_new

            # Update search direction
            p = z + beta * p

            # Reorthogonalize if needed (every few iterations)
            if i % 50 == 0 and i > 0:
                r = b - torch.mm(A, x.unsqueeze(1)).squeeze(1)
                if self.use_preconditioner:
                    z = M_inv * r
                else:
                    z = r
                p = z.clone()
                r_norm_sq = r @ z

        return x


class SparseConjugateGradientSolver(LinearSolver):
    """Iterative linear solver using preconditioned conjugate gradient method."""

    def __init__(self, max_iterations=100, tolerance=1e-10, use_preconditioner=True):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_preconditioner = use_preconditioner

    def solve(self, A, b, device=None):
        if device is None:
            device = A.device

        n = A.shape[0]
        # Create sparse x with same structure as b
        x = torch.zeros_like(b)

        # Initial residual (keep sparse)
        r = b - torch.sparse.mm(A, x)  # Result is sparse

        # Jacobi preconditioner: M = diag(A)
        if self.use_preconditioner:
            M_inv = 1.0 / (sparse_diagonal(A) + 1e-10)  # Add small constant for stability
            M_inv = M_inv.unsqueeze(1)  # Shape: (N, 1)
            # Convert M_inv to sparse for multiplication
            M_inv_indices = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
            M_inv_sparse = torch.sparse_coo_tensor(M_inv_indices, M_inv.squeeze(), size=(n, n))
            z = torch.sparse.mm(M_inv_sparse, r)  # Result is sparse
        else:
            z = r

        p = z.clone()
        r_norm_sq = torch.sparse.sum(r * z)
        initial_residual = r_norm_sq

        if torch.sqrt(r_norm_sq) < self.tolerance:
            return x

        for i in range(self.max_iterations):
            Ap = torch.sparse.mm(A, p)
            alpha = r_norm_sq / (torch.sparse.sum(p * Ap) + 1e-10)

            # Update solution and residual (keeping sparse format)
            x = x + alpha * p  # Result stays sparse
            r = r - alpha * Ap

            # Apply preconditioner
            if self.use_preconditioner:
                z = torch.sparse.mm(M_inv_sparse, r)
            else:
                z = r

            r_norm_sq_new = torch.sparse.sum(r * z)

            # Improved convergence check using relative residual
            relative_residual = torch.sqrt(r_norm_sq_new) / (torch.sqrt(initial_residual) + 1e-10)
            if relative_residual < self.tolerance:
                break

            beta = r_norm_sq_new / (r_norm_sq + 1e-10)
            r_norm_sq = r_norm_sq_new

            # Update search direction
            p = z + beta * p

            # Reorthogonalize if needed
            if i % 50 == 0 and i > 0:
                r = b - torch.sparse.mm(A, x)
                if self.use_preconditioner:
                    z = torch.sparse.mm(M_inv_sparse, r)
                else:
                    z = r
                p = z.clone()
                r_norm_sq = torch.sparse.sum(r * z)

        return x


# -------- Linear Solvers --------
# Extract the diagonal, filling in missing values with 0
def sparse_diagonal(sparse_tensor):
    assert (
        sparse_tensor.layout == torch.sparse_coo
    ), "Input must be a sparse tensor in COOrdinate format"

    # Extract the indices and values
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()

    # Determine the size of the sparse tensor (assuming it's square)
    size = sparse_tensor.size(0)

    # Initialize a tensor for the diagonal filled with zeros
    diagonal_values = torch.zeros(size, dtype=values.dtype, device=sparse_tensor.device)

    # Identify the diagonal elements (row index == column index)
    diagonal_mask = indices[0] == indices[1]

    # Get the indices of diagonal elements
    diagonal_indices = indices[0][diagonal_mask]

    # Fill in the diagonal values where they exist
    diagonal_values[diagonal_indices] = values[diagonal_mask]

    return diagonal_values
