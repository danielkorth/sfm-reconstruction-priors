from abc import ABC, abstractmethod

import torch

from opt.linear_solve import LinearSolver
from opt.stopping_criterion import StoppingCriterionFactory
from utils.torch_sparse import slice_torch_sparse_coo_tensor, sparse_diagonal


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    def __init__(self, **kwargs):
        """Initialize the optimizer with stopping criteria."""
        optimizer_type = self.__class__.__name__
        self.stopping_criterion = StoppingCriterionFactory.create(
            optimizer_type=optimizer_type, **kwargs
        )

    def check_stopping(self, params, cost=None, **kwargs):
        """Check if optimization should stop."""
        return self.stopping_criterion.check(params, cost, **kwargs)

    def reset_stopping(self):
        """Reset stopping criteria for a new optimization run."""
        self.stopping_criterion.reset()

    @abstractmethod
    def step(
        self,
        params,
        residual_closure=None,
        manual_jac=None,
        params_mask=None,
        **kwargs,
    ):
        """Optimize the parameters.

        :params: the parameters to be optimized :residual_closure: closure function for the
            residuals taking only params :manual_jac: manual jacobian for the residuals taking only
            params (used instead of residual_closure, eg if you want to batch the computation)
        :params_mask: mask over the parameters, which to optimize and which not
        """
        pass


class GradientDescent(Optimizer):
    def __init__(self, step_size=1e-3, **kwargs):
        """Initialize the optimizer.

        Args:
            step_size: Initial step size (mostly for historical reasons, use line search)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)  # Initialize stopping criteria
        self.step_size = step_size

    def step(
        self,
        params,
        cost_function,
        params_mask=None,
        line_search_kwargs=None,
        use_parallel=False,
        **kwargs,
    ):
        """Optimization step using direct gradient computation through residuals.

        Args:
            params: The parameters to be optimized
            cost_function: Function for computing the cost and its gradients
            params_mask: Mask over the parameters, which to optimize and which not
            line_search_kwargs: Optional parameters to pass to the line search function
            use_parallel: Whether to use the parallel line search (default: True)

        Returns:
            Tuple of (updated parameters, info dictionary)
        """
        params_with_grad = params.clone().detach().requires_grad_(True)

        res = cost_function.residuals(params_with_grad)

        # Compute cost from residuals
        if isinstance(res, list):
            residuals_tensor = torch.cat(res)
        else:
            residuals_tensor = res

        cost = cost_function.forward_from_residuals(
            residuals_tensor, verbose=True, **cost_function.kwargs, **kwargs
        )

        # Compute gradients
        cost.backward()
        gradient = params_with_grad.grad.clone()

        # Apply mask if needed
        if params_mask is not None:
            gradient[~params_mask] = 0

        # Set default line search parameters if none provided
        if line_search_kwargs is None:
            line_search_kwargs = {}

        # Additional parameters for parallel line search
        if use_parallel and "batch_size" not in line_search_kwargs:
            line_search_kwargs["batch_size"] = 8
            line_search_kwargs["fine_search_batch_size"] = 4

        # Use line search to find the optimal step size - use parallel version if requested

        if use_parallel:
            params_new, search_info = parallel_line_search(
                params,
                -gradient,
                cost_function,
                **line_search_kwargs,
                **kwargs,
            )
        else:
            params_new, search_info = ternary_line_search(
                params,
                -gradient,
                cost_function,
                **line_search_kwargs,
                **kwargs,
            )

        stop_info = self.check_stopping(params_new, cost.detach(), gradient=gradient)

        # Store the residuals for use in other components (detached for memory efficiency)
        stop_info["residuals"] = residuals_tensor.detach()

        # Add optimizer-specific info
        stop_info.update(
            {
                "search_info": search_info,
            }
        )

        return params_new, stop_info


class Adam(Optimizer):
    def __init__(self, lr=3e-4, **kwargs):
        super().__init__(**kwargs)  # Initialize stopping criteria
        # Don't initialize optimizer here since we don't have params yet
        self.optimizer = None
        self.lr = lr
        self.scheduler = kwargs.get("scheduler", None)
        self.scheduler_instance = None

    def step(self, params, cost_function, params_mask=None, **kwargs):
        """
        :params: the parameters to be optimized
        :cost_function: function for computing the cost taking only params
        :params_mask: mask over the parameters, which to optimize and which not
        """
        # Create a copy of params that requires gradients
        params_to_optimize = params.clone().detach().requires_grad_(True)

        # Initialize optimizer if not already done
        if self.optimizer is None or len(self.optimizer.param_groups[0]["params"]) == 0:
            self.optimizer = torch.optim.Adam([params_to_optimize], lr=self.lr)
            # Initialize scheduler if specified
            if self.scheduler is not None:
                self.scheduler_instance = self.scheduler(self.optimizer)
        else:
            # Update the parameter in the optimizer
            self.optimizer.param_groups[0]["params"][0] = params_to_optimize

        self.optimizer.zero_grad()
        res = cost_function.residuals(params_to_optimize)
        cost = cost_function.forward_from_residuals(
            torch.cat(res) if isinstance(res, list) else res
        )
        cost.backward()

        # Get gradient for stopping criterion
        gradient = params_to_optimize.grad.clone()

        # Apply mask to gradients if provided
        if params_mask is not None:
            params_to_optimize.grad[~params_mask] = 0.0

        self.optimizer.step()

        # Step the scheduler if it exists
        if self.scheduler_instance is not None:
            self.scheduler_instance.step(cost)

        # Update original params with optimized values
        new_params = params.clone()
        with torch.no_grad():
            if params_mask is not None:
                # Only update parameters that are not masked
                new_params[params_mask] = params_to_optimize.detach()[params_mask]
            else:
                new_params = params_to_optimize.detach()

        # Check stopping conditions
        stop_info = self.check_stopping(new_params, cost, gradient=gradient)

        # Add optimizer-specific info
        stop_info.update(
            {
                "residuals": torch.cat(res).detach() if isinstance(res, list) else res.detach(),
                "lr": self.optimizer.param_groups[0]["lr"],
            }
        )
        return new_params, stop_info


class GaussNewton(Optimizer):
    def __init__(self, step_size=1e-5, linear_solver: LinearSolver = None, **kwargs):
        self.step_size = step_size
        self.linear_solver = linear_solver

    def step(self, params, cost_function, params_mask=None, **kwargs):
        """
        :params: the parameters to be optimized
        :residual_closure: closure function for the residuals taking only params
        :params_mask: mask over the parameters, which to optimize and which not
        """

        if params_mask is None:
            params_mask = torch.ones_like(params, dtype=bool)

        jac, res = cost_function.jacobian(params)

        res.requires_grad = True
        cost = cost_function.forward_from_residuals(res)

        cost.backward()
        res_grad = res.grad

        if params_mask is not None:
            slice_params = torch.arange(params.shape[0])[params_mask]
            jac = slice_torch_sparse_coo_tensor(jac, [torch.arange(res.shape[0]), slice_params])

        # Compute J^T J using sparse operations
        jtj = torch.sparse.mm(jac.t(), jac)

        # pretty significant differences in the two operations below
        jtf = jac.t() @ res_grad.unsqueeze(1).to_sparse_coo()

        hlm = self.linear_solver.solve(jtj.coalesce(), -jtf)

        hlm_dense = torch.zeros_like(params)
        hlm_dense[params_mask] = hlm.to_dense().squeeze()

        params_new, info = ternary_line_search(params, hlm_dense, cost_function)

        info["cost"] = cost
        info["residuals"] = res
        return params_new, info


# Example usage:
# optimizer = GaussNewtonOptimizer(step_size=1e-5)
# new_params, info = optimizer.optimize(params, residual_closure, params_mask)

# LevenbergMarquardt implemented similar to Ceres. Resources:
# https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf
# https://github.com/ceres-solver/ceres-solver/blob/c29b5257e23f91d6a47c4db9d57350ed4985ea46/internal/ceres/levenberg_marquardt_strategy.cc#L4
# https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/trust_region_strategy.h
# https://github.com/kashif/ceres-solver/blob/master/internal/ceres/levenberg_marquardt.cc
# http://ceres-solver.org/nnls_solving.html#section-levenberg-marquardt

# class LevenbergMarquardt(Optimizer):
#     tensor_type = "dense"

#     def __init__(
#         self,
#         mu=1e-4,
#         nu=2,
#         max_inner_iterations=10,
#         eps1=1e-9,
#         eps2=1e-9,
#         min_diagonal=1e-6,
#         max_diagonal=1e32,
#         q_tolerance=0.1,
#         function_tolerance=1e-6,
#         linear_solver: LinearSolver = None,
#         loss_function: LossFunction = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)  # Initialize with second-order stopping criterion

#         self.mu = torch.tensor(mu)
#         self.nu = torch.tensor(nu)
#         self.nu_init = torch.tensor(nu)
#         self.max_inner_iterations = max_inner_iterations
#         self.eps1 = torch.tensor(eps1)
#         self.eps2 = torch.tensor(eps2)
#         self.min_diagonal = torch.tensor(min_diagonal)
#         self.max_diagonal = torch.tensor(max_diagonal)
#         self.q_tolerance = torch.tensor(q_tolerance)
#         self.function_tolerance = torch.tensor(function_tolerance)
#         self.linear_solver = linear_solver
#         self.loss_function = loss_function

#     def step(self, params, residual_closure, jac_closure, params_mask=None, **kwargs):
#         """Implementation of the LM algorithm with dense matrices."""
#         dtype = params.dtype
#         device = params.device

#         #         if params_mask is None:
#         #             params_mask = torch.ones_like(params, dtype=bool)

#         # Compute jacobian and residuals
#         jac, res = jac_closure(params)
#         cost = self.loss_function(res**2, split_value=kwargs["split_value"]).sum() * 0.5
#         cost_value = cost.item()

#         if params_mask is not None:
#             jac[:, ~params_mask] = 0

#         #         # jtj
#         #         jtj = jac.T @ jac

#         # calculate jtf
#         res_grad = res * self.loss_function.backward(res**2, split_value=kwargs["split_value"])
#         jtf = jac.T @ res_grad

#         # Check stopping criteria - update iteration and check maximum iterations and gradient
#         stop_info = self.check_stopping(
#             params=params, cost=cost_value, jtf=jtf, update_iteration=True
#         )

#         if stop_info["stop_opt"]:
#             stop_info.update({"residuals": res, "mu": self.mu.item(), "nu": self.nu.item()})
#             return params, stop_info

#         #         step_accepted = False
#         #         inner_iterations = 0

#         # precompute the diagonal
#         diag = torch.min(torch.max(torch.diag(jtj), self.min_diagonal), self.max_diagonal)
#         diag = torch.eye(jtj.shape[0], dtype=dtype, device=device) * diag

#         # Store values for reporting
#         used_mu = self.mu.item()
#         used_nu = self.nu.item()

#         while not step_accepted and inner_iterations < self.max_inner_iterations:
#             # Solve (A + µdiag(jtj))hlm = -g
#             try:
#                 dampening = torch.sqrt(diag * self.mu)
#                 hlm = self.linear_solver.solve(jtj + dampening, -jtf)
#             except torch.linalg.LinAlgError:
#                 print("Linear solve failed: exiting optimization")
#                 break

#             # Check step size only
#             stop_info = self.check_stopping(
#                 params=params, hlm=hlm, update_iteration=False, update_cost=False
#             )

#             if stop_info["stop_opt"]:
#                 stop_info.update(
#                     {
#                         "residuals": res,
#                         "cost": cost_value,
#                         "mu": used_mu,
#                         "nu": used_nu,
#                     }
#                 )
#                 return params, stop_info

#             #             predicted_reduction = -jtf @ hlm - 0.5 * (hlm @ (jtj @ hlm))

#             #             # Try the step
#             #             new_params = params + hlm

#             # Compute new cost
#             tmp_res = residual_closure(new_params)
#             new_cost = (
#                 self.loss_function(tmp_res**2, split_value=kwargs["split_value"]).sum() * 0.5
#             )
#             new_cost_value = new_cost.item()

#             # Compute gain ratio ρ
#             rho = (cost_value - new_cost_value) / predicted_reduction

#             # Update stored values for reporting
#             used_mu = self.mu.item()
#             used_nu = self.nu.item()

#             if rho > 0:  # Step is acceptable
#                 params = new_params

#                 # Update damping parameters
#                 self.mu = self.mu * max(1 / 3, 1 - (2 * rho - 1) ** 3)
#                 self.nu = self.nu_init
#                 step_accepted = True

#                 # Check cost reduction after accepting step
#                 stop_info = self.check_stopping(
#                     params=params, cost=new_cost_value, update_iteration=False, update_cost=True
#                 )

#                 if stop_info["stop_opt"]:
#                     stop_info.update(
#                         {
#                             "residuals": res,
#                             "mu": used_mu,
#                             "nu": used_nu,
#                         }
#                     )
#                     return params, stop_info

#                 cost_value = new_cost_value
#             else:
#                 # Increase damping parameters
#                 self.mu = self.mu * self.nu
#                 self.nu = self.nu_init * self.nu

#             inner_iterations += 1

#         # Check inner iteration limit
#         stop_info = self.check_stopping(
#             params=params,
#             cost=cost_value,
#             inner_iterations=inner_iterations,
#             update_iteration=False,
#         )

#         if stop_info["stop_opt"]:
#             stop_info.update(
#                 {
#                     "residuals": res,
#                     "mu": used_mu,
#                     "nu": used_nu,
#                 }
#             )
#             return params, stop_info

#         # Final info
#         stop_info = self.check_stopping(params=params, cost=cost_value, update_iteration=False)

#         stop_info.update(
#             {
#                 "residuals": res,
#                 "mu": used_mu,
#                 "nu": used_nu,
#             }
#         )
#         return params, stop_info


class SparseLevenbergMarquardt(Optimizer):
    tensor_type = "sparse"

    def __init__(
        self,
        # optimizer parameters
        mu=1e-4,
        nu=2,
        min_diagonal=1e-6,
        max_diagonal=1e32,
        linear_solver: LinearSolver = None,
        # stopping criterion parameters
        max_iterations=50,
        max_inner_iterations=10,
        gradient_threshold=1e-9,
        function_tolerance=1e-6,
        step_size_threshold=1e-9,
        **kwargs,
    ):
        super().__init__(
            max_iterations=max_iterations,
            max_inner_iterations=max_inner_iterations,
            gradient_threshold=gradient_threshold,
            function_tolerance=function_tolerance,
            step_size_threshold=step_size_threshold,
            **kwargs,
        )  # Initialize with second-order stopping criterion

        self.mu = torch.tensor(mu)
        self.nu = torch.tensor(nu)
        self.max_inner_iterations = max_inner_iterations
        self.nu_init = torch.tensor(nu)
        self.min_diagonal = torch.tensor(min_diagonal)
        self.max_diagonal = torch.tensor(max_diagonal)
        self.linear_solver = linear_solver

    def step(self, params, cost_function, params_mask=None, **kwargs):
        """Implementation of the LM algorithm with sparse matrices."""
        dtype = params.dtype
        device = params.device

        if params_mask is None:
            params_mask = torch.ones_like(params, dtype=bool)

        # Compute jacobian and residuals
        jac, res = cost_function.jacobian(params)
        res.requires_grad = True
        cost = cost_function.forward_from_residuals(res)
        cost.backward()
        res_grad = res.grad

        # Apply mask if needed
        if params_mask is not None:
            slice_params = torch.arange(params.shape[0], device=params.device)[params_mask]
            jac = slice_torch_sparse_coo_tensor(jac, [torch.arange(res.shape[0]), slice_params])

        # Compute J^T J and J^T f
        jtj = torch.sparse.mm(jac.t(), jac)
        jtf = jac.t() @ res_grad.unsqueeze(1).to_sparse_coo()

        # Detach tensors
        res = res.detach()
        cost_value = cost.detach().item()

        # initialize the cost
        if self.stopping_criterion.previous_cost == torch.inf:
            self.stopping_criterion.previous_cost = cost_value

        # Check stopping criteria - update iteration and check maximum iterations and gradient
        stop_info = self.check_stopping(
            params=params, jtf=jtf, update_iteration=False  # gradient step mag
        )

        if stop_info["stop_opt"]:
            stop_info.update(
                {
                    "residuals": res,
                    "jac": jac,
                    "mu": self.mu.item(),
                    "nu": self.nu.item(),
                    "cost": cost_value,
                }
            )
            return params, stop_info

        # Prepare for inner loop
        step_accepted = False
        inner_iterations = 0

        # Calculate diagonal for damping
        diag = sparse_diagonal(jtj)
        diag = torch.clamp(
            diag, min=self.min_diagonal.to(device), max=self.max_diagonal.to(device)
        )
        diag_matrix = torch.sparse_coo_tensor(
            indices=torch.stack([torch.arange(len(diag)), torch.arange(len(diag))]).to(device),
            values=diag,
            size=jtj.size(),
        )

        # Store values for reporting
        used_mu = self.mu.item()
        used_nu = self.nu.item()

        # Inner loop for trust region adjustments
        while not step_accepted and inner_iterations < self.max_inner_iterations:
            # Try to solve the linear system
            try:
                dampening = torch.sqrt(self.mu) * diag_matrix
                damped_jtj = jtj + dampening
                hlm = self.linear_solver.solve(damped_jtj.coalesce(), -jtf)
            except torch.linalg.LinAlgError:
                break

            # Check step size only
            stop_info = self.check_stopping(
                params=params, hlm=hlm, update_iteration=False, update_cost=False
            )

            if stop_info["stop_opt"]:
                stop_info.update(
                    {
                        "residuals": res,
                        "jac": jac,
                        "cost": cost_value,
                        "mu": used_mu,
                        "nu": used_nu,
                    }
                )
                return params, stop_info

            # Calculate predicted reduction
            predicted_reduction = -jtf.t() @ hlm - 0.5 * (hlm.t() @ (jtj @ hlm))
            predicted_reduction = predicted_reduction.values()[0]

            # Try the step
            hlm_dense = torch.zeros_like(params)
            hlm_dense[params_mask] = hlm.to_dense().squeeze()
            new_params = params + hlm_dense

            # Compute new cost
            new_cost_value = cost_function(new_params, wandb_log=False)

            # Calculate gain ratio
            rho = (cost_value - new_cost_value) / predicted_reduction

            # Update stored values for reporting
            used_mu = self.mu.item()
            used_nu = self.nu.item()

            if rho > 0:  # Step is acceptable
                params = new_params

                # Update damping parameters
                self.mu = self.mu * max(1 / 3, 1 - (2 * rho - 1) ** 3)
                self.nu = self.nu_init
                step_accepted = True

                cost_value = new_cost_value
            else:
                # Increase damping
                self.mu = self.mu * self.nu
                self.nu = self.nu_init * self.nu

            inner_iterations += 1

        stop_info = self.check_stopping(
            params=params,
            cost=cost_value,
            inner_iterations=inner_iterations,
            update_iteration=True,
        )

        if stop_info["stop_opt"]:
            stop_info.update(
                {"residuals": res, "jac": jac, "mu": used_mu, "nu": used_nu, "cost": cost_value}
            )

        return params, stop_info


def ternary_line_search(
    params,
    direction,
    cost_closure,
    depth=3,
    lower=1e-12,
    higher=1.0,
    ratio_closeness=0.8,
    verbose=False,
    **kwargs,
):
    """Optimized ternary line search for finding the optimal step size.

    Args:
        params: Current parameters
        direction: Direction to search in (typically negative gradient)
        cost_closure: Function to evaluate cost at a point, must accept params kwarg
        depth: Number of iterations for fine-grained search
        lower: Lower bound for step size (in log space)
        higher: Upper bound for step size (in log space)
        ratio_closeness: Interpolation ratio for golden-section-like search
        verbose: Whether to print status messages

    Returns:
        Tuple of (new_parameters, info_dict)
    """
    # Compute initial cost
    initial_cost = cost_closure(params=params, wandb_log=False, **kwargs)

    # PHASE 1: Coarse search to find good initial bounds
    # Convert to log space for better numerical properties
    log_low_step_size = torch.log10(torch.tensor(lower))
    log_high_step_size = torch.log10(torch.tensor(higher))
    log_range = torch.arange(int(log_low_step_size), int(log_high_step_size) + 1)

    # Pre-compute step sizes for all iterations to batch computation
    step_sizes = 10.0**log_range

    # Find the best step size in the coarse search
    best_cost = float("inf")
    best_i = 0
    best_step_size = 0

    for i, step_size in enumerate(step_sizes):
        tmp_params = params + step_size * direction
        tmp_cost = cost_closure(params=tmp_params, wandb_log=False, **kwargs)
        if tmp_cost < best_cost:
            best_cost = tmp_cost
            best_i = int(log_range[i])
            best_step_size = step_size
    # PHASE 2: Fine-grained search
    # Set new bounds around the best coarse step size
    log_low_step_size = best_i - 1
    log_high_step_size = best_i + 1
    i = 0

    # Iterative binary-like search in log space
    while i < depth:
        # Find midpoint in log space
        log_mid_step_size = (log_low_step_size + log_high_step_size) / 2

        # Interpolate step sizes in log space
        log_midlow_step_size = (
            ratio_closeness * log_mid_step_size + (1 - ratio_closeness) * log_low_step_size
        )
        log_midhigh_step_size = (
            ratio_closeness * log_mid_step_size + (1 - ratio_closeness) * log_high_step_size
        )

        # Convert back to linear space
        midlow_step_size = 10**log_midlow_step_size
        midhigh_step_size = 10**log_midhigh_step_size

        # Compute costs for both step sizes
        midlow_params = params + midlow_step_size * direction
        midhigh_params = params + midhigh_step_size * direction

        left_cost = cost_closure(params=midlow_params, wandb_log=False, **kwargs)
        right_cost = cost_closure(params=midhigh_params, wandb_log=False, **kwargs)

        # Update bounds based on cost comparison
        if left_cost < right_cost:
            log_high_step_size = log_mid_step_size
            if left_cost < best_cost:
                best_cost = left_cost
                best_step_size = midlow_step_size
        elif left_cost > right_cost:
            log_low_step_size = log_mid_step_size
            if right_cost < best_cost:
                best_cost = right_cost
                best_step_size = midhigh_step_size
        else:
            # Equal costs, can stop early
            break

        i += 1

    # Check if we found a better solution
    stop_opt = initial_cost <= best_cost

    if verbose and not stop_opt:
        print("Applying step size:", best_step_size)

    # Apply the best step size found
    params_new = params + best_step_size * direction

    return params_new, {"step_size": best_step_size, "stop_opt": stop_opt}


def parallel_line_search(
    params,
    direction,
    cost_closure,
    depth=5,
    lower=1e-10,
    higher=1.0,
    batch_size=8,
    fine_search_batch_size=4,
    verbose=False,
    **kwargs,
):
    """Parallelized line search for finding the optimal step size.

    This function improves GPU utilization by evaluating multiple step sizes
    in parallel batches, significantly reducing the number of sequential operations.

    Args:
        params: Current parameters
        direction: Direction to search in (typically negative gradient)
        cost_closure: Function to evaluate cost at a point, must accept params kwarg
        depth: Number of iterations for fine-grained search
        lower: Lower bound for step size (in log space)
        higher: Upper bound for step size (in log space)
        batch_size: Number of step sizes to evaluate in parallel during coarse search
        fine_search_batch_size: Number of step sizes to evaluate in parallel during fine search
        verbose: Whether to print status messages

    Returns:
        Tuple of (new_parameters, info_dict)
    """
    # Compute initial cost
    initial_cost = cost_closure(params=params, wandb_log=False, **kwargs)

    # PHASE 1: Coarse search with parallel evaluation
    # Convert to log space for better numerical properties
    log_low_step_size = torch.log10(torch.tensor(lower, device=params.device))
    log_high_step_size = torch.log10(torch.tensor(higher, device=params.device))
    log_range = torch.linspace(
        log_low_step_size, log_high_step_size, steps=batch_size, device=params.device
    )

    # Pre-compute step sizes
    step_sizes = 10.0**log_range

    # Create a batch of parameters by applying different step sizes all at once
    # Shape: [batch_size, n_params]
    params_batch = params.unsqueeze(0) + step_sizes.unsqueeze(1) * direction.unsqueeze(0)

    costs = cost_closure.forward_batch(params_batch, wandb_log=False, **kwargs)

    # Find best step size
    best_idx = torch.argmin(costs).item()
    best_cost = costs[best_idx].item()
    best_step_size = step_sizes[best_idx].item()
    best_log_step = log_range[best_idx].item()

    # PHASE 2: Fine-grained search around the best coarse step size
    # Define a narrower range around the best step size from phase 1
    log_low = best_log_step - 0.5
    log_high = best_log_step + 0.5

    for _ in range(depth):
        # Generate evenly spaced step sizes in the current range
        fine_log_range = torch.linspace(
            log_low, log_high, steps=fine_search_batch_size, device=params.device
        )
        fine_step_sizes = 10.0**fine_log_range

        # Create a batch of parameters for fine-grained search
        fine_params_batch = params.unsqueeze(0) + fine_step_sizes.unsqueeze(
            1
        ) * direction.unsqueeze(0)

        fine_costs = cost_closure.forward_batch(fine_params_batch, wandb_log=False, **kwargs)

        # Find best fine step size
        fine_best_idx = torch.argmin(fine_costs).item()
        fine_best_cost = fine_costs[fine_best_idx].item()
        fine_best_step = fine_step_sizes[fine_best_idx].item()
        fine_best_log = fine_log_range[fine_best_idx].item()

        # Update overall best if better found
        if fine_best_cost < best_cost:
            best_cost = fine_best_cost
            best_step_size = fine_best_step

        # Narrow search range around best point for next iteration
        span = (log_high - log_low) / 4
        log_low = max(fine_best_log - span, log_low)
        log_high = min(fine_best_log + span, log_high)

    # Check if we found a better solution
    stop_opt = initial_cost <= best_cost

    if verbose and not stop_opt:
        print(f"Applying step size: {best_step_size}")

    # Apply the best step size found
    params_new = params + best_step_size * direction

    # Return new parameters and information dictionary
    return params_new, {"step_size": best_step_size, "stop_opt": stop_opt}
