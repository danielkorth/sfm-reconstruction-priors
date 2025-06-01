"""The Cost function combines everything needed for the optimizer to calculate the residuals and
the jacobian.

It handles weighting of different terms, applying loss functions, and everything. In essence, it is
a wrapper around residuals.py, residuals_closure.py, losses.py, and is being passed to the
optimizer.
"""
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from jaxtyping import Float

import wandb
from opt.loss import LossFunction, TrivialLoss


class CostFunction(torch.nn.Module):
    """Abstract base class for cost functions.

    A cost function calculates residuals and their jacobians for optimization.
    """

    name = "CostFunction"

    @abstractmethod
    def forward(self, params: Float[torch.Tensor, "n_params"], **kwargs) -> torch.Tensor:
        """Calculate the cost (sum of squared residuals after applying loss function).

        Args:
            params: Parameters to evaluate
            **kwargs: Additional arguments

        Returns:
            Total cost value (scalar)
        """
        raise NotImplementedError("Subclasses must implement this method")

    def forward_batch(
        self, params_batch: Float[torch.Tensor, "batch_size n_params"], **kwargs
    ) -> torch.Tensor:
        """Calculate the cost for a batch of parameters.

        Default implementation loops through the batch and calls forward for each parameter set.
        Subclasses should override this with a more efficient implementation when possible.

        Args:
            params_batch: Batch of parameter tensors to evaluate with shape [batch_size, n_params]
            **kwargs: Additional arguments

        Returns:
            Tensor of cost values with shape [batch_size]
        """
        batch_size = params_batch.shape[0]
        costs = torch.zeros(batch_size, device=params_batch.device)

        for i in range(batch_size):
            costs[i] = self.forward(params_batch[i], **kwargs)

        return costs

    @abstractmethod
    def forward_from_residuals(self, residuals: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the cost from residuals.

        Args:
            residuals: Residuals tensor
            **kwargs: Additional arguments to pass to loss function

        Returns:
            Total cost value (scalar)
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def residuals(self, params: Float[torch.Tensor, "n_params"], **kwargs) -> torch.Tensor:
        """Calculate the raw residuals.

        Args:
            params: Parameters to evaluate
            **kwargs: Additional arguments

        Returns:
            Residuals tensor
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def jacobian(
        self, params: Float[torch.Tensor, "n_params"], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the jacobian and residuals.

        Args:
            params: Parameters to evaluate
            **kwargs: Additional arguments

        Returns:
            Tuple of (jacobian, residuals)
        """
        raise NotImplementedError("Subclasses must implement this method")


class BaseCostFunction(CostFunction):
    """Base implementation of a cost function that combines residuals and loss functions.

    This class provides the basic functionality to:
    1. Calculate residuals using a closure function
    2. Apply loss functions to the residuals
    3. Calculate jacobians for optimization
    """

    name = "BaseCostFunction"

    def __init__(
        self,
        residual_closure: Callable,
        jac_closure: Callable,
        loss_function: Optional[LossFunction] = None,
        sparse_jac: bool = False,
        **kwargs,
    ):
        """Initialize the cost function.

        Args:
            residual_closure: Function that takes parameters and returns residuals
            jac_closure: Function that takes parameters and returns (jacobian, residuals)
            loss_function: Optional loss function to apply to residuals
            sparse_jac: Whether to use sparse jacobians
            **kwargs: Additional arguments to pass to closures
        """
        super().__init__()
        self.residual_closure = residual_closure
        self.jac_closure = jac_closure
        self.loss_function = loss_function if loss_function is not None else TrivialLoss()
        self.sparse_jac = sparse_jac
        self.kwargs = kwargs

        self._pass_down_prefix()

    def _pass_down_prefix(self):
        """Pass down the cost function name to the loss function.

        (necessary for wandb logging)
        """
        curr_loss_obj = self.loss_function
        while curr_loss_obj is not None:
            curr_loss_obj.name_prefix = self.name
            curr_loss_obj = curr_loss_obj.loss_function

    def forward(self, params: Float[torch.Tensor, "n_params"], **kwargs) -> torch.Tensor:
        """Calculate the cost (sum of squared residuals after applying loss function).

        Args:
            params: Parameters to evaluate
            **kwargs: Additional arguments to pass to residual_closure
            residuals: Residuals tensor: if provided, we attach gradient to them so you can just call backward on them.

        Returns:
            Total cost value (scalar)
        """
        # Merge instance kwargs with call kwargs
        all_kwargs = {**self.kwargs, **kwargs}

        residuals = self.residual_closure(params, **all_kwargs)

        # kwargs
        cost = self.forward_from_residuals(residuals, **all_kwargs)

        return cost

    def forward_from_residuals(
        self, residuals: torch.Tensor, reduction="sum", **kwargs
    ) -> torch.Tensor:
        """Calculate the cost from residuals.

        Args:
            residuals: Residuals tensor
            **kwargs: Additional arguments to pass to loss function
        """
        if reduction == "sum":
            cost = self.loss_function(residuals, **kwargs).sum() * 0.5
        elif reduction == "mean":
            cost = self.loss_function(residuals, **kwargs).mean() * 0.5
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

        if kwargs.get("weight", 1.0) != 1.0:
            cost = cost * kwargs["weight"]

        if kwargs.get("verbose", False):
            print(f"Cost ({self.name}): {cost}")

        if wandb.run is not None:
            wandb.log({f"cost/{self.name}": cost})

        return cost

    def residuals(self, params: Float[torch.Tensor, "n_params"], **kwargs) -> torch.Tensor:
        """Calculate the raw residuals.

        Args:
            params: Parameters to evaluate
            **kwargs: Additional arguments to pass to residual_closure

        Returns:
            Residuals tensor
        """
        all_kwargs = {**self.kwargs, **kwargs}
        return self.residual_closure(params, **all_kwargs)

    def jacobian(
        self, params: Float[torch.Tensor, "n_params"], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the jacobian and residuals.

        Args:
            params: Parameters to evaluate
            **kwargs: Additional arguments to pass to jac_closure

        Returns:
            Tuple of (jacobian, residuals)
        """
        all_kwargs = {**self.kwargs, **kwargs}
        return self.jac_closure(params, **all_kwargs)

    def forward_batch(
        self, params_batch: Float[torch.Tensor, "batch_size n_params"], **kwargs
    ) -> torch.Tensor:
        """Calculate the cost for a batch of parameters efficiently.

        This method tries to avoid sequential processing by checking if the residual_closure
        can handle batched inputs directly.

        Args:
            params_batch: Batch of parameter tensors to evaluate
            **kwargs: Additional arguments to pass to the residual_closure

        Returns:
            Tensor of cost values with shape [batch_size]
        """
        # Merge instance kwargs with call kwargs
        all_kwargs = {**self.kwargs, **kwargs}
        batch_size = params_batch.shape[0]

        # Check if we can process the batch all at once
        try:
            # Attempt to compute all residuals at once
            # This will work if residual_closure can handle batched inputs
            batched_residuals = self.residual_closure(params_batch, **all_kwargs)
            is_batched = True
        except Exception:
            # Fall back to sequential processing
            is_batched = False

        if is_batched:
            # If the residual_closure returned batched results, process them
            if isinstance(batched_residuals, list):
                # Handle case where we get a list of residual tensors
                costs = torch.zeros(batch_size, device=params_batch.device)
                for i in range(batch_size):
                    costs[i] = self.loss_function(batched_residuals[i]).sum() * 0.5
            else:
                # Assume batched_residuals has shape [batch_size, n_residuals]
                # Apply loss function to each batch element
                costs = self.loss_function(batched_residuals).sum(dim=1) * 0.5

            # Apply weight if needed
            if kwargs.get("weight", 1.0) != 1.0:
                costs = costs * kwargs["weight"]

            # Only log if wandb is active and logging is enabled
            if wandb.run is not None and kwargs.get("wandb_log", True):
                wandb.log({f"cost/{self.name}_batch_mean": costs.mean()})

            return costs
        else:
            # Fall back to the default implementation if batched processing failed
            return super().forward_batch(params_batch, **kwargs)


class BACostFunction(BaseCostFunction):
    """Cost function specifically for Bundle Adjustment.

    This is a specialized cost function that uses the BA residual and jacobian closures.
    """

    name = "BA"

    def __init__(
        self,
        K: Float[torch.Tensor, "3*3"],
        n_cameras: int,
        n_points: int,
        camera_indices: torch.Tensor,
        track_indices: torch.Tensor,
        points_2d: Float[torch.Tensor, "n_points*2"],
        loss_function: Optional[LossFunction] = None,
        chunk_size: int = 1024,
        sparse_jac: bool = False,
        image_space_residuals: bool = True,
        **kwargs,
    ):
        """Initialize the BA cost function.

        Args:
            K: Camera intrinsic matrix
            n_cameras: Number of cameras
            n_points: Number of 3D points
            camera_indices: Indices of cameras for each observation
            track_indices: Indices of 3D points for each observation
            points_2d: 2D point observations
            loss_function: Optional loss function to apply to residuals
            chunk_size: Chunk size for batched operations
            sparse_jac: Whether to use sparse jacobians
            image_space_residuals: Whether residuals are in image space
            **kwargs: Additional arguments
        """
        from opt.residuals_closure import ba_jac_closure, ba_residuals_closure

        # Create closures
        residual_closure = ba_residuals_closure(
            K=K,
            n_cameras=n_cameras,
            n_points=n_points,
            camera_indices=camera_indices,
            track_indices=track_indices,
            points_2d=points_2d,
            image_space_residuals=image_space_residuals,
            chunk_size=chunk_size,
        )

        jac_closure = ba_jac_closure(
            K=K,
            n_cameras=n_cameras,
            n_points=n_points,
            camera_indices=camera_indices,
            track_indices=track_indices,
            points_2d=points_2d,
            chunk_size=chunk_size,
            sparse_jac=sparse_jac,
            image_space_residuals=image_space_residuals,
        )

        # Initialize base class
        super().__init__(
            residual_closure=residual_closure,
            jac_closure=jac_closure,
            loss_function=loss_function,
            sparse_jac=sparse_jac,
            **kwargs,
        )

        # Store parameters for parameter extraction
        self.n_cameras = n_cameras
        self.n_points = n_points


class DUSt3RCostFunction(BaseCostFunction):
    """Cost function specifically for DUSt3R point-to-point alignment.

    This is a specialized cost function that uses the DUSt3R residual and jacobian closures.
    """

    name = "P2P"

    def __init__(
        self,
        n_points: int,
        pose_indices: torch.Tensor,
        pointmaps: Float[torch.Tensor, "n_tracks*3"],
        dust3r_track_indices: torch.Tensor,
        loss_function: Optional[LossFunction] = None,
        chunk_size: int = 1024,
        sparse_jac: bool = False,
        **kwargs,
    ):
        """Initialize the DUSt3R cost function.

        Args:
            n_points: Number of 3D points
            pose_indices: Indices of poses for each observation
            pointmaps: 3D point observations from DUSt3R
            dust3r_track_indices: Indices of 3D points for each observation
            loss_function: Optional loss function to apply to residuals
            chunk_size: Chunk size for batched operations
            sparse_jac: Whether to use sparse jacobians
            **kwargs: Additional arguments
        """
        from opt.residuals_closure import dust3r_jac_closure, dust3r_residuals_closure

        # Create closures
        residual_closure = dust3r_residuals_closure(
            n_points=n_points,
            pose_indices=pose_indices,
            pointmaps=pointmaps,
            dust3r_track_indices=dust3r_track_indices,
            chunk_size=chunk_size,
        )

        jac_closure = dust3r_jac_closure(
            n_points=n_points,
            pose_indices=pose_indices,
            pointmaps=pointmaps,
            dust3r_track_indices=dust3r_track_indices,
            chunk_size=chunk_size,
            sparse_jac=sparse_jac,
        )

        # Initialize base class
        super().__init__(
            residual_closure=residual_closure,
            jac_closure=jac_closure,
            loss_function=loss_function,
            sparse_jac=sparse_jac,
            **kwargs,
        )

        # Store parameters for parameter extraction
        self.n_points = n_points


class CompositeCostFunction(CostFunction):
    """A cost function that combines multiple cost functions with different weights.

    This allows combining different types of cost functions (e.g., BA and DUSt3R) with different
    weights and parameter mappings.
    """

    name = "CompositeCost"

    def __init__(
        self,
        cost_functions: List[CostFunction],
        param_mappings: Optional[List[Dict[str, Union[int, slice]]]] = None,
        weights: Optional[List[float]] = None,
        scale_jac: bool = False,
        **kwargs,
    ):
        """Initialize the composite cost function.

        Args:
            cost_functions: List of cost function objects
            weights: Optional list of weights for each cost function (defaults to 1.0)
            param_mappings: Optional list of parameter mappings for each cost function
                Each mapping is a dictionary that maps parameter blocks to indices or slices
                in the full parameter vector
            **kwargs: Additional arguments to pass to all cost functions
        """
        super().__init__()
        self.cost_functions = cost_functions
        self.param_mappings = param_mappings
        self.weights = weights

        if self.param_mappings is not None:
            assert len(self.param_mappings) == len(
                self.cost_functions
            ), "Number of parameter mappings must match number of cost functions"

        self.kwargs = kwargs
        self.scale_jac = scale_jac

        # Cache for residual sizes
        self._residual_sizes = None

    def _extract_params(
        self, params: torch.Tensor, mapping: Dict[str, Union[int, slice]]
    ) -> torch.Tensor:
        """Extract parameters for a specific cost function based on mapping.

        Args:
            params: Full parameter vector
            mapping: Parameter mapping dictionary

        Returns:
            Extracted parameters for the cost function
        """
        if mapping is None:
            return params

        # Create a new parameter vector with the mapped parameters
        extracted_params = []
        for key, index_or_slice in mapping.items():
            if isinstance(index_or_slice, slice):
                extracted_params.append(params[index_or_slice].reshape(-1))
            else:
                extracted_params.append(params[index_or_slice : index_or_slice + 1])

        return torch.cat(extracted_params)

    def _extract_batch_params(
        self, params_batch: torch.Tensor, mapping: Dict[str, Union[int, slice]]
    ) -> torch.Tensor:
        """Extract parameters for a specific cost function based on mapping for batched input.

        Args:
            params_batch: Batch of parameter vectors [batch_size, n_params]
            mapping: Parameter mapping dictionary

        Returns:
            Extracted parameters for the cost function [batch_size, n_mapped_params]
        """
        if mapping is None:
            return params_batch

        batch_size = params_batch.shape[0]
        extracted_batch_params_list = []

        for key, index_or_slice in mapping.items():
            if isinstance(index_or_slice, slice):
                extracted_batch_params_list.append(
                    params_batch[:, index_or_slice].reshape(batch_size, -1)
                )
            else:
                extracted_batch_params_list.append(
                    params_batch[:, index_or_slice : index_or_slice + 1]
                )

        return torch.cat(extracted_batch_params_list, dim=-1)

    def forward(self, params: Float[torch.Tensor, "n_params"], **kwargs) -> torch.Tensor:
        """Calculate the combined cost.

        Args:
            params: Parameters to evaluate
            **kwargs: Additional arguments to pass to cost functions

        Returns:
            Total weighted cost value (scalar)
        """
        all_kwargs = {**self.kwargs, **kwargs}

        # Calculate cost for each cost function and apply weights
        total_cost = 0.0
        for i, cost_fn in enumerate(self.cost_functions):
            # Extract parameters for this cost function if mapping is provided
            if self.param_mappings is not None:
                cost_params = self._extract_params(params, self.param_mappings[i])
            else:
                cost_params = params

            cost = cost_fn(cost_params, weight=self.weights[i], **all_kwargs)
            total_cost += cost

        return total_cost

    def forward_batch(
        self, params_batch: Float[torch.Tensor, "batch_size n_params"], **kwargs
    ) -> torch.Tensor:
        """Calculate the combined cost for a batch of parameters.

        Args:
            params_batch: Batch of parameters to evaluate [batch_size, n_params]
            **kwargs: Additional arguments to pass to cost functions

        Returns:
            Tensor of cost values with shape [batch_size]
        """
        all_kwargs = {**self.kwargs, **kwargs}
        batch_size = params_batch.shape[0]

        # Initialize output costs tensor
        total_costs = torch.zeros(batch_size, device=params_batch.device)

        # Calculate cost for each cost function and apply weights
        for i, cost_fn in enumerate(self.cost_functions):
            # Extract parameters for this cost function if mapping is provided
            if self.param_mappings is not None:
                cost_params_batch = self._extract_batch_params(
                    params_batch, self.param_mappings[i]
                )
            else:
                cost_params_batch = params_batch

            # Check if this cost function supports forward_batch
            if hasattr(cost_fn, "forward_batch"):
                # Use batched version if available
                costs = cost_fn.forward_batch(
                    cost_params_batch, weight=self.weights[i], **all_kwargs
                )
            else:
                # Fall back to sequential evaluation
                costs = torch.zeros(batch_size, device=params_batch.device)
                for b in range(batch_size):
                    costs[b] = cost_fn(cost_params_batch[b], weight=self.weights[i], **all_kwargs)

            # Add weighted costs to the total
            total_costs += costs

        return total_costs

    def forward_from_residuals(self, residuals: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the cost from residuals.

        Args:
            residuals: Residuals tensor
            residuals_sizes: List of sizes of residuals for each cost function
            **kwargs: Additional arguments to pass to loss function
        """
        all_kwargs = {**self.kwargs, **kwargs}
        assert (
            self._residual_sizes is not None
        ), "residual_sizes must be set before calling forward_from_residuals"
        total_cost = 0.0
        prev_size = 0
        for i, (cost_fn, res_size) in enumerate(zip(self.cost_functions, self._residual_sizes)):
            residuals_masked = residuals[prev_size : prev_size + res_size]

            if (
                wandb.run is not None
                and wandb.run.summary["opt_step"]
                % wandb.run.config["sfm"]["reconstruction"]["global_optimization"][
                    "logging_frequency"
                ]
                == 0
            ):
                if cost_fn.name == "P2P":
                    residuals_masked_reshaped = residuals_masked.reshape(-1, 3)
                elif cost_fn.name == "BA":
                    residuals_masked_reshaped = residuals_masked.reshape(-1, 2)
                else:
                    raise ValueError(f"Unknown cost function: {cost_fn.name}")

                # Can be interpreted as pixel reprojection error OR 3D point error
                wandb.log(
                    {
                        f"residuals/{cost_fn.name}_mean": residuals_masked_reshaped.norm(
                            dim=-1
                        ).mean()
                    }
                )
                wandb.log(
                    {f"residuals/{cost_fn.name}_max": residuals_masked_reshaped.norm(dim=-1).max()}
                )
                wandb.log(
                    {f"residuals/{cost_fn.name}_min": residuals_masked_reshaped.norm(dim=-1).min()}
                )
                wandb.log(
                    {f"residuals/{cost_fn.name}_std": residuals_masked_reshaped.norm(dim=-1).std()}
                )

            cost = cost_fn.forward_from_residuals(
                residuals_masked, weight=self.weights[i], **all_kwargs
            )
            total_cost += cost
            prev_size += res_size

        if wandb.run is not None:
            wandb.log(
                {f"cost/{self.name}": total_cost},
            )

        return total_cost

    def residuals(self, params: Float[torch.Tensor, "n_params"], **kwargs) -> List[torch.Tensor]:
        """Calculate residuals for each cost function.

        Args:
            params: Parameters to evaluate
            **kwargs: Additional arguments to pass to cost functions

        Returns:
            List of residual tensors
        """
        all_kwargs = {**self.kwargs, **kwargs}

        residuals_list = []
        residual_sizes = []
        for i, cost_fn in enumerate(self.cost_functions):
            # Extract parameters for this cost function if mapping is provided
            if self.param_mappings is not None:
                cost_params = self._extract_params(params, self.param_mappings[i])
            else:
                cost_params = params

            residuals_list.append(cost_fn.residuals(cost_params, **all_kwargs))
            residual_sizes.append(residuals_list[-1].shape[0])

        self._residual_sizes = residual_sizes
        return residuals_list

    def jacobian(
        self, params: Float[torch.Tensor, "n_params"], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the combined jacobian and residuals.

        This method combines the jacobians from all cost functions into a single jacobian.

        Args:
            params: Parameters to evaluate
            **kwargs: Additional arguments to pass to cost functions

        Returns:
            Tuple of (combined jacobian, combined residuals)
        """
        all_kwargs = {**self.kwargs, **kwargs}

        # Get jacobians and residuals from each cost function
        jacobians = []
        residuals_list = []
        residual_sizes = []

        for i, cost_fn in enumerate(self.cost_functions):
            # Extract parameters for this cost function if mapping is provided
            if self.param_mappings is not None:
                cost_params = self._extract_params(params, self.param_mappings[i])
            else:
                cost_params = params

            jac, res = cost_fn.jacobian(cost_params, **all_kwargs)

            jac = jac.coalesce()

            # scale jacobian by std to make it more numerically stable
            if self.scale_jac:
                jac = jac / jac.values().std()

            # If we have parameter mappings, we need to map the jacobian columns
            # to the correct positions in the full jacobian
            if self.param_mappings is not None:
                # Create a mapping from local parameter indices to global parameter indices
                local_to_global = torch.zeros(
                    cost_params.shape[0], dtype=torch.long, device=params.device
                )
                current_idx = 0

                for _, index_or_slice in self.param_mappings[i].items():
                    if isinstance(index_or_slice, slice):
                        # Calculate how many parameters this slice represents
                        start = index_or_slice.start or 0
                        stop = index_or_slice.stop or params.shape[0]
                        step = index_or_slice.step or 1
                        slice_length = len(range(start, stop, step))

                        # Map local indices to global indices
                        local_indices = torch.arange(
                            current_idx, current_idx + slice_length, device=params.device
                        )
                        global_indices = torch.arange(start, stop, step, device=params.device)
                        local_to_global[local_indices] = global_indices

                        current_idx += slice_length
                    else:
                        # Single index case
                        local_to_global[current_idx] = index_or_slice
                        current_idx += 1

                # Map jacobian column indices from local to global
                jac_indices = jac.coalesce().indices()
                jac_values = jac.coalesce().values()

                # Map each column index to its global parameter index
                new_col_indices = local_to_global[jac_indices[1]]

                # Create new indices tensor with mapped column indices
                new_indices = torch.stack([jac_indices[0], new_col_indices])

                # Create new sparse jacobian with mapped indices
                jac = torch.sparse_coo_tensor(
                    new_indices, jac_values, (jac.shape[0], params.shape[0]), device=params.device
                )

            jacobians.append(jac)
            residuals_list.append(res)
            residual_sizes.append(res.shape[0])

        # Combine jacobians and residuals
        # For now, we'll just stack them vertically
        # In a more sophisticated implementation, we would need to handle
        # the parameter mappings properly
        combined_jac = torch.cat(jacobians, dim=0)
        combined_residuals = torch.cat(residuals_list, dim=0)

        self._residual_sizes = residual_sizes

        return combined_jac, combined_residuals
