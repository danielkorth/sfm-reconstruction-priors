"""We see a loss function similar to the Ceres Solver.

(http://ceres-solver.org/nnls_modeling.html)
A loss function is a scalar valued function that operates on the (squared) residuals and is (mostly) used to reduce the influnce of outliers
"""

from abc import abstractmethod
from typing import Optional

import torch

import wandb


class LossFunction(torch.nn.Module):
    """Abstract base class for loss functions.

    A loss function is applied to squared residuals to reduce the influence of outliers.
    """

    name = "LossFunction"

    def __init__(
        self,
        name_prefix: str = "",
        **kwargs,
    ):
        """Initialize the loss function with optional logging capabilities.

        Args:
            name: Name of this loss term for logging
            enable_logging: Whether to enable logging
            logging_dir: Directory to save logs
            log_interval: How often to print log messages (in steps)
            save_interval: How often to save metrics to disk (in steps)
            **kwargs: Additional arguments
        """
        super().__init__()
        self.name_prefix = name_prefix

    def _log_wandb(
        self,
        x: torch.Tensor,
    ):
        """Log the loss function to wandb.

        Args:
            x: Loss function output
        """
        return
        # if wandb.run is None or not wandb.config[""]:
        #     return

        # wandb.log({f"loss/{self.name_prefix}_{self.name}_forward_mean": x.abs().mean().item()})
        # wandb.log({f"loss/{self.name_prefix}_{self.name}_forward_std": x.abs().std().item()})
        # wandb.log({f"loss/{self.name_prefix}_{self.name}_forward_max": x.abs().max().item()})
        # wandb.log({f"loss/{self.name_prefix}_{self.name}_forward_sum": x.abs().sum().item()})

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the loss function to squared residuals.

        Args:
            x: Squared residuals
            **kwargs: Additional arguments

        Returns:
            Transformed residuals
        """
        raise NotImplementedError("Subclasses must implement this method")


class TrivialLoss(LossFunction):
    """Identity loss function that doesn't modify the residuals."""

    name = "TrivialLoss"

    def __init__(self, loss_function: Optional[LossFunction] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = loss_function

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.loss_function is None:
            result = x
            return result

        result = self.loss_function(x, **kwargs)
        return result


class CauchyLoss(LossFunction):
    """Cauchy loss function that reduces the influence of outliers.

    The Cauchy loss function is defined as:
    rho(s) = log(1 + s)
    """

    name = "CauchyLoss"

    def __init__(self, loss_function: Optional[LossFunction] = None, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = loss_function
        self.scale = scale

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.loss_function is None:
            result = torch.log(1 + x / self.scale)
            return result

        cauchy_out = torch.log(1 + x / self.scale)
        return self.loss_function(cauchy_out, **kwargs)


class SquaredLoss(LossFunction):
    """Squared loss function that doesn't modify the residuals."""

    name = "SquaredLoss"

    def __init__(self, loss_function: Optional[LossFunction] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = loss_function

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.loss_function is None:
            result = x**2
            return result

        squared = x**2
        return self.loss_function(squared, **kwargs)


class ScaledLoss(LossFunction):
    """Wrapper around a loss function that scales the input residuals.

    This is useful for normalizing residuals of different scales before applying the loss function.
    """

    name = "ScaledLoss"

    def __init__(self, loss_function: Optional[LossFunction] = None, scale: float = 1.0, **kwargs):
        """Initialize the scaled loss function.

        Args:
            loss_function: Base loss function to apply after scaling
            scale: Scale factor to apply to residuals
        """
        super().__init__(**kwargs)
        self.loss_function = loss_function
        self.scale = scale

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply scaling and then the base loss function.

        Args:
            x: Squared residuals
            **kwargs: Additional arguments

        Returns:
            Transformed residuals
        """
        if self.loss_function is None:
            scaled_x = x * self.scale
            return scaled_x

        scaled_x = x * self.scale
        return self.loss_function(scaled_x, **kwargs)


class ConfWeightedLoss(LossFunction):
    """Wrapper around a loss function that applies a weight to the output.

    This is useful for balancing different loss terms in a composite cost function.
    """

    name = "ConfWeightedLoss"

    def __init__(self, loss_function: Optional[LossFunction] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = loss_function

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if "conf_scale" not in kwargs:
            raise ValueError("Confidence must be provided")
        conf_scale = kwargs["conf_scale"]
        if self.loss_function is None:
            res = x * conf_scale
            return res

        res = x * conf_scale
        return self.loss_function(res, **kwargs)
