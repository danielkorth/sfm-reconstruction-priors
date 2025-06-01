from abc import ABC, abstractmethod

import torch


class StoppingCriterion(ABC):
    """Base stopping criterion framework."""

    def __init__(self, **kwargs):
        """Initialize base stopping criteria parameters."""
        self.max_iterations = kwargs.get("max_iterations", 1000)
        self.iteration = 0
        self.stop = False
        self.convergence_message = ""

    @abstractmethod
    def check(self, params, cost, **kwargs):
        """Check if optimization should stop."""
        pass

    def reset(self):
        """Reset internal state for a new optimization."""
        self.iteration = 0
        self.stop = False
        self.convergence_message = ""


class FirstOrderStoppingCriterion(StoppingCriterion):
    """Stopping criterion for first-order optimizers like Adam and GradientDescent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rel_decrease_threshold = kwargs.get("rel_decrease_threshold", 1e-6)
        self.patience = kwargs.get("patience", 10)
        self.gradient_norm_threshold = kwargs.get("gradient_norm_threshold", 1e-5)
        self.param_change_threshold = kwargs.get("param_change_threshold", 1e-5)

        # Internal state tracking
        self.prev_cost = None
        self.prev_params = None
        self.small_decrease_counter = 0

    def check(self, params, cost, gradient=None, **kwargs):
        """Check first-order stopping conditions."""
        self.iteration += 1
        cost_value = cost.item() if torch.is_tensor(cost) else cost

        # 1. Maximum iterations check
        if self.iteration >= self.max_iterations:
            self.stop = True
            self.convergence_message = f"Maximum iterations ({self.max_iterations}) reached"
            return self._get_info(cost_value)

        # 2. Relative improvement check
        if self.prev_cost is not None:
            rel_decrease = (
                abs(self.prev_cost - cost_value) / self.prev_cost if self.prev_cost > 0 else 0
            )

            if rel_decrease < self.rel_decrease_threshold:
                self.small_decrease_counter += 1
                if self.small_decrease_counter >= self.patience:
                    self.stop = True
                    self.convergence_message = (
                        f"Relative decrease below threshold {self.rel_decrease_threshold} "
                        f"for {self.patience} consecutive iterations"
                    )
                    return self._get_info(cost_value, rel_decrease)
            else:
                self.small_decrease_counter = 0

        # 3. Gradient norm check
        if gradient is not None:
            grad_norm = torch.norm(gradient).item()
            if grad_norm < self.gradient_norm_threshold:
                self.stop = True
                self.convergence_message = f"Gradient norm ({grad_norm:.6f}) below threshold ({self.gradient_norm_threshold})"
                return self._get_info(cost_value, rel_decrease=0, grad_norm=grad_norm)

        # 4. Parameter change check
        if self.prev_params is not None:
            param_change = torch.norm(params - self.prev_params).item()
            if param_change < self.param_change_threshold:
                self.stop = True
                self.convergence_message = f"Parameter change ({param_change:.6f}) below threshold ({self.param_change_threshold})"
                return self._get_info(cost_value, rel_decrease=0, param_change=param_change)

        # Update state for next iteration
        self.prev_cost = cost_value
        self.prev_params = params.clone().detach() if torch.is_tensor(params) else params.copy()

        return self._get_info(
            cost_value,
            rel_decrease=(
                abs(self.prev_cost - cost_value) / self.prev_cost
                if self.prev_cost is not None and self.prev_cost > 0
                else 0
            ),
        )

    def reset(self):
        """Reset internal state for a new optimization."""
        super().reset()
        self.prev_cost = None
        self.prev_params = None
        self.small_decrease_counter = 0

    def _get_info(self, cost_value, rel_decrease=0, grad_norm=None, param_change=None):
        """Construct info dictionary for the optimizer."""
        info = {
            "stop_opt": self.stop,
            "convergence_message": self.convergence_message,
            "iteration": self.iteration,
            "cost": cost_value,
            "rel_decrease": rel_decrease,
            "small_decrease_counter": self.small_decrease_counter,
        }
        if grad_norm is not None:
            info["grad_norm"] = grad_norm
        if param_change is not None:
            info["param_change"] = param_change
        return info


class SecondOrderStoppingCriterion(StoppingCriterion):
    """Stopping criterion for second-order optimizers like Gauss-Newton and Levenberg-Marquardt."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Gradient-based stopping
        self.gradient_threshold = kwargs.get("gradient_threshold", 1e-9)
        # Relative decrease stopping
        self.rel_decrease_threshold = kwargs.get("rel_decrease_threshold", 1e-6)
        # Step size stopping
        self.step_size_threshold = kwargs.get("step_size_threshold", 1e-9)
        # Trust region parameters
        self.max_inner_iterations = kwargs.get("max_inner_iterations", 10)

        # Internal state
        self.previous_cost = float("inf")
        self.just_updated_iteration = False

    def check(self, params=None, cost=None, **kwargs):
        """Check stopping conditions, with control over which checks to perform.

        Parameters:
            params: Current parameter values
            cost: Current cost value
            kwargs: Additional parameters that control which checks to perform:
                - jtf: Gradient for gradient convergence check
                - hlm: Step vector for step size check
                - inner_iterations: Number of inner iterations for trust region check
                - update_iteration: Whether to increment iteration counter
                - update_cost: Whether to update the stored cost value
        """
        # Process control flags
        update_iteration = kwargs.pop("update_iteration", True)
        update_cost = kwargs.pop("update_cost", cost is not None)

        # Create result info dictionary
        stop_info = {"stop_opt": False, "iteration": self.iteration}

        # Update iteration counter if requested
        if update_iteration and not self.just_updated_iteration:
            self.iteration += 1
            self.just_updated_iteration = True

            # Maximum iterations check
            if self.iteration >= self.max_iterations:
                self.stop = True
                self.convergence_message = f"Maximum iterations ({self.max_iterations}) reached"
                stop_info["stop_opt"] = True
                stop_info["convergence_message"] = self.convergence_message
                return stop_info
        else:
            self.just_updated_iteration = False

        # Add cost to info if provided
        if cost is not None:
            stop_info["cost"] = cost

        # Gradient convergence check if jtf provided
        jtf = kwargs.get("jtf", None)
        if jtf is not None:
            if torch.is_tensor(jtf):
                if hasattr(jtf, "is_sparse") and jtf.is_sparse:
                    grad_max = torch.max(torch.abs(jtf.to_dense())).item()
                else:
                    grad_max = torch.max(torch.abs(jtf)).item()

                stop_info["gradient_max"] = grad_max

                if grad_max < self.gradient_threshold:
                    self.stop = True
                    self.convergence_message = f"Gradient max value ({grad_max:.6f}) below threshold ({self.gradient_threshold})"
                    stop_info["stop_opt"] = True
                    stop_info["convergence_message"] = self.convergence_message
                    return stop_info

        # Step size check if hlm provided
        hlm = kwargs.get("hlm", None)
        if hlm is not None and params is not None:
            if torch.is_tensor(hlm):
                if hasattr(hlm, "is_sparse") and hlm.is_sparse:
                    hlm_norm = torch.norm(hlm.to_dense()).item()
                else:
                    hlm_norm = torch.norm(hlm).item()

                params_norm = torch.norm(params).item()
                threshold = self.step_size_threshold * (params_norm + self.step_size_threshold)

                stop_info["step_size"] = hlm_norm

                if hlm_norm < threshold:
                    self.stop = True
                    self.convergence_message = (
                        f"Step size too small: {hlm_norm:.6f} < {threshold:.6f}"
                    )
                    stop_info["stop_opt"] = True
                    stop_info["convergence_message"] = self.convergence_message
                    return stop_info

        # Cost reduction check if cost is provided and we should update
        if cost is not None and update_cost and self.previous_cost != float("inf"):
            relative_decrease = (
                abs(self.previous_cost - cost) / self.previous_cost
                if self.previous_cost > 0
                else 0
            )
            stop_info["relative_decrease"] = relative_decrease

            if relative_decrease < self.rel_decrease_threshold:
                self.stop = True
                self.convergence_message = f"Cost reduction too small: {relative_decrease:.6f} < {self.rel_decrease_threshold}"
                stop_info["stop_opt"] = True
                stop_info["convergence_message"] = self.convergence_message
                return stop_info

        # Inner iterations check if provided
        inner_iterations = kwargs.get("inner_iterations", None)
        if inner_iterations is not None:
            stop_info["inner_iterations"] = inner_iterations

            if inner_iterations >= self.max_inner_iterations:
                self.stop = True
                self.convergence_message = (
                    f"Trust region too small after {inner_iterations} iterations"
                )
                stop_info["stop_opt"] = True
                stop_info["convergence_message"] = self.convergence_message
                return stop_info

        # Update cost tracking if requested
        if cost is not None and update_cost:
            self.previous_cost = cost

        return stop_info

    def reset(self):
        """Reset internal state."""
        super().reset()
        self.previous_cost = float("inf")
        self.just_updated_iteration = False


class StoppingCriterionFactory:
    """Factory for creating appropriate stopping criteria based on optimizer type."""

    @staticmethod
    def create(optimizer_type="first_order", **kwargs):
        """Create a stopping criterion instance based on optimizer type."""
        if optimizer_type.lower() in ["first_order", "adam", "gradientdescent"]:
            return FirstOrderStoppingCriterion(**kwargs)
        elif optimizer_type.lower() in [
            "second_order",
            "gaussnewton",
            "levenbergmarquardt",
            "sparselevenbergmarquardt",
        ]:
            return SecondOrderStoppingCriterion(**kwargs)
        else:
            # Default to first-order criterion
            return FirstOrderStoppingCriterion(**kwargs)
