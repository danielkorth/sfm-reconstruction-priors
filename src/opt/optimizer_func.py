import functools

import torch


# necessary to keep state but remain in the functional programming paradigm
def create_lm_optimizer(mu=1, nu=2, max_inner_iterations=10):
    state = {"mu": mu, "nu": nu}

    def lm_step(params, residual_closure, params_mask=None):
        out, info = levenberg_marquardt_single_step(
            params=params,
            residual_closure=residual_closure,
            params_mask=params_mask,
            mu=state["mu"],
            nu=state["nu"],
            max_inner_iterations=max_inner_iterations,
        )
        state["mu"] = info["mu"]
        state["nu"] = info["nu"]
        return out, info

    return lm_step


def instantiate_optimizer(name, options):
    if name == "gradient_descent":
        return functools.partial(gradient_descent, **options)
    elif name == "gauss_newton":
        return functools.partial(gauss_newton, **options)
    elif name == "levenberg_marquardt":
        lm_step = create_lm_optimizer(**options)
        return lm_step
    else:
        raise ValueError("Please provide a valid optimizer.")


def gradient_descent(
    params, residual_closure, params_mask=None, line_search_fn=None, step_size=1e-3, **kwargs
):
    """
    :params: the parameters to be optimized
    :residual_closure: closure function for the residuals taking only params
    :params_mask: mask over the parameters, which to optimize and which not
    """
    stop_opt = False

    # Compute gradients using jacrev to match the gauss_newton interface
    out = torch.func.jacrev(residual_closure, argnums=(0), has_aux=True)(params)
    jac, res = out[0], out[1]["residuals"]

    # Gradient is J^T * r
    grad = jac.T @ res

    # Apply mask if provided
    if params_mask is not None:
        grad_tmp = torch.zeros_like(params, dtype=res.dtype)
        grad_tmp[params_mask] = grad[params_mask]
        grad = grad_tmp

    if line_search_fn is not None:
        params_new, info = line_search_fn(params, grad, residual_closure)
        step_size = info["step_size"]
        stop_opt = info["stop_opt"]
    else:
        params_new = params - step_size * grad

    info = dict(step_size=step_size, stop_opt=stop_opt) | out[1]
    return params_new, info


def gauss_newton(
    params,
    residual_closure,
    params_mask=None,
    line_search_fn=None,
    step_size=1e-5,
    **kwargs,
):
    """
    :params: the parameters to be optimized
    :residual_closure: closure function for the residuals taking only params
    :params_opt_mask: mask over the parameters, which to optimize and which not (e.g. useful for fixing the first camera)
    """
    stop_opt = False

    out = torch.func.jacfwd(residual_closure, argnums=(0), has_aux=True)(params)
    jac, res = out[0], out[1]["residuals"]
    # apply opt mask
    if params_mask is not None:
        jac = jac[:, params_mask]
    jtj = jac.T @ jac
    jtf = jac.T @ res
    delta = torch.linalg.solve(jtj, -jtf)
    if params_mask is not None:
        delta_tmp = torch.zeros_like(params_mask, dtype=res.dtype)
        delta_tmp[params_mask] = delta
        delta = delta_tmp

    if line_search_fn is not None:
        params_new, info = line_search_fn(params, delta, residual_closure)
        step_size = info["step_size"]
        stop_opt = info["stop_opt"]
    else:
        params_new = params + step_size * delta

    info = dict(step_size=step_size, stop_opt=stop_opt) | out[1]
    return params_new, info


# https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf
def levenberg_marquardt_single_step(
    params,
    residual_closure,
    params_mask=None,
    eps1=1e-8,  # Gradient convergence criterion
    eps2=1e-8,  # Step size convergence criterion
    mu=1,  # Current damping parameter
    nu=2,  # Current step size multiplier
    max_inner_iterations=10,  # Maximum attempts to find acceptable step
    **kwargs,
):
    stop_opt = False
    convergence_criterion = 0

    # Compute initial Jacobian and gradient
    out = torch.func.jacfwd(residual_closure, argnums=(0), has_aux=True)(params)
    jac, res = out[0], out[1]["residuals"]
    if params_mask is not None:
        jac = jac[:, params_mask]

    # Compute A = J^T J and g = J^T f
    jtj = jac.T @ jac
    jtf = jac.T @ res

    # gradient convergence criterion
    grad_criterion = torch.max(jtf).item()
    # print("Gradient Norm:", torch.max(jtf).item(), "| Convergence Criterion: ", eps1)
    if torch.max(jtf) < eps1:
        convergence_criterion = 1
        step_accepted = True
        stop_opt = True

    current_cost = out[1]["cost"]
    step_accepted = False
    inner_iterations = 0

    step_size_mag = None
    step_size_crit = None

    while not step_accepted and inner_iterations < max_inner_iterations:
        # Solve (A + µdiag(jtj))hlm = -g
        try:
            diag = torch.eye(jtj.shape[0]) * torch.diag(jtj)
            hlm = torch.linalg.solve(jtj + mu * diag, -jtf)
        except torch.linalg.LinAlgError:
            print("Linear solve failed: exiting optimization")
            break

        # step size convergence criterion
        # print("Step size norm:", torch.norm(hlm).item(), "Step size conv criterion: ", (eps2 * (torch.norm(params[params_mask]) + eps2)).item())
        step_size_mag = torch.norm(hlm).item()
        step_size_crit = (eps2 * (torch.norm(params[params_mask]) + eps2)).item()
        if torch.norm(hlm) < eps2 * (torch.norm(params[params_mask]) + eps2):
            convergence_criterion = 2
            stop_opt = True
            break
        # Compute predicted reduction (L(0) - L(hlm))
        predicted_reduction = -jtf @ hlm - 0.5 * hlm @ jtj @ hlm

        # Expand step to full parameter size if mask is used
        if params_mask is not None:
            hlm_full = torch.zeros_like(params, dtype=res.dtype)
            hlm_full[params_mask] = hlm
            hlm = hlm_full

        # Try the step
        new_params = params + hlm

        # Compute new cost
        new_out = torch.func.jacfwd(residual_closure, argnums=(0), has_aux=True)(new_params)
        new_cost = new_out[1]["cost"]

        # Compute gain ratio ρ
        rho = (current_cost - new_cost) / predicted_reduction

        if rho > 0:  # Step is acceptable
            # Update params
            params = new_params
            # Update damping parameter
            mu = mu * max(1 / 3, 1 - (2 * rho - 1) ** 3)
            nu = 2
            step_accepted = True
        else:
            # Increase damping parameter and try again
            mu = mu * nu
            nu = 2 * nu

        inner_iterations += 1

    if not step_accepted:
        print(f"Warning: Could not find acceptable step after {max_inner_iterations} attempts")
        stop_opt = True
        convergence_criterion = 3

    info = {
        "mu": mu,
        "nu": nu,
        "stop_opt": stop_opt,
        "inner_iterations": inner_iterations,
        "grad_criterion": grad_criterion,
        "step_size_mag": step_size_mag,
        "step_size_crit": step_size_crit,
        "convergence_criterion": convergence_criterion,
    } | out[1]
    return params, info
