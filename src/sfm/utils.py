import torch


def get_optimization_mask(cfg, params_opt):
    """Utility function to get the optimization mask based on configuration.

    :param cfg: A DictConfig configuration composed by Hydra.
    :param params_opt: The parameters to optimize.
    :return: A boolean mask indicating which parameters to optimize.
    """
    opt_mask = None
    if cfg.get("optim_params") == "all":
        opt_mask = torch.ones_like(params_opt, dtype=bool)
        opt_mask[:6] = False
    elif cfg.get("optim_params") == "points":
        opt_mask = torch.ones_like(params_opt, dtype=bool)
        opt_mask[:12] = False
    elif cfg.get("optim_params") == "cameras":
        opt_mask = torch.ones_like(params_opt, dtype=bool)
        opt_mask[12:] = 0
        opt_mask[:6] = 0
    else:
        raise ValueError("Nothing to optimize...!")

    return opt_mask
