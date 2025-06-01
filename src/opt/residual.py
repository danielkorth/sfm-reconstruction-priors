"""
Intended workflow with this:

1. Define residuals
2. vmap(residuals) -> List[Jacobian]
3. JacComposer.compose(List[Jacobian]) -> Jacobian
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import omegaconf
import roma
import torch
from jaxtyping import Float, Int


class JacComposer:
    """Composes a jacobian from the output of vmap(jacfwd)."""

    def __init__(self, n_residuals: int, n_params: int, block_start: list[int]):
        self.n_residuals = n_residuals
        self.n_params = n_params
        self.block_start = block_start

    def get_jac(self, dtype, device):
        return torch.zeros(self.n_residuals, self.n_params, dtype=dtype, device=device)

    def compose(
        self,
        jac: Float[torch.Tensor, "n_residuals n_params"],
        partials: List[
            Float[torch.Tensor, "n_residuals/res_per_forward res_per_forward single_params_size"]
        ],
        indices: List[Int[torch.Tensor, "n_residuals/res_per_forward"]] | List[List[int]],
    ) -> Float[torch.Tensor, "n_residuals n_params"]:
        """
        ALERT: it is important that partials / indices are ordered by block correctly.
        Composes a jacobian inplace.
        """
        for i, (partial, index, block_start) in enumerate(
            zip(partials, indices, self.block_start)
        ):
            # Handle case where index is a list by converting to numpy array and using repeat
            if isinstance(index, list) or isinstance(index, omegaconf.listconfig.ListConfig):
                col_start = (
                    torch.from_numpy(np.repeat(index, partial.shape[-2])) * partial.shape[-1]
                    + block_start
                )
            else:
                col_start = (
                    torch.repeat_interleave(index, partial.shape[-2]) * partial.shape[-1]
                    + block_start
                )
            cols = col_start[:, None] + torch.arange(partial.shape[-1])
            jac[torch.arange(self.n_residuals).unsqueeze(1), cols] = partial.reshape(
                -1, partial.shape[-1]
            )
        return jac

    def compose_sparse(
        self,
        partials: List[
            Float[torch.Tensor, "n_residuals/res_per_forward res_per_forward single_params_size"]
        ],
        indices: List[Int[torch.Tensor, "n_residuals/res_per_forward"]] | List[List[int]],
    ) -> torch.Tensor:
        """Composes a sparse jacobian in Coordinate format.

        Returns a sparse tensor of shape (n_residuals, n_params)
        """
        rows = []
        cols = []
        values = []

        for i, (partial, index, block_start) in enumerate(
            zip(partials, indices, self.block_start)
        ):
            # Handle case where index is a list
            if isinstance(index, torch.Tensor):
                col_start = (
                    torch.repeat_interleave(index, partial.shape[-2]) * partial.shape[-1]
                    + block_start
                )
            else:
                col_start = (
                    torch.from_numpy(np.repeat(index, partial.shape[-2])) * partial.shape[-1]
                    + block_start
                )

            # Generate column indices
            cols_block = col_start[:, None] + torch.arange(
                partial.shape[-1], device=col_start.device
            )
            # Generate row indices
            rows_block = (
                torch.arange(self.n_residuals, device=cols_block.device)
                .unsqueeze(1)
                .expand_as(cols_block)
            )

            # Append to lists
            rows.append(rows_block.reshape(-1))
            cols.append(cols_block.reshape(-1))
            values.append(partial.reshape(-1))

        # Concatenate all indices and values
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        # Create sparse tensor
        indices = torch.stack([rows, cols])
        return torch.sparse_coo_tensor(
            indices, values, size=(self.n_residuals, self.n_params), device=values.device
        )


class Residuals(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, params: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_aux(self, params: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, dict]:
        pass


class ReprojectionResiduals(Residuals):
    """Residuals for reprojection error.

    Assumes intrinsics is fixed/known.
    """

    def __init__(self, K: Float[torch.Tensor, "3 3"], image_space_residuals: bool = True):
        super().__init__()
        self.K = K
        self.res_per_forward = 2
        self.image_space_residuals = image_space_residuals

    def forward(
        self,
        cam_params: Float[torch.Tensor, "6"],
        point_3d: Float[torch.Tensor, "3"],
        point_2d: Float[torch.Tensor, "2"],
        **kwargs
    ) -> Float[torch.Tensor, "2"]:
        proj = roma.rotvec_to_rotmat(cam_params[:3]) @ point_3d
        proj = proj + cam_params[3:]
        proj = proj / proj[2]
        if self.image_space_residuals:
            proj = self.K @ proj
        residuals = proj[:2] - point_2d
        # else:
        #     # Convert point_2d to homogeneous coordinates
        #     point_2d_h = torch.cat([point_2d, torch.ones_like(point_2d[0:1])], dim=0)
        #     # Transform to normalized coordinates
        #     point_2d_norm = self.K.inverse() @ point_2d_h
        #     residuals = proj[:2] - point_2d_norm[:2]
        return residuals.flatten()

    # useful when we vmap the jacobian computation but still want to return the residuals
    def forward_aux(
        self,
        cam_params: Float[torch.Tensor, "6"],
        point_3d: Float[torch.Tensor, "3"],
        point_2d: Float[torch.Tensor, "2"],
        **kwargs
    ) -> Tuple[Float[torch.Tensor, "2"], dict]:
        residuals = self.forward(cam_params, point_3d, point_2d)
        return residuals, {"residuals": residuals}


class Point3DResiduals(Residuals):
    """Used to align the tracks to DUSt3r pointmaps."""

    def __init__(self):
        super().__init__()
        self.res_per_forward = 3

    def forward(
        self,
        track: Float[torch.Tensor, "3"],
        sim_params: Float[torch.Tensor, "7"],
        pointmap: Float[torch.Tensor, "3"],
        **kwargs
    ) -> Float[torch.Tensor, "3"]:
        R = roma.rotvec_to_rotmat(sim_params[:3])
        t = sim_params[3:6]
        s = sim_params[6]

        aligned_point = R @ pointmap
        aligned_point = aligned_point * s
        aligned_point = aligned_point + t
        residuals = track - aligned_point
        return residuals.flatten()

    def forward_aux(
        self,
        track: Float[torch.Tensor, "3"],
        sim_params: Float[torch.Tensor, "7"],
        pointmap: Float[torch.Tensor, "3"],
        **kwargs
    ) -> Tuple[Float[torch.Tensor, "3"], dict]:
        residuals = self.forward(track, sim_params, pointmap)
        return residuals, {"residuals": residuals}


class DebugPointmapReprojectionResiduals(Residuals):
    """Used for debugging only!

    We want to see what the "perfect" pointmap reprojection residuals are.
    """

    def __init__(self, reprojection_residual: ReprojectionResiduals):
        super().__init__()
        self.res_per_forward = 2
        self.reprojection_residual = reprojection_residual

    def forward(
        self,
        pointmap: Float[torch.Tensor, "3"],
        sim_params: Float[torch.Tensor, "7"],
        point_2d: Float[torch.Tensor, "2"],
        cam_params: Float[torch.Tensor, "6"],
        **kwargs
    ) -> Float[torch.Tensor, "2"]:
        R = roma.rotvec_to_rotmat(sim_params[:3])
        t = sim_params[3:6]
        s = sim_params[6]

        aligned_point = R @ pointmap
        aligned_point = aligned_point * s
        aligned_point = aligned_point + t

        residuals = self.reprojection_residual.forward(cam_params, aligned_point, point_2d)

        return residuals

    def forward_aux(
        self,
        pointmap: Float[torch.Tensor, "3"],
        sim_params: Float[torch.Tensor, "7"],
        point_2d: Float[torch.Tensor, "2"],
        cam_params: Float[torch.Tensor, "6"],
        **kwargs
    ) -> Tuple[Float[torch.Tensor, "2"], dict]:
        residuals = self.forward(pointmap, sim_params, point_2d, cam_params)
        return residuals, {"residuals": residuals}
