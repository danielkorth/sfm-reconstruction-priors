"""At the end of the day what we want is a closure function that takes the params, and returns the
residuals (and possibly jacobian) we want a residuals closure (params -> residuals) and a jac.

closure (residuals -> jacobian) (efficient) and therefore a closure that takes params -> residuals
-> jacobian.

The main issue is that we want to vmap the output, so it is non-trivial to just use the
residuals_closure to get the jacobian. So we have two different functions here, one that returns
only residuals, and one that returns both residuals and jacobian.
"""
import functools
from collections import defaultdict

import roma
import torch
from jaxtyping import Float, Int

import wandb
from opt.residual import JacComposer, Point3DResiduals, ReprojectionResiduals

# --------- BUNDLE ADJUSTMENT ---------


def ba_jac_closure(
    K: Float[torch.Tensor, "3*3"],
    n_cameras: int,
    n_points: int,
    camera_indices: Int[torch.Tensor, "n_points"],
    track_indices: Int[torch.Tensor, "n_points"],
    points_2d: Float[torch.Tensor, "n_points*2"],
    chunk_size: int = 1024,
    sparse_jac: bool = False,
    image_space_residuals: bool = True,
    **kwargs,
):
    residual_fn = ReprojectionResiduals(K, image_space_residuals=image_space_residuals)
    return functools.partial(
        ba_jac,
        residual_fn=residual_fn,
        n_cameras=n_cameras,
        n_points=n_points,
        camera_indices=camera_indices,
        track_indices=track_indices,
        points_2d=points_2d,
        chunk_size=chunk_size,
        sparse_jac=sparse_jac,
    )


def ba_residuals_closure(
    K: Float[torch.Tensor, "3*3"],
    n_cameras: int,
    n_points: int,
    camera_indices: Int[torch.Tensor, "n_points"],
    track_indices: Int[torch.Tensor, "n_points"],
    points_2d: Float[torch.Tensor, "n_points*2"],
    image_space_residuals: bool = True,
    chunk_size: int = 2048,
    **kwargs,
):
    residual_fn = ReprojectionResiduals(K, image_space_residuals=image_space_residuals)
    return functools.partial(
        ba_residuals,
        residual_fn=residual_fn,
        n_cameras=n_cameras,
        n_points=n_points,
        camera_indices=camera_indices,
        track_indices=track_indices,
        points_2d=points_2d,
        chunk_size=chunk_size,
    )


def ba_residuals(
    params: Float[torch.Tensor, "n_cameras*6+n_points*3"],
    residual_fn: ReprojectionResiduals,
    n_cameras: int,
    n_points: int,
    camera_indices: Int[torch.Tensor, "n_points"],
    track_indices: Int[torch.Tensor, "n_points"],
    points_2d: Float[torch.Tensor, "n_points*2"],
    chunk_size: int = 1024,
    **kwargs,
):
    if params.ndim == 2:
        # batched mode
        batched = True
        batch_size = params.shape[0]
    else:
        batched = False

    def _extract_from_params(params, n_cameras, batched=False):
        if batched:
            camera_params = params[:, : n_cameras * 6]
            camera_params = camera_params.view(params.shape[0], -1, 6)
            points = params[:, n_cameras * 6 :]
            points = points.view(params.shape[0], -1, 3)
        else:
            camera_params = params[: n_cameras * 6]
            camera_params = camera_params.view(-1, 6)
            points = params[n_cameras * 6 :]
            points = points.view(-1, 3)
        return dict(points=points, camera_params=camera_params)

    data = _extract_from_params(params, n_cameras, batched)
    points, camera_params = data["points"], data["camera_params"]

    # prepare data
    if not batched:
        residuals = torch.func.vmap(residual_fn.forward, in_dims=(0, 0, 0), chunk_size=chunk_size)(
            camera_params[camera_indices],
            points[track_indices],
            points_2d,
        )
    else:
        residuals = torch.func.vmap(residual_fn.forward, in_dims=(0, 0, 0), chunk_size=chunk_size)(
            camera_params[:, camera_indices].view(-1, 6),
            points[:, track_indices].view(-1, 3),
            points_2d.view(batch_size, 1),
        )

    return residuals.reshape(batch_size, -1) if batched else residuals.flatten()


def ba_jac(
    params: Float[torch.Tensor, "n_cameras*6+n_points*3"],
    residual_fn: ReprojectionResiduals,
    n_cameras: int,
    n_points: int,
    camera_indices: Int[torch.Tensor, "n_points"],
    track_indices: Int[torch.Tensor, "n_points"],
    points_2d: Float[torch.Tensor, "n_points*2"],
    # loss_fn: LossFunction = lambda x: x,
    chunk_size: int = 1024,
    sparse_jac: bool = False,
    **kwargs,
):
    def _extract_from_params(params, n_cameras):
        camera_params = params[: n_cameras * 6]
        camera_params = camera_params.reshape(-1, 6)
        points = params[n_cameras * 6 :]
        points = points.reshape(-1, 3)
        return dict(points=points, camera_params=camera_params)

    data = _extract_from_params(params, n_cameras)
    points, camera_params = data["points"], data["camera_params"]

    # prepare data
    camera_params_input = camera_params[camera_indices]
    points_input = points[track_indices]

    partials, residuals = torch.func.vmap(
        torch.func.jacfwd(residual_fn.forward_aux, argnums=(0, 1), has_aux=True),
        in_dims=(0, 0, 0),
        chunk_size=chunk_size,
    )(
        camera_params_input,
        points_input,
        points_2d,
    )

    composer = JacComposer(
        n_residuals=points_2d.flatten().shape[0],
        n_params=params.shape[0],
        block_start=[0, n_cameras * 6, n_cameras * 6 + n_points * 3],
    )

    if sparse_jac:
        jac = composer.compose_sparse(
            partials=partials,
            indices=[camera_indices, track_indices],
        )
    else:
        jac = composer.compose(
            jac=composer.get_jac(params.dtype, params.device),
            partials=partials,
            indices=[camera_indices, track_indices],
        )
    return jac, residuals["residuals"].flatten()


# --------- DUST3R ---------
def dust3r_jac_closure(
    n_points: int,
    pose_indices: Int[torch.Tensor, "n_tracks"],
    pointmaps: Float[torch.Tensor, "n_tracks*3"],
    dust3r_track_indices: Int[torch.Tensor, "n_tracks"],
    chunk_size: int = 1024,
    sparse_jac: bool = False,
    **kwargs,
):
    residual_fn = Point3DResiduals()
    return functools.partial(
        dust3r_jac,
        residual_fn=residual_fn,
        n_points=n_points,
        pose_indices=pose_indices,
        pointmaps=pointmaps,
        dust3r_track_indices=dust3r_track_indices,
        chunk_size=chunk_size,
        sparse_jac=sparse_jac,
    )


def dust3r_residuals_closure(
    n_points: int,
    pose_indices: Int[torch.Tensor, "n_tracks"],
    pointmaps: Float[torch.Tensor, "n_tracks*3"],
    dust3r_track_indices: Int[torch.Tensor, "n_tracks"],
    chunk_size: int = 2048,
    **kwargs,
):
    residual_fn = Point3DResiduals()
    return functools.partial(
        dust3r_residuals,
        residual_fn=residual_fn,
        n_points=n_points,
        pose_indices=pose_indices,
        pointmaps=pointmaps,
        dust3r_track_indices=dust3r_track_indices,
        chunk_size=chunk_size,
    )


def dust3r_residuals(
    params: Float[torch.Tensor, "n_points*3+n_edges*7"],
    residual_fn: Point3DResiduals,
    n_points: int,
    pose_indices: Int[torch.Tensor, "n_tracks"],
    pointmaps: Float[torch.Tensor, "n_tracks*3"],
    dust3r_track_indices: Int[torch.Tensor, "n_tracks"],
    chunk_size: int = 1024,
    **kwargs,
):
    if params.ndim == 2:
        # batched mode
        batched = True
        batch_size = params.shape[0]
    else:
        batched = False

    def _extract_from_params(params, n_points, batched=False):
        if batched:
            points = params[:, : n_points * 3]
            points = points.view(params.shape[0], -1, 3)
            similarity_params = params[:, n_points * 3 :]
            similarity_params = similarity_params.view(params.shape[0], -1, 7)
        else:
            points = params[: n_points * 3]
            points = points.view(-1, 3)
            similarity_params = params[n_points * 3 :]
            similarity_params = similarity_params.view(-1, 7)

        return dict(points=points, similarity_params=similarity_params)

    data = _extract_from_params(params, n_points, batched)
    points, similarity_params = data["points"], data["similarity_params"]

    # early out if there are no tracks to align
    if similarity_params.shape[0] == 0:
        return torch.empty(0)

    # prepare data
    if not batched:
        residuals = torch.func.vmap(residual_fn.forward, in_dims=(0, 0, 0), chunk_size=chunk_size)(
            points[dust3r_track_indices],
            similarity_params[pose_indices],
            pointmaps,
        )
    else:
        residuals = torch.func.vmap(residual_fn.forward, in_dims=(0, 0, 0), chunk_size=chunk_size)(
            points[:, dust3r_track_indices].view(-1, 3),
            similarity_params[:, pose_indices].view(-1, 7),
            pointmaps.repeat(batch_size, 1),
        )

    return residuals.reshape(batch_size, -1) if batched else residuals.flatten()


def dust3r_jac(
    params: Float[torch.Tensor, "n_points*3+n_edges*7"],
    residual_fn: Point3DResiduals,
    n_points: int,
    pose_indices: Int[torch.Tensor, "n_tracks"],
    pointmaps: Float[torch.Tensor, "n_tracks*3"],
    dust3r_track_indices: Int[torch.Tensor, "n_tracks"],
    chunk_size: int = 1024,
    sparse_jac: bool = False,
    **kwargs,
):
    def _extract_from_params(params, n_points):
        points = params[: n_points * 3]
        points = points.reshape(-1, 3)
        similarity_params = params[n_points * 3 :]
        similarity_params = similarity_params.reshape(-1, 7)
        return dict(points=points, similarity_params=similarity_params)

    data = _extract_from_params(params, n_points)
    points, similarity_params = data["points"], data["similarity_params"]

    partials, residuals = torch.func.vmap(
        torch.func.jacfwd(residual_fn.forward_aux, argnums=(0, 1), has_aux=True),
        in_dims=(0, 0, 0),
        chunk_size=chunk_size,
    )(
        points[dust3r_track_indices],
        similarity_params[pose_indices],
        pointmaps,
    )

    n_edges = similarity_params.shape[0]
    jac = JacComposer(
        n_residuals=pointmaps.flatten().shape[0],
        n_params=params.shape[0],
        block_start=[0, n_points * 3, n_points * 3 + n_edges * 7],
    )

    if sparse_jac:
        jac = jac.compose_sparse(
            partials=partials,
            indices=[dust3r_track_indices, pose_indices],
        )
    else:
        jac = jac.compose(
            jac=jac.get_jac(params.dtype, params.device),
            partials=partials,
            indices=[dust3r_track_indices, pose_indices],
        )
    return jac, residuals["residuals"].flatten()


def debug_pointmap_reprojection_residuals_closure(
    K: Float[torch.Tensor, "3*3"],
    n_cameras: int,
    n_points: int,
    n_poses: int,
    camera_indices: Int[torch.Tensor, "n_points"],
    track_indices: Int[torch.Tensor, "n_points"],
    points_2d: Float[torch.Tensor, "n_points*2"],
    pointmaps: Float[torch.Tensor, "n_tracks*3"],
    pose_indices: Int[torch.Tensor, "n_tracks"],
    dust3r_track_indices: Int[torch.Tensor, "n_tracks"],
    image_space_residuals: bool = True,
    **kwargs,
):
    return functools.partial(
        debug_pointmap_reprojection_residuals,
        K=K,
        n_cameras=n_cameras,
        n_points=n_points,
        n_poses=n_poses,
        camera_indices=camera_indices,
        track_indices=track_indices,
        points_2d=points_2d,
        pointmaps=pointmaps,
        pose_indices=pose_indices,
        dust3r_track_indices=dust3r_track_indices,
        image_space_residuals=image_space_residuals,
    )


def debug_pointmap_reprojection_residuals(
    params: Float[torch.Tensor, "n_cameras*6+n_points*3+n_edges*7"],
    K: Float[torch.Tensor, "3*3"],
    n_cameras: int,
    n_points: int,
    n_poses: int,
    camera_indices: Int[torch.Tensor, "n_points"],
    track_indices: Int[torch.Tensor, "n_points"],
    points_2d: Float[torch.Tensor, "n_points*2"],
    pointmaps: Float[torch.Tensor, "n_tracks*3"],
    pose_indices: Int[torch.Tensor, "n_tracks"],
    dust3r_track_indices: Int[torch.Tensor, "n_tracks"],
    image_space_residuals: bool = True,
    **kwargs,
):
    residual_fn = ReprojectionResiduals(K, image_space_residuals=image_space_residuals)

    def _extract_from_params(params, n_cameras, n_points, n_poses):
        camera_params = params[: n_cameras * 6]
        camera_params = camera_params.reshape(-1, 6)
        similarity_params = params[n_cameras * 6 + n_points * 3 :]
        similarity_params = similarity_params.reshape(-1, 7)
        return dict(camera_params=camera_params, similarity_params=similarity_params)

    data = _extract_from_params(params, n_cameras, n_points, n_poses)
    camera_params = data["camera_params"]
    similarity_params = data["similarity_params"]

    # get a mask of which reprojections we can actually compute (since dust3r does the cropping)
    mask = torch.zeros(len(track_indices), dtype=torch.bool)
    for i in range(len(track_indices)):
        mask[i] = track_indices[i] in dust3r_track_indices

    camera_indices_masked = [camera_indices[i] for i in range(len(camera_indices)) if mask[i]]
    track_indices_masked = [track_indices[i] for i in range(len(track_indices)) if mask[i]]
    points_2d_masked = [points_2d[i] for i in range(len(points_2d)) if mask[i]]

    all_poses = similarity_params[torch.unique(pose_indices, sorted=False)]
    all_R = roma.rotvec_to_rotmat(all_poses[:, :3])
    all_t = all_poses[:, 3:6]
    all_s = all_poses[:, 6]

    pointmaps_aligned = []
    for i in range(len(pointmaps)):
        new_aligned_point = (
            all_R[pose_indices[i]] @ pointmaps[i] * all_s[pose_indices[i]] + all_t[pose_indices[i]]
        )
        pointmaps_aligned.append(new_aligned_point)

    pointmaps_aligned = torch.stack(pointmaps_aligned)

    pointmap_for_track_idx = defaultdict(list)
    for idx in range(len(pointmaps)):
        pointmap_for_track_idx[dust3r_track_indices[idx].item()].append(pointmaps_aligned[idx])

    # average the pointmaps for each track
    pointmaps_average_per_track = {
        k: torch.stack(v).mean(dim=0) for k, v in pointmap_for_track_idx.items()
    }

    residuals = []
    for i in range(len(track_indices_masked)):
        residuals.append(
            residual_fn(
                cam_params=camera_params[camera_indices_masked[i].item()],
                point_3d=pointmaps_average_per_track[track_indices_masked[i].item()],
                point_2d=points_2d_masked[i],
            )
        )

    residuals = torch.stack(residuals)

    if wandb.run is not None:
        wandb.log({"residuals/pointmap_reprojection_mean(px)": residuals.norm(dim=-1).mean()})
        wandb.log({"residuals/pointmap_reprojection_max(px)": residuals.norm(dim=-1).max()})
        wandb.log({"residuals/pointmap_reprojection_min(px)": residuals.norm(dim=-1).min()})
        wandb.log({"residuals/pointmap_reprojection_std(px)": residuals.norm(dim=-1).std()})


def log_reprojection_residuals_in_px_closure(
    K: Float[torch.Tensor, "3*3"],
    n_cameras: int,
    n_points: int,
    camera_indices: Int[torch.Tensor, "n_points"],
    track_indices: Int[torch.Tensor, "n_points"],
    points_2d: Float[torch.Tensor, "n_points*2"],
    **kwargs,
):
    def log_reprojection_residuals_in_px(
        params: Float[torch.Tensor, "n_cameras*6+n_points*3"],
        K: Float[torch.Tensor, "3*3"],
        n_cameras: int,
        n_points: int,
        camera_indices: Int[torch.Tensor, "n_points"],
        track_indices: Int[torch.Tensor, "n_points"],
        points_2d: Float[torch.Tensor, "n_points*2"],
        **kwargs,
    ):
        residual_fn = ReprojectionResiduals(K, image_space_residuals=True)

        # filter pointmaps
        params = params[: n_cameras * 6 + n_points * 3]

        residuals = ba_residuals(
            params, residual_fn, n_cameras, n_points, camera_indices, track_indices, points_2d
        )

        residuals = residuals.reshape(-1, 2)

        if wandb.run is not None:
            wandb.log({"residuals/keypoint_pixel_error_mean(px)": residuals.norm(dim=-1).mean()})
            wandb.log({"residuals/keypoint_pixel_error_max(px)": residuals.norm(dim=-1).max()})
            wandb.log({"residuals/keypoint_pixel_error_min(px)": residuals.norm(dim=-1).min()})
            wandb.log({"residuals/keypoint_pixel_error_std(px)": residuals.norm(dim=-1).std()})

    return functools.partial(
        log_reprojection_residuals_in_px,
        K=K,
        n_cameras=n_cameras,
        n_points=n_points,
        camera_indices=camera_indices,
        track_indices=track_indices,
        points_2d=points_2d,
    )
