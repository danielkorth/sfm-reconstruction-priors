import roma
import torch
from jaxtyping import Float, Int

from opt.loss import LossFunction

# ----------- Bundle Adjustment -----------


def ba_residuals(
    params: Float[torch.Tensor, "n_cameras*6+n_points*3"],
    K: Float[torch.Tensor, "3*3"],
    n_cameras: int,
    n_points: int,
    camera_indices: Int[torch.Tensor, "n_points"],
    track_indices: Int[torch.Tensor, "n_points"],
    points_2d: Float[torch.Tensor, "n_points*2"],
    loss_fn: LossFunction = lambda x: x,
    **kwargs,
):
    # extract the params necessary
    def _extract_from_params(params, n_cameras):
        camera_params = params[: n_cameras * 6]
        camera_params = camera_params.reshape(-1, 6)
        rotations, translations = camera_params[:, :3], camera_params[:, 3:]
        points = params[n_cameras * 6 :]
        points = points.reshape(-1, 3)
        return dict(points=points, rotations=rotations, translations=translations)

    data = _extract_from_params(params, n_cameras)
    points, rotations, translations = data["points"], data["rotations"], data["translations"]

    projections = project(
        points[track_indices], rotations[camera_indices], translations[camera_indices], K
    )

    # Calculate residuals
    residuals = projections.flatten() - points_2d.flatten()
    cost = loss_fn(residuals.pow(2)).sum() * 0.5
    return (residuals, dict(projections=projections, cost=cost, residuals=residuals))


def project(points, rotations, translations, K):
    # push everything to the same type
    points = points.to(K)
    rotations = rotations.to(K)
    translations = translations.to(K)
    proj = roma.rotvec_to_rotmat(rotations) @ points[..., None]  # Nx3x3 @ Nx3x1 => Nx3x1
    proj = proj + translations[..., None]  # Nx3x1 + Nx3x1 => Nx3x1
    proj = proj / proj[:, 2, None]  # Nx3x1 / Nx1x1 => Nx3x1
    proj = K @ proj  # 3x3 @ Nx3x1 => Nx3x1
    return proj[:, :2].squeeze(2)  # Nx3x1 => Nx2


def project_chunked(points, rotations, translations, K, batch_size=1024):
    n_points = points.shape[0]
    projections = []

    for i in range(0, n_points, batch_size):
        # Get the current batch of points
        batch_points = points[i : i + batch_size]
        batch_rotations = rotations[i : i + batch_size]
        batch_translations = translations[i : i + batch_size]

        # Perform the projection for the current batch
        proj = (
            roma.rotvec_to_rotmat(batch_rotations) @ batch_points[..., None]
        )  # Nx3x3 @ Nx3x1 => Nx3x1
        proj = proj + batch_translations[..., None]  # Nx3x1 + Nx3x1 => Nx3x1
        proj = proj / proj[:, 2, None]  # Nx3x1 / Nx1x1 => Nx3x1
        proj = K @ proj  # 3x3 @ Nx3x1 => Nx3x1

        # Store the projections
        projections.append(proj[:, :2].squeeze(2))  # Nx3x1 => Nx2

    return torch.cat(projections, dim=0)  # Concatenate all batches


# ----------- Dust3r Energy -----------
def dust3r_residuals(
    params: Float[torch.Tensor, "n_points*3+n_edges*7"],
    n_points: int,
    pose_indices: Int[
        torch.Tensor, "n_tracks"
    ],  # refers to the index of the pose for the current track/pointmap
    pointmaps: Float[torch.Tensor, "n_tracks*3"],  # (x, y, z)
    dust3r_track_indices: Int[
        torch.Tensor, "n_tracks"
    ],  # index of the track that is being aligned to
    loss_fn: LossFunction = lambda x: x,
):
    # extract the params necessary
    def _extract_from_params(params, n_points):
        points = params[: n_points * 3]
        points = points.reshape(-1, 3)
        similarity_params = params[n_points * 3 :]
        similarity_params = similarity_params.reshape(-1, 7)
        return dict(points=points, similarity_params=similarity_params)

    data = _extract_from_params(params, n_points)
    points, similarity_params = data["points"], data["similarity_params"]

    # align pointmap with similarity transformation
    aligned_pointmaps = dust3r_pointmaps_align(pointmaps, similarity_params[pose_indices])

    # project the pointmaps
    residuals = points[dust3r_track_indices].flatten() - aligned_pointmaps.flatten()
    cost = loss_fn(residuals.pow(2)).sum() * 0.5
    return (residuals, dict(aligned_pointmaps=aligned_pointmaps, cost=cost, residuals=residuals))


def dust3r_pointmaps_align(
    pointmap: Float[torch.Tensor, "3"],
    similarity_params: Float[torch.Tensor, "7"],
):
    R = roma.rotvec_to_rotmat(similarity_params[:, :3]).float()
    t = similarity_params[:, 3:6].float()
    s = similarity_params[:, 6].float()

    # project the pointmaps
    aligned_point = torch.bmm(R, pointmap.unsqueeze(-1)).squeeze()
    aligned_point = aligned_point * s.unsqueeze(-1)
    aligned_point = aligned_point + t

    return aligned_point


# ----------- Combined Energy -----------
def combined_residuals(
    # params
    # (rotvec, translation), (points), (R, t, s)
    params: Float[
        torch.Tensor, "n_cameras*6+n_points*3+n_edges*7"
    ],  # concatenation of "regular" energy and dust3r energy
    # stuff for ba
    K: Float[torch.Tensor, "3*3"],
    camera_indices: Int[torch.Tensor, "n_points"],
    track_indices: Int[torch.Tensor, "n_points"],
    points_2d: Float[torch.Tensor, "n_points*2"],
    # stuff for dust3r
    pose_indices: Int[
        torch.Tensor, "n_tracks"
    ],  # refers to the index of the pose for the current track/pointmap
    pointmaps: Float[torch.Tensor, "n_tracks*3"],  # (x, y, z)
    dust3r_track_indices: Int[
        torch.Tensor, "n_tracks"
    ],  # index of the track that is being aligned to
    # information for params
    n_cameras: int,
    n_points: int,
    loss_fn: LossFunction = lambda x: x,
):
    # extract the params necessary
    def separate_params(params, n_cameras, n_points):
        ba_params = params[: n_cameras * 6 + n_points * 3]
        dust3r_params = params[n_cameras * 6 :]
        return dict(ba_params=ba_params, dust3r_params=dust3r_params)

    data = separate_params(params, n_cameras, n_points)
    ba_params, dust3r_params = data["ba_params"], data["dust3r_params"]

    # calculate the residuals
    ba_res, ba_info = ba_residuals(
        ba_params,
        K.to(ba_params),
        n_cameras,
        n_points,
        camera_indices,
        track_indices,
        points_2d,
        loss_fn,
    )

    # dust3r residuals
    dust3r_res, dust3r_info = dust3r_residuals(
        dust3r_params, n_points, pose_indices, pointmaps, dust3r_track_indices, loss_fn
    )

    # combine the residuals
    residuals = torch.cat([ba_res, dust3r_res], dim=0)

    # combine the costs
    cost = ba_info["cost"] + dust3r_info["cost"]

    # until when are residuals from ba, when do they belong to dust3r?
    residual_split = torch.tensor(len(ba_res))

    # print both costs
    print(f"BA Cost: {ba_info['cost']}, Dust3r Cost: {dust3r_info['cost']}, Total Cost: {cost}")

    return residuals, dict(
        cost=cost,
        residuals=residuals,
        ba_info=ba_info,
        dust3r_info=dust3r_info,
        residual_split=residual_split,
    )
