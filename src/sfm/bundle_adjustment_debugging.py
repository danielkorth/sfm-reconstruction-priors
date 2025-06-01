import cv2
import matplotlib.pyplot as plt
import numpy as np
import roma
import torch
from plotly import graph_objects as go
from torchtyping import TensorType

import wandb
from utils.camera import Camera
from utils.visualization import fig2img, visualize_reprojections_2

synthetic_data = {
    "1p2c": {
        "points": torch.tensor([[0, 1, 0]]),
        "look_at": torch.tensor(
            [[1, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0]], dtype=torch.float32
        ),
    },
    "3p2c": {
        "points": torch.tensor([[0, 1, 0], [0.1, 1.1, -0.2], [0.3, 0.8, 0.4]]),
        "look_at": torch.tensor(
            [[1, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0]], dtype=torch.float32
        ),
    },
    "8p2c": {
        "points": torch.tensor(
            [
                [0, 1, 0],
                [0.1, 1.1, -0.2],
                [0.3, 0.8, 0.4],
                [-0.15, 0.96, 0.4],
                [-0.4, 1.3, 0],
                [0.2, 0.7, -0.25],
                [-0.05, 1.1, -0.06],
                [0.4164, 0.5803, -0.0598],
            ]
        ),
        "look_at": torch.tensor(
            [[1, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0]], dtype=torch.float32
        ),
    },
    "8p2c_nonorthogonal": {
        "points": torch.tensor(
            [
                [0, 1, 0],
                [0.1, 1.1, -0.2],
                [0.3, 0.8, 0.4],
                [-0.15, 0.96, 0.4],
                [-0.4, 1.3, 0],
                [0.2, 0.7, -0.25],
                [-0.05, 1.1, -0.06],
                [0.4164, 0.5803, -0.0598],
            ]
        ),
        "look_at": torch.tensor(
            [[1, 1, 0, 0, 1, 0, 0, 1, 0], [0.1, 1, 0.8, 0, 1, 0, 0, 1, 0]], dtype=torch.float32
        ),
    },
    "8p3c": {
        "points": torch.tensor(
            [
                [0, 1, 0],
                [0.1, 1.1, -0.2],
                [0.3, 0.8, 0.4],
                [-0.15, 0.96, 0.4],
                [-0.4, 1.3, 0],
                [0.2, 0.7, -0.25],
                [-0.05, 1.1, -0.06],
                [0.4164, 0.5803, -0.0598],
            ]
        ),
        "look_at": torch.tensor(
            [
                [1, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 0, 0, 1, 0],
                [-0.7, 1, 0.1, 0, 1, 0, 0, 1, 0],  # not on a perfect circle
            ],
            dtype=torch.float32,
        ),
    },
    "8p4c": {
        "points": torch.tensor(
            [
                [0, 1, 0],
                [0.1, 1.1, -0.2],
                [0.3, 0.8, 0.4],
                [-0.15, 0.96, 0.4],
                [-0.4, 1.3, 0],
                [0.2, 0.7, -0.25],
                [-0.05, 1.1, -0.06],
                [0.4164, 0.5803, -0.0598],
            ]
        ),
        "look_at": torch.tensor(
            [
                [1, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 0, 0, 1, 0],
                [-0.7, 1, 0.1, 0, 1, 0, 0, 1, 0],  # not on a perfect circle
                [0.1, 0.9, 0.8, 0, 1, 0, 0, 1, 0],  # not on a perfect circle
            ],
            dtype=torch.float32,
        ),
    },
    "15p2c": {
        "points": torch.tensor(
            [
                [0, 1, 0],
                [0.1, 1.1, -0.2],
                [0.3, 0.8, 0.4],
                [-0.15, 0.96, 0.4],
                [-0.4, 1.3, 0],
                [0.2, 0.7, -0.25],
                [-0.05, 1.1, -0.06],
                [-0.3099, 1.0980, -0.3747],
                [0.2513, 1.2930, 0.4982],
                [0.2494, 1.0290, -0.2957],
                [0.1863, 0.7310, 0.3132],
                [-0.2250, 1.0813, 0.3717],
                [0.1789, 1.2653, 0.4200],
                [-0.4985, 0.6466, -0.1329],
                [0.4164, 0.5803, -0.0598],
            ]
        ),
        "look_at": torch.tensor(
            [[1, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0]], dtype=torch.float32
        ),
    },
    "15p3c": {
        "points": torch.tensor(
            [
                [0, 1, 0],
                [0.1, 1.1, -0.2],
                [0.3, 0.8, 0.4],
                [-0.15, 0.96, 0.4],
                [-0.4, 1.3, 0],
                [0.2, 0.7, -0.25],
                [-0.05, 1.1, -0.06],
                [-0.3099, 1.0980, -0.3747],
                [0.2513, 1.2930, 0.4982],
                [0.2494, 1.0290, -0.2957],
                [0.1863, 0.7310, 0.3132],
                [-0.2250, 1.0813, 0.3717],
                [0.1789, 1.2653, 0.4200],
                [-0.4985, 0.6466, -0.1329],
                [0.4164, 0.5803, -0.0598],
            ]
        ),
        "look_at": torch.tensor(
            [
                [1, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 0, 0, 1, 0],
                [-0.7, 1, 0.1, 0, 1, 0, 0, 1, 0],  # not on a perfect circle
            ],
            dtype=torch.float32,
        ),
    },
    "15p4c": {
        "points": torch.tensor(
            [
                [0, 1, 0],
                [0.1, 1.1, -0.2],
                [0.3, 0.8, 0.4],
                [-0.15, 0.96, 0.4],
                [-0.4, 1.3, 0],
                [0.2, 0.7, -0.25],
                [-0.05, 1.1, -0.06],
                [-0.3099, 1.0980, -0.3747],
                [0.2513, 1.2930, 0.4982],
                [0.2494, 1.0290, -0.2957],
                [0.1863, 0.7310, 0.3132],
                [-0.2250, 1.0813, 0.3717],
                [0.1789, 1.2653, 0.4200],
                [-0.4985, 0.6466, -0.1329],
                [0.4164, 0.5803, -0.0598],
            ]
        ),
        "look_at": torch.tensor(
            [
                [1, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 0, 0, 1, 0],
                [-0.7, 1, 0.1, 0, 1, 0, 0, 1, 0],  # not on a perfect circle
                [0.1, 0.9, 0.8, 0, 1, 0, 0, 1, 0],  # not on a perfect circle
            ],
            dtype=torch.float32,
        ),
    },
    "30p2c": {
        "points": torch.tensor(
            [
                [0, 1, 0],
                [0.1, 1.1, -0.2],
                [0.3, 0.8, 0.4],
                [-0.15, 0.96, 0.4],
                [-0.4, 1.3, 0],
                [0.2, 0.7, -0.25],
                [-0.05, 1.1, -0.06],
                [-0.3099, 1.0980, -0.3747],
                [0.2513, 1.2930, 0.4982],
                [0.2494, 1.0290, -0.2957],
                [0.1863, 0.7310, 0.3132],
                [-0.2250, 1.0813, 0.3717],
                [0.1789, 1.2653, 0.4200],
                [-0.4985, 0.6466, -0.1329],
                [0.4164, 0.5803, -0.0598],
                [-0.2670, 0.9646, 0.4352],
                [0.2464, 0.5562, 0.0733],
                [-0.4727, 0.5419, -0.3461],
                [-0.0639, 1.2305, 0.4891],
                [-0.1809, 0.7359, 0.1120],
                [0.3237, 1.2661, -0.4619],
                [-0.1005, 1.2517, 0.0713],
                [-0.0490, 0.9948, -0.3091],
                [0.1701, 0.9483, 0.0851],
                [0.2320, 0.8175, -0.0777],
                [0.0688, 1.4442, -0.0274],
                [-0.1970, 1.3707, -0.4632],
                [0.3978, 0.7958, -0.4714],
                [0.4831, 1.1169, -0.2411],
            ]
        ),
        "look_at": torch.tensor(
            [[1, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0]], dtype=torch.float32
        ),
    },
    "30p3c": {
        "points": torch.tensor(
            [
                [0, 1, 0],
                [0.1, 1.1, -0.2],
                [0.3, 0.8, 0.4],
                [-0.15, 0.96, 0.4],
                [-0.4, 1.3, 0],
                [0.2, 0.7, -0.25],
                [-0.05, 1.1, -0.06],
                [-0.3099, 1.0980, -0.3747],
                [0.2513, 1.2930, 0.4982],
                [0.2494, 1.0290, -0.2957],
                [0.1863, 0.7310, 0.3132],
                [-0.2250, 1.0813, 0.3717],
                [0.1789, 1.2653, 0.4200],
                [-0.4985, 0.6466, -0.1329],
                [0.4164, 0.5803, -0.0598],
                [-0.2670, 0.9646, 0.4352],
                [0.2464, 0.5562, 0.0733],
                [-0.4727, 0.5419, -0.3461],
                [-0.0639, 1.2305, 0.4891],
                [-0.1809, 0.7359, 0.1120],
                [0.3237, 1.2661, -0.4619],
                [-0.1005, 1.2517, 0.0713],
                [-0.0490, 0.9948, -0.3091],
                [0.1701, 0.9483, 0.0851],
                [0.2320, 0.8175, -0.0777],
                [0.0688, 1.4442, -0.0274],
                [-0.1970, 1.3707, -0.4632],
                [0.3978, 0.7958, -0.4714],
                [0.4831, 1.1169, -0.2411],
            ]
        ),
        "look_at": torch.tensor(
            [
                [1, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 0, 0, 1, 0],
                [-0.7, 1, 0.1, 0, 1, 0, 0, 1, 0],  # not on a perfect circle
            ],
            dtype=torch.float32,
        ),
    },
    "30p4c": {
        "points": torch.tensor(
            [
                [0, 1, 0],
                [0.1, 1.1, -0.2],
                [0.3, 0.8, 0.4],
                [-0.15, 0.96, 0.4],
                [-0.4, 1.3, 0],
                [0.2, 0.7, -0.25],
                [-0.05, 1.1, -0.06],
                [-0.3099, 1.0980, -0.3747],
                [0.2513, 1.2930, 0.4982],
                [0.2494, 1.0290, -0.2957],
                [0.1863, 0.7310, 0.3132],
                [-0.2250, 1.0813, 0.3717],
                [0.1789, 1.2653, 0.4200],
                [-0.4985, 0.6466, -0.1329],
                [0.4164, 0.5803, -0.0598],
                [-0.2670, 0.9646, 0.4352],
                [0.2464, 0.5562, 0.0733],
                [-0.4727, 0.5419, -0.3461],
                [-0.0639, 1.2305, 0.4891],
                [-0.1809, 0.7359, 0.1120],
                [0.3237, 1.2661, -0.4619],
                [-0.1005, 1.2517, 0.0713],
                [-0.0490, 0.9948, -0.3091],
                [0.1701, 0.9483, 0.0851],
                [0.2320, 0.8175, -0.0777],
                [0.0688, 1.4442, -0.0274],
                [-0.1970, 1.3707, -0.4632],
                [0.3978, 0.7958, -0.4714],
                [0.4831, 1.1169, -0.2411],
            ]
        ),
        "look_at": torch.tensor(
            [
                [1, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 0, 0, 1, 0],
                [-0.7, 1, 0.1, 0, 1, 0, 0, 1, 0],  # not on a perfect circle
                [0.1, 0.9, 0.8, 0, 1, 0, 0, 1, 0],  # not on a perfect circle
            ],
            dtype=torch.float32,
        ),
    },
}


def get_synthetic_scene(scene="1p2c"):
    return synthetic_data[scene]


# Goal: define points; define cameras (lookAt method); return data
def create_experimental_data(points, look_at_input, rotation_parameterization="quat"):
    # get intrinsics
    # W = H = 32
    f = 0.5
    p_x = 0.5
    p_y = 0.5
    K = Camera.compose_K(f, f, p_x, p_y)
    m_x = m_y = 32
    K = Camera.K_to_pixel_dimensions(K, m_x, m_y)

    # check if single track or camera
    if len(points.shape) == 1:
        points = points.unsqueeze(0)
    if len(look_at_input.shape) == 1:
        look_at_input = look_at_input.unsqueeze(0)

    # calculate look at camera rotation & translation
    extrinsics = list()
    camera_centres = list()
    translations = list()
    for look_at in look_at_input:
        camera_centres.append(look_at[:3])
        ex = Camera.look_at_matrix(look_at[:3], look_at[3:6], look_at[6:])
        extrinsics.append(ex)
        translations.append(ex[:3, 3])
    translations = torch.stack(translations)
    extrinsics = torch.stack(extrinsics)

    # calculate projection matrices
    P_matrices = list()
    for R_t in extrinsics:
        P_matrices.append(Camera.compose_P(K=K, R_t=R_t))

    # extract rotations
    quats = roma.rotmat_to_unitquat(extrinsics[:, :3, :3])
    euler_angles = roma.rotmat_to_euler("xyz", extrinsics[:, :3, :3])
    rotvecs = roma.rotmat_to_rotvec(extrinsics[:, :3, :3])
    rotmats = extrinsics[:, :3, :3]

    # make points homogeneous
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1)

    rotations = None
    if rotation_parameterization == "quat":
        rotations = quats
    elif rotation_parameterization == "rotvec":
        rotations = rotvecs
    elif rotation_parameterization == "euler":
        rotations = euler_angles
    else:
        raise ValueError(f"Unknown rotation parameterization: {rotation_parameterization}")

    # calculate gt 2d positions
    rotations = Camera.to_R(rotations, rotation_parameterization=rotation_parameterization)
    translations_temp = translations.unsqueeze(-1)
    ex = torch.cat((rotations, translations_temp), dim=-1)
    P = torch.matmul(K, ex)
    y = P @ points_homo.T
    reprojections_gt = y / y[:, 2].unsqueeze(1)
    # try out:
    # reprojections_gt = y / y[:, None, 2]

    return {
        "projections_gt": reprojections_gt,
        # SO(3) / SE(3)
        "euler": euler_angles,
        "quat": quats,
        "rotvec": rotvecs,
        "rotmat": rotmats,
        "R_t": extrinsics,
        # Projection
        "P": torch.stack(P_matrices),
        # Intrinsics
        "K": K,
        "translations": translations,
        "camera_locations": torch.stack(camera_centres),
        "points": points.float(),
    }


def noise_input(x, noise_level=0.1, noise_type="gaussian"):
    if noise_type == "gaussian":
        noise = torch.randn_like(x) * noise_level
    elif noise_type == "uniform":
        noise = torch.zeros_like(x, dtype=float).uniform_(-1, 1) * noise_level
    return noise


# from typeguard import typechecked


# @typechecked
def calc_reprojection_error(
    points: TensorType["N", 3],  # N points
    rotations: TensorType["M", "rotation"],  # M cameras
    translations: TensorType["M", 3],
    K: TensorType[3, 3],
    reprojections_gt: TensorType["M", 2, "N"],  # homogeneous coordinates
    reprojections_gt_filter: TensorType["M", "N"] = None,
    return_reprojections=False,
    rotation_parameterization="rotvec",
):
    # make points homogeneous
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1)

    if rotation_parameterization == "quat":
        # make sure we have a unit quaterion
        rotations.data = rotations.data / torch.linalg.norm(rotations.data, dim=1, keepdim=True)
        # rotations = rotations / torch.linalg.norm(rotations, dim=1, keepdim=True)

    rotations = Camera.to_R(rotations, rotation_parameterization=rotation_parameterization)
    translations = translations.unsqueeze(-1)
    ex = torch.cat((rotations, translations), dim=-1)
    P = torch.matmul(K, ex)
    y_hat = P @ points_homo.T
    reprojections_hat = y_hat / y_hat[:, 2].unsqueeze(1)
    # try out:
    # reprojections_gt = y / y[:, None, 2]

    # BOTH ARE EQUAL
    # loss = torch.mean((torch.norm((reprojections_hat - reprojections_gt)[:, :2], dim=1)**2))
    loss = (
        torch.nn.functional.mse_loss(
            reprojections_hat[:, :2], reprojections_gt[:, :2], reduction="none"
        )
        .sum(dim=1)[reprojections_gt_filter]
        .mean()
    )
    # ALternative more robust
    # loss = (
    #     torch.nn.functional.smooth_l1_loss(
    #         reprojections_hat[:, :2], reprojections_gt[:, :2], reduction="none"
    #     )
    #     .sum(dim=1)[reprojections_gt_filter]
    #     .mean()
    # )

    if return_reprojections:
        return loss, reprojections_hat, reprojections_gt
    return loss


def adjust_bundles(
    points,
    rotations,
    translations,
    K,
    projections_gt,
    optim,
    projections_gt_filter=None,
    scheduler=None,
    data_gt=dict(),
    logger=None,
    max_iter=100,
    inner_iter=1,  # if you do line search or something
    verbose=False,
    fix_first_camera=False,
    fix_first_point=False,
    logging_freqency=5,
    log_over_time=False,
    rotation_parameterization="rotvec",
    log_images=False,
    resect_intersect=False,
    **kwargs,
):
    if log_over_time:
        points_over_time = list()
        rotations_over_time = list()
        translations_over_time = list()

    if resect_intersect:
        resect_intersect_alternation = True

    # optimization loop
    i = 0
    while i < max_iter:
        # print(rotations)
        # log stuff over time
        if log_over_time:
            points_over_time.append(points.clone().detach())
            rotations_over_time.append(rotations.clone().detach())
            translations_over_time.append(translations.clone().detach())

        # calculate loss and reprojection error
        loss, reprojections_hat, reprojections_gt = calc_reprojection_error(
            points,
            rotations,
            translations,
            K,
            projections_gt,
            projections_gt_filter,
            return_reprojections=True,
            rotation_parameterization=rotation_parameterization,
        )

        # log stuff
        if i % logging_freqency == 0 and logger:
            with torch.no_grad():
                logger.log({"loss": loss.item()}, step=i)

                # visualize reprojections
                if log_images:
                    fig = visualize_reprojections_2(reprojections_hat, reprojections_gt)
                    logger.log({"view_1": wandb.Image(fig2img(fig))}, step=i)

                # log other stuff if available
                if "points" in data_gt:
                    point_dist = torch.mean(
                        torch.norm(points[:, :3] - data_gt["points"][:, :3], dim=1)
                    )
                    logger.log({"point_dist": point_dist.item()}, step=i)

                if "rotvec" in data_gt:
                    geodesic_dist = roma.rotvec_geodesic_distance(
                        rotations, data_gt["rotvec"]
                    ).mean()
                    logger.log({"geodesic_dist_rotvec": geodesic_dist.item()}, step=i)

                    # log relative rotation
                    r1, r2 = roma.rotvec_to_rotmat(rotations)
                    r1_gt, r2_gt = roma.rotvec_to_rotmat(data_gt["rotvec"])
                    r1_rel = r2 @ r1.T
                    r1_rel_gt = r2_gt @ r1_gt.T
                    theta = roma.rotmat_geodesic_distance(r1_rel, r1_rel_gt)  # In radian
                    logger.log({"geodesic_relative_rot": theta.item()}, step=i)

                    # log relative translation
                    t1, t2 = translations
                    t1_gt, t2_gt = data_gt["translations"]
                    t_diff = r1.T @ (t2 - t1)
                    t_diff_gt = r1_gt.T @ (t2_gt - t1_gt)
                    t_diff_norm = torch.norm(t_diff - t_diff_gt)
                    t_diff_cos = 1 - (t_diff @ t_diff_gt) / (
                        torch.norm(t_diff) * torch.norm(t_diff_gt)
                    )
                    logger.log({"t_diff_norm": t_diff_norm.item()}, step=i)
                    logger.log({"t_diff_cos": t_diff_cos.item()}, step=i)

                if "rotmat" in data_gt:
                    geodesic_distances = Camera.geodesic_distance(
                        rotations,
                        data_gt["rotmat"],
                        rotation_parameterization_1=rotation_parameterization,
                        rotation_parameterization_2="rotmat",
                    )
                    logger.log({"geodesic_distance": geodesic_distances.mean().item()}, step=i)

                if "translations" in data_gt:
                    translation_dist = torch.zeros(1)
                    for t_hat, t_gt in zip(translations, data_gt["translations"]):
                        translation_dist += torch.norm(t_hat - t_gt)

                    translation_dist = translation_dist / len(data_gt["translations"])
                    logger.log({"translation_dist": translation_dist.item()}, step=i)

                # average pixel distance:
                avg_pixel_dist = torch.zeros(1)
                for y_hat, y_gt in zip(reprojections_hat, reprojections_gt):
                    avg_pixel_dist += torch.mean(torch.norm((y_hat[:2] - y_gt[:2]).T, dim=1))
                avg_pixel_dist = avg_pixel_dist / len(reprojections_gt)
                logger.log({"avg_pixel_dist": avg_pixel_dist.item()}, step=i)

        elif verbose:
            print(f"Step {i} - Loss: {loss.item()}")

        # take optimization step
        if isinstance(optim, torch.optim.Optimizer):
            loss.backward()
            if rotations.grad is not None and fix_first_camera:
                rotations.grad[0].data.zero_()
            if translations.grad is not None and fix_first_camera:
                translations.grad[0].data.zero_()
            if points.grad is not None and fix_first_point:
                points.grad[0].data.zero_()

            # resect_intersect
            if resect_intersect and resect_intersect_alternation:
                rotations.grad.data.zero_()
                translations.grad.data.zero_()
                resect_intersect_alternation = not resect_intersect_alternation
            elif resect_intersect and not resect_intersect_alternation:
                points.grad.data.zero_()
                resect_intersect_alternation = not resect_intersect_alternation

            optim.step()
            # LOG GRADIENT INFORMATION
            # log gradient magnitude
            for group in optim.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if (fix_first_camera and group["name"] in ["rotations", "translations"]) or (
                        fix_first_point and group["name"] == "points"
                    ):
                        grad_norm = torch.mean(torch.norm(p.grad[1:], dim=1))
                    else:
                        grad_norm = torch.mean(torch.norm(p.grad, dim=1))
                    if logger:
                        logger.log({f"grad_norm_{group['name']}": grad_norm.item()}, step=i)

            optim.zero_grad()
            if scheduler:
                scheduler.step()

        else:
            import sys

            sys.exit("Not implemented yet")
        i += 1 * inner_iter

    if log_over_time:
        return dict(
            loss=loss,
            points_over_time=points_over_time,
            rotations_over_time=rotations_over_time,
            translations_over_time=translations_over_time,
        )
    return dict(loss=loss)
