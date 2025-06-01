"""SCIPY Implementation for scannet camera model."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params, intrinsics):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    points_proj *= intrinsics[:2]
    points_proj += intrinsics[2:]
    return points_proj


def fun(
    params,
    camera_params_first,
    n_cameras,
    n_points,
    camera_indices,
    point_indices,
    points_2d,
    intrinsics,
):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    # fix first camera
    n_cameras = n_cameras - 1
    camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
    camera_params = np.concatenate((camera_params_first[None], camera_params))

    points_3d = params[n_cameras * 6 :].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], intrinsics)
    return (points_proj - points_2d).ravel()


def get_gt(params, camera_params_first, n_cameras, n_points, camera_indices, point_indices):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    # fix first camera
    n_cameras = n_cameras - 1
    camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
    camera_params = np.concatenate((camera_params_first[None], camera_params))

    points_3d = params[n_cameras * 6 :].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return points_proj


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


def ba_scipy(params, data):
    # recalculate the GT points (numerical differences between numpy and torch when conversion)
    points_2d = get_gt(
        params=data["params_gt"][6:].numpy(),
        camera_params_first=data["params_gt"][:6],
        n_cameras=data["n_cameras"],
        n_points=data["n_points"],
        camera_indices=data["camera_indices"],
        point_indices=data["point_indices"],
    )

    res_new = least_squares(
        fun,
        params[6:].numpy(),
        verbose=2,
        x_scale="jac",
        method="lm",
        args=(
            data["params_gt"][:6],
            data["n_cameras"],
            data["n_points"],
            data["camera_indices"],
            data["point_indices"],
            points_2d,
        ),
    )

    return res_new
