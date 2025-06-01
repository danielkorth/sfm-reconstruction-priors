"""Rerun Visualization Utilities.

To use Rerun with remote visualization:

1. Start the Rerun Viewer and TCP server on your remote machine:
       rerun --serve

2. Set up port forwarding on your local machine:
       ssh -L 9876:localhost:9876 -L 9877:localhost:9877 -L 9090:localhost:9090 -N user@remote_server

3. Access the web viewer at:
       http://localhost:9090?url=ws://localhost:9877

This module provides utilities for:
- Visualizing Structure from Motion reconstructions
- Managing Rerun server connections
- Testing visualization setup
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

RERUN_KWARGS = {
    "points": {"colors": [0x8839EFFF], "radii": 0.01},
    "pointmaps": {"colors": [0xFE640BFF], "radii": 0.005},
    "points_pointmaps_strips": {"colors": [0x00FF00FF], "radii": 0.0005},
    "reference_mesh": {"radii": 0.001},
    "background": [239, 241, 245],
}


def connect_server(name: str = "SfM", spawn: bool = False) -> None:
    """Initialize and connect to the Rerun server.

    Args:
        spawn: If True, spawns a new viewer. If False, connects to an existing viewer.
    """
    rr.init(name, spawn=spawn)
    if not spawn:
        rr.connect_tcp()


def setup_blueprint() -> None:
    # Create a Spatial3D view to display the points.
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/",
            name="3D Scene",
            # Set the background color to light blue.
            background=RERUN_KWARGS["background"],
            # Configure the line grid.
            line_grid=rrb.archetypes.LineGrid3D(
                visible=True,  # The grid is enabled by default, but you can hide it with this property.
                spacing=0.1,  # Makes the grid more fine-grained.
                # By default, the plane is inferred from view coordinates setup, but you can set arbitrary planes.
                plane=rr.components.Plane3D(np.array([0, 0, 1])),
                stroke_width=2.0,  # Makes the grid lines twice as thick as usual.
                color=[76, 79, 105, 128],  # Colors the grid a half-transparent white.
            ),
        ),
        collapse_panels=True,
    )

    rr.send_blueprint(blueprint)


def setup_blueprint_special_plane() -> None:
    # Create a Spatial3D view to display the points.
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/",
            name="3D Scene",
            # Set the background color to light blue.
            background=RERUN_KWARGS["background"],
            # Configure the line grid.
            line_grid=rrb.archetypes.LineGrid3D(
                visible=True,  # The grid is enabled by default, but you can hide it with this property.
                spacing=0.1,  # Makes the grid more fine-grained.
                # By default, the plane is inferred from view coordinates setup, but you can set arbitrary planes.
                plane=rr.components.Plane3D(np.array([0, 1, 0])),
                stroke_width=2.0,  # Makes the grid lines twice as thick as usual.
                color=[76, 79, 105, 128],  # Colors the grid a half-transparent white.
            ),
        ),
        collapse_panels=True,
    )

    rr.send_blueprint(blueprint)


def log_points(points: np.ndarray, name: str = "points", points_kwargs: Dict = {}) -> None:
    rr.log(name, rr.Points3D(points, **points_kwargs))


def visualize_reconstruction(data: Union[Dict, Path], name: str = "SfM") -> None:
    """Visualize a Structure from Motion reconstruction using Rerun.

    Args:
        data: Either a dictionary containing reconstruction parameters or a Path to a .pth file
    """
    connect_server(name=name)

    # Load data if path is provided
    if isinstance(data, Path):
        data = torch.load(data)

    # If data doesn't have multiple reconstruction cycles, wrap it in a dict
    if not isinstance(data, dict) or (isinstance(data, dict) and "params_per_step" in data):
        data = {0: data}

    K = data[0]["K"]
    reconstruction_cycles = len(data.keys())

    overall_time = 0
    for i in range(reconstruction_cycles):
        rr.set_time_sequence("/rec_cycle", i)

        # Extract camera parameters
        camera_params_len = data[i]["n_cameras"] * 6
        camera_params = data[i]["params_per_step"][:, :camera_params_len]
        point_params = data[i]["params_per_step"][:, camera_params_len:]

        # Reshape parameters
        rotations = camera_params.reshape(-1, 6)[:, :3]
        translations = camera_params.reshape(-1, 6)[:, 3:]
        n_steps = point_params.shape[0]

        # Reshape into proper dimensions
        point_params = point_params.reshape(n_steps, -1, 3)
        rotations = rotations.reshape(n_steps, -1, 3)
        translations = translations.reshape(n_steps, -1, 3)

        n_ba_steps = point_params.shape[0]
        for j in range(n_ba_steps):
            rr.set_time_sequence("/ba_step", overall_time)
            rr.set_time_sequence(f"/cycle_{i:03d}/ba_step", j)

            # Log 3D points
            rr.log("points", rr.Points3D(point_params[j]))

            # Log cameras
            for k, (translation, rotation) in enumerate(zip(translations[j], rotations[j])):
                rr.log(
                    f"camera_{k:03d}",
                    rr.Transform3D(
                        translation=translation,
                        rotation=rr.RotationAxisAngle(axis=rotation, radians=torch.norm(rotation)),
                        from_parent=True,
                        scale=1,
                    ),
                )
                rr.log(f"camera_{k:03d}", rr.ViewCoordinates.RDF, static=True)

                rr.log(
                    f"camera_{k:03d}",
                    rr.Pinhole(
                        resolution=[K[0][2] * 2, K[1][2] * 2],
                        focal_length=[K[0][0], K[1][1]],
                        principal_point=[K[0][2], K[1][2]],
                        image_plane_distance=0.15,
                    ),
                )
            overall_time += 1
        rr.disable_timeline(f"/cycle_{i:03d}/ba_step")

    rr.disconnect()


def test_connection(show_test_images: bool = True) -> None:
    """Test the Rerun server connection by logging some test images.

    Args:
        show_test_images: If True, logs test images to verify the connection
    """
    rr.init("rerun_test", spawn=False)
    rr.connect_tcp()

    if show_test_images:
        # Create test images
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image1[:, :] = [255, 0, 0]  # Red

        image2 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2[:, :] = [0, 255, 0]  # Green

        # Log test images
        rr.log("test/image1", rr.Image(image1))
        rr.log("test/image2", rr.Image(image2))

        print("Test images logged successfully. Check the viewer to confirm connection.")
    else:
        print("Connection test successful.")


def visualize_reconstruction_colmap(data: Union[Dict, Path], name: str = "SfM_COLMAP") -> None:
    """Visualize a Structure from Motion reconstruction from COLMAP-style data using Rerun. This
    function expects a single reconstruction state without time sequences.

    Args:
        data: Either a dictionary containing reconstruction parameters or a Path to a .pth file.
              The dictionary should contain 'n_cameras', 'params_per_step', and 'K'.
        name: Name for the recording/visualization
    """
    connect_server(name=name)

    # Load data if path is provided
    if isinstance(data, Path):
        data = torch.load(data)

    K = data["K"]

    # Extract camera parameters
    camera_params_len = data["n_cameras"] * 6
    params = data["params"]

    if len(params.shape) == 1:
        # If params is a 1D tensor, reshape it to 2D with a single step
        params = params.unsqueeze(0)

    camera_params = params[:, :camera_params_len]
    point_params = params[:, camera_params_len:]

    # Reshape parameters
    rotations = camera_params.reshape(-1, 6)[:, :3]
    translations = camera_params.reshape(-1, 6)[:, 3:]

    # Reshape point parameters
    point_params = point_params.reshape(-1, 3)

    # Log 3D points
    rr.log("points", rr.Points3D(point_params))

    # Log cameras
    for k, (translation, rotation) in enumerate(zip(translations, rotations)):
        rr.log(
            f"camera_{k:03d}",
            rr.Transform3D(
                translation=translation,
                rotation=rr.RotationAxisAngle(axis=rotation, radians=torch.norm(rotation)),
                from_parent=True,
                scale=1,
            ),
        )
        rr.log(f"camera_{k:03d}", rr.ViewCoordinates.RDF, static=True)

        rr.log(
            f"camera_{k:03d}",
            rr.Pinhole(
                resolution=[K[0][2] * 2, K[1][2] * 2],
                focal_length=[K[0][0], K[1][1]],
                principal_point=[K[0][2], K[1][2]],
                image_plane_distance=0.15,
            ),
        )

    rr.disconnect()


def visualize_final_reconstruction(rec, name="SfM_final") -> None:
    connect_server(name=name)
    setup_blueprint()

    rec._log_mesh_to_rerun()

    rotations, translations = rec._align_poses_with_gt()
    rotations_gt = rec.rotations_gt
    translations_gt = rec.translations_gt

    import roma

    # Log estimated cameras
    log_cameras(roma.rotmat_to_rotvec(rotations), translations, rec, log_images=True)
    # Log GT cameras
    log_cameras(rotations_gt, translations_gt, rec, prefix="gt_")

    points = rec.points
    log_points(points, name="points")

    rr.disconnect()


def log_cameras(rotations, translations, rec, log_images=False, prefix="") -> None:
    K = rec.scenegraph.views[0].K
    # LOG CAMERAS
    for k, (translation, rotation, view_idx) in enumerate(
        zip(translations, rotations, rec.active_view_idxs)
    ):
        rr.log(
            f"{prefix}camera_{k:03d}",
            rr.Transform3D(
                translation=translation.cpu().numpy(),
                rotation=rr.RotationAxisAngle(
                    axis=rotation.cpu().numpy(), radians=torch.norm(rotation).cpu().numpy()
                ),
                from_parent=True,
                scale=1,
            ),
        )

        # Only log camera parameters and images for the pre-optimization step
        rr.log(f"{prefix}camera_{k:03d}", rr.ViewCoordinates.RDF, static=True)
        rr.log(
            f"{prefix}camera_{k:03d}/image",
            rr.Pinhole(
                resolution=[K[0][2] * 2, K[1][2] * 2],
                focal_length=[K[0][0], K[1][1]],
                principal_point=[K[0][2], K[1][2]],
                image_plane_distance=0.1,
            ),
        )

        # LOG IMAGES
        if log_images:
            view = rec.scenegraph.views[view_idx]
            rr.log(f"{prefix}camera_{k:03d}/image", rr.Image(view.img))
