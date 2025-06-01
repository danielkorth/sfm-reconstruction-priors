"""
Goal: integrate/map the output of COLMAP into my own reconstruction pipeline.
This serves both as a baseline and a means to debug my own pipeline by comparing it to what COLMAP produces in the end.
"""
import os
import sqlite3
import struct
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pycolmap

from data.colmap import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_points3D_binary,
)
from data.scannet_scene import ScannetppScene
from opt.alignment_solver import procrustes
from sfm.reconstruction import Reconstruction
from sfm.scenegraph import SceneGraph
from sfm.track import Track, TrackState
from sfm.view import View
from utils.camera import undistort_fisheye_intrinsics


def map_colmap_into_reconstruction(
    output_path: Path, scene: ScannetppScene, add_tracks=True, use_colmap_aligner=True
):
    """
    1. load in COLMAP output
    2. Create SceneGraph
    3. Create Reconstruction
    4. Profit (can visualize & evaluate metrics similar to my approaches)

    Following a minimalistic approach. I only add the data to the SceneGraph/Reconstruction that is deemed necessary (ie no pairwise relationships, tracks only consist of a 3d point)
    somewhat hacky
    """

    # align the coordinates of the colmap output with the scannetpp coordinates
    images = read_images_text(scene.dslr_colmap_dir / "images.txt")
    if use_colmap_aligner:
        # COLMAP ALIGNER
        # Check if the images_aligner.txt file already exists
        aligner_file_path = scene.dslr_colmap_dir / "images_aligner.txt"
        if not os.path.exists(aligner_file_path):
            # write the image name and camera center to a file
            with open(aligner_file_path, "w") as f:
                for image in images.values():
                    camera_center = image.world_to_camera[:3, :3].T @ -image.world_to_camera[:3, 3]
                    f.write(
                        f"{image.name} {camera_center[0]} {camera_center[1]} {camera_center[2]}\n"
                    )

        if not (output_path / "aligned").exists():
            os.mkdir(output_path / "aligned")

        # run colmap aligner
        subprocess.run(
            f"colmap model_aligner \
                        --input_path {output_path / '0'} \
                        --output_path {output_path / 'aligned'} \
                        --alignment_max_error 3.0 \
                        --ref_images_path {scene.dslr_colmap_dir / 'images_aligner.txt'} \
                        --ref_is_gps 0",
            shell=True,
            check=True,
        )

        # calculated by myself
        image_data = read_images_binary(output_path / "aligned" / "images.bin")
        image_data = {v.name: v for _, v in image_data.items()}
        camera_data = read_cameras_binary(output_path / "aligned" / "cameras.bin")
        points_3d = read_points3D_binary(output_path / "aligned" / "points3D.bin")

        # get all initial images - not only the ones that were used for the reconstruction / registered
        db = pycolmap.Database()
        db.open(output_path / "database.db")
        cameras = db.read_all_images()

        print(image_data.keys())

        # scannetpp data
        camera_data_scannet = read_cameras_text(scene.dslr_colmap_dir / "cameras.txt")
        image_data_scannet = read_images_text(scene.dslr_colmap_dir / "images.txt")
        image_data_scannet = {v.name: v for _, v in image_data_scannet.items()}

        print(image_data_scannet.keys())
        K = undistort_fisheye_intrinsics(camera_data_scannet[1])

    else:
        # https://github.com/colmap/colmap/issues/1507
        # -> COLMAP creates multiple folders. we need to search for the one with the biggest connected component
        best_folder = None
        best_num_images = 0
        for file in output_path.iterdir():
            if file.is_dir() and file.name.isdigit():
                image_data = read_images_binary(file / "images.bin")
                if len(image_data) > best_num_images:
                    best_num_images = len(image_data)
                    best_folder = file

        image_data = read_images_binary(best_folder / "images.bin")
        image_data = {v.name: v for _, v in image_data.items()}
        image_data_scannet = read_images_text(scene.dslr_colmap_dir / "images.txt")
        image_data_scannet = {v.name: v for _, v in image_data_scannet.items()}

        # collect camera centers
        X, Y = [], []
        for image_name, data in image_data.items():
            X.append(data.world_to_camera[:3, :3].T @ -data.world_to_camera[:3, 3])
            Y.append(
                image_data_scannet[image_name].world_to_camera[:3, :3].T
                @ -image_data_scannet[image_name].world_to_camera[:3, 3]
            )

        X = np.array(X)
        Y = np.array(Y)

        s, R, t = procrustes(X, Y, verbose=True, scaling=True)

        poses = [image_data[image_name].world_to_camera for image_name in image_data.keys()]

        # For world-to-camera poses:
        # The rotation part needs to be transformed as R_new = R_old @ R.T
        # The translation part needs to be transformed as c_new = s * R @ c + t
        new_rotations = [x[:3, :3] @ R.T for x in poses]
        camera_centers = [-x[:3, :3].T @ x[:3, 3] for x in poses]  # Extract camera centers
        new_camera_centers = [s * R @ c + t for c in camera_centers]  # Transform camera centers

        # Update the image data with new aligned poses
        for idx, image_name in enumerate(image_data.keys()):
            # Construct new world_to_camera matrix using the aligned rotation and translation
            new_world_to_camera = np.eye(4)
            new_world_to_camera[:3, :3] = new_rotations[idx]
            # Convert camera center back to translation vector
            new_world_to_camera[:3, 3] = -new_rotations[idx] @ new_camera_centers[idx]

            # Update the image data
            image_data[image_name].world_to_camera_aligned = new_world_to_camera

        # get all initial images - not only the ones that were used for the reconstruction / registered
        db = pycolmap.Database()
        db.open(output_path / "database.db")
        cameras = db.read_all_images()

        # scannetpp data
        camera_data_scannet = read_cameras_text(scene.dslr_colmap_dir / "cameras.txt")
        K = undistort_fisheye_intrinsics(camera_data_scannet[1])

    # create all views
    views = []
    active_view_idxs = []
    idx = 0
    for cam in cameras:
        view = View(scene.dslr_undistorted_images_dir / cam.name, process_image=True)
        # gt camera
        view.add_camera(kind="gt", K=K, R_t=image_data_scannet[cam.name].world_to_camera)

        if cam.name in image_data.keys():
            if use_colmap_aligner:
                R_colmap = image_data[cam.name].world_to_camera[:3, :3]
                t_colmap = image_data[cam.name].world_to_camera[:3, 3]
            else:
                R_colmap = image_data[cam.name].world_to_camera_aligned[:3, :3]
                t_colmap = image_data[cam.name].world_to_camera_aligned[:3, 3]

            # Update the view with aligned camera parameters
            view.add_camera(kind="opt", K=K, R_t=np.hstack((R_colmap, t_colmap.reshape(-1, 1))))
            active_view_idxs.append(idx)

        idx += 1
        views.append(view)

        # this part is correct, the translation I wasn't able to figure out yet
        # R_rel_local = R_ref @ R_colmap.T # map from currenct rotation to the reference rotation in COLMAP space
        # R_rel = R_rel_local.T @ R_transform @ R_rel_local # full relative rotation
        # R_aligned = R_rel @ R_colmap

    # add all tracks
    if add_tracks:
        tracks = []
        for point in points_3d.values():
            # Apply the global transformation to the point
            track = Track()
            track.point = point.xyz  # Use the aligned point
            track.state = TrackState.ACTIVE
            tracks.append(track)

    scenegraph = SceneGraph(
        views=views,
        pairs_mat=None,
        tracks=tracks if add_tracks else None,
        feature_to_track=None,
        view_to_tracks=None,
        cfg=None,
        scene=scene,
    )
    active_tracks = list(range(len(tracks))) if add_tracks else []
    reconstruction = Reconstruction(
        active_view_idxs=active_view_idxs,
        active_track_idxs=active_tracks,
        scenegraph=scenegraph,
        cfg=None,
    )
    return reconstruction


def run_colmap_reconstruction(
    image_dir: Path,
    output_path: Path,
    image_list: List[str] = [],
    camera_params=(621.9321, 622.0554, 876.0, 584.0),
    mask_path=None,
):
    """
    Run the complete COLMAP reconstruction pipeline:
    1. Extract features
    2. Fix intrinsics in the database
    3. Match features
    4. Run reconstruction
    5. Convert model to text format

    Args:
        image_dir (Path): Directory containing input images
        image_list (List[str]): List of images to use
        output_path (Path): Directory for COLMAP output
        camera_params (tuple): Camera parameters (fx, fy, cx, cy) for PINHOLE model

    Returns:
        Path: Path to the output directory containing the reconstruction
    """
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    database_path = output_path / "database.db"

    # Write image list to a file
    image_list_path = output_path / "image_list.txt"
    with open(image_list_path, "w") as f:
        for img in image_list:
            f.write(f"{img}\n")

    if mask_path is not None:
        reader_options = pycolmap.ImageReaderOptions(mask_path=mask_path)
        # rewrite all images to the correct format
        for img_name in mask_path.iterdir():
            # Only rename if it doesn't already have .JPG.png extension
            if not img_name.name.endswith(".JPG.png"):
                # Add .JPG before .png instead of replacing .png
                img_name_new = img_name.parent / (img_name.stem + ".JPG.png")
                img_name.rename(img_name_new)
    else:
        reader_options = pycolmap.ImageReaderOptions()

    # Extract features
    pycolmap.extract_features(
        database_path,
        image_dir,
        image_list=image_list,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="PINHOLE",
        reader_options=reader_options,
    )


    # Fix intrinsics in database
    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        camera_id = 1
        # Convert parameters to binary format (float64)
        new_params = struct.pack("dddd", *camera_params)
        cursor.execute(
            """
            UPDATE cameras
            SET params = ?
            WHERE camera_id = ?
        """,
            (new_params, camera_id),
        )
        conn.commit()

    # Verify intrinsics (optional)
    db = pycolmap.Database()
    db.open(database_path)
    camera = db.read_all_cameras()

    # Match features
    pycolmap.match_exhaustive(database_path)

    # Run reconstruction and convert to text format
    commands = [
        f"colmap mapper --database_path {database_path} --image_path {image_dir} "
        f"--output_path {output_path} --Mapper.ba_refine_focal_length 0 "
        f"--Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0",
        # f"colmap model_converter --input_path {output_path} "
        # f"--output_path {output_path} --output_type TXT"
    ]

    for command in commands:
        subprocess.run(command, shell=True, check=True)

    return output_path
