import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import lightning as L
import numpy as np
import rootutils
from omegaconf import DictConfig, open_dict


# Custom stream class to redirect print statements to a file
class PrintToFile:
    def __init__(self, file_path):
        self.file = open(file_path, "a")  # Open the file in append mode

    def write(self, message):
        self.file.write(message)  # Write the message to the file
        self.file.flush()  # Ensure the message is written immediately

    def flush(self):
        pass  # This is needed for compatibility with the print function


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from concurrent.futures import ThreadPoolExecutor

from data.scannet_scene import ScannetppScene
from sfm.pair import Pair
from utils import extras


def precompute_stuff(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    print("Loading Scene...")
    print(cfg.data.get("scene_id"))
    print(cfg.data.get("scannet_path"))

    scene = ScannetppScene(str(cfg.data.get("scene_id")), cfg.data.get("scannet_path"))
    image_paths = list(
        scene.dslr_undistorted_images_dir.glob("*.JPG")
    )  # Adjust the pattern as needed

    # Sample a subset of the images
    n_images = sum(1 for _ in image_paths)
    if cfg.data.num_images > 0:
        n_images = min(cfg.data.num_images, n_images)  # Ensure we don't exceed available images
    else:
        n_images = int(n_images * cfg.data.get("pct_images") * 0.01)

    # use random.shuffle! bugs.python.org/issue33114
    random.shuffle(image_paths)
    selected_images = image_paths[:n_images]
    selected_images = [Path(img.name) for img in selected_images]

    ### --- SIFT --- ###
    if cfg.sfm.feature_matching.name == "sift":
        # load data
        from data.colmap import read_cameras_text, read_images_text
        from sfm.view import View
        from utils.camera import undistort_fisheye_intrinsics

        camera_data = read_cameras_text(scene.dslr_colmap_dir / "cameras.txt")
        image_data = read_images_text(scene.dslr_colmap_dir / "images.txt")
        image_data = {v.name: v for _, v in image_data.items()}

        # undistort intrinsics
        K = undistort_fisheye_intrinsics(camera_data[1])

        # load images and extract features
        views = []
        for path in selected_images:
            view = View(
                scene.dslr_undistorted_images_dir / path,
                parameterization=cfg.sfm.feature_matching.view.parameterization,
                descriptor=cfg.sfm.feature_matching.view.descriptor,
                mask_path=scene.dslr_undistorted_anon_masks_dir / (path.name + ".png")
                if cfg.sfm.feature_matching.view.mask
                else None,
                mask_blur_iterations=cfg.sfm.feature_matching.view.mask_blur_iterations,
            )
            view.add_camera(kind="gt", K=K, R_t=image_data[path.name].world_to_camera)
            views.append(view)

        # write that shit to disk
        (scene.dslr_dir / "keypoints").mkdir(parents=True, exist_ok=True)
        for view in views:
            view._serialize_keypoints(
                scene.dslr_dir / "keypoints" / "sift" / (view.img_path.name + ".npy")
            )

        # --- precompute the matches type shiit ---
        # copied over from scenegraph.py
        # # Initialize pairs matrix
        n_views = len(views)
        pairs_mat = [[None for _ in range(n_views)] for _ in range(n_views)]

        def run_parallel():
            pairs_mat = [[None for _ in range(n_views)] for _ in range(n_views)]

            def process_pair(i, j):
                pair = Pair(views[j], views[i])
                pair.find_matches(
                    ratio=cfg.sfm.feature_matching.pair.sift_lowe_ratio,
                    max_distance=cfg.sfm.feature_matching.pair.sift_max_distance,
                    distance_metric=cfg.sfm.feature_matching.pair.sift_distance_metric,
                )
                return (j, i, pair)

            pairs_to_process = [(i, j) for i in range(1, n_views) for j in range(i)]
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda p: process_pair(*p), pairs_to_process))

            for j, i, pair in results:
                if pair is not None:
                    pairs_mat[j][i] = pair

            return pairs_mat

        pairs_mat = run_parallel()
        (scene.dslr_dir / "matches").mkdir(parents=True, exist_ok=True)
        for i in range(n_views):
            for j in range(i):
                if pair := pairs_mat[j][i]:
                    pair._serialize_matches(
                        scene.dslr_dir
                        / "matches"
                        / "sift"
                        / (pair.v1.img_path.name + "_" + pair.v2.img_path.name + ".npy")
                    )

    ### --- SuperGlue --- ###
    elif cfg.sfm.feature_matching.name == "spsg":
        matcher = hydra.utils.instantiate(cfg.sfm.feature_matching.superglue_wrapper)

        # create keypoints directory
        (scene.dslr_dir / "keypoints").mkdir(parents=True, exist_ok=True)
        (scene.dslr_dir / "keypoints" / "spsg").mkdir(parents=True, exist_ok=True)
        (scene.dslr_dir / "matches" / "spsg").mkdir(parents=True, exist_ok=True)

        for i in range(len(selected_images)):
            # i images are the diagonal
            for j in range(i):
                img0_path = scene.dslr_undistorted_images_dir / selected_images[j]
                img1_path = scene.dslr_undistorted_images_dir / selected_images[i]
                output = matcher.match_image_pair(
                    str(img0_path),
                    str(img1_path),
                    resize=cfg.sfm.feature_matching.resize,
                    resize_float=cfg.sfm.feature_matching.resize_float,
                )

                # write keypoints to disk if they don't exist yet
                kp_path = (
                    scene.dslr_dir / "keypoints" / "spsg" / (str(selected_images[j]) + ".npy")
                )
                if not kp_path.exists() or cfg.sfm.feature_matching.overwrite:
                    matcher._serialize_keypoints(output["keypoints0"], kp_path)

                kp_path = (
                    scene.dslr_dir / "keypoints" / "spsg" / (str(selected_images[i]) + ".npy")
                )
                if not kp_path.exists() or cfg.sfm.feature_matching.overwrite:
                    matcher._serialize_keypoints(output["keypoints1"], kp_path)

                # write matches to disk if they don't exist yet
                match_path = (
                    scene.dslr_dir
                    / "matches"
                    / "spsg"
                    / (str(selected_images[j]) + "_" + str(selected_images[i]) + ".npy")
                )
                if not match_path.exists() or cfg.sfm.feature_matching.overwrite:
                    matcher._serialize_matches(output["matches"], match_path)

    ### --- MASt3R --- ###
    elif cfg.sfm.feature_matching.name == "mast3r":
        mast3r = hydra.utils.instantiate(cfg.sfm.feature_matching.mast3r_wrapper)

        selected_images = [scene.dslr_undistorted_images_dir / img for img in selected_images]

        # create keypoints directory
        (scene.dslr_dir / "keypoints").mkdir(parents=True, exist_ok=True)
        (scene.dslr_dir / "keypoints" / "mast3r").mkdir(parents=True, exist_ok=True)
        (scene.dslr_dir / "matches" / "mast3r").mkdir(parents=True, exist_ok=True)

        output, res = mast3r.match_image_pairs(selected_images)

        # Create a mapping from 2D indices to 1D indices
        x_coords, y_coords = np.meshgrid(np.arange(res[1]), np.arange(res[0]))
        indices_2d = np.stack([x_coords, y_coords], axis=-1)
        indices_2d = indices_2d.reshape(-1, 2)

        # Create a dictionary that maps from 2D coordinates to 1D index
        index_map = {}
        for i, (x, y) in enumerate(indices_2d):
            index_map[(x, y)] = i

        # Also keep the original keypoints format for compatibility
        keypoints_dict = {
            "x": indices_2d[:, 0],
            "y": indices_2d[:, 1],
        }

        # we want to collect all keypoints (px locations) with a match
        view_to_kp_idxs = defaultdict(list)
        matches = []
        for pair in output:
            if pair[0] < pair[1]:
                # write matches to disk if they don't exist yet
                match_path = (
                    scene.dslr_dir
                    / "matches"
                    / "mast3r"
                    / (
                        str(selected_images[pair[0]].name)
                        + "_"
                        + str(selected_images[pair[1]].name)
                        + ".npy"
                    )
                )

                if not match_path.exists() or cfg.sfm.feature_matching.overwrite:
                    matches_dict = {"query_idx": [], "database_idx": []}
                    for kp0, kp1 in zip(output[pair]["keypoints0"], output[pair]["keypoints1"]):
                        matches_dict["query_idx"].append(index_map[(kp0[0], kp0[1])])
                        matches_dict["database_idx"].append(index_map[(kp1[0], kp1[1])])

                    matches.append((pair, matches_dict))

                # collect keypoints
                view_to_kp_idxs[pair[0]].append(matches_dict["query_idx"])
                view_to_kp_idxs[pair[1]].append(matches_dict["database_idx"])

        # save the keypoints
        new_kp_idx_mapping = []
        for view, kp_idxs in view_to_kp_idxs.items():
            kp_idxs = np.concatenate(kp_idxs).astype(np.int32).tolist()
            new_mapping = {x: i for i, x in enumerate(kp_idxs)}
            new_kp_idx_mapping.append(new_mapping)

            # write keypoints to disk
            path = scene.dslr_dir / "keypoints" / "mast3r" / (selected_images[view].name + ".npy")
            keypoints_dict = {
                "x": indices_2d[kp_idxs][:, 0],
                "y": indices_2d[kp_idxs][:, 1],
            }
            np.save(path, keypoints_dict, allow_pickle=True)

        # reread all pairwise and update the idxs
        for pair, matches_dict in matches:
            match_path = (
                scene.dslr_dir
                / "matches"
                / "mast3r"
                / (
                    str(selected_images[pair[0]].name)
                    + "_"
                    + str(selected_images[pair[1]].name)
                    + ".npy"
                )
            )

            matches_dict["query_idx"] = [
                new_kp_idx_mapping[pair[0]][x] for x in matches_dict["query_idx"]
            ]
            matches_dict["database_idx"] = [
                new_kp_idx_mapping[pair[1]][x] for x in matches_dict["database_idx"]
            ]

            np.save(match_path, matches_dict, allow_pickle=True)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="sfm.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    scene_ids = cfg.data.scene_id
    from tqdm import tqdm

    if isinstance(scene_ids, str):
        scene_ids = [scene_ids]

    for scene_id in tqdm(scene_ids):
        with open_dict(cfg):
            cfg.data.scene_id = scene_id
            # optimize the scene
            precompute_stuff(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
