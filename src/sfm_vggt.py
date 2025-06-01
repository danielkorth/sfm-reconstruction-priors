# https://github.com/facebookresearch/vggt/issues/39
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import lightning as L
import omegaconf
import roma
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb
from data.colmap import read_cameras_text
from sfm.view import View
from utils.camera import undistort_fisheye_intrinsics
from utils.rerun import visualize_final_reconstruction
from utils.scene import get_camera_centers


def calculate_average_metrics(metrics_list):
    """Calculate the average values for each metric across a list of metric dictionaries.

    Args:
        metrics_list: List of dictionaries containing metric values

    Returns:
        Dictionary with the average value for each metric
    """
    if not metrics_list:
        return {}

    # Initialize a dictionary to store the sum of each metric
    sum_metrics = {}

    # Sum up all values for each metric
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key in sum_metrics:
                sum_metrics[key] += value
            else:
                sum_metrics[key] = value

    # Calculate the average for each metric
    avg_metrics = {key: value / len(metrics_list) for key, value in sum_metrics.items()}

    return avg_metrics


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

from data.scannet_scene import ScannetppScene
from sfm.reconstruction import Reconstruction
from sfm.scenegraph import SceneGraph
from utils import extras
from utils.visualization import create_image_grid


def is_debugging():
    """Check if running in debug mode."""
    import sys

    print("in debugging mode")
    is_debugging = hasattr(sys, "gettrace") and sys.gettrace() is not None
    print(f"is_debugging: {is_debugging}")
    return is_debugging


def optimize_scene(cfg: DictConfig) -> Dict[str, Any]:
    """Process a single scene with the given configuration.

    Args:
        cfg: A DictConfig configuration composed by Hydra.

    Returns:
        A dictionary with metrics from the optimization.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    print("Loading Scene...")
    print(cfg.data.get("scene_id"))
    print(cfg.data.get("scannet_path"))

    scene = ScannetppScene(str(cfg.data.get("scene_id")), cfg.data.get("scannet_path"))
    # select images to use of the scene randomly
    image_paths = list(
        scene.dslr_undistorted_images_dir.glob("*.JPG")
    )  # Adjust the pattern as needed

    # Sample a subset of the images
    n_images = sum(1 for _ in image_paths)
    if cfg.data.get("num_images") > 0:
        n_images = min(
            cfg.data.get("num_images"), n_images
        )  # Ensure we don't exceed available images
    else:
        n_images = int(n_images * cfg.data.get("pct_images") * 0.01)

    # use random.shuffle! bugs.python.org/issue33114
    random.shuffle(image_paths)
    selected_images = image_paths[:n_images]
    selected_images = [Path(img.name) for img in selected_images]

    selected_images_paths = [str(scene.dslr_undistorted_images_dir / x) for x in selected_images]

    model = hydra.utils.instantiate(cfg.model)
    extrinsics, _ = model.predict_camera_extrinsics(selected_images_paths)

    # extrinsics is world-to-camera format
    extrinsics = extrinsics.cpu().squeeze()

    # Extract world-to-camera rotations and camera centers
    rotmats_w2c = extrinsics[:, :3, :3].to(torch.float64)
    camera_centers_vggt = -torch.bmm(
        rotmats_w2c.transpose(1, 2), extrinsics[:, :3, 3:].to(torch.float64)
    ).squeeze(-1)

    image_data_scannet, camera_centers = get_camera_centers(scene)
    camera_centers_gt = torch.tensor(
        [camera_centers[x.name] for x in selected_images], dtype=torch.float64
    )

    R, t, s = roma.rigid_points_registration(
        camera_centers_vggt, camera_centers_gt, weights=None, compute_scaling=True
    )

    # Transform camera centers
    new_camera_centers = torch.stack([s * R @ x + t for x in camera_centers_vggt], dim=0)

    # Transform rotations (world-to-camera)
    updated_rots = torch.stack([x[:3, :3] @ R.T for x in rotmats_w2c], dim=0)

    # Compute new translations (world-to-camera)
    new_translations = torch.bmm(updated_rots, -new_camera_centers.unsqueeze(-1)).squeeze(-1)

    camera_data_scannet = read_cameras_text(scene.dslr_colmap_dir / "cameras.txt")
    K = undistort_fisheye_intrinsics(camera_data_scannet[1])

    views = []
    for i in range(len(selected_images)):
        view = View(Path(selected_images_paths[i]), process_image=True)
        view.add_camera(
            kind="gt", K=K, R_t=image_data_scannet[selected_images[i].name].world_to_camera
        )
        view.add_camera(
            kind="opt",
            K=K,
            R_t=torch.hstack((updated_rots[i], new_translations[i].reshape(-1, 1))),
        )
        views.append(view)

    scenegraph = SceneGraph(
        views=views,
        pairs_mat=None,
        tracks=None,
        feature_to_track=None,
        view_to_tracks=None,
        cfg=None,
        scene=scene,
    )

    rec = Reconstruction(
        active_view_idxs=list(range(len(views))),
        active_track_idxs=[],
        scenegraph=scenegraph,
        cfg=None,
    )

    if cfg.rerun_log_final_reconstruction:
        visualize_final_reconstruction(rec)

    # save final params
    metrics = rec.calculate_metrics()
    metrics_file_path = Path(cfg.paths.output_dir) / "metrics.txt"
    with open(metrics_file_path, "w") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
    print(f"Metrics saved to {metrics_file_path}")

    return metrics


def log_metrics_to_wandb(metrics, to_summary=True):
    """Log metrics to wandb.

    Args:
        metrics: Dictionary containing metrics to log.
    """
    # does not work
    for key in wandb.run.summary.keys():
        del wandb.run.summary[key]

    if to_summary:
        wandb.run.summary["_ATE"] = metrics.get("ate", 0)
        wandb.run.summary["_RRA@5"] = metrics.get("rra5", 0)
        wandb.run.summary["_RTA@5"] = metrics.get("rta5", 0)
        wandb.run.summary["_RRA@15"] = metrics.get("rra15", 0)
        wandb.run.summary["_RTA@15"] = metrics.get("rta15", 0)
        wandb.run.summary["_RAUC"] = metrics.get("rotation_auc", 0)
        wandb.run.summary["_TAUC"] = metrics.get("translation_auc", 0)
        wandb.run.summary["_mAA(30)"] = metrics.get("combined_maa", 0)
        wandb.run.summary["_#Points"] = metrics.get("n_tracks", 0)
        wandb.run.summary["_#Images_Registered"] = metrics.get("n_images", 0)
        wandb.run.summary["_Registration_Rate"] = metrics.get("registration_rate", 0)

        # Overall fitness score (used for sweeps)
        if "combined_maa" in metrics and "registration_rate" in metrics:
            wandb.run.summary["_Sweep_Score"] = (
                metrics["combined_maa"] * metrics["registration_rate"]
            )
    else:
        wandb.log(metrics)


@hydra.main(version_base="1.3", config_path="../configs", config_name="sfm_vggt.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for SfM processing.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Optional[float] with optimized metric value.
    """
    # Hacky way to rerun a specific run from a multirun directory (and use the exactly same config)
    if cfg.get("viz_run", False):
        cfg = OmegaConf.load(Path(cfg.viz_run) / ".hydra/config.yaml")
        cfg.rerun_log_rec = True

    # Apply extra utilities
    extras(cfg)

    # Initialize wandb if logger is configured
    if cfg.get("logger") is not None:
        wandb.init(
            **cfg.logger, config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

    try:
        # Check if we're processing multiple scenes or a single scene
        is_multi_scene = (
            isinstance(cfg.data.scene_id, omegaconf.listconfig.ListConfig)
            and len(cfg.data.scene_id) > 1
        ) or (
            isinstance(cfg.data.num_images, omegaconf.listconfig.ListConfig)
            and len(cfg.data.num_images) > 1
        )

        if is_multi_scene:
            # Process multiple scenes
            scene_ids = cfg.data.scene_id
            num_images = cfg.data.num_images

            if not isinstance(scene_ids, omegaconf.listconfig.ListConfig):
                scene_ids = [scene_ids]
            if not isinstance(num_images, omegaconf.listconfig.ListConfig):
                num_images = [num_images]

            output_dir = cfg.paths.output_dir
            metrics = []
            wandb_table_data = []

            for scene_id in scene_ids:
                for num_image in num_images:
                    with open_dict(cfg):
                        cfg.data.scene_id = scene_id
                        cfg.data.num_images = num_image
                        cfg.paths.output_dir = f"{output_dir}/{scene_id}/{num_image}"

                    output_dir_path = Path(cfg.paths.output_dir)
                    if not output_dir_path.exists():
                        output_dir_path.mkdir(parents=True)

                    # Optimize the scene
                    metric_curr_run = optimize_scene(cfg)
                    metrics.append(metric_curr_run)
                    wandb_table_data.append([f"{scene_id}/{num_image}", *metric_curr_run.values()])
                    if wandb.run is not None:
                        log_metrics_to_wandb(metric_curr_run, to_summary=False)

            # Create a summary table
            import pandas as pd

            if metrics:
                keyz = metrics[0].keys()
                df = pd.DataFrame(wandb_table_data, columns=["Scene/Images", *keyz])
                df.to_csv(Path(output_dir) / "metrics_table.csv", index=False)

            # Average the metrics
            metrics_avg = calculate_average_metrics(metrics)

            # Log average metrics to wandb
            if wandb.run is not None:
                log_metrics_to_wandb(metrics_avg)

            # Write average metrics to a file
            with open(Path(output_dir) / "metrics_all.txt", "w") as file:
                for key, value in metrics_avg.items():
                    file.write(f"{key}: {value}\n")

            return 0  # Success
        else:
            # Process a single scene
            metrics = optimize_scene(cfg)

            # Log metrics to wandb
            if wandb.run is not None:
                log_metrics_to_wandb(metrics)

            return 0  # Success
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        return 1  # Error
    finally:
        # Restore the original stdout
        sys.stdout = sys.__stdout__
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
