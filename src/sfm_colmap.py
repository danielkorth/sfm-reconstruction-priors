import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import omegaconf
import pandas as pd
import rootutils
from omegaconf import DictConfig, open_dict

from data.colmap import read_cameras_text
from data.scannet_scene import ScannetppScene
from sfm.colmap_integration import (
    map_colmap_into_reconstruction,
    run_colmap_reconstruction,
)
from utils import RankedLogger, extras, task_wrapper
from utils.camera import undistort_fisheye_intrinsics
from utils.rerun import visualize_final_reconstruction
from utils.visualization import create_image_grid


# Custom stream class to redirect print statements to a file
class PrintToFile:
    def __init__(self, file_path):
        self.file = open(file_path, "a")  # Open the file in append mode

    def write(self, message):
        self.file.write(message)  # Write the message to the file
        self.file.flush()  # Ensure the message is written immediately

    def flush(self):
        pass  # This is needed for compatibility with the print function


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


log = RankedLogger(__name__, rank_zero_only=True)


def optimize(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    # Redirect print statements to a log file
    log_file = PrintToFile(
        cfg.paths.get("output_dir") + "/output.log"
    )  # Specify your output log file
    sys.stdout = log_file  # Redirect stdout to the log file

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    print("Loading Scene...")
    print(cfg.data.get("scene_id"))
    print(cfg.data.get("scannet_path"))

    scene = ScannetppScene(cfg.data.get("scene_id"), cfg.data.get("scannet_path"))
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

    # load data
    camera_data = read_cameras_text(scene.dslr_colmap_dir / "cameras.txt")

    # undistort intrinsics
    K = undistort_fisheye_intrinsics(camera_data[1])
    camera_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    run_colmap_reconstruction(
        image_dir=scene.dslr_undistorted_images_dir,
        output_path=Path(cfg.paths.output_dir),  # where to put the database
        image_list=selected_images,
        camera_params=camera_params,
        mask_path=scene.dslr_undistorted_anon_masks_dir
        if cfg.use_masks
        else None,  # TODO not the best way to do this
    )
    rec = map_colmap_into_reconstruction(
        output_path=Path(cfg.paths.output_dir),
        scene=scene,
        add_tracks=cfg.add_tracks,
        use_colmap_aligner=cfg.use_colmap_aligner,
    )

    if cfg.rerun_log_final_reconstruction:
        visualize_final_reconstruction(rec)

    # save metrics
    metrics = rec.calculate_metrics()
    metrics_file_path = Path(cfg.paths.output_dir) / "metrics.txt"
    with open(metrics_file_path, "w") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
    print(f"Metrics saved to {metrics_file_path}")
    return metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="sfm_colmap.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    script_start_time = time.time()

    # apply extra utilities
    extras(cfg)

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
                metric_curr_run = optimize(cfg)
                metrics.append(metric_curr_run)

        # Create a summary table
        if metrics:
            keyz = metrics[0].keys()
            df = pd.DataFrame(wandb_table_data, columns=["Scene/Images", *keyz])
            df.to_csv(Path(output_dir) / "metrics_table.csv", index=False)

        # Average the metrics
        metrics_avg = calculate_average_metrics(metrics)

        # Write average metrics to a file
        with open(Path(output_dir) / "metrics_all.txt", "w") as file:
            for key, value in metrics_avg.items():
                file.write(f"{key}: {value}\n")

        # Log total execution time
        script_end_time = time.time()
        total_time = script_end_time - script_start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        return 0  # Success
    else:
        # Process a single scene
        metrics = optimize(cfg)

        # Log total execution time
        script_end_time = time.time()
        total_time = script_end_time - script_start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        return 0  # Success


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)

# Restore the original stdout
sys.stdout = sys.__stdout__  # Restore original stdout
