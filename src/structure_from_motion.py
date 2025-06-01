import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import lightning as L
import omegaconf
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb
from sfm.pointmap import MultiviewPointmap, TwoViewPointmapWrapper
from utils.rerun import visualize_final_reconstruction


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

from data.scannet_scene import ScannetppScene
from sfm.reconstruction import Reconstruction
from sfm.scenegraph import SceneGraph
from utils import RankedLogger, extras, task_wrapper
from utils.visualization import create_image_grid

# Force deterministic operations
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use double precision if accuracy is critical
torch.set_default_dtype(torch.float32)
# Set CUBLAS workspace configuration for deterministic behavior
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

log = RankedLogger(__name__, rank_zero_only=True)


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

    scene_start_time = time.time()

    # Redirect print statements to a log file
    log_file = PrintToFile(
        cfg.paths.get("output_dir") + "/output.log"
    )  # Specify your output log file
    sys.stdout = log_file  # Redirect stdout to the log file

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

    if "model" in cfg:
        # extract pointmaps
        model = hydra.utils.instantiate(cfg.model)

        print("Extracting pointmaps...")
        if cfg.pointmap_type == "two_view":
            pointmaps, resolution = model.extract_pointmaps(selected_images_paths)
            pointmaps = TwoViewPointmapWrapper.from_dict(pointmaps)
        else:
            data, resolution = model.extract_global_pointmaps(selected_images_paths)
            pointmaps = MultiviewPointmap.from_dict(data)

        with open_dict(cfg):
            if 224 in resolution:
                cfg.sfm.reconstruction.global_optimization.dust3r_output_size = (224, 224)
            else:
                cfg.sfm.reconstruction.global_optimization.dust3r_output_size = (
                    int(resolution[0]),
                    int(resolution[1]),
                )
    else:
        pointmaps = None

    print("Building SceneGraph...")
    # feature matching
    graph = SceneGraph.build_from_images_colmap(
        selected_images,
        scene=scene,
        cfg=cfg.sfm.feature_matching,
        use_precomputed_matches=cfg.use_precomputed_matches,
        pointmaps=pointmaps,
    )
    if cfg.save_matches_and_images:
        (Path(cfg.paths.output_dir) / "matches").mkdir(parents=True, exist_ok=True)
        graph.save_verified_matches_png(Path(cfg.paths.output_dir) / "matches")
        (Path(cfg.paths.output_dir) / "images").mkdir(parents=True, exist_ok=True)
        graph.save_images(Path(cfg.paths.output_dir) / "images")
        create_image_grid(
            Path(cfg.paths.output_dir) / "images",
            Path(cfg.paths.output_dir) / "images" / "images_grid.png",
        )

        if wandb.run is not None:
            for img in (Path(cfg.paths.output_dir) / "images").glob("*JPG.png"):
                wandb.log({"images": wandb.Image(str(img))})

            wandb.log(
                {
                    "all_images": wandb.Image(
                        str(Path(cfg.paths.output_dir) / "images" / "images_grid.png")
                    )
                }
            )

            for img in (Path(cfg.paths.output_dir) / "matches").glob("*.png"):
                wandb.log({"matches": wandb.Image(str(img))})

    # Pass True to write statistics to a file and specify the file path
    graph.compute_statistics(
        write_to_file=True, file_path=f"{cfg.paths.output_dir}/scenegraph_statistics.txt"
    )

    print("Starting Reconstruction...")
    rec = Reconstruction.init_from_twoview(graph, cfg=cfg.sfm.reconstruction)
    if cfg.get("rerun_log_rec"):
        print("Connecting to rerun")
        rec._connect_rerun(
            name=f"{Path(cfg.paths.get('output_dir')).parent.name}-{Path(cfg.paths.get('output_dir')).name}-mine"
        )

    try:
        info = rec.incremental_reconstruction()
    except Exception as e:
        import traceback

        exception_file = Path(cfg.paths.output_dir) / "exception.txt"
        with open(exception_file, "w") as f:
            traceback.print_exc(file=f)

    # save metrics
    metrics = rec.calculate_metrics()

    # Add timing information to metrics
    scene_end_time = time.time()
    metrics["scene_execution_time"] = scene_end_time - scene_start_time

    metrics_file_path = Path(cfg.paths.output_dir) / "metrics.txt"
    with open(metrics_file_path, "w") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
    print(f"Metrics saved to {metrics_file_path}")

    if cfg.rerun_log_final_reconstruction:
        visualize_final_reconstruction(rec, name=str(scene.scene_id))

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


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="sfm.yaml" if not is_debugging() else "sfm_debug.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for SfM processing.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Optional[float] with optimized metric value.
    """
    script_start_time = time.time()

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

            original_cfg = cfg.copy()

            for scene_id in scene_ids:
                for num_image in num_images:
                    cfg = original_cfg.copy()
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

            # Log total execution time
            script_end_time = time.time()
            total_time = script_end_time - script_start_time
            print(f"\nTotal execution time: {total_time:.2f} seconds")
            if wandb.run is not None:
                wandb.run.summary["total_execution_time"] = total_time

            return 0  # Success
        else:
            # Process a single scene
            metrics = optimize_scene(cfg)

            # Log metrics to wandb
            if wandb.run is not None:
                log_metrics_to_wandb(metrics)

            # Log total execution time
            script_end_time = time.time()
            total_time = script_end_time - script_start_time
            print(f"\nTotal execution time: {total_time:.2f} seconds")
            if wandb.run is not None:
                wandb.run.summary["total_execution_time"] = total_time

            return 0  # Success
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
        return 1  # Error
    finally:
        # Restore the original stdout
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
