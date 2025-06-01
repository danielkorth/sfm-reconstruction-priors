"""Preprocess the masks data by rewriting all images to the correct format.

(required for COLMAP reconstruction) This should be run AFTER the data has been downloaded and
undistorted.
"""
from pathlib import Path
from typing import Optional

import hydra
import rootutils
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def preprocess_data(mask_path: Path):
    """Preprocess the masks data by rewriting all images to the correct format.

    (required for COLMAP reconstruction)
    """
    if mask_path is not None:
        # rewrite all images to the correct format
        for img_name in mask_path.iterdir():
            # Only rename if it doesn't already have .JPG.png extension
            if not img_name.name.endswith(".JPG.png"):
                # Add .JPG before .png instead of replacing .png
                img_name_new = img_name.parent / (img_name.stem + ".JPG.png")
                img_name.rename(img_name_new)


@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess_data.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    scene_path = Path(cfg.data.scannet_path)
    for scene_id in tqdm(list(scene_path.iterdir()), desc="Processing scenes"):
        mask_path = scene_id / "dslr/undistorted_anon_masks"
        if mask_path.exists():
            preprocess_data(mask_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
