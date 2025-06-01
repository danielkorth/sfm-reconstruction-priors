import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import numpy as np
import rootutils
import torch

# Setup root and add spsg to path
root = rootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
)

# Add the external directory to path (not just spsg)
external_path = Path(root) / "external"
if str(external_path) not in sys.path:
    sys.path.append(str(external_path))

# Now import using the full package path
from spsg.models.matching import Matching
from spsg.models.utils import (
    AverageTimer,
    compute_epipolar_error,
    compute_pose_error,
    error_colormap,
    estimate_pose,
    make_matching_plot,
    pose_auc,
    read_image,
    rotate_intrinsics,
    rotate_pose_inplane,
    scale_intrinsics,
)

# set grad off
torch.set_grad_enabled(False)


class SuperGlueWrapper:
    """Wrapper for SuperGlue matching functionality."""

    def __init__(
        self,
        config: Dict,
        device: Optional[str] = None,
    ):
        """Initialize the SuperGlue wrapper.

        Args:
            config: Configuration dictionary containing SuperPoint and SuperGlue parameters
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure config has required structure
        if "superpoint" not in config:
            config["superpoint"] = {}
        if "superglue" not in config:
            config["superglue"] = {}

        # Set default values if not provided
        config["superpoint"].setdefault("nms_radius", 4)
        config["superpoint"].setdefault("keypoint_threshold", 0.005)
        config["superpoint"].setdefault("max_keypoints", 1024)
        config["superglue"].setdefault("weights", "indoor")
        config["superglue"].setdefault("sinkhorn_iterations", 20)
        config["superglue"].setdefault("match_threshold", 0.2)

        self.matching = Matching(config).eval().to(self.device)

    def match_image_pair(
        self,
        image0: Union[str, Path, np.ndarray],
        image1: Union[str, Path, np.ndarray],
        resize: Optional[Tuple[int, int]] = [-1],  # no resize
        resize_float: bool = False,
        rot0: int = 0,
        rot1: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Match a pair of images using SuperGlue.

        Args:
            image0: First image (path or numpy array)
            image1: Second image (path or numpy array)
            resize: Optional resize dimensions (width, height)
            resize_float: Whether to resize after casting to float
            rot0: Rotation for first image
            rot1: Rotation for second image

        Returns:
            Dictionary containing:
                - keypoints0: Keypoints in first image
                - keypoints1: Keypoints in second image
                - matches: Matches between keypoints
                - match_confidence: Confidence scores for matches
        """
        # Load and preprocess images
        if isinstance(image0, (str, Path)):
            image0, inp0, scales0 = read_image(image0, self.device, resize, rot0, resize_float)
        else:
            # Handle numpy array input
            image0, inp0, scales0 = read_image(image0, self.device, resize, rot0, resize_float)

        if isinstance(image1, (str, Path)):
            image1, inp1, scales1 = read_image(image1, self.device, resize, rot1, resize_float)
        else:
            # Handle numpy array input
            image1, inp1, scales1 = read_image(image1, self.device, resize, rot1, resize_float)

        if image0 is None or image1 is None:
            raise ValueError("Failed to load one or both images")

        with torch.no_grad():
            # Perform matching
            pred = self.matching({"image0": inp0, "image1": inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        return {
            "keypoints0": pred["keypoints0"],
            "keypoints1": pred["keypoints1"],
            "matches": pred["matches0"],
            "match_confidence": pred["matching_scores0"],
            "scales0": scales0,
            "scales1": scales1,
        }

    def estimate_relative_pose(
        self,
        matches: Dict[str, np.ndarray],
        K0: np.ndarray,
        K1: np.ndarray,
        thresh: float = 1.0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate relative pose from matches.

        Args:
            matches: Dictionary from match_image_pair
            K0: Camera matrix for first image
            K1: Camera matrix for second image
            thresh: Threshold for pose estimation

        Returns:
            Tuple of (R, t, inliers) or (None, None, None) if estimation fails
        """
        valid = matches["matches"] > -1
        mkpts0 = matches["keypoints0"][valid]
        mkpts1 = matches["keypoints1"][matches["matches"][valid]]

        # Scale intrinsics if images were resized
        K0 = scale_intrinsics(K0, matches["scales0"])
        K1 = scale_intrinsics(K1, matches["scales1"])

        return estimate_pose(mkpts0, mkpts1, K0, K1, thresh)

    def visualize_matches(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        matches: Dict[str, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        show_keypoints: bool = False,
        fast_viz: bool = False,
        opencv_display: bool = False,
    ) -> None:
        """Visualize matches between image pair.

        Args:
            image0: First image
            image1: Second image
            matches: Dictionary from match_image_pair
            output_path: Optional path to save visualization
            show_keypoints: Whether to show keypoints
            fast_viz: Use faster OpenCV visualization
            opencv_display: Display via OpenCV before saving
        """
        valid = matches["matches"] > -1
        mkpts0 = matches["keypoints0"][valid]
        mkpts1 = matches["keypoints1"][matches["matches"][valid]]
        mconf = matches["match_confidence"][valid]

        color = cm.jet(mconf)
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(len(matches["keypoints0"]), len(matches["keypoints1"])),
            f"Matches: {len(mkpts0)}",
        ]

        small_text = [
            "Keypoint Threshold: {:.4f}".format(
                self.matching.superpoint.config["keypoint_threshold"]
            ),
            "Match Threshold: {:.2f}".format(self.matching.superglue.config["match_threshold"]),
        ]

        if output_path:
            make_matching_plot(
                image0,
                image1,
                matches["keypoints0"],
                matches["keypoints1"],
                mkpts0,
                mkpts1,
                color,
                text,
                output_path,
                show_keypoints,
                fast_viz,
                opencv_display,
                "Matches",
                small_text,
            )

    def _serialize_keypoints(self, keypoints, path):
        """Serialize keypoints to disk (expects output from match_image_pair)"""
        keypoints_dict = {
            "x": keypoints[:, 0],
            "y": keypoints[:, 1],
        }
        np.save(path, keypoints_dict, allow_pickle=True)

    def _serialize_matches(self, matches, path):
        """Serialize matches to disk (expects output from match_image_pair)"""
        matches_dict = {"query_idx": [], "database_idx": []}
        for i, idx in enumerate(matches):
            if idx == -1:
                continue
            matches_dict["query_idx"].append(i)
            matches_dict["database_idx"].append(idx)

        np.save(path, matches_dict, allow_pickle=True)
