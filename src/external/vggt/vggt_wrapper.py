import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import numpy as np
import rootutils
import torch

# Setup root and add mast3r to path
root = rootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
)

# Add the paths explicitly
external_path = Path(root) / "external" / "vggt"

if str(external_path) not in sys.path:
    sys.path.append(str(external_path))

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VGGTWrapper:
    """Wrapper for VGG-T matching functionality."""

    def __init__(
        self,
        model_path: str = "/home/korth/guided-research/checkpoints/vggt/model.pt",
        device: Optional[str] = "cuda",
        model_dtype: Optional[torch.dtype] = torch.float16,
    ):
        """Initialize the VGG-T wrapper.

        Args:
            model_path: Path to the pretrained model to use
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        model = VGGT()
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        if model_dtype is not None:
            model = model.to(model_dtype)
        self.model = model
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8 and self.device == "cuda"
            else torch.float16
        )
        torch.cuda.empty_cache()

    @torch.no_grad()
    def match_image_pairs(
        self,
        image_pairs: List[Tuple[Union[str, Path, np.ndarray], Union[str, Path, np.ndarray]]],
    ) -> Dict[str, torch.Tensor]:
        """Match multiple pairs of images using VGGT.

        Args:
            image_pairs: List of tuples containing pairs of images (path or numpy array)

        Returns:
            Dictionary containing model predictions
        """
        # Convert all image paths to strings
        all_images = [str(img) for img in image_pairs]

        # Load all images at once
        images = load_and_preprocess_images(all_images).to(self.device)

        # Ensure correct shape: (B, S, C, H, W)
        # For multiple images from the same scene, we use batch=1 and sequence=num_images
        if images.dim() == 3:  # Single image (C, H, W)
            images = images.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        elif images.dim() == 4:  # Multiple images (N, C, H, W)
            # Treat all images as a sequence from one scene
            N, C, H, W = images.shape
            images = images.unsqueeze(0)  # (1, N, C, H, W)

        with torch.amp.autocast(dtype=self.dtype, device_type=self.device):
            predictions = self.model(images)

        return predictions

    @torch.no_grad()
    def predict_camera_extrinsics(
        self,
        images: Union[List[Union[str, Path, np.ndarray]], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict only camera extrinsics for images using VGGT.

        Args:
            images: List of image paths or tensor of images from the same scene

        Returns:
            Tuple containing (extrinsic, intrinsic) matrices
        """
        # Load images if they're not already a tensor
        if not isinstance(images, torch.Tensor):
            all_images = [str(img) for img in images]
            images = load_and_preprocess_images(all_images).to(self.device)

        # Ensure correct shape: (B, S, C, H, W)
        # For multiple images from the same scene, we use batch=1 and sequence=num_images
        if images.dim() == 3:  # Single image (C, H, W)
            images = images.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        elif images.dim() == 4:  # Multiple images (N, C, H, W)
            # Treat all images as a sequence from one scene
            N, C, H, W = images.shape
            images = images.unsqueeze(0)  # (1, N, C, H, W)

        # Get aggregated tokens - only compute what's needed for camera prediction
        with torch.amp.autocast(dtype=self.dtype, device_type=self.device):
            aggregated_tokens_list, _ = self.model.aggregator(images)
            # Predict cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]

        # Get extrinsic and intrinsic matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        return extrinsic, intrinsic

    @torch.no_grad()
    def predict_detailed(
        self,
        images: Union[List[Union[str, Path, np.ndarray]], torch.Tensor],
        query_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict detailed attributes for images using VGGT components.

        Args:
            images: List of image paths or tensor of images from the same scene
            query_points: Optional tensor of query points with shape (B, N, 2)

        Returns:
            Dictionary containing detailed predictions including cameras, depth maps, point maps
        """
        # Load images if they're not already a tensor
        if not isinstance(images, torch.Tensor):
            all_images = [str(img) for img in images]
            images = load_and_preprocess_images(all_images).to(self.device)

        # Ensure correct shape: (B, S, C, H, W)
        # For multiple images from the same scene, we use batch=1 and sequence=num_images
        if images.dim() == 3:  # Single image (C, H, W)
            images = images.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        elif images.dim() == 4:  # Multiple images (N, C, H, W)
            # Treat all images as a sequence from one scene
            N, C, H, W = images.shape
            images = images.unsqueeze(0)  # (1, N, C, H, W)

        # Store results
        results = {}

        # First get aggregated tokens
        with torch.amp.autocast(dtype=self.dtype, device_type=self.device):
            aggregated_tokens_list, ps_idx = self.model.aggregator(images)

            # Predict cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            # Get extrinsic and intrinsic matrices
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            results["extrinsic"] = extrinsic
            results["intrinsic"] = intrinsic

            # Predict depth maps
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
            results["depth_map"] = depth_map
            results["depth_conf"] = depth_conf

            # Predict point maps
            point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)
            results["point_map"] = point_map
            results["point_conf"] = point_conf

        # Construct 3D points from depth maps and cameras
        point_map_by_unprojection = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )
        results["point_map_by_unprojection"] = point_map_by_unprojection

        # Predict tracks if query points are provided
        if query_points is not None:
            if query_points.dim() == 2:
                query_points = query_points[None]  # add batch dimension
            with torch.amp.autocast(dtype=self.dtype, device_type=self.device):
                track_list, vis_score, conf_score = self.model.track_head(
                    aggregated_tokens_list, images, ps_idx, query_points=query_points
                )
            results["track_list"] = track_list
            results["vis_score"] = vis_score
            results["conf_score"] = conf_score

        return results

    @torch.no_grad()
    def process_scene(
        self,
        images: Union[List[Union[str, Path, np.ndarray]], torch.Tensor],
        query_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process all images from a scene together, leveraging the sequence dimension.

        This method is a convenience wrapper that passes all images to predict_detailed
        to process them as a single scene rather than individual images.

        Args:
            images: List of image paths or tensor of images from the same scene
            query_points: Optional tensor of query points with shape (B, N, 2)

        Returns:
            Dictionary containing detailed predictions for the whole scene
        """
        return self.predict_detailed(images, query_points)

    def _serialize_matches(self, points_map, match_path, index_map):
        """Serialize matches to a file."""
        matches_dict = {"query_idx": [], "database_idx": []}
        for kp0, kp1 in zip(points_map["keypoints0"], points_map["keypoints1"]):
            matches_dict["query_idx"].append(index_map[(kp0[0], kp0[1])])
            matches_dict["database_idx"].append(index_map[(kp1[0], kp1[1])])
        np.save(match_path, matches_dict, allow_pickle=True)

    @torch.no_grad()
    def extract_pointmaps(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 2,
    ) -> Tuple[Dict[Tuple[int, int], Dict[str, torch.Tensor]], Tuple[int, int]]:
        """Extract 3D point maps from a list of images in pairwise mode.

        Args:
            image_paths: List of paths to images from the same scene
            batch_size: Number of pairs to process at once

        Returns:
            Tuple containing:
                - Dictionary mapping (i,j) pairs to their pointmaps and confmaps
                - Tuple of (height, width) of the images
        """
        # Load and preprocess images
        images = load_and_preprocess_images(image_paths).to(self.device)

        # Add batch dimension if needed
        if images.dim() == 3:  # Single image (C, H, W)
            images = images.unsqueeze(0)  # (1, C, H, W)

        # Get all pairwise combinations - a la dust3r
        N = images.shape[0]  # number of images (10 in your case)
        pairs = []
        pair_indices = []  # Store the indices for each pair
        for i in range(N):
            for j in range(i + 1, N):
                # Stack the pair of images together
                pair = torch.stack([images[i], images[j]], dim=0)  # shape: (2, 3, 350, 518)
                pairs.append(pair)
                pair_indices.append((i, j))

        # Stack all pairs into a single batch
        # Each pair becomes a "scene" with 2 images
        all_pairs = torch.stack(pairs, dim=0)  # shape: (num_pairs, 2, 3, H, W)

        # Process in batches
        num_pairs = len(pairs)
        pointmaps_list = []
        confmaps_list = []
        pair_indices_list = []

        for start_idx in range(0, num_pairs, batch_size):
            end_idx = min(start_idx + batch_size, num_pairs)
            batch_pairs = all_pairs[start_idx:end_idx]
            batch_indices = pair_indices[start_idx:end_idx]

            with torch.amp.autocast(dtype=self.dtype, device_type=self.device):
                # Get aggregated tokens
                aggregated_tokens_list, ps_idx = self.model.aggregator(batch_pairs)

                # Get point maps and confidence
                pointmaps, confmaps = self.model.point_head(
                    aggregated_tokens_list, batch_pairs, ps_idx
                )

                # Convert to numpy and squeeze batch dimension
                pointmaps = pointmaps.cpu()
                confmaps = confmaps.cpu()

                # Add to list
                pointmaps_list.append(pointmaps)
                confmaps_list.append(confmaps)
                pair_indices_list.extend(batch_indices)

        # Get image resolution from first pair
        _, _, _, H, W = all_pairs.shape
        resolution = (H, W)

        # Concatenate all batches
        pointmaps = torch.cat(pointmaps_list, dim=0)
        confmaps = torch.cat(confmaps_list, dim=0)

        # Create mapping dictionary
        mapping = {}
        for idx, (i, j) in enumerate(pair_indices_list):
            mapping[(i, j)] = {
                "pts3d_1": pointmaps[idx, 0],  # First image in pair
                "pts3d_2": pointmaps[idx, 1],  # Second image in pair
                "conf_1": confmaps[idx, 0],  # Confidence for first image
                "conf_2": confmaps[idx, 1],  # Confidence for second image
                "conf_value": confmaps[idx].mean(),  # Overall confidence
            }

        return mapping, resolution

    @torch.no_grad()
    def extract_global_pointmaps(
        self,
        image_paths: List[Union[str, Path]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Extract 3D point maps from a list of images as a single global scene.

        This method processes all images together as a single scene, similar to VGGT/MUSt3R/MV-DUSt3R.

        Args:
            image_paths: List of paths to images from the same scene

        Returns:
            Tuple containing:
                - Point maps tensor
                - Confidence maps tensor
                - Tuple of (height, width) of the images
        """
        # Load and preprocess images
        images = load_and_preprocess_images(image_paths).to(self.device)

        # Add batch dimension if needed
        if images.dim() == 3:  # Single image (C, H, W)
            images = images.unsqueeze(0)  # (1, C, H, W)

        # Add sequence dimension for the model
        images = images.unsqueeze(0)  # (1, N, C, H, W)

        with torch.amp.autocast(dtype=self.dtype, device_type=self.device):
            # Get aggregated tokens
            aggregated_tokens_list, ps_idx = self.model.aggregator(images)

            # Get point maps and confidence
            pointmaps, confmaps = self.model.point_head(
                aggregated_tokens_list, images, ps_idx, frames_chunk_size=2
            )

            # Convert to numpy and squeeze batch dimension
            pointmaps = pointmaps.cpu().squeeze()
            confmaps = confmaps.cpu().squeeze()

            # Get image resolution
            _, _, _, H, W = images.shape
            resolution = (H, W)

        data = {
            "pts3d": pointmaps,
            "conf": confmaps,
            "view_indices": list(range(len(image_paths))),
        }

        return data, resolution
