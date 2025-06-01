import os
import shutil
import sys
import tempfile
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
external_path = Path(root) / "external" / "mast3r"

if str(external_path) not in sys.path:
    sys.path.append(str(external_path))

dust3r_path = external_path / "dust3r"

if str(dust3r_path) not in sys.path:
    sys.path.append(str(dust3r_path))

# # dust3r_dust3r_path = dust3r_path / "dust3r"

dust3r_croco_path = dust3r_path / "croco"

if str(dust3r_croco_path) not in sys.path:
    sys.path.append(str(dust3r_croco_path))


from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.fast_nn import fast_reciprocal_NNs

# Now import using the full package path
from mast3r.model import AsymmetricMASt3R


class DUSt3RWrapper:
    """Wrapper for DUSt3R."""

    @classmethod
    def get_model_class(cls):
        """Get the model class to use for this wrapper."""
        return AsymmetricCroCo3DStereo

    def __init__(
        self,
        model_name: str = "/home/korth/guided-research/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        device: Optional[str] = None,
        resize: Optional[Tuple[int, int]] = (512, 384),
        # global optimization parameters
        go_niter: int = 1000,
        go_lr: float = 0.01,
        go_init: str = "mst",
        go_schedule: str = "cosine",
        go_device: str = "cpu",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_class = self.get_model_class()
        self.model = model_class.from_pretrained(model_name).to(self.device)
        self.resize = resize
        self.go_niter = go_niter
        self.go_lr = go_lr
        self.go_init = go_init
        self.go_schedule = go_schedule
        self.go_device = go_device

    @torch.no_grad()
    def inference(
        self,
        image_pairs: List[Tuple[Union[str, Path, np.ndarray], Union[str, Path, np.ndarray]]],
        symmetrize: bool = True,
    ):
        """Run inference on a list of image pairs."""
        # Convert all image paths to strings
        all_images = [str(img) for img in image_pairs]

        # Load all images at once
        images = load_images(all_images, size=self.resize[0], verbose=False)

        # Create pairs using the loaded images
        pairs = make_pairs(images, symmetrize=symmetrize)

        # Run inference on all pairs at once
        output = inference(pairs, self.model, self.device, 3, verbose=False)

        return output

    def extract_pointmaps(
        self,
        image_pairs: List[Tuple[Union[str, Path, np.ndarray], Union[str, Path, np.ndarray]]],
    ) -> List[Dict[str, np.ndarray]]:
        """Extract 3D point maps from pairs of images using DUSt3R.

        Args:
            image_pairs: List of tuples containing pairs of images (path or numpy array)

        Returns:
            List of dictionaries, each containing 3D point maps and confidence values for each pair
        """
        output = self.inference(image_pairs)

        resolution = output["view1"]["true_shape"][0]
        # Create mapping from indices to output index
        mapping = {}
        for i, (idx1, idx2) in enumerate(zip(output["view1"]["idx"], output["view2"]["idx"])):
            mapping[(idx1, idx2)] = i

        # Process each pair and organize results
        mapping_to_matches = {}
        for pair, idx in mapping.items():
            # Get predictions for this pair
            pred1 = output["pred1"]
            pred2 = output["pred2"]

            # Get other outputs
            pts3d_im0 = pred1["pts3d"][idx].squeeze(0).detach().cpu()
            pts3d_im1 = pred2["pts3d_in_other_view"][idx].squeeze(0).detach().cpu()

            conf_im0 = pred1["conf"][idx].squeeze(0).detach().cpu()
            conf_im1 = pred2["conf"][idx].squeeze(0).detach().cpu()

            # log the conf values
            conf_im0 = torch.log(conf_im0)
            conf_im1 = torch.log(conf_im1)

            # Calculate overall confidence value (similar to DUSt3R)
            conf_value = conf_im0.mean() * conf_im1.mean()

            match_data = {
                "pts3d_1": pts3d_im0,
                "pts3d_2": pts3d_im1,
                "conf_1": conf_im0,
                "conf_2": conf_im1,
                "conf_value": conf_value,
            }

            mapping_to_matches[pair] = match_data

        return mapping_to_matches, resolution

    def global_optimization(self, output: torch.Tensor):
        """Global optimization of the point maps."""
        scene = global_aligner(output, device="cuda", mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(
            init=self.go_init, niter=self.go_niter, schedule=self.go_schedule, lr=self.go_lr
        )
        return scene

    def reconstruct_poses(self, image_paths: List[Union[str, Path]], **kwargs):
        """Reconstruct the scene from a list of image paths.

        Args:
            image_paths: List of paths to input images

        Returns:
            List of camera poses
        """
        output = self.inference(image_paths, symmetrize=False)
        del self.model
        torch.cuda.empty_cache()
        scene = self.global_optimization(output)
        return scene.get_im_poses().detach().cpu().numpy()


class MASt3RWrapper(DUSt3RWrapper):
    """Wrapper for MASt3R matching functionality."""

    @classmethod
    def get_model_class(cls):
        """Get the model class to use for this wrapper."""
        return AsymmetricMASt3R

    def __init__(
        self,
        model_name: str = "/home/korth/guided-research/checkpoints/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        device: Optional[str] = None,
        resize: Optional[Tuple[int, int]] = (512, 384),
        subsample: int = 8,  # how to subsample the image for initial reciprocal matching (to speed it up)
        border: int = 3,  # how many pixels to ignore at the border of the image
        # Global optimization parameters
        go_niter: int = 1000,
        go_lr: float = 0.01,
        go_init: str = "mst",
        go_schedule: str = "cosine",
        go_device: str = "cpu",
    ):
        """Initialize the MASt3R wrapper.

        Args:
            model_name: Name of the pretrained model to use
            device: Device to run on ('cuda' or 'cpu')
            resize: Optional resize dimensions (width, height)
            subsample: Subsampling factor for matching (default: 8)
            border: Border to ignore for matches (default: 3)
            go_niter: Number of iterations for global optimization (default: 1000)
            go_lr: Learning rate for optimization (default: 0.01)
            go_init: Initialization method (default: "mst")
            go_schedule: Learning rate schedule (default: "cosine")
            go_device: Device for optimization (default: "cpu")
        """
        super().__init__(
            model_name=model_name,
            device=device,
            resize=resize,
            go_niter=go_niter,
            go_lr=go_lr,
            go_init=go_init,
            go_schedule=go_schedule,
            go_device=go_device,
        )
        self.subsample = subsample
        self.border = border

    @torch.no_grad()
    def match_image_pairs(
        self,
        image_pairs: List[Tuple[Union[str, Path, np.ndarray], Union[str, Path, np.ndarray]]],
    ) -> List[Dict[str, np.ndarray]]:
        """Match multiple pairs of images using MASt3R.

        Args:
            image_pairs: List of tuples containing pairs of images (path or numpy array)

        Returns:
            List of dictionaries, each containing matching information for each pair
        """
        # manual inference to avoid CPU OOM
        # Convert all image paths to strings
        all_images = [str(img) for img in image_pairs]

        # Load all images at once
        images = load_images(all_images, size=self.resize[0], verbose=False)

        # Create pairs using the loaded images
        pairs = make_pairs(images, symmetrize=True)

        # Store all matches
        mapping_to_matches = {}
        batch_size = 3  # Process 3 pairs at a time - gotta prevent CPU OOM

        # Run inference in batches to avoid memory issues
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            output = inference(batch, self.model, self.device, 3, verbose=False)

            # Process each pair in the batch
            for j, (idx1, idx2) in enumerate(zip(output["view1"]["idx"], output["view2"]["idx"])):
                # Get predictions for this pair
                view1, pred1 = output["view1"], output["pred1"]
                view2, pred2 = output["view2"], output["pred2"]

                # Get descriptors
                desc1 = pred1["desc"][j].squeeze(0).detach()
                desc2 = pred2["desc"][j].squeeze(0).detach()

                # Find matches using fast reciprocal nearest neighbors
                matches_im0, matches_im1 = fast_reciprocal_NNs(
                    desc1,
                    desc2,
                    subsample_or_initxy1=self.subsample,
                    device=self.device,
                    dist="dot",
                    block_size=2**13,
                )

                # Get image dimensions
                H0, W0 = view1["true_shape"][j]
                H1, W1 = view2["true_shape"][j]

                # Filter out matches near image borders
                valid_matches_im0 = (
                    (matches_im0[:, 0] >= self.border)
                    & (matches_im0[:, 0] < int(W0) - self.border)
                    & (matches_im0[:, 1] >= self.border)
                    & (matches_im0[:, 1] < int(H0) - self.border)
                )

                valid_matches_im1 = (
                    (matches_im1[:, 0] >= self.border)
                    & (matches_im1[:, 0] < int(W1) - self.border)
                    & (matches_im1[:, 1] >= self.border)
                    & (matches_im1[:, 1] < int(H1) - self.border)
                )

                valid_matches = valid_matches_im0 & valid_matches_im1
                matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

                # Convert to matches format
                matches = np.zeros((len(matches_im0), 2), dtype=np.int32)
                matches[:, 0] = np.arange(len(matches_im0))
                matches[:, 1] = np.arange(len(matches_im1))

                match_data = {
                    "keypoints0": matches_im0,
                    "keypoints1": matches_im1,
                    "matches": matches,
                }

                mapping_to_matches[(idx1, idx2)] = match_data

        return mapping_to_matches, (H0.item(), W0.item())

    def _serialize_matches(self, points_map, match_path, index_map):
        """Serialize matches to a file."""
        matches_dict = {"query_idx": [], "database_idx": []}
        for kp0, kp1 in zip(points_map["keypoints0"], points_map["keypoints1"]):
            matches_dict["query_idx"].append(index_map[(kp0[0], kp0[1])])
            matches_dict["database_idx"].append(index_map[(kp1[0], kp1[1])])
        np.save(match_path, matches_dict, allow_pickle=True)


class MASt3RSfMWrapper:
    """A wrapper for the MASt3R reconstruction pipeline."""

    def __init__(
        self,
        model_name: str = "/home/korth/guided-research/checkpoints/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        device: Optional[str] = None,
        image_size: Tuple[int, int] = (512, 384),
        cache_dir: Optional[str] = None,
        # Global optimization parameters
        lr1: float = 0.2,  # Coarse learning rate
        niter1: int = 500,  # Coarse iterations
        lr2: float = 0.02,  # Fine learning rate
        niter2: int = 500,  # Fine iterations
        matching_conf_thr: float = 5.0,
        shared_intrinsics: bool = False,
        optim_level: str = "refine+depth",
        # Scene graph parameters
        scenegraph_type: str = "complete",
        winsize: Optional[int] = None,
        win_cyclic: bool = False,
        refid: Optional[int] = None,
    ):
        """Initialize the MASt3R pipeline.

        Args:
            model_name: Name of the pretrained model to use
            device: Device to run on ('cuda' or 'cpu')
            image_size: Size to resize images to (width, height)
            cache_dir: Directory to store temporary files
            lr1: Learning rate for coarse optimization
            niter1: Number of iterations for coarse optimization
            lr2: Learning rate for fine optimization
            niter2: Number of iterations for fine optimization
            matching_conf_thr: Confidence threshold for matching
            shared_intrinsics: Whether to share intrinsics across views
            optim_level: Optimization level ('coarse', 'refine', or 'refine+depth')
            scenegraph_type: Type of scene graph to use ('complete', 'swin', 'logwin', 'oneref')
            winsize: Window size for sliding window scene graph
            win_cyclic: Whether to use cyclic window
            refid: Reference image ID for one-reference scene graph
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AsymmetricMASt3R.from_pretrained(model_name).to(self.device)
        self.image_size = image_size
        self.cache_dir = cache_dir or tempfile.mkdtemp(suffix="_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Store optimization parameters
        self.lr1 = lr1
        self.niter1 = niter1
        self.lr2 = lr2
        self.niter2 = niter2
        self.matching_conf_thr = matching_conf_thr
        self.shared_intrinsics = shared_intrinsics
        self.optim_level = optim_level

        # Store scene graph parameters
        self.scenegraph_type = scenegraph_type
        self.winsize = winsize
        self.win_cyclic = win_cyclic
        self.refid = refid

        # Initialize scene state
        self.scene_state = None

    def __del__(self):
        """Clean up temporary files."""
        if self.cache_dir and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def _prepare_scene_graph_params(self) -> List[str]:
        """Prepare scene graph parameters."""
        scene_graph_params = [self.scenegraph_type]
        if self.scenegraph_type in ["swin", "logwin"]:
            scene_graph_params.append(str(self.winsize))
        elif self.scenegraph_type == "oneref":
            scene_graph_params.append(str(self.refid))
        if self.scenegraph_type in ["swin", "logwin"] and not self.win_cyclic:
            scene_graph_params.append("noncyclic")
        return scene_graph_params

    def reconstruct_poses(
        self,
        image_paths: List[Union[str, Path]],
        K: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Run the reconstruction pipeline on a set of images.

        Args:
            image_paths: List of paths to input images
            min_conf_thr: Minimum confidence threshold for points
            as_pointcloud: Whether to output as pointcloud
            mask_sky: Whether to mask sky regions
            clean_depth: Whether to clean depth maps
            transparent_cams: Whether to make cameras transparent
            cam_size: Size of camera visualization
            TSDF_thresh: TSDF threshold for post-processing
            focal: Fixed focal length in pixels (if None, will be estimated)
            K: Custom 3x3 intrinsics matrix (if None, will be estimated or created from focal)
            init: Dictionary of initialization parameters for each image

        Returns:
            Dictionary containing reconstruction results
        """
        # Load and preprocess images
        imgs = load_images(image_paths, size=self.image_size[0], verbose=False)

        # Create image pairs based on scene graph
        scene_graph = "-".join(self._prepare_scene_graph_params())
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

        # Adjust optimization parameters based on level
        if self.optim_level == "coarse":
            niter2 = 0
        else:
            niter2 = self.niter2

        # Initialize camera parameters
        init_dict = {}

        # If a custom intrinsics matrix is provided, use it for all images
        if K is not None:
            # Ensure K is on the correct device
            K = K.to(self.device)
            target_size = 512
            scale = target_size / (max(K[1, 2], K[0, 2]) * 2)
            K[:2, :2] *= scale  # Scale focal lengths
            K[:2, 2] *= scale  # Scale principal point
            for img in image_paths:
                init_dict[img] = {"intrinsics": K}

        # Run sparse global alignment
        scene = sparse_global_alignment(
            image_paths,
            pairs,
            self.cache_dir,
            self.model,
            lr1=self.lr1,
            niter1=self.niter1,
            lr2=self.lr2,
            niter2=niter2,
            device=self.device,
            opt_depth="depth" in self.optim_level,
            shared_intrinsics=self.shared_intrinsics,
            matching_conf_thr=self.matching_conf_thr,
            init=init_dict,  # Pass initialization dictionary
        )

        return scene.get_im_poses().detach().cpu().numpy()
