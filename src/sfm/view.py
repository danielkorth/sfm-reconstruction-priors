from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import opt.residuals_func as ba
from sfm.keypoint import Keypoint
from utils.camera import Camera
from utils.visualization import visualize_reprojections


class View:
    def __init__(
        self,
        img_path: Path,
        mask_path: Path = None,
        mask_blur_iterations=0,
        K=None,
        R_t=None,
        parameterization="rotvec",
        descriptor="sift",
        process_image=True,
        keypoints_path=None,
        **kwargs
    ):
        self.img_path = img_path
        self.mask_path = mask_path
        self.name = img_path.name
        if process_image:
            self.img = self._load_image()
            self.mask = self._load_mask(mask_blur_iterations)
            if keypoints_path is not None:
                self._deserialize_keypoints(keypoints_path)
            else:
                self.kp = self._extract_features(type=descriptor)

            self.kp_active = [False] * len(self.kp)

        # this is only used for constructing the scene graph
        self.kp_visited = None

        self._active_keypoint_idxs = list()

        self._K = K
        self._scales = (1.0, 1.0)  # Initialize scales to 1.0 (no scaling)

        self.parameterization = parameterization
        self.descriptor = descriptor

    def resize(self, target_size: int) -> None:
        """Resize the image while maintaining aspect ratio and adjust camera intrinsics.

        Args:
            target_size (int): Target size for the longest dimension
        """
        if target_size <= 0:
            raise ValueError("Target size must be positive")
        if self.img is None:
            raise ValueError("No image loaded to resize")

        h, w = self.img.shape[:2]
        scale = target_size / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))

        # Store the scale factors
        self._scales = (float(w) / float(w_new), float(h) / float(h_new))

        # Resize image
        self.img = cv2.resize(self.img, (w_new, h_new))

        # Resize mask if it exists
        if self.mask is not None:
            self.mask = cv2.resize(self.mask, (w_new, h_new))

        # Adjust camera intrinsics if they exist
        if self._K is not None:
            K_new = self._K.clone()
            K_new[:2, :2] *= scale  # Scale focal lengths
            K_new[:2, 2] *= scale  # Scale principal point
            self._K = K_new

    @property
    def scales(self):
        """Get the scale factors between original and current image size."""
        return self._scales

    def add_camera(
        self, K, R_t=None, rotation=None, translation=None, kind="gt", parameterization="rotvec"
    ):
        if kind == "gt":
            if R_t is not None:
                self.camera_gt = Camera.from_Rt(K=K, R_t=R_t, parameterization=parameterization)
            elif rotation is not None and translation is not None:
                self.camera_gt = Camera.from_params(
                    K=self._K,
                    rotation=rotation,
                    translation=translation,
                    parameterization=parameterization,
                )
            else:
                raise ValueError("You need to provide parameters for the camera")
        elif kind == "opt":
            if R_t is not None:
                self.camera = Camera.from_Rt(K=K, R_t=R_t, parameterization=parameterization)
            elif rotation is not None and translation is not None:
                self.camera = Camera.from_params(
                    K=self._K,
                    rotation=rotation,
                    translation=translation,
                    parameterization=parameterization,
                )
            else:
                raise ValueError("You need to provide parameters for the camera")
        else:
            raise ValueError("Camera kind not available.")

    def _load_image(self):
        img = cv2.imread(str(self.img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, mask_blur_iterations):
        if self.mask_path is not None:
            mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
            mask = self.iterative_mask(mask, iterations=mask_blur_iterations)
            return mask
        else:
            return None

    def iterative_mask(self, mask, iterations=0):
        result = np.zeros_like(mask)
        result[mask == 0] = 1
        for _ in range(iterations):
            result = cv2.filter2D(result, -1, kernel=np.ones((3, 3)))
            result[result > 0] = 1
        return result

    def _extract_features(self, type="sift"):
        if type == "sift":
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(self.img, None)
            des /= np.linalg.norm(des, axis=1, keepdims=True) + 1e-7
        elif type == "rootsift":
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(self.img, None)
            des /= des.sum(axis=1, keepdims=True) + 1e-7
            des = np.sqrt(des)
        else:
            raise NotImplementedError("Feature extraction method not implemented")

        if self.mask is not None:
            kp, des = zip(
                *[
                    (kp[i], des[i])
                    for i in range(len(kp))
                    if self.mask[int(kp[i].pt[1]), int(kp[i].pt[0])] == 0
                ]
            )
            kp = tuple(kp)
            des = np.stack(des)

        # convert to keypoints objects
        kp = [Keypoint(kp[i].pt[0], kp[i].pt[1], des[i]) for i in range(len(kp))]
        return kp

    def _serialize_keypoints(self, path):
        kp_list = {
            "x": [kp.pt[0] for kp in self.kp],
            "y": [kp.pt[1] for kp in self.kp],
            "des": [kp.des for kp in self.kp],
        }
        np.save(path, kp_list, allow_pickle=True)

    def _deserialize_keypoints(self, path):
        kp_list = np.load(path, allow_pickle=True).item()
        if "des" in kp_list:
            self.kp = [
                Keypoint(kp_list["x"][i], kp_list["y"][i], kp_list["des"][i])
                for i in range(len(kp_list["x"]))
            ]
        else:
            self.kp = [
                Keypoint(kp_list["x"][i], kp_list["y"][i]) for i in range(len(kp_list["x"]))
            ]

    def draw_keypoints_cv2(self):
        viz = cv2.drawKeypoints(
            self.img, self.kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        plt.imshow(viz)
        plt.show()
        plt.close()

    # camera properties
    @property
    def K(self):
        if self._K is not None:
            return self._K
        elif self.camera_gt is not None:
            return self.camera_gt.K
        elif self.camera is not None:
            return self.camera.K
        else:
            raise ValueError("K has not been provided.")

    # keypoints
    @property
    def active_keypoint_idxs(self):
        return self._active_keypoint_idxs

    @active_keypoint_idxs.setter
    def active_keypoint_idxs(self, keypoints):
        tmp = self._active_keypoint_idxs
        tmp += keypoints
        tmp = list(set(tmp))
        self._active_keypoint_idxs = sorted(tmp)

    def filter_active_keypoint_idxs(self, filter):
        self._active_keypoint_idxs = [x for x, y in zip(self._active_keypoint_idxs, filter) if y]

    @property
    def projections_2d(self):
        return self._projections_2d

    @projections_2d.setter
    def projections_2d(self, projections_2d):
        self._projections_2d = projections_2d

    @property
    def points_3d_filter(self):
        return self._points_3d_filter

    @points_3d_filter.setter
    def points_3d_filter(self, points_3d_filter):
        self._points_3d_filter = points_3d_filter

    @property
    def shape(self):
        return self.img.shape[:2]

    def _get_active_keypoints(self):
        pts = list()
        for idx in self._active_keypoint_idxs:
            pts.append(self.kp[idx].pt)
        return np.array(pts)

    def _get_keypoints_from_idx(self, keypoint_idxs):
        pts = list()
        for idx in keypoint_idxs:
            pts.append(self.kp[idx].pt)
        return np.array(pts)

    def draw_active_keypoints(self, predictions=None, use_projections=True):
        """:use_projections: currently, we are using GT matches by assigning 2d_projections value.

        In the future, this will be replaced by the active_keypoints that are found, but are not
        ground truth.
        """
        if use_projections:
            return visualize_reprojections(predictions, self.projections_2d, img=self.img)
        else:
            return visualize_reprojections(predictions, self._get_active_keypoints(), img=self.img)

    def get_gt_points_from_keypoints(self, mesh_aligned_path="", pixels=None):
        camera_center = self.camera_gt.R.T @ -self.camera_gt.t
        pixels = (
            self._get_active_keypoints() if pixels is None else pixels
        )  # consider adding +0.5 and +0.5 to get the center of the pixel
        pixels_homo = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
        pixels_camera = self.camera_gt.K_inv @ pixels_homo.T
        pixels_world = self.camera_gt.R.T @ pixels_camera
        pixels_world /= np.linalg.norm(pixels_world, axis=0, keepdims=1)
        cc_directions = np.hstack(
            (np.repeat(camera_center.T, pixels_world.shape[1], axis=0), pixels_world.T)
        )

        mesh = o3d.io.read_triangle_mesh(mesh_aligned_path)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        raycasting_scene = o3d.t.geometry.RaycastingScene()
        raycasting_scene.add_triangles(mesh)

        out = raycasting_scene.cast_rays(cc_directions.astype(np.float32))

        points_3d = cc_directions[:, :3] + (out["t_hit"].numpy()[:, None] * cc_directions[:, 3:])
        return points_3d

    def project_onto_image(self, points_3d, visualize=False, use_gt_cam=True):
        if len(points_3d.shape) == 1:
            points_3d = points_3d[None, :]

        if use_gt_cam:
            proj = ba.project(
                points_3d, self.camera_gt.rotation, self.camera_gt.translation, self.camera_gt.K
            )
        else:
            proj = ba.project(
                points_3d, self.camera.rotation, self.camera.translation, self.camera.K
            )

        if visualize:
            return visualize_reprojections(proj, img=self.img)
        else:
            return proj

    def draw_keypoints(
        self,
        keypoints=None,
        keypoint_idxs=None,
        keypoints_gt=None,
        keypoint_gt_idxs=None,
        markersize=2,
    ):
        if keypoint_gt_idxs is not None:
            keypoints_gt = self._get_keypoints_from_idx(keypoint_gt_idxs)
        return visualize_reprojections(
            keypoints, keypoints_gt, img=self.img, markersize=markersize
        )
