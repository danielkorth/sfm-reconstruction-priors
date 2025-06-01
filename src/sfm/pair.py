from collections import defaultdict
from typing import List

import cv2
import numpy as np
import roma
import torch

from sfm.matching import Match, Matcher
from sfm.view import View
from utils.visualization import (
    plot_3d,
    plot_matches,
    visualize_cameras_and_points,
    visualize_points_3d,
)


class Pair:
    def __init__(self, v1: View, v2: View):
        self.v1 = v1  # query image
        self.v2 = v2  # "database image"
        self.base_R = np.eye(3)
        self.base_t = np.zeros((3, 1))
        self.scenegraph = None

        self.query_to_database = None
        self.database_to_query = None

    def find_matches(
        self,
        ratio=0.75,
        verbose=False,
        max_distance=0.7,
        distance_metric="euclidean",
        symmetric=False,
    ):
        # snavely ratio: 0.6
        matcher = Matcher()

        # all matches from v1 to v2
        matches_v1_to_v2 = matcher.match(
            np.array([kp.des for kp in self.v1.kp]),
            np.array([kp.des for kp in self.v2.kp]),
            topk=2,
            max_distance=max_distance,
            distance_metric=distance_metric,
        )
        if verbose:
            print(f"Found {len(matches_v1_to_v2)} initial matches from v1 to v2.")

        # Lowe's ratio test for v1 to v2
        lowe_ratio = ratio
        matches_v1_to_v2_ratio = [
            m[0] for m in matches_v1_to_v2 if m[0].distance < lowe_ratio * m[1].distance
        ]
        removed_matches_v1_to_v2 = len(matches_v1_to_v2) - len(matches_v1_to_v2_ratio)
        if verbose:
            print(
                f"Removed {removed_matches_v1_to_v2} matches after ratio test, {len(matches_v1_to_v2_ratio)} remaining."
            )

        # If symmetric matching is enabled
        if symmetric:
            # all matches from v2 to v1
            matches_v2_to_v1 = matcher.match(
                np.array([kp.des for kp in self.v2.kp]),
                np.array([kp.des for kp in self.v1.kp]),
                topk=2,
                max_distance=max_distance,
                distance_metric=distance_metric,
            )
            if verbose:
                print(f"Found {len(matches_v2_to_v1)} initial matches from v2 to v1.")

            # Lowe's ratio test for v2 to v1
            matches_v2_to_v1_ratio = [
                m[0] for m in matches_v2_to_v1 if m[0].distance < lowe_ratio * m[1].distance
            ]
            removed_matches_v2_to_v1 = len(matches_v2_to_v1) - len(matches_v2_to_v1_ratio)
            if verbose:
                print(
                    f"Removed {removed_matches_v2_to_v1} matches after ratio test, {len(matches_v2_to_v1_ratio)} remaining."
                )

            # Keep only matches that are consistent in both directions
            matches = []
            for match in matches_v1_to_v2_ratio:
                if match.database_idx in [m.query_idx for m in matches_v2_to_v1_ratio]:
                    matches.append(match)
        else:
            matches = matches_v1_to_v2_ratio

        self.matches = matches

    def geometric_verification(
        self,
        kind="essential",
        prob=0.999,
        max_pixel_distance=1.0,
        min_inlier_ratio=0.25,
        verbose=False,
    ):
        if self.matches is None:
            raise Exception("Matches have not been found yet.")

        # extract matching keypoints
        pts1, pts2 = self._extract_points_from_matches()

        # Normalize coordinates for better numerical stability
        # performs the following: u = (x - cx) / fx, v = (y - cy) / fy
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), np.array(self.v1.K), None)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), np.array(self.v2.K), None)

        # Get the focal length from the intrinsic matrix
        # Double checked this, seems correct
        f = (
            self.v1.K[0, 0] + self.v1.K[1, 1]
        ) / 2  # Assuming K is in the form [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        pixel_threshold = float(
            max_pixel_distance / f
        )  # Convert 1 pixel to normalized coordinates

        if kind == "fundamental":
            raise NotImplementedError("not implemented")
            F, mask = cv2.findFundamentalMat(
                pts1_norm, pts2_norm, method=cv2.FM_RANSAC, confidence=0.999
            )
        elif kind == "essential":
            # https://github.com/colmap/colmap/blob/eb54a3c5e919581308066f0a913176378f86110b/src/colmap/estimators/two_view_geometry.h#L42
            self.E, mask = cv2.findEssentialMat(
                pts1_norm,
                pts2_norm,
                np.eye(3),
                method=cv2.RANSAC,
                prob=prob,
                threshold=pixel_threshold,
            )

            inlier_ratio = mask.sum() / len(mask)
            if inlier_ratio < min_inlier_ratio:
                # set all matches to zero, the pair is not valid / geometrically verified
                self.matches = []
                return

            # Check if the geometrically verified matches form a homography
            # self._homography = self._is_homography(pts1_norm, pts2_norm, mask, pixel_threshold, prob, verbose)

        else:
            raise Exception(
                "Provide a valid geometric verification technique: [fundamental | essential]"
            )

        if verbose:
            print(
                f"Out of {len(mask)} potential matches, {mask.sum()} matches were geometrically verified using {kind} matrix."
            )

        new_matches = list()
        for match, verified in zip(self.matches, mask):
            if verified:
                new_matches.append(match)

        self.matches = new_matches

    def _is_homography(self, pts1, pts2, essential_mask, pixel_threshold, prob, verbose=False):
        """Check if the pair of views forms a homography by comparing inlier counts between
        homography and essential matrix transformations.

        Args:
            pts1: Points from the first view
            pts2: Points from the second view
            essential_mask: Inlier mask from essential matrix computation
            pixel_threshold: RANSAC reprojection threshold
            prob: RANSAC confidence level
            verbose: Whether to print additional information

        Returns:
            bool: True if the pair forms a homography, False otherwise
        """
        # Compute homography and get inlier count
        H, homography_mask = cv2.findHomography(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=pixel_threshold,
            maxIters=2000,
            confidence=prob,
        )

        homography_inliers = np.sum(homography_mask)
        essential_inliers = np.sum(essential_mask)

        # If essential matrix has no inliers, consider it a homography
        if essential_inliers == 0:
            if verbose:
                print("Essential matrix has no inliers, likely a homography")
            return True

        # Compare inlier counts
        inlier_ratio = homography_inliers / essential_inliers
        is_homography = inlier_ratio > 0.8  # Threshold for considering homography

        if verbose:
            print(f"Homography inliers: {homography_inliers}")
            print(f"Essential matrix inliers: {essential_inliers}")
            print(f"Inlier ratio (homography/essential): {inlier_ratio:.3f}")
            print(f"Is homography: {is_homography}")

        return is_homography

    def _remove_spurious_matches(self, keep_one_to_one=False, verbose=False):
        """A spurious match is a match with a one-to-many mapping, ie kp1 is matched to multlipe
        kp2's :keep_one_to_one: flag whether we should reduce the one-to-many matches to a one-to-
        one match, or remove all matches in this set completely.

        This makes most sense if the spurious match removal happens after geometric verification.
        The keep_one_to_one strategy is currently to just keep one of them (the first match found - ie random).
        """

        # Remove Spurious Matches
        # a spurious match is a match where we have a many-to-one mapping of features
        database_to_query = defaultdict(list)

        # add matches (the other way around)
        for match in self.matches:
            database_to_query[match.database_idx].append(match.query_idx)

        # identify spurious features/matches
        spurious_features = list()
        for database_idx, query_idxs in database_to_query.items():
            if len(query_idxs) > 1:
                # detected spurious one
                spurious_features.append(database_idx)
        spurious_features = set(spurious_features)

        new_matches = list()

        n_removed = 0

        if keep_one_to_one:
            spurious_features_used = {x: True for x in spurious_features}

        for match in self.matches:
            if match.database_idx not in spurious_features:
                new_matches.append(match)
            elif keep_one_to_one and spurious_features_used[match.database_idx]:
                # choose the first time it is
                spurious_features_used[match.database_idx] = False
                new_matches.append(match)
            else:
                n_removed += 1

        self.matches = new_matches

        if verbose:
            print(f"Removed {n_removed} spurious matches, {len(self.matches)} remaining.")

    def set_kp_active(self):
        """Set keypoints as active or not based on whether they are used / matched with a keypoint
        from another image (based on this pair)"""
        for match in self.matches:
            self.v1.kp_active[match.query_idx] = True
            self.v2.kp_active[match.database_idx] = True

    def estimate_relative_pose(self, kp1_idxs=None, kp2_idxs=None, use_tracks=False):
        if use_tracks:
            pts1, pts2 = self._extract_points_from_keypoints(kp1_idxs, kp2_idxs)
        else:
            pts1, pts2 = self._extract_points_from_matches()

        n_points, R, t, mask = cv2.recoverPose(self.E, pts1, pts2, self.v1.K.numpy())
        return dict(R_rel=R, t_rel=t, mask=mask, pts1=pts1, pts2=pts2, K=self.v1.K)

    def reconstruct_two_views(
        self,
        R1=None,
        t1=None,
        translation_scale=1,
        kp1_idxs=None,
        kp2_idxs=None,
        use_tracks=False,  # whether to use all tracks for recoverPose or only the two-view matches from the initial pair
        use_all_matches=False,  # whether to use all matches for reconstruct_two_views or only the two-view matches from the initial pair
        use_gt_first_view=True,
        max_reprojection_error=4.0,  # maximum allowed reprojection error in pixels
        min_triangulation_angle=3.0,
        verbose=False,
    ):
        # extracting points from keypoints
        if use_all_matches:
            pts1, pts2 = self._extract_points_from_matches()
        else:
            pts1, pts2 = self._extract_points_from_keypoints(kp1_idxs, kp2_idxs)
        pts1_norm = cv2.undistortPoints(
            pts1.reshape(-1, 1, 2), np.array(self.v1.K), distCoeffs=None
        )
        pts2_norm = cv2.undistortPoints(
            pts2.reshape(-1, 1, 2), np.array(self.v2.K), distCoeffs=None
        )

        info = self.estimate_relative_pose(
            kp1_idxs=kp1_idxs, kp2_idxs=kp2_idxs, use_tracks=use_tracks
        )

        if R1 is None:
            R1 = np.eye(3)
        if t1 is None:
            t1 = np.zeros((3, 1))

        # scale in world space:
        cc = info["R_rel"].T @ -info["t_rel"]
        cc *= (
            translation_scale.numpy()
            if isinstance(translation_scale, torch.Tensor)
            else translation_scale
        )
        t = -(info["R_rel"] @ cc)
        R1, R2, t1, t2 = self.relative_to_global(
            info["R_rel"], t, use_gt=False  # True #False
        ).values()
        P1 = np.hstack((R1, t1))
        P1_homo = np.vstack((P1, np.zeros((1, 4))))
        P2 = np.hstack((R2, t2))
        P2_homo = np.vstack((P2, np.zeros((1, 4))))

        points = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm).T

        # apply mask to remove points that won't pass Chirality Condition
        points /= points[:, -1][:, None]
        # cheirality check
        depth_1 = (P1_homo @ points.T)[2]
        depth_2 = (P2_homo @ points.T)[2]
        mask = (depth_1 > 0) & (depth_2 > 0)
        info["mask"] = mask

        # Triangulation angle
        cc1 = P1[:3, :3].T @ -P1[:3, 3]  # camera center
        cc2 = P2[:3, :3].T @ -P2[:3, 3]  # camera center

        points = points[:, :3]
        ray1 = points[:, :3] - cc1  # vector from camera center 1 to point
        ray2 = points[:, :3] - cc2  # vector from camera center 2 to point
        ray1_norm = ray1 / np.linalg.norm(ray1, axis=1)[:, None]
        ray2_norm = ray2 / np.linalg.norm(ray2, axis=1)[:, None]
        dot_product = np.sum(ray1_norm * ray2_norm, axis=1)
        angle = np.arccos(
            dot_product / (np.linalg.norm(ray1_norm, axis=1) * np.linalg.norm(ray2_norm, axis=1))
        )
        angle = np.rad2deg(angle)
        print("Mean & Median Triangulation angle: ", np.mean(angle[mask]), np.median(angle[mask]))

        if min_triangulation_angle > 0:
            mask = mask & (angle > min_triangulation_angle)
            info["mask"] = mask

        # Filter points based on reprojection error
        if max_reprojection_error > 0:
            # Project 3D points back to both images
            points_homo = np.hstack((points, np.ones((points.shape[0], 1))))

            # Project to first image
            proj_points1 = (self.v1.K @ (P1 @ points_homo.T)).T
            proj_points1 = proj_points1[:, :2] / proj_points1[:, 2:3]

            # Project to second image
            proj_points2 = (self.v2.K @ (P2 @ points_homo.T)).T
            proj_points2 = proj_points2[:, :2] / proj_points2[:, 2:3]

            # Calculate reprojection errors in pixel coordinates
            error1 = np.linalg.norm(pts1 - proj_points1.numpy(), axis=1)
            error2 = np.linalg.norm(pts2 - proj_points2.numpy(), axis=1)

            # Combined error (average of both views)
            reproj_error = (error1 + error2) / 2

            # Filter points with high reprojection error
            reproj_mask = reproj_error < max_reprojection_error

            # Update the mask to include both cheirality and reprojection error checks
            mask = mask & reproj_mask
            info["mask"] = mask

            # Print statistics
            if verbose:
                print(f"Mean reprojection error: {np.mean(reproj_error):.2f} pixels")
                print(f"Max reprojection error: {np.max(reproj_error):.2f} pixels")
                print(f"Points filtered by reprojection error: {np.sum(~reproj_mask)}")
                print(f"Total points after filtering: {np.sum(mask)}")

        return dict(
            points=points,
            **info,
            R1=R1,
            t1=t1,
            R2=R2,
            t2=t2,
            mean_triangulation_angle=np.mean(angle),
            median_triangulation_angle=np.median(angle),
        )

    def _serialize_matches(self, path="tmp.npy"):
        match_list = {
            "query_idx": [m.query_idx for m in self.matches],
            "database_idx": [m.database_idx for m in self.matches],
        }
        np.save(path, match_list, allow_pickle=True)

    def _deserialize_matches(self, path="tmp.npy"):
        match_list = np.load(path, allow_pickle=True).item()
        self.matches = [
            Match(
                query_idx=match_list["query_idx"][i],
                database_idx=match_list["database_idx"][i],
            )
            for i in range(len(match_list["query_idx"]))
        ]

    def draw_twoview_reconstruction(self, show_gt=False, map_into_gt_space=False, **kwargs):
        info = self.estimate_relative_pose(**kwargs)
        r1, r2 = roma.rotmat_to_rotvec(info["R1"]), roma.rotmat_to_rotvec(
            torch.from_numpy(info["R2"])
        )
        rotations = torch.stack((r1, r2))
        translations = torch.stack((info["t1"].squeeze(), torch.from_numpy(info["t2"].squeeze())))
        points = torch.tensor(info["points"])
        if show_gt and self.scenegraph:
            rotations_gt = self.scenegraph.rotations_gt
            translations_gt = self.scenegraph.translations_gt
            points_3d_gt = self.scenegraph.points_3d_gt

            return plot_3d(
                points,
                rotations,
                translations,
                self.v1.K,
                points_3d_gt,
                rotations_gt,
                translations_gt,
            )
        return plot_3d(points, rotations, translations, self.v1.K)

    def draw_3d_points(self, with_camera=False):
        if with_camera:
            pass
            # get global camera positions
            np.stack()

            visualize_cameras_and_points(self.points_3d[:3].T)
        else:
            visualize_points_3d(self.points_3d[:3].T)

    def _extract_points_from_matches(self):
        """Extracts the 2d locations of the points form the matches."""
        pts1 = list()
        pts2 = list()
        for match in self.matches:
            pts1.append(self.v1.kp[match.query_idx].pt)
            pts2.append(self.v2.kp[match.database_idx].pt)
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)
        return pts1, pts2

    def _extract_points_from_keypoints(self, kp1_idxs, kp2_idxs):
        """Extracts the 2d location of the points from the keypoint idxs for each image."""
        pts1 = list()
        pts2 = list()
        for kp1, kp2 in zip(kp1_idxs, kp2_idxs):
            pts1.append(self.v1.kp[kp1].pt)
            pts2.append(self.v2.kp[kp2].pt)
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)
        return pts1, pts2

    def relative_to_global(self, R_rel, t_rel, use_gt=False, **kwargs):
        """Takes a relative rotation and translation and aligns it into the "global" coordinate
        frame."""
        # R = R2 @ R1.T
        R1 = self.v1.camera_gt.R if use_gt else self.base_R[:3, :3]
        t1 = self.v1.camera_gt.t if use_gt else self.base_t
        R1 = R1.numpy() if isinstance(R1, torch.Tensor) else R1
        t1 = t1.numpy() if isinstance(t1, torch.Tensor) else t1
        R_new = R_rel @ R1
        t_new = R_rel @ t1 + t_rel
        # t_new = R_new @ -(self.v1.camera_gt.camera_center + t)
        return dict(R1=R1, R2=R_new, t1=t1, t2=t_new)

    def create_dicts_from_matches(self):
        self.query_to_database = {m.query_idx: m.database_idx for m in self.matches}
        self.database_to_query = {m.database_idx: m.query_idx for m in self.matches}

    # ---------------------------- GETTERS/SETTERS ----------------------------
    @property
    def n_matches(self):
        return len(self.matches)

    # ---------------------------- VISUALIZATION ----------------------------
    def draw_matches(self, first_n=np.inf, title=""):
        first_n = min(first_n, len(self.matches))
        return plot_matches(
            self.v1.img, self.v2.img, self.v1.kp, self.v2.kp, self.matches[:first_n]
        )

    def draw_match_by_idx(self, idx0, title=""):
        return plot_matches(
            self.v1.img,
            self.v2.img,
            self.v1.kp,
            self.v2.kp,
            [self.matches[idx0]],
            linewidth=1.0,
        )

    def draw_match_by_idxs(self, idxs: List[int]):
        return plot_matches(
            self.v1.img,
            self.v2.img,
            self.v1.kp,
            self.v2.kp,
            [self.matches[idx] for idx in idxs],
            linewidth=1.0,
        )
