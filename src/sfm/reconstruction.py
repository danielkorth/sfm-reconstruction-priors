import itertools
from collections import defaultdict
from typing import List

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import roma
import torch
from omegaconf import DictConfig, open_dict

import utils.visualization as viz
import wandb
from opt.residuals_func import dust3r_pointmaps_align, project
from sfm.scenegraph import SceneGraph
from sfm.track import TrackState
from utils.dust3r import transform_coordinates, transform_coordinates_resize_crop
from utils.metrics import (
    average_trajectory_error,
    combined_auc,
    combined_maa,
    mean_average_accuracy,
    relative_accuracy,
    relative_rotation_auc,
    relative_rotation_error,
    relative_translation_auc,
    relative_translation_error,
)
from utils.rerun import connect_server, setup_blueprint

RERUN_KWARGS = {
    "points": {"colors": [0x8839EFFF], "radii": 0.01},
    "pointmaps": {"colors": [0xFE640BFF], "radii": 0.005},
    "points_pointmaps_strips": {"colors": [0x00FF00FF], "radii": 0.0005},
    "reference_mesh": {"radii": 0.001},
    "background": [239, 241, 245],
}


class Reconstruction:
    """The Reconstruction class maintains and updates the scenegraph to incrementally have a
    reconstruction of the scene."""

    def __init__(
        self,
        active_view_idxs: List[int],
        active_track_idxs: List[int],
        scenegraph: SceneGraph,
        cfg: DictConfig = None,
    ):
        self.active_view_idxs = active_view_idxs
        self.active_track_idxs = active_track_idxs
        self.scenegraph = scenegraph
        self.cfg = cfg
        print(
            f"Initialized Reconstruction with {len(active_view_idxs)} active views and {len(active_track_idxs)} active tracks."
        )
        self.rec_cycle = 0
        self.rec_steps = 0

        if self.cfg is not None:
            self._setup()

    def _setup(self):
        if "similar_scale" in self.cfg.global_optimization.cost.desc:
            with open_dict(self.cfg.global_optimization.cost):
                self.cfg.global_optimization.cost.dust3r_scale = (
                    self.scenegraph.views[0].K[0][0].item()
                    * self.cfg.global_optimization.cost.dust3r_scale
                )

    # INITIALIZATION & ENTRY POINT
    @staticmethod
    def init_from_twoview(scenegraph: SceneGraph, cfg: DictConfig = None):
        print("Initializing from two-view reconstruction...")
        # TODO currently, this is a naive reconstruction that does not use all available tracks from these two views, since it only considers
        # the relationship locally. However, it can happen that there are even more tracks that could be triangulated from these two views, because
        # eg 1-2 and 3-4 and 4-1 are connected, and we reconstructe 2-3 (which are connected, but initially not via a direct relationship)

        max_overlap = -1
        best_pair = (-1, -1)
        pairs = defaultdict(int)

        if cfg.initialization.criterion == "matches":
            # considers 2-view relationship
            for i in range(1, len(scenegraph.views)):
                for j in range(i):
                    pairs[(j, i)] = scenegraph.pairs_mat[j][i].n_matches
        elif cfg.initialization.criterion == "tracks":
            # find best pair based on overlap in tracks
            for track in scenegraph.tracks:
                views_in_track = [x[0] for x in track.views]
                for two_view_comb in itertools.combinations(views_in_track, 2):
                    pairs[(min(two_view_comb), max(two_view_comb))] += 1
        elif cfg.initialization.criterion == "tracksgt2":
            # find best pair based on overlap in tracks
            for track in scenegraph.tracks:
                if len(track.views) > 2:
                    views_in_track = [x[0] for x in track.views]
                    for two_view_comb in itertools.combinations(views_in_track, 2):
                        pairs[(min(two_view_comb), max(two_view_comb))] += 1
        else:
            raise NotImplementedError(
                "initial criterion for two view reconstruction not implemented."
            )

        # Sort pairs by the number of matches
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)

        active_view_idxs = []
        active_track_idxs = []

        min_median_triangulation_angle = cfg.initialization.min_median_triangulation_angle
        min_inlier_count = cfg.initialization.min_inlier_count

        success = False
        for _ in range(cfg.initialization.max_tries):
            for best_pair in sorted_pairs:
                # best_pair = [(13, 14), 31]
                view_1_idx, view_2_idx = best_pair[0]
                print(f"Trying best pair: {best_pair[0]} with {best_pair[1]} matches.")

                # estimate the relative pose from it
                two_views = scenegraph.pairs_mat[view_1_idx][view_2_idx]

                # get all tracks that the two views share
                kp1_idxs = list()
                kp2_idxs = list()
                tracks = list()

                # using all tracks for initial two-view reconstruction
                for track in scenegraph.view_to_tracks[view_1_idx]:
                    # check if the other view also sees that track
                    if track.contains(view_2_idx):
                        # extract keypoint info for both views
                        kp1_idx = track.find_feature_for_view(view_1_idx)
                        kp2_idx = track.find_feature_for_view(view_2_idx)

                        kp1_idxs.append(kp1_idx)
                        kp2_idxs.append(kp2_idx)
                        tracks.append(track)

                if len(tracks) <= min_inlier_count:
                    continue

                info = two_views.reconstruct_two_views(
                    kp1_idxs=kp1_idxs,
                    kp2_idxs=kp2_idxs,
                    use_tracks=cfg.initialization.use_tracks,
                    max_reprojection_error=cfg.initialization.max_reprojection_error,
                    min_triangulation_angle=cfg.initialization.min_triangulation_angle,
                )

                median_triangulation_angle = info["median_triangulation_angle"]

                n_inliers = info["mask"].sum()

                if (
                    median_triangulation_angle >= min_median_triangulation_angle
                    and n_inliers > min_inlier_count
                ):
                    print(
                        "Two View Reconstruction successful! Triangulation angle: ",
                        median_triangulation_angle,
                        "with ",
                        n_inliers,
                        "inliers",
                    )
                    success = True
                    break

            if success:
                break
            else:
                print("Trying again with lower thresholds...")
                min_median_triangulation_angle /= 2
                min_inlier_count /= 2

        v1 = two_views.v1
        v2 = two_views.v2
        R_t1 = np.hstack((info["R1"], info["t1"]))
        R_t2 = np.hstack((info["R2"], info["t2"]))
        v1.add_camera(K=info["K"], R_t=R_t1, kind="opt")
        v2.add_camera(K=info["K"], R_t=R_t2, kind="opt")

        # initialize the views and tracks
        active_view_idxs = [view_1_idx, view_2_idx]
        active_track_idxs = list()

        for track, mask, point in zip(tracks, info["mask"], info["points"]):
            if mask:
                track.state = TrackState.ACTIVE
                track.point = point

                track.set_view_active(view_1_idx)
                track.set_view_active(view_2_idx)

                active_track_idxs.append(scenegraph.tracks.index(track))
            elif cfg.initialization.delete_failed_tracks:
                # track will not be considered in the future, since it did not pass the geometric verification/chirality condition of this stage
                track.state = TrackState.REMOVED

        # activate other tracks that are however only seen by a single view
        for track in scenegraph.view_to_tracks[view_1_idx]:
            track.set_view_active(view_1_idx)
            if track.state == TrackState.INACTIVE:
                track.state = TrackState.OBSERVED

        for track in scenegraph.view_to_tracks[view_2_idx]:
            track.set_view_active(view_2_idx)
            if track.state == TrackState.INACTIVE:
                track.state = TrackState.OBSERVED

        print(f"Reconstructed views: {view_1_idx}, {view_2_idx}")

        print(f"Initialized {len(tracks)} tracks from the two views.")

        return Reconstruction(
            active_view_idxs=active_view_idxs,
            active_track_idxs=active_track_idxs,
            scenegraph=scenegraph,
            cfg=cfg,
        )

    def incremental_reconstruction(self):
        print("--- Starting incremental reconstruction ---")
        i = 0

        info = {}

        while True:
            print("--- Adjusting bundles ---")
            info_global_opt = self._adjust_bundles()

            self.rec_cycle += 1

            print("--- Adding a new view ---")
            if len(self.active_view_idxs) < len(self.scenegraph.views):
                add_view_result = self._add_view()
                if not add_view_result["success"]:
                    print(add_view_result["message"])
                    break
            else:
                print("Reconstruction Completed!")
                break

            i += 1

        return info

    # ADDING VIEWS
    def _add_view(self):
        print("--- Searching for next view ---")
        info_next_view = self._find_best_next_view()
        if not info_next_view["success"]:
            print("No suitable view found to add.")
            return dict(success=False, message=info_next_view["message"])

        ranking = sorted(
            [(k, len(v)) for (k, v) in info_next_view["view_candidates"].items()],
            key=lambda x: -x[1],
        )
        print(f"Next view candidates ranked: {ranking}")

        for next_view_idx, _ in ranking:
            print(f"Attempting to register view {next_view_idx}...")
            visible_tracks = info_next_view["view_candidates"][next_view_idx]
            info_registration = self._register_view(next_view_idx, visible_tracks)
            if info_registration["success"]:
                print(f"View {next_view_idx} registered successfully.")
                print("--- Triangulating new tracks ---")
                if self.cfg.triangulation.triangulation_method == "multiview":
                    _ = self._triangulate_new_tracks_dlt(next_view_idx)
                elif self.cfg.triangulation.triangulation_method == "pairwise":
                    _ = self._triangulate_new_tracks(next_view_idx)
                return dict(success=True, message="View added successfully")
            else:
                print(f"Registration of view {next_view_idx} failed. Trying next candidate.")

        # sanity check
        for track_idx in self.active_track_idxs:
            if self.scenegraph.tracks[track_idx].state != TrackState.ACTIVE:
                print("ALERT: Track is not active after adding a new view.")
                print(self.scenegraph.tracks[track_idx].state)
            # assert self.scenegraph.tracks[track_idx].state == TrackState.ACTIVE
        return dict(success=False, message="Failed to register any candidate views")

    # IMAGE REGISTRATION
    def _find_best_next_view(self):
        # find the next best view based on the number of **visible** points
        if self.cfg.image_registration.next_best_view_criterion == "tracks":
            view_candidates = defaultdict(list)

            for track_idx in self.active_track_idxs:
                # only consider active tracks
                if self.scenegraph.tracks[track_idx].state == TrackState.ACTIVE:
                    inactive_views = self.scenegraph.tracks[track_idx].inactive_views
                    for inactive_view in inactive_views:
                        # consider only tracks that have not been removed beforehand
                        view_candidates[inactive_view].append(track_idx)

            if len(view_candidates) == 0:
                return dict(
                    success=False,
                    message="All views of the scenegraph have been added to the reconstruction.",
                )

            best_view = max(view_candidates, key=lambda x: view_candidates.get(x).__len__())

            print(
                f"Found Next Best View: {best_view} with {len(view_candidates[best_view])} existing tracks in the scene."
            )
            return dict(
                success=True,
                best_view_idx=best_view,
                visible_tracks=view_candidates[best_view],
                view_candidates=view_candidates,
            )
        else:
            raise NotImplementedError("Next Best View criterion not implemented")

    def _register_view(self, view_idx, visible_tracks, update_scenegraph=True):
        points_2d = list()
        points_3d = list()
        for track_idx in visible_tracks:
            track = self.scenegraph.tracks[track_idx]
            points_3d.append(track.point)

            feature = track.find_feature_for_view(view_idx)
            points_2d.append(self.scenegraph.views[view_idx].kp[feature].pt)

        points_2d = np.array(points_2d).astype(np.float32)
        points_3d = np.array(points_3d).astype(np.float32)

        try:
            info = cv2.solvePnPRansac(
                objectPoints=points_3d,
                imagePoints=points_2d,
                cameraMatrix=self.scenegraph.views[0].K.numpy(),
                distCoeffs=None,
                useExtrinsicGuess=False,
                iterationsCount=self.cfg.image_registration.pnp.iterations_count,
                reprojectionError=self.cfg.image_registration.pnp.reprojection_error,  # in pixels
                confidence=self.cfg.image_registration.pnp.confidence,
                flags=cv2.SOLVEPNP_ITERATIVE,  # uses LM
            )
        except cv2.error as e:
            print(f"OpenCV error during solvePnPRansac: {str(e)}")
            return dict(success=False, message=f"OpenCV error during solvePnPRansac: {str(e)}")

        info_status = ["success", "rvec", "tvec", "inliers"]
        info = {st: x for st, x in zip(info_status, info)}
        # TODO: check if inliers are empty

        if not info["success"]:
            print(f"Failed to register view {view_idx}.")
            return dict(success=False, message="PnPRansac failed to find a valid solution.")

        print(f"View {view_idx} registered successfully with {len(info['inliers'])} inliers.")

        if update_scenegraph:
            print(
                f"PnPRansac success status: {info['success']}. Identified {len(info['inliers'])} inlier 2d-3d correspondences."
            )
            # add camera parameters
            rotation = torch.from_numpy(info["rvec"]).squeeze()  # rotvec
            translation = torch.from_numpy(info["tvec"]).squeeze()  # trnsvec
            self.scenegraph.views[view_idx].add_camera(
                K=self.scenegraph.views[0].K,
                rotation=rotation,
                translation=translation,
                kind="opt",
            )

            # update active views list
            self.active_view_idxs.append(view_idx)

            for track in self.scenegraph.view_to_tracks[view_idx]:
                track.set_view_active(view_idx)
                if track.state == TrackState.INACTIVE:
                    track.state = TrackState.OBSERVED
                # if it was active before, it stays active

            if self.cfg.image_registration.pnp.remove_outliers:
                print("Removing outliers...")
                # remove tracks from reconstruction that were deemed outliers via PnP.
                idx = 0
                for i in range(len(points_3d)):
                    if idx < len(info["inliers"]) and i == info["inliers"][idx]:
                        # inlier
                        idx += 1
                    else:
                        # outlier
                        invalid_track_idx = visible_tracks[i]
                        invalid_track = self.scenegraph.tracks[invalid_track_idx]
                        invalid_track.state = TrackState.REMOVED
                        # remove track from reconstruciton
                        self.active_track_idxs.remove(invalid_track_idx)

            elif self.cfg.image_registration.pnp.deactivate_outliers:
                print(
                    f"Deactivating outliers... with a reprojection error of {self.cfg.image_registration.pnp.reprojection_error} px"
                )
                # remove tracks from reconstruction that were deemed outliers via PnP.
                idx = 0
                for i in range(len(points_3d)):
                    if idx < len(info["inliers"]) and i == info["inliers"][idx]:
                        # inlier
                        idx += 1
                    else:
                        # outlier
                        invalid_track_idx = visible_tracks[i]
                        invalid_track = self.scenegraph.tracks[invalid_track_idx]
                        invalid_track.state = TrackState.OBSERVED
                        # remove track from reconstruciton
                        self.active_track_idxs.remove(invalid_track_idx)

        # TODO careful, points_3d might not be more than all the points that were triangulated
        return dict(
            message="View registered successfully.",
            **info,
            points_3d=points_3d,
            points_2d=points_2d,
        )

    # TRIANGULATION
    def _triangulate_new_tracks(self, view_idx, P1=None, update_scenegraph=True):
        print(f"Triangulating new tracks for view {view_idx}...")
        # after a view is added, it is possible that we now add new points to the scene (because we have new tracks)
        # search the new tracks, and triangulate them!
        # it is guaranteed that the new triangulation
        # find all tracks that the view is part of
        tracks_to_be_triangulated = defaultdict(list)

        for track in self.scenegraph.view_to_tracks[view_idx]:
            # only those in the observed state are interesting to be triangulated in the future
            if (track.state == TrackState.OBSERVED) and (
                len(track.views_observing) == 2
            ):  # should always be 1 or 2
                # TODO this fails sometimes. Why?
                assert len(track.views_observing) in [1, 2]

                v1, v2 = list(track.views_observing)
                kp1 = track.find_feature_for_view(v1)
                kp2 = track.find_feature_for_view(v2)

                # make sure that the first element in the dict is always the view that is added
                if v2 == view_idx:
                    v1, v2 = v2, v1
                    kp1, kp2 = kp2, kp1

                tracks_to_be_triangulated[(v1, v2)].append((kp1, kp2))

        if P1 is None:
            P1 = self.scenegraph.views[view_idx].camera.P.numpy()

        new_triangulated_points = list()
        new_triangulated_mask = list()  # chirality mask

        for pair, feature_matches in tracks_to_be_triangulated.items():
            print(
                f"Triangulating tracks between views {pair} with {len(feature_matches)} matches."
            )
            P2 = self.scenegraph.views[pair[1]].camera.P.numpy()

            # get the points
            pts1 = list()
            pts2 = list()
            for ft1, ft2 in feature_matches:
                pts1.append(self.scenegraph.views[pair[0]].kp[ft1].pt)
                pts2.append(self.scenegraph.views[pair[1]].kp[ft2].pt)

            pts1 = np.array(pts1).astype(np.float32)
            pts2 = np.array(pts2).astype(np.float32)

            pts1_normalized = cv2.undistortPoints(
                pts1.reshape(-1, 1, 2), np.array(self.scenegraph.views[pair[0]].K), distCoeffs=None
            )
            pts2_normalized = cv2.undistortPoints(
                pts2.reshape(-1, 1, 2), np.array(self.scenegraph.views[pair[1]].K), distCoeffs=None
            )

            points = cv2.triangulatePoints(P1, P2, pts1_normalized, pts2_normalized).T
            # make homogeneous
            points /= points[:, -1][:, None]

            # Check chirality constraints
            points_in_cam1 = P1 @ points.T
            points_in_cam2 = P2 @ points.T

            # Check if the point is in front of both cameras
            cheirality_mask = (points_in_cam1[2] > 0) & (points_in_cam2[2] > 0)

            # Reprojection error
            points_in_cam1 = self.scenegraph.views[pair[0]].K @ points_in_cam1
            points_in_cam2 = self.scenegraph.views[pair[1]].K @ points_in_cam2
            points_in_cam1 = points_in_cam1 / points_in_cam1[2]
            points_in_cam2 = points_in_cam2 / points_in_cam2[2]
            points_in_cam1 = points_in_cam1.T[:, :2]
            points_in_cam2 = points_in_cam2.T[:, :2]
            reprojection_error_cam1 = np.sqrt(
                np.power(pts1 - points_in_cam1.numpy(), 2).sum(axis=1)
            )
            reprojection_error_cam2 = np.sqrt(
                np.power(pts2 - points_in_cam2.numpy(), 2).sum(axis=1)
            )
            reprojection_error_mean = np.stack(
                [reprojection_error_cam1, reprojection_error_cam2]
            ).mean(axis=0)
            reprojection_error_mask = (
                reprojection_error_mean < self.cfg.triangulation.max_mean_reprojection_error
            )

            # Triangulation angle
            cc1 = P1[:3, :3] @ -P1[:3, 3]  # camera center
            cc2 = P2[:3, :3] @ -P2[:3, 3]  # camera center
            t1 = cc1 - points[:, :3]
            t2 = cc2 - points[:, :3]
            angle = np.arccos(
                np.diag(t1 @ t2.T) / (np.linalg.norm(t1, axis=1) * np.linalg.norm(t2, axis=1))
            )
            angle = np.rad2deg(angle)
            tri_angle_mask = angle > self.cfg.triangulation.min_median_triangulation_angle
            valid_mask = cheirality_mask & reprojection_error_mask & tri_angle_mask

            # Count points filtered by each criterion alone
            only_chirality = sum(~cheirality_mask & reprojection_error_mask & tri_angle_mask)
            only_reprojection = sum(cheirality_mask & ~reprojection_error_mask & tri_angle_mask)
            only_angle = sum(cheirality_mask & reprojection_error_mask & ~tri_angle_mask)

            # Count points filtered by multiple criteria
            multiple_criteria = sum(~valid_mask) - only_chirality - only_reprojection - only_angle

            print(f"Removing {sum(~valid_mask)} tracks in total:")
            print(f" - {only_chirality} tracks removed only due to chirality")
            print(
                f" - {only_reprojection} tracks removed only due to high ({self.cfg.triangulation.max_mean_reprojection_error} px) reprojection error"
            )
            print(
                f" - {only_angle} tracks removed only due to low ({self.cfg.triangulation.min_median_triangulation_angle} deg) triangulation angle"
            )
            print(f" - {multiple_criteria} tracks removed due to multiple criteria")

            points = points[:, :3]

            new_triangulated_points.append(points)
            new_triangulated_mask.append(valid_mask)

            if update_scenegraph:
                # add new tracks
                for (ft1, ft2), point, mask in zip(feature_matches, points, valid_mask):
                    if mask:
                        assert (
                            self.scenegraph.feature_to_track[(pair[0], ft1)]
                            == self.scenegraph.feature_to_track[(pair[1], ft2)]
                        )
                        track = self.scenegraph.feature_to_track[(pair[0], ft1)]
                        track.state = TrackState.ACTIVE
                        track.point = point

                        self.active_track_idxs.append(self.scenegraph.tracks.index(track))
                        track.set_view_active(pair[0])
                        track.set_view_active(pair[1])
                    elif self.cfg.triangulation.remove_failed_from_reconstruction:
                        track = self.scenegraph.feature_to_track[(pair[0], ft1)]
                        track.state = TrackState.REMOVED
                    elif (
                        self.cfg.triangulation.remove_max_reprojection_failed_from_reconstruction
                        and not reprojection_error_mask
                    ):
                        track = self.scenegraph.feature_to_track[(pair[0], ft1)]
                        track.state = TrackState.REMOVED
                    else:
                        pass

        points = np.concatenate(new_triangulated_points)
        mask = np.concatenate(new_triangulated_mask)

        print(
            f"Triangulation completed. Out of {points.shape[0]} possible points, {sum(mask)} passed all filters."
        )

        return dict(points=points, chirality_mask=mask)

    def _triangulate_new_tracks_dlt(self, view_idx, P1=None, update_scenegraph=True):
        """Triangulates new tracks for a given view.

        uses DLT and checks for cheirality, reprojection error and triangulation angle.
        reference: Hartley & Zisserman, Multiple View Geometry, 2003, p. 312
        reference #2: https://amytabb.com/tips/tutorials/2021/10/31/triangulation-DLT-2-3/
        """
        print(f"Triangulating new tracks for view {view_idx}...")
        tracks_to_triangulate = []

        # Collect tracks with >=2 observations that are observed
        for track in self.scenegraph.view_to_tracks[view_idx]:
            if track.state == TrackState.OBSERVED and len(track.views_observing) >= 2:
                tracks_to_triangulate.append(track)

        new_triangulated_points = []
        new_triangulated_mask = []

        K = self.scenegraph.views[view_idx].K

        # Counters for failure tracking
        only_chirality = 0
        only_reprojection = 0
        only_angle = 0
        multiple_criteria = 0
        total_tracks = len(tracks_to_triangulate)

        for track in tracks_to_triangulate:
            # Collect all observations for this track
            view_indices = list(track.views_observing)
            kp_indices = [track.find_feature_for_view(v) for v in view_indices]

            # Get normalized points and projection matrices
            points_normalized = []
            points_unnormalized = []
            projection_matrices = []
            for v, kp in zip(view_indices, kp_indices):
                # Get 2D point and normalize
                pt = np.array(self.scenegraph.views[v].kp[kp].pt, dtype=np.float32)
                points_unnormalized.append(pt)
                pt_norm = cv2.undistortPoints(
                    pt.reshape(-1, 1, 2), self.scenegraph.views[v].K.numpy(), None
                ).reshape(-1, 2)

                # Get projection matrix [R|t] (without K since points are normalized)
                R = self.scenegraph.views[v].camera.R.numpy()
                t = self.scenegraph.views[v].camera.t.numpy()
                P = np.hstack([R, t.reshape(-1, 1)])

                points_normalized.append(pt_norm)
                projection_matrices.append(P)

            # Multi-view triangulation using DLT
            A = []
            for P, kp in zip(projection_matrices, points_normalized):
                x, y = kp[0][0], kp[0][1]
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])

            A = np.array(A).reshape(-1, 4)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]  # Last row of Vt
            X /= X[3]  # Convert to inhomogeneous coordinates
            point_3d = X[:3]

            # Validation checks - track individual failure reasons
            chirality_valid = True
            reprojection_valid = True
            angle_valid = True

            # 1. Cheirality check (positive depth)
            for P in projection_matrices:
                P = np.vstack([P, [0, 0, 0, 1]])  # Make homogeneous
                point_cam = P @ np.hstack([point_3d, 1])
                if point_cam[2] <= 0:
                    chirality_valid = False
                    break

            # 2. Reprojection error check
            reproj_errors = []
            for P, kp in zip(projection_matrices, points_unnormalized):
                proj = P @ np.hstack([point_3d, 1])
                proj = K @ proj
                proj = proj[:2] / proj[2]
                error = (proj - kp).norm()
                reproj_errors.append(error)

            if np.mean(reproj_errors) > self.cfg.triangulation.max_mean_reprojection_error:
                reprojection_valid = False

            # 3. Triangulation angle check
            # Get camera centers
            cam_centers = [
                self.scenegraph.views[v].camera.camera_center.numpy() for v in view_indices
            ]
            angles = []
            for i, j in itertools.combinations(range(len(cam_centers)), 2):
                vec1 = point_3d - cam_centers[i]
                vec2 = point_3d - cam_centers[j]
                angle = np.arccos(
                    np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                )
                angles.append(np.rad2deg(angle))

            if np.median(angles) < self.cfg.triangulation.min_median_triangulation_angle:
                angle_valid = False

            # Determine overall validity and count failure types
            valid = chirality_valid and reprojection_valid and angle_valid

            if not valid:
                # Count individual failure reasons
                if not chirality_valid and reprojection_valid and angle_valid:
                    only_chirality += 1
                elif chirality_valid and not reprojection_valid and angle_valid:
                    only_reprojection += 1
                elif chirality_valid and reprojection_valid and not angle_valid:
                    only_angle += 1
                else:
                    multiple_criteria += 1

            # Store results
            new_triangulated_points.append(point_3d)
            new_triangulated_mask.append(valid)

            # Update track state
            if update_scenegraph:
                if valid:
                    track.state = TrackState.ACTIVE
                    track.point = point_3d
                    self.active_track_idxs.append(self.scenegraph.tracks.index(track))
                    for v in view_indices:
                        track.set_view_active(v)
                elif self.cfg.triangulation.remove_failed_from_reconstruction:
                    track.state = TrackState.REMOVED

        # Convert results to numpy arrays
        points = np.array(new_triangulated_points)
        mask = np.array(new_triangulated_mask)

        # Print detailed failure statistics
        total_failed = only_chirality + only_reprojection + only_angle + multiple_criteria
        print(f"Removing {total_failed} tracks in total:")
        print(f" - {only_chirality} tracks removed only due to chirality")
        print(
            f" - {only_reprojection} tracks removed only due to high ({self.cfg.triangulation.max_mean_reprojection_error} px) reprojection error"
        )
        print(
            f" - {only_angle} tracks removed only due to low ({self.cfg.triangulation.min_median_triangulation_angle} deg) triangulation angle"
        )
        print(f" - {multiple_criteria} tracks removed due to multiple criteria")
        print(
            f"Triangulation completed. Out of {total_tracks} possible points, {sum(mask)} passed all filters."
        )

        return dict(points=points, chirality_mask=mask)

    def _retriangulate_tracks(self):
        """It is possible that drift accumulates over time.

        therefore, we sometimes want to perform a new multi-view triangulation before running
        bundle adjustment
        """
        pass

    # RERUN logging
    def _setup_rerun_tracking(self, cycle):
        """Set up rerun time sequences for the current optimization cycle."""
        if self.cfg.rerun_log_rec:
            print(f"Setting time sequence for cycle {cycle}...")
            rr.set_time_sequence("/rec_cycle", cycle)

    def _log_to_rerun_pre_optimization(self, params, info_data):
        """Log initial state to rerun before optimization begins.

        Args:
            params: The current optimization parameters
            info_data: Dictionary containing optimization data
        """
        if not self.cfg.rerun_log_rec:
            return

        # Set time sequences
        rr.set_time_sequence("/rec_steps", self.rec_steps)
        rr.set_time_sequence(f"/cycle_{self.rec_cycle:03d}/ba_step", 0)

        # Log the reference mesh if it hasn't been logged yet
        if not hasattr(self, "_logged_mesh"):
            self._log_mesh_to_rerun()
            self._logged_mesh = True

        # Extract parameters
        self._log_cameras_and_points_to_rerun(params, info_data)

    def _log_to_rerun_during_optimization(self, params, info_data, step):
        """Log current state to rerun during an optimization step.

        Args:
            params: The current optimization parameters
            info_data: Dictionary containing optimization data
            step: Current optimization step number
        """
        if not (self.cfg.rerun_log_rec and self.cfg.rerun_log_opt_steps):
            return

        # Set time sequences
        rr.set_time_sequence("/rec_steps", self.rec_steps)
        rr.set_time_sequence(f"/cycle_{self.rec_cycle:03d}/ba_step", step + 1)

        # Extract parameters and log to rerun
        self._log_cameras_and_points_to_rerun(params, info_data)

    def _log_to_rerun_post_optimization(self, params, info_data):
        """Log final state to rerun after optimization completes.

        Args:
            params: The final optimization parameters
            info_data: Dictionary containing optimization data
        """
        if not self.cfg.rerun_log_rec:
            return

        # Set time sequences
        rr.set_time_sequence("/rec_steps", self.rec_steps)
        rr.set_time_sequence(
            f"/cycle_{self.rec_cycle:03d}/ba_step",
            self.cfg.global_optimization.optimizer.max_iterations,
        )

        # Extract parameters and log to rerun
        self._log_cameras_and_points_to_rerun(params, info_data)

        self._logged_camera_params = False
        # Disable the timeline for this cycle
        rr.disable_timeline(f"/cycle_{self.rec_cycle:03d}")

    def _log_mesh_to_rerun(self, entity_path: str = "/reference_mesh"):
        """Log a mesh to rerun for visualization.

        Args:
            mesh_path: Path to the PLY mesh file
            entity_path: Path where the mesh should be logged in rerun (default: "/reference_mesh")
        """

        mesh_path = str(self.scenegraph.scene.scans_dir / "mesh_aligned_0.05.ply")
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        points = np.asarray(mesh.vertices)
        colors = (
            np.asarray(mesh.vertex_colors)
            if mesh.has_vertex_colors()
            else np.ones_like(points) * 0.7
        )

        # Log the mesh as points with colors
        rr.log(
            entity_path,
            rr.Points3D(positions=points, colors=colors, **RERUN_KWARGS["reference_mesh"]),
        )

    def _log_cameras_and_points_to_rerun(self, params, info_data):
        """Extract parameters and log cameras, points, and pointmaps to rerun.

        Args:
            params: The current optimization parameters
            info_data: Dictionary containing optimization data
        """
        # Extract camera parameters
        params_cameras = params[: info_data["n_cameras"] * 6]
        params_tracks = params[
            info_data["n_cameras"] * 6 : info_data["n_cameras"] * 6 + info_data["n_points"] * 3
        ]

        rotations = params_cameras.reshape(-1, 6)[:, :3]
        translations = params_cameras.reshape(-1, 6)[:, 3:]

        point_params = params_tracks.reshape(-1, 3)
        rotations = rotations.reshape(-1, 3)
        translations = translations.reshape(-1, 3)

        # LOG CAMERAS
        for k, (translation, rotation, view_idx) in enumerate(
            zip(translations, rotations, self.active_view_idxs)
        ):
            rr.log(
                f"camera_{k:03d}",
                rr.Transform3D(
                    translation=translation.cpu().numpy(),
                    rotation=rr.RotationAxisAngle(
                        axis=rotation.cpu().numpy(), radians=torch.norm(rotation).cpu().numpy()
                    ),
                    from_parent=True,
                    scale=1,
                ),
            )

            # Set flag to avoid redundant logging of static camera parameters
            if not hasattr(self, "_logged_camera_params"):
                self._logged_camera_params = False

            # Only log camera parameters and images for the pre-optimization step
            if not self._logged_camera_params:
                K_numpy = info_data["K"].cpu().numpy()
                rr.log(f"camera_{k:03d}", rr.ViewCoordinates.RDF, static=True)
                rr.log(
                    f"camera_{k:03d}/image",
                    rr.Pinhole(
                        resolution=[K_numpy[0][2] * 2, K_numpy[1][2] * 2],
                        focal_length=[K_numpy[0][0], K_numpy[1][1]],
                        principal_point=[K_numpy[0][2], K_numpy[1][2]],
                        image_plane_distance=0.1,
                    ),
                )

                # LOG IMAGES
                view = self.scenegraph.views[view_idx]
                rr.log(f"camera_{k:03d}/image", rr.Image(view.img))

                # LOG IMAGE KEYPOINTS
                keypoints = list()
                tracks = self.scenegraph.view_to_tracks[view_idx]
                for track in tracks:
                    if track.state == TrackState.ACTIVE:
                        keypoint_idx = track.find_feature_for_view(view_idx)
                        kp = view.kp[keypoint_idx]
                        keypoints.append(kp.pt)
                keypoints = np.stack(keypoints)
                # rr.log(f"camera_{k:03d}/image/keypoints", rr.Points2D(keypoints))

        self._logged_camera_params = True

        # LOG POINTS AND POINTMAPS
        has_aligned_pointmaps = (
            info_data["pose_indices"] is not None and info_data["pose_indices"].shape[0] > 0
        )

        if has_aligned_pointmaps:
            # Log aligned pointmaps if we have pose parameters
            params_poses = params[info_data["n_cameras"] * 6 + info_data["n_points"] * 3 :]
            pointmaps = info_data["pointmaps_dust3r"]
            pose_indices = info_data["pose_indices"]

            aligned_pointmaps = dust3r_pointmaps_align(
                pointmaps, params_poses.reshape(-1, 7)[pose_indices]
            )
            aligned_pointmaps = aligned_pointmaps.cpu().numpy()
            point_params = point_params.cpu().numpy()

            rr.log("/pointmaps", rr.Points3D(aligned_pointmaps, **RERUN_KWARGS["pointmaps"]))

            # Create line strips connecting points to their corresponding pointmaps
            strip = np.stack(
                [
                    np.stack((aligned_pointmaps[pointmap_idx], point_params[track_idx]))
                    for pointmap_idx, track_idx in enumerate(info_data["dust3r_track_indices"])
                ],
                axis=0,
            )

            rr.log(
                "/points_pointmaps_strips",
                rr.LineStrips3D(strip, **RERUN_KWARGS["points_pointmaps_strips"]),
            )
            rr.log("/points", rr.Points3D(point_params, **RERUN_KWARGS["points"]))
        else:
            # Just log the points if we don't have pose parameters
            rr.log("/points", rr.Points3D(point_params.cpu().numpy(), **RERUN_KWARGS["points"]))

    # BUNDLE ADJUSTMENT
    def _adjust_bundles(self, **kwargs):
        device = self.cfg.global_optimization.device
        # define our custom x axis metric
        if wandb.run is not None:
            wandb.define_metric("opt_step")
            # set all other train/ metrics to use this step - doesn't work ffs
            wandb.define_metric("residuals/*", step_metric="opt_step")
            wandb.define_metric("cost/*", step_metric="opt_step")
            wandb.define_metric("metrics/*", step_metric="opt_step")

        # Set up rerun tracking
        self._setup_rerun_tracking(self.rec_cycle)

        # Free unused memory and monitor memory usage
        torch.cuda.empty_cache()

        info_data = self._assemble_data_for_opt()

        torch.save(info_data, f"{self.cfg.path}/info_data_{self.rec_cycle}.pth")

        info = {}

        self._send_to_device_and_adjust_type(info_data, device, torch.float32)

        # noising the params
        # camera params
        # set gt camera param - IMPORTANT. NEED TO REMOVE THIS WHEN OPTIMIZING FOR SIFT/DUST#R stuff
        if self.cfg.global_optimization.get("debug_gt", False):
            if (
                self.cfg.global_optimization.sift_weight == 0
                and self.cfg.global_optimization.dust3r_weight == 0
            ):
                for view_idx in self.active_view_idxs:
                    self.scenegraph.views[view_idx].camera_gt = self.scenegraph.views[
                        view_idx
                    ].camera

            random_rotvecs = roma.random_rotvec(info_data["n_cameras"])
            random_translations = torch.randn(info_data["n_cameras"], 3)
            random_camera_params = torch.cat(
                (random_rotvecs, random_translations), dim=1
            ).flatten()
            random_camera_params[:6] = 0
            info_data["params_cameras"] = info_data[
                "params_cameras"
            ] + random_camera_params * float(self.cfg.global_optimization.camera_noise)

            # point params
            info_data["params_tracks"] = info_data["params_tracks"] + torch.randn_like(
                info_data["params_tracks"]
            ) * float(self.cfg.global_optimization.points_noise)

            # rotation params
            random_rotvecs = roma.random_rotvec(info_data["n_poses"]).flatten()
            random_translations = torch.randn(info_data["n_poses"], 3).flatten()
            random_scales = torch.zeros(info_data["n_poses"])
            random_poses = torch.hstack(
                (random_rotvecs, random_translations, random_scales)
            ).flatten()
            info_data["params_poses"] = info_data["params_poses"] + random_poses * float(
                self.cfg.global_optimization.align_noise
            )

        # Instantiate the optimizer from the config
        optim = hydra.utils.instantiate(self.cfg.global_optimization.optimizer)

        # Determine if we're using sparse or dense tensors
        if hasattr(optim, "tensor_type"):
            sparse = optim.tensor_type == "sparse"
        else:
            sparse = False

        debug_closures = []
        # Initialize cost function and parameters based on the energy type
        if self.cfg.global_optimization.cost.desc == "ba":
            # Use the BACostFunction
            cost_function = hydra.utils.instantiate(
                self.cfg.global_optimization.cost,
                K=info_data["K"],
                n_cameras=info_data["n_cameras"],
                n_points=info_data["n_points"],
                camera_indices=info_data["camera_indices"],
                track_indices=info_data["track_indices"],
                points_2d=info_data["points_2d"]
                if self.cfg.global_optimization.image_space_residuals
                else info_data["points_2d_nic"],
                sparse_jac=sparse,
                image_space_residuals=self.cfg.global_optimization.image_space_residuals,
            )

            # Prepare parameters
            params = torch.hstack((info_data["params_cameras"], info_data["params_tracks"])).to(
                device
            )
            opt_mask = torch.ones_like(params, dtype=bool, device=device)
            # fix first camera
            opt_mask[:6] = 0

        elif self.cfg.global_optimization.cost.desc == "dust3r":
            # Use the DUSt3RCostFunction
            cost_function = hydra.utils.instantiate(
                self.cfg.global_optimization.cost,
                n_points=info_data["n_points"],
                pose_indices=info_data["pose_indices"],
                pointmaps=info_data["pointmaps_dust3r"],
                dust3r_track_indices=info_data["dust3r_track_indices"],
                sparse_jac=sparse,
            )

            # Prepare parameters
            params = torch.hstack((info_data["params_tracks"], info_data["params_poses"])).to(
                device
            )
            opt_mask = torch.ones_like(params, dtype=bool, device=device)
            info_data["n_cameras"] = 0
            # cursed :) - fix scale
            # fix scale
            if self.cfg.global_optimization.alignment_fix_scale:
                opt_mask[info_data["n_points"] * 3 + 6 :: 7] = 0

        elif "combined" in self.cfg.global_optimization.cost.desc:
            # Prepare parameters
            params = torch.hstack(
                (
                    info_data["params_cameras"],
                    info_data["params_tracks"],
                    info_data["params_poses"],
                )
            ).to(device)
            opt_mask = torch.ones_like(params, dtype=bool, device=device)
            # fix first camera
            opt_mask[:6] = 0

            # fix scale
            if not self.cfg.global_optimization.opt_scale:
                opt_mask[info_data["n_cameras"] * 6 + info_data["n_points"] * 3 + 6 :: 7] = 0
            if (
                self.cfg.global_optimization.get("debug_gt", False)
                and self.cfg.global_optimization.opt_only_poses
            ):
                opt_mask[: info_data["n_cameras"] * 6 + info_data["n_points"] * 3] = 0
            if not self.cfg.global_optimization.opt_poses:
                opt_mask[info_data["n_cameras"] * 6 + info_data["n_points"] * 3 :] = 0

            # Create parameter mappings for each cost function
            ba_mapping = {
                "cameras": slice(0, info_data["n_cameras"] * 6),
                "points": slice(
                    info_data["n_cameras"] * 6,
                    info_data["n_cameras"] * 6 + info_data["n_points"] * 3,
                ),
            }

            dust3r_mapping = {
                "points": slice(
                    info_data["n_cameras"] * 6,
                    info_data["n_cameras"] * 6 + info_data["n_points"] * 3,
                ),
                "similarity": slice(
                    info_data["n_cameras"] * 6 + info_data["n_points"] * 3,
                    info_data["n_cameras"] * 6
                    + info_data["n_points"] * 3
                    + info_data["n_poses"] * 7,
                ),
            }

            # injecting stuff
            if "nic" in self.cfg.global_optimization.cost.desc:
                with open_dict(self.cfg.global_optimization.cost):
                    self.cfg.global_optimization.cost.cauchy_scale = (
                        1 / (info_data["K"][0][0] ** 2).item()
                    )

            ba_cost_function = hydra.utils.instantiate(
                self.cfg.global_optimization.cost.cost_functions[0],
                K=info_data["K"],
                n_cameras=info_data["n_cameras"],
                n_points=info_data["n_points"],
                camera_indices=info_data["camera_indices"],
                track_indices=info_data["track_indices"],
                points_2d=info_data["points_2d"]
                if self.cfg.global_optimization.image_space_residuals
                else info_data["points_2d_nic"],
                image_space_residuals=self.cfg.global_optimization.image_space_residuals,
            )

            from opt.residuals_closure import (
                debug_pointmap_reprojection_residuals_closure,
                log_reprojection_residuals_in_px_closure,
            )

            debug_pointmap_reprojection_residuals = debug_pointmap_reprojection_residuals_closure(
                K=info_data["K"],
                n_cameras=info_data["n_cameras"],
                n_points=info_data["n_points"],
                n_poses=info_data["n_poses"],
                camera_indices=info_data["camera_indices"],
                track_indices=info_data["track_indices"],
                points_2d=info_data["points_2d"],
                pointmaps=info_data["pointmaps_dust3r"],
                pose_indices=info_data["pose_indices"],
                dust3r_track_indices=info_data["dust3r_track_indices"],
            )
            debug_closures.append(debug_pointmap_reprojection_residuals)

            if not self.cfg.global_optimization.image_space_residuals:
                # log the reprojection error in pixels (if not done already)
                log_pixel_residuals = log_reprojection_residuals_in_px_closure(
                    K=info_data["K"],
                    n_cameras=info_data["n_cameras"],
                    n_points=info_data["n_points"],
                    camera_indices=info_data["camera_indices"],
                    track_indices=info_data["track_indices"],
                    points_2d=info_data["points_2d"],
                )
                debug_closures.append(log_pixel_residuals)

            dust3r_cost_function = hydra.utils.instantiate(
                self.cfg.global_optimization.cost.cost_functions[1],
                n_points=info_data["n_points"],
                pose_indices=info_data["pose_indices"],
                pointmaps=info_data["pointmaps_dust3r"],
                dust3r_track_indices=info_data["dust3r_track_indices"],
            )

            # Instantiate the composite cost function
            cost_function = hydra.utils.instantiate(
                self.cfg.global_optimization.cost,
                cost_functions=[ba_cost_function, dust3r_cost_function],
                param_mappings=[ba_mapping, dust3r_mapping],
            )

        else:
            raise ValueError(f"Unknown energy function: {self.cfg.global_optimization.cost.desc}")

        prev_cost = float("inf")

        # Log initial state to rerun
        self._log_to_rerun_pre_optimization(params, info_data)

        if wandb.run is not None:
            self._log_points_to_wandb(params, info_data)

        self.rec_steps += 1

        for i in range(self.cfg.global_optimization.optimizer.max_iterations):
            # anchor the step
            if wandb.run is not None:
                if wandb.run.summary.get("opt_step") is not None:
                    wandb.log({"opt_step": wandb.run.summary["opt_step"] + 1})
                else:
                    wandb.log({"opt_step": 0})

            # to get some info
            if i % self.cfg.global_optimization.logging_frequency == 0 and wandb.run is not None:
                for closure in debug_closures:
                    closure(params)

            params, info_optimizer = optim.step(
                params,
                residual_closure=cost_function.residuals,
                jac_closure=cost_function.jacobian,
                params_mask=opt_mask,
                cost_function=cost_function,
                info_data=info_data,
                conf_scale=info_data["confmaps_per_residual"]
                if "confmaps_per_residual" in info_data
                else 1.0,
            )

            if i % self.cfg.global_optimization.logging_frequency == 0 and wandb.run is not None:
                # Compute relative decrease based on the previous cost.
                self._write_back_params(
                    params.cpu(),
                    0
                    if self.cfg.global_optimization.cost.desc == "dust3r"
                    else info_data["n_cameras"],
                    info_data["n_points"],
                    0 if self.cfg.global_optimization.cost.desc == "ba" else info_data["n_poses"],
                )

                # Compute metrics
                metrics = self.calculate_metrics()
                if wandb.run is not None:
                    wandb.log(
                        {
                            "metrics/rotation_error": metrics["rotation_error_degrees"],
                            "metrics/translation_error": metrics["translation_error_degrees"],
                            "metrics/rotation_auc": metrics["rotation_auc"],
                            "metrics/translation_auc": metrics["translation_auc"],
                            "metrics/combined_maa": metrics["combined_maa"],
                            "metrics/ate": metrics["ate"],
                        }
                    )

            if i == 0:
                rel_decrease = float("inf")
            else:
                rel_decrease = abs(info_optimizer["cost"] - prev_cost) / prev_cost

            print(
                f"Step: {i} | Cost: {info_optimizer['cost']:.6f} | Relative decrease: {rel_decrease:.6f}"
            )

            # Log to rerun during optimization if enabled
            if i % 200 == 0:
                self._log_to_rerun_during_optimization(params, info_data, i)

            # Update prev_cost for the next iteration
            prev_cost = info_optimizer["cost"]

            # Early stopping logic
            if info_optimizer["stop_opt"]:
                print(f"Optimization stopped: {info_optimizer['convergence_message']}")
                break

            self.rec_steps += 1

        # write back the update parameters
        self._write_back_params(
            params.cpu(),
            0 if self.cfg.global_optimization.cost.desc == "dust3r" else info_data["n_cameras"],
            info_data["n_points"],
            0 if self.cfg.global_optimization.cost.desc == "ba" else info_data["n_poses"],
        )

        if self.cfg.global_optimization.pointmaps_filter_tracks:
            # VERY HACKY AND NOT RECOMMENDED LOL
            ransac_threshold = self.cfg.global_optimization.ransac_threshold
            with open_dict(self.cfg.global_optimization):
                self.cfg.global_optimization.ransac_threshold = (
                    self.cfg.global_optimization.pointmaps_filter_tracks_threshold
                )
            info_for_filtering_only = self._assemble_data_for_opt()
            # filter all tracks that are NOT inliers based on pointmap alignment (for at least one pair)
            dt_idx = info_for_filtering_only["dust3r_track_indices"]
            t_idx = info_for_filtering_only["track_indices"]
            tracks_to_be_removed_pointmap_alignment = torch.tensor(
                [x for x in t_idx if x not in dt_idx]
            ).tolist()
            with open_dict(self.cfg.global_optimization):
                self.cfg.global_optimization.ransac_threshold = ransac_threshold

        # Log final state to rerun
        self._log_to_rerun_post_optimization(params, info_data)

        # AFTER OPTIMIZATION
        if wandb.run is not None:
            self._log_points_to_wandb(params, info_data)

        # FILTERS TRACKS BASED ON REPROJECTION ERROR
        if (
            "combined" in self.cfg.global_optimization.cost.desc
            or "ba" in self.cfg.global_optimization.cost.desc
        ) and self.cfg.global_filtering.filter_tracks:
            if "combined" in self.cfg.global_optimization.cost.desc:
                ba_residual_vals = info_optimizer["residuals"][
                    : len(info_data["points_2d"].flatten())
                ]
                dust3r_residual_vals = info_optimizer["residuals"][
                    len(info_data["points_2d"].flatten()) :
                ]
            else:
                ba_residual_vals = info_optimizer["residuals"]

            # remove tracks with high *reprojection* error
            reprojection_error = ba_residual_vals.reshape(-1, 2)
            if self.cfg.global_optimization.image_space_residuals:
                reprojection_error = reprojection_error.norm(dim=-1)
            else:
                reprojection_error = (reprojection_error * info_data["K"][0][0]).norm(dim=-1)
            reprojection_error_indices = torch.where(
                reprojection_error > self.cfg.global_filtering.filter_max_reproj_error
            )[0]
            track_indices = [
                info_data["track_indices"][i] for i in reprojection_error_indices.cpu().tolist()
            ]
            # Convert track_indices to list if it's an int
            remove_reprojection_error = [
                self.active_track_idxs[track_idx] for track_idx in track_indices
            ]

            # filter points with low triangulation angle
            # https://github.com/colmap/colmap/blob/main/src/colmap/sfm/observation_manager.cc#L362
            # -> calculate pairwise triangulation angle. detlete only if none of combinations has sufficient triangulation angle
            remove_triangulation_angle = []
            for track_idx in self.active_track_idxs:
                remove = True
                track = self.scenegraph.tracks[track_idx]
                # extract camera center for all current views
                view_idxs = track.active_views
                camera_centers = [
                    self.scenegraph.views[view_idx].camera.camera_center.squeeze()
                    for view_idx in view_idxs
                ]
                comb = list(itertools.combinations(camera_centers, 2))
                for c1, c2 in comb:
                    ray1 = c1 - track.point
                    ray2 = c2 - track.point
                    dot_product = torch.dot(ray1, ray2)
                    magnitude_product = torch.norm(ray1) * torch.norm(ray2)
                    angle = torch.acos(dot_product / magnitude_product)
                    angle = torch.rad2deg(angle)
                    if angle > self.cfg.global_filtering.filter_min_triangulation_angle:
                        remove = False
                        break
                if remove:
                    remove_triangulation_angle.append(track_idx)

            remove_reprojection_error = set(remove_reprojection_error)
            remove_triangulation_angle = set(remove_triangulation_angle)
            to_be_removed = remove_reprojection_error | remove_triangulation_angle

            # if self.cfg.global_optimization.pointmaps_filter_tracks:
            #     remove_pointmap_alignment = {
            #         self.active_track_idxs[i] for i in tracks_to_be_removed_pointmap_alignment
            #     }
            #     to_be_removed = to_be_removed | remove_pointmap_alignment

            # Calculate statistics about removals
            only_triangulation = remove_triangulation_angle - remove_reprojection_error
            only_reprojection = remove_reprojection_error - remove_triangulation_angle
            both = remove_triangulation_angle & remove_reprojection_error

            print(f"We had {len(self.active_track_idxs)} tracks before filtering")
            for track_idx in to_be_removed:
                track = self.scenegraph.tracks[track_idx]
                # should I remove it here?
                if self.cfg.global_filtering.filter_remove_from_reconstruction:
                    track.state = TrackState.REMOVED
                elif (
                    self.cfg.global_filtering.filter_remove_max_reprojection_failed_from_reconstruction
                    and track_idx in remove_reprojection_error
                ):
                    track.state = TrackState.REMOVED
                else:
                    # reset track state
                    track.state = TrackState.OBSERVED
                self.active_track_idxs.remove(track_idx)

            print(f"Removing {len(to_be_removed)} tracks in total:")
            if self.cfg.global_optimization.pointmaps_filter_tracks:
                print(
                    f" - {len(remove_pointmap_alignment)} tracks removed due to pointmap alignment"
                )
            print(
                f" - {len(only_reprojection)} tracks removed only due to high reprojection error ({self.cfg.global_filtering.filter_max_reproj_error} px) - with a mean of {reprojection_error[reprojection_error_indices].mean()} px"
            )
            print(
                f" - {len(only_triangulation)} tracks removed only due to low triangulation angle ({self.cfg.global_filtering.filter_min_triangulation_angle} deg)"
            )
            print(f" - {len(both)} tracks removed due to both criteria")
            print(f"Number of tracks remaining: {len(self.active_track_idxs)}")

        return info

    def _assemble_data_for_opt(self):
        """Assemble the data to start reconstruction."""
        params_cameras = torch.hstack((self.rotations, self.translations))
        params_tracks = self.points

        # extract the keypoint values
        camera_indices = list()
        track_indices = list()
        points_2d = list()  # in pixel coordinates
        points_2d_nic = list()  # in normalized image coordinates

        # dust3r edgemap
        dust3r_edgemap = defaultdict(
            list
        )  # (edge1, edge2) -> [track_idx, pointmap_view1, pointmap_view2, confmap_view1, confmap_view2]

        # over all tracks
        for track_idx in self.active_track_idxs:
            # over all views for a track
            for camera_idx in self.scenegraph.tracks[track_idx].active_views:
                camera_indices.append(
                    self.active_view_idxs.index(camera_idx)
                )  # could be more efficient... and elegant
                keypoint_idx = self.scenegraph.tracks[track_idx].find_feature_for_view(camera_idx)
                if self.cfg.global_optimization.get("use_gt_projections", False):
                    keypoint_location_gt = (
                        self.scenegraph.views[camera_idx]
                        .project_onto_image(
                            torch.from_numpy(self.scenegraph.tracks[track_idx].point),
                            use_gt_cam=False,
                        )
                        .squeeze()
                        .tolist()
                    )
                    keypoint_location = self.scenegraph.views[camera_idx].kp[keypoint_idx].pt
                    # linear interpolation
                    keypoint_location = [
                        (1 - self.cfg.global_optimization.sift_weight) * keypoint_location_gt[0]
                        + self.cfg.global_optimization.sift_weight * keypoint_location[0],
                        (1 - self.cfg.global_optimization.sift_weight) * keypoint_location_gt[1]
                        + self.cfg.global_optimization.sift_weight * keypoint_location[1],
                    ]
                else:
                    keypoint_location = self.scenegraph.views[camera_idx].kp[keypoint_idx].pt

                points_2d.append(keypoint_location)
                track_indices.append(self.active_track_idxs.index(track_idx))

                if not self.cfg.global_optimization.image_space_residuals:
                    # undo the image scaling - TODO for later to not scale in the first place
                    keypoint_location = torch.linalg.inv(
                        self.scenegraph.views[camera_idx].K
                    ).float() @ torch.hstack(
                        (torch.tensor(keypoint_location), torch.ones(1))
                    ).unsqueeze(
                        1
                    )
                    keypoint_location = keypoint_location[:2].squeeze().tolist()
                    points_2d_nic.append(keypoint_location)

            if self.cfg.global_optimization.cost.desc != "ba":
                if self.scenegraph.pointmaps.pointmap_type == "two_view":
                    # go over edge pairwise edge that the track exposes
                    for edge in itertools.combinations(
                        self.scenegraph.tracks[track_idx].active_views, 2
                    ):
                        if edge[0] > edge[1]:
                            edge = (edge[1], edge[0])
                        pair = self.scenegraph.pairs_mat[edge[0]][edge[1]]
                        if pair is None:
                            # this can happen when the edge was not geometrically verified but a track still connects the two views - we ignore this for now
                            continue
                        keypoint_idx_1 = self.scenegraph.tracks[track_idx].find_feature_for_view(
                            edge[0]
                        )
                        keypoint_idx_2 = self.scenegraph.tracks[track_idx].find_feature_for_view(
                            edge[1]
                        )
                        keypoint_location_1 = torch.tensor(
                            self.scenegraph.views[edge[0]].kp[keypoint_idx_1].pt
                        ).reshape(1, 2)
                        keypoint_location_2 = torch.tensor(
                            self.scenegraph.views[edge[1]].kp[keypoint_idx_2].pt
                        ).reshape(1, 2)

                        # not elegant, but it works :)
                        if 224 in self.cfg.global_optimization.dust3r_output_size:
                            # transform location into dust3r space
                            dust3r_keypoint_location_1 = transform_coordinates_resize_crop(
                                keypoint_location_1,
                                input_shape=self.scenegraph.views[edge[0]].img.shape[:2],
                                output_size=224,
                            ).squeeze()
                            dust3r_keypoint_location_2 = transform_coordinates_resize_crop(
                                keypoint_location_2,
                                input_shape=self.scenegraph.views[edge[1]].img.shape[:2],
                                output_size=224,
                            ).squeeze()
                        else:
                            dust3r_keypoint_location_1 = transform_coordinates(
                                keypoint_location_1,
                                input_shape=self.scenegraph.views[edge[0]].img.shape[:2],
                                output_shape=self.cfg.global_optimization.dust3r_output_size,
                            ).squeeze()
                            dust3r_keypoint_location_2 = transform_coordinates(
                                keypoint_location_2,
                                input_shape=self.scenegraph.views[edge[1]].img.shape[:2],
                                output_shape=self.cfg.global_optimization.dust3r_output_size,
                            ).squeeze()

                        dust3r_pointmap_1 = self.scenegraph.pointmaps[edge].get_point_at(
                            float(dust3r_keypoint_location_1[0]),
                            float(dust3r_keypoint_location_1[1]),
                            view_idx=0,
                            use_interpolation=self.cfg.global_optimization.dust3r_bilinear_interpolation,
                        )
                        dust3r_pointmap_2 = self.scenegraph.pointmaps[edge].get_point_at(
                            float(dust3r_keypoint_location_2[0]),
                            float(dust3r_keypoint_location_2[1]),
                            view_idx=1,
                            use_interpolation=self.cfg.global_optimization.dust3r_bilinear_interpolation,
                        )
                        dust3r_confmap_1 = self.scenegraph.pointmaps[edge].get_confidence_at(
                            float(dust3r_keypoint_location_1[0]),
                            float(dust3r_keypoint_location_1[1]),
                            view_idx=0,
                        )
                        dust3r_confmap_2 = self.scenegraph.pointmaps[edge].get_confidence_at(
                            float(dust3r_keypoint_location_2[0]),
                            float(dust3r_keypoint_location_2[1]),
                            view_idx=1,
                        )

                        track_value = self.scenegraph.tracks[track_idx].point

                        # add to edgemap
                        dust3r_edgemap[(edge[0], edge[1])].append(
                            [
                                track_idx,
                                track_value,
                                dust3r_pointmap_1,
                                dust3r_pointmap_2,
                                dust3r_confmap_1,
                                dust3r_confmap_2,
                            ]
                        )

        points_2d = torch.tensor(points_2d)
        points_2d_nic = torch.tensor(points_2d_nic)
        # for dust3r energy
        # algorithm (two_view):
        # 1. for each edge in the edgemap, check if enough tracks are available
        # 2. if yes, get the tracks, and do a rigid alignment of the pointmaps
        alignment_errors = []
        whole_pointmaps_aligned = []
        if self.cfg.global_optimization.cost.desc != "ba":
            if self.scenegraph.pointmaps.pointmap_type == "two_view":
                params_dust3r = (
                    []
                )  # 7d tensor representing rotation (rotvec), translation and scaling
                pose_indices = []  # index of the pose (see line above) for each residual
                pointmaps_dust3r = []  # the pointmaps that are to be aligned with the tracks (Nx3)
                aligned_pointmaps = []  # the aligned pointmaps (Nx3)
                confmaps_dust3r = []  # the confmaps that are to be aligned with the tracks (Nx1)
                dust3r_track_indices = []  # index of the track that is being aligned to

                idx = 0
                for edge in dust3r_edgemap:
                    if len(dust3r_edgemap[edge]) < self.cfg.global_optimization.dust3r_min_tracks:
                        continue

                    # get tracks
                    tracks = [dust3r_edgemap[edge][i][1] for i in range(len(dust3r_edgemap[edge]))]
                    if self.cfg.global_optimization.dust3r_average_pointmaps:
                        if self.cfg.global_optimization.use_confmaps:
                            pointmaps = torch.stack(
                                [
                                    # Weighted average of the two pointmaps based on their confmaps
                                    (
                                        dust3r_edgemap[edge][i][2] * dust3r_edgemap[edge][i][4]
                                        + dust3r_edgemap[edge][i][3] * dust3r_edgemap[edge][i][5]
                                    )
                                    / (dust3r_edgemap[edge][i][4] + dust3r_edgemap[edge][i][5])
                                    for i in range(len(dust3r_edgemap[edge]))
                                ]
                            )

                            # Update confmaps to be the sum of the two confmaps
                            confmaps = torch.tensor(
                                [
                                    dust3r_edgemap[edge][i][4] + dust3r_edgemap[edge][i][5]
                                    for i in range(len(dust3r_edgemap[edge]))
                                ]
                            )
                        else:
                            pointmaps = torch.stack(
                                [
                                    torch.stack(
                                        [dust3r_edgemap[edge][i][2], dust3r_edgemap[edge][i][3]]
                                    ).mean(0)
                                    for i in range(len(dust3r_edgemap[edge]))
                                ]
                            )
                            confmaps = torch.stack(
                                [
                                    torch.mean(
                                        torch.tensor(
                                            [
                                                dust3r_edgemap[edge][i][4],
                                                dust3r_edgemap[edge][i][5],
                                            ]
                                        )
                                    )
                                    for i in range(len(dust3r_edgemap[edge]))
                                ]
                            )
                        tracks = torch.from_numpy(np.array(tracks)).float()
                    else:
                        pointmaps = [
                            torch.stack([dust3r_edgemap[edge][i][2], dust3r_edgemap[edge][i][3]])
                            for i in range(len(dust3r_edgemap[edge]))
                        ]
                        pointmaps = torch.cat(pointmaps, dim=0).float()
                        confmaps = torch.cat(
                            [
                                torch.tensor(
                                    [dust3r_edgemap[edge][i][4], dust3r_edgemap[edge][i][5]]
                                )
                                for i in range(len(dust3r_edgemap[edge]))
                            ],
                            dim=0,
                        ).float()
                        tracks = torch.from_numpy(np.array(tracks)).float()
                        tracks = np.repeat(tracks, 2, axis=0)

                    # Perform alignment using either RANSAC or direct method based on config
                    if self.cfg.global_optimization.rigid_alignment_ransac:
                        R, t, s, inliers, alignment_error = self._rigid_alignment_ransac(
                            pointmaps,
                            tracks,
                            min_points=self.cfg.global_optimization.ransac_min_points,
                            iterations=self.cfg.global_optimization.ransac_iterations,
                            threshold=self.cfg.global_optimization.ransac_threshold,
                            confmaps=confmaps
                            if self.cfg.global_optimization.use_confmaps
                            else None,
                        )
                        aligned_pts = (s * R @ pointmaps.T).T + t
                        if not self.cfg.global_optimization.pointmaps_align_only_inliers:
                            inliers = torch.ones(len(pointmaps), dtype=torch.bool)
                        print(
                            f"average residual error: {(tracks[inliers] - aligned_pts[inliers]).abs().mean()}"
                        )
                        print(
                            f"Number of inliers (based on threshold {self.cfg.global_optimization.ransac_threshold}m): {inliers.sum()}/{len(pointmaps)}"
                        )
                    else:
                        # Direct alignment without RANSAC
                        R, t, s = roma.rigid_points_registration(
                            pointmaps,
                            tracks,
                            compute_scaling=True,
                            weights=confmaps
                            if self.cfg.global_optimization.use_confmaps
                            else None,
                        )
                        aligned_pts = (s * R @ pointmaps.T).T + t
                        alignment_error = torch.norm(aligned_pts - tracks, dim=1)
                        if self.cfg.global_optimization.pointmaps_align_only_inliers:
                            inliers = (
                                alignment_error
                                < self.cfg.global_optimization.pointmaps_align_threshold
                            )
                        else:
                            inliers = torch.ones(len(pointmaps), dtype=torch.bool)

                        alignment_error = alignment_error.mean()
                        print(
                            f"Number of inliers (based on threshold {self.cfg.global_optimization.pointmaps_align_threshold}m): {inliers.sum()}/{len(pointmaps)}"
                        )

                    print(f"Alignment error for edge {edge}: {alignment_error}")

                    temp = (s * R @ self.scenegraph.pointmaps[edge].pts3d.reshape(-1, 3).T).T + t
                    whole_pointmaps_aligned.append(temp)

                    # add to params
                    if inliers.sum() > 0:
                        params_dust3r.append(
                            torch.cat((roma.rotmat_to_rotvec(R), t, s.unsqueeze(0)))
                        )
                        pointmaps_dust3r.append(pointmaps[inliers])
                        confmaps_dust3r.append(confmaps[inliers])
                        aligned_pointmaps.append(aligned_pts[inliers])
                        alignment_errors.append(alignment_error)
                        pose_indices += torch.tensor(idx).repeat(len(pointmaps[inliers])).tolist()
                        if self.cfg.global_optimization.dust3r_average_pointmaps:
                            dust3r_track_indices.append(
                                torch.tensor(
                                    [
                                        self.active_track_idxs.index(dust3r_edgemap[edge][i][0])
                                        for i, inlier in enumerate(inliers)
                                        if inlier
                                    ]
                                )
                            )
                        else:
                            dust3r_track_indices.append(
                                torch.tensor(
                                    [
                                        self.active_track_idxs.index(
                                            dust3r_edgemap[edge][i // 2][0]
                                        )
                                        for i, inlier in enumerate(inliers)
                                        if inlier
                                    ]
                                )
                            )
                        idx += 1

                params_poses = (
                    torch.cat(params_dust3r).reshape(-1, 7) if params_dust3r else torch.empty(0)
                )
                pointmaps_dust3r = torch.cat(pointmaps_dust3r)
                dust3r_track_indices = torch.cat(dust3r_track_indices).tolist()
                confmaps_dust3r = torch.cat(confmaps_dust3r)
            elif self.scenegraph.pointmaps.pointmap_type == "multiview":
                view_idxs = camera_indices
                # not elegant, but it works :) - used for the small dust3r where cropping squares the image
                if 224 in self.cfg.global_optimization.dust3r_output_size:
                    # transform location into dust3r space
                    query_keypoints = transform_coordinates_resize_crop(
                        points_2d,
                        input_shape=self.scenegraph.views[0].img.shape[:2],
                        output_size=224,
                    ).squeeze()
                else:
                    query_keypoints = transform_coordinates(
                        points_2d,
                        input_shape=self.scenegraph.views[0].img.shape[:2],
                        output_shape=self.cfg.global_optimization.dust3r_output_size,
                    ).squeeze()

                pointmaps = self.scenegraph.pointmaps.get_point_at(
                    x=query_keypoints[:, 0],
                    y=query_keypoints[:, 1],
                    view_idx=torch.tensor([self.active_view_idxs[i] for i in view_idxs]),
                    use_interpolation=self.cfg.global_optimization.dust3r_bilinear_interpolation,
                )
                confmaps = self.scenegraph.pointmaps.get_confidence_at(
                    x=query_keypoints[:, 0],
                    y=query_keypoints[:, 1],
                    view_idx=torch.tensor([self.active_view_idxs[i] for i in view_idxs]),
                    use_interpolation=self.cfg.global_optimization.dust3r_bilinear_interpolation,
                )

                params_poses = (
                    []
                )  # 7d tensor representing rotation (rotvec), translation and scaling
                pose_indices = []  # index of the pose (see line above) for each residual
                pointmaps_dust3r = []  # the pointmaps that are to be aligned with the tracks (Nx3)
                aligned_pointmaps = []  # the aligned pointmaps (Nx3)
                confmaps_dust3r = []  # the confmaps that are to be aligned with the tracks (Nx1)
                dust3r_track_indices = []  # index of the track that is being aligned to

                points = self.points
                if self.cfg.global_optimization.dust3r_average_pointmaps:
                    if self.cfg.global_optimization.use_confmaps:
                        # For each coordinate (x,y,z), do weighted bincount using confmaps as weights
                        pointmaps = torch.stack(
                            [
                                torch.bincount(
                                    torch.tensor(track_indices), weights=pointmaps[:, i] * confmaps
                                )  # weight by confmaps
                                / torch.bincount(
                                    torch.tensor(track_indices), weights=confmaps
                                )  # normalize by sum of weights
                                for i in range(3)
                            ],
                            dim=1,
                        )
                        # Sum the confmaps for each track
                        confmaps = torch.bincount(torch.tensor(track_indices), weights=confmaps)
                    else:
                        # collect all tracks and align pointmaps to them
                        counts = torch.bincount(torch.tensor(track_indices))
                        pointmaps = torch.stack(
                            [
                                torch.bincount(
                                    torch.tensor(track_indices), weights=pointmaps[:, i]
                                )
                                / counts
                                for i in range(3)
                            ],
                            dim=1,
                        )
                        confmaps = (
                            torch.bincount(torch.tensor(track_indices), weights=confmaps) / counts
                        )
                # Perform alignment using either RANSAC or direct method based on config
                if self.cfg.global_optimization.rigid_alignment_ransac:
                    R, t, s, inliers, alignment_error = self._rigid_alignment_ransac(
                        pointmaps,
                        points,
                        min_points=self.cfg.global_optimization.ransac_min_points,
                        iterations=self.cfg.global_optimization.ransac_iterations,
                        threshold=self.cfg.global_optimization.ransac_threshold,
                        confmaps=confmaps if self.cfg.global_optimization.use_confmaps else None,
                    )
                    aligned_pts = (s * R @ pointmaps.T).T + t
                    if not self.cfg.global_optimization.pointmaps_align_only_inliers:
                        inliers = torch.ones(len(pointmaps), dtype=torch.bool)
                    print(
                        f"average residual error: {(points[inliers] - aligned_pts[inliers]).abs().mean()}"
                    )
                    print(
                        f"Number of inliers (based on threshold {self.cfg.global_optimization.ransac_threshold}m): {inliers.sum()}/{len(pointmaps)}"
                    )
                else:
                    # Direct alignment without RANSAC
                    R, t, s = roma.rigid_points_registration(
                        pointmaps,
                        points,
                        compute_scaling=True,
                        weights=confmaps if self.cfg.global_optimization.use_confmaps else None,
                    )
                    aligned_pts = (s * R @ pointmaps.T).T + t
                    alignment_error = torch.norm(aligned_pts - points, dim=1)
                    if self.cfg.global_optimization.pointmaps_align_only_inliers:
                        inliers = (
                            alignment_error
                            < self.cfg.global_optimization.pointmaps_align_threshold
                        )
                    else:
                        inliers = torch.ones(len(pointmaps), dtype=torch.bool)

                    alignment_error = alignment_error.mean()
                    print(
                        f"Number of inliers (based on threshold {self.cfg.global_optimization.pointmaps_align_threshold}m): {inliers.sum()}/{len(pointmaps)}"
                    )

                params_poses = (
                    torch.cat((roma.rotmat_to_rotvec(R), t, s.unsqueeze(0)))
                    if inliers.sum() > 0
                    else torch.empty(0)
                )
                pose_indices = torch.tensor(0).repeat(len(pointmaps[inliers])).tolist()
                pointmaps_dust3r = pointmaps[inliers]
                confmaps_dust3r = confmaps[inliers]
                aligned_pointmaps = aligned_pts[inliers]
                # this assumes that the track_indices are monotonically increasing - which should be the çase ;) :thinking:
                if self.cfg.global_optimization.dust3r_average_pointmaps:
                    # Get unique track indices for inliers
                    dust3r_track_indices = np.unique(track_indices)[inliers]
                else:
                    dust3r_track_indices = track_indices[inliers]
                alignment_errors = [alignment_error]

        if wandb.run is not None and self.cfg.global_optimization.dust3r_average_pointmaps:
            wandb.log(
                {
                    "other/dust3r_rigid_alignment_error": sum(alignment_errors)
                    / len(alignment_errors)
                }
            )

        return dict(
            # param blocks
            params_cameras=params_cameras.flatten(),
            params_tracks=params_tracks.flatten(),
            params_poses=params_poses.flatten()
            if self.cfg.global_optimization.cost.desc != "ba"
            else None,
            # info
            n_cameras=len(self.active_view_idxs),
            n_points=len(self.active_track_idxs),
            n_poses=len(params_poses) if self.cfg.global_optimization.cost.desc != "ba" else None,
            # ba specific data
            camera_indices=torch.tensor(camera_indices, dtype=torch.int32),
            track_indices=torch.tensor(track_indices, dtype=torch.int32),
            K=self.scenegraph.views[0].K,  # assume that all intrinsics are shared
            points_2d=points_2d,
            points_2d_nic=points_2d_nic,
            # dust3r specific data
            pose_indices=torch.tensor(pose_indices, dtype=torch.int32)
            if self.cfg.global_optimization.cost.desc != "ba"
            else None,
            dust3r_track_indices=torch.tensor(dust3r_track_indices, dtype=torch.int32)
            if self.cfg.global_optimization.cost.desc != "ba"
            else None,
            pointmaps_dust3r=pointmaps_dust3r
            if self.cfg.global_optimization.cost.desc != "ba"
            else None,
            aligned_pointmaps=aligned_pointmaps
            if self.cfg.global_optimization.cost.desc != "ba"
            else None,
            confmaps_per_residual=confmaps_dust3r.repeat_interleave(3)
            if self.cfg.global_optimization.cost.desc != "ba"
            else None,
            whole_pointmaps=whole_pointmaps_aligned,
        )

    def get_params(self):
        """Get current set of params (used to save them)"""
        return self._assemble_data_for_opt()

    # METRICS
    def calculate_metrics(self):
        """Calculate relative pose metrics between all pairs of cameras in a batched manner.

        Returns various metrics including:
        - RRA/RTA (Relative Rotation/Translation Accuracy) at 5 and 15 degrees
        - Rotation and Translation AUC (Area Under the Curve)
        - Combined AUC (min of rotation and translation accuracy at each threshold)
        - Mean Average Accuracy for rotation and translation
        - Combined Mean Average Accuracy (minimum of rotation and translation)
        - ATE (Average Trajectory Error)
        - Registration rate
        """
        # Convert rotations to rotation matrices in batch
        rotations_rotmat = roma.rotvec_to_rotmat(self.rotations)  # (N, 3, 3)
        rotations_rotmat_gt = roma.rotvec_to_rotmat(self.rotations_gt)  # (N, 3, 3)

        ### PAIRWISE
        n_cams = len(self.active_view_idxs)
        pairs = torch.combinations(torch.arange(n_cams), r=2)  # (P, 2)

        # Align estimated poses with ground truth using rigid registration
        rotations_rotmat, translations = self._align_poses_with_gt()

        # Index the rotations and translations for all pairs at once
        R1 = rotations_rotmat[pairs[:, 0]]  # (P, 3, 3)
        R2 = rotations_rotmat[pairs[:, 1]]  # (P, 3, 3)
        R1_gt = rotations_rotmat_gt[pairs[:, 0]]  # (P, 3, 3)
        R2_gt = rotations_rotmat_gt[pairs[:, 1]]  # (P, 3, 3)

        t1 = translations[pairs[:, 0]]  # (P, 3)
        t2 = translations[pairs[:, 1]]  # (P, 3)
        t1_gt = self.translations_gt[pairs[:, 0]]  # (P, 3)
        t2_gt = self.translations_gt[pairs[:, 1]]  # (P, 3)

        # Convert translations to world space
        t1_world = -R1.transpose(1, 2) @ t1.unsqueeze(-1)  # (P, 3, 1)
        t2_world = -R2.transpose(1, 2) @ t2.unsqueeze(-1)  # (P, 3, 1)
        t1_gt_world = -R1_gt.transpose(1, 2) @ t1_gt.unsqueeze(-1)  # (P, 3, 1)
        t2_gt_world = -R2_gt.transpose(1, 2) @ t2_gt.unsqueeze(-1)  # (P, 3, 1)

        # Remove extra dimension
        t1_world = t1_world.squeeze(-1)  # (P, 3)
        t2_world = t2_world.squeeze(-1)  # (P, 3)
        t1_gt_world = t1_gt_world.squeeze(-1)  # (P, 3)
        t2_gt_world = t2_gt_world.squeeze(-1)  # (P, 3)

        # Calculate errors for all pairs at once
        rotation_error_degrees = relative_rotation_error(R1, R2, R1_gt, R2_gt)  # (P,)
        translation_error_degrees = relative_translation_error(
            t1_world, t2_world, t1_gt_world, t2_gt_world
        )  # (P,)

        # Calculate relative accuracies at specific thresholds
        rra5 = relative_accuracy(rotation_error_degrees, tau=5)
        rta5 = relative_accuracy(translation_error_degrees, tau=5)
        rra15 = relative_accuracy(rotation_error_degrees, tau=15)
        rta15 = relative_accuracy(translation_error_degrees, tau=15)

        # Calculate AUC metrics
        rot_auc = relative_rotation_auc(R1, R2, R1_gt, R2_gt)
        trans_auc = relative_translation_auc(t1_world, t2_world, t1_gt_world, t2_gt_world)
        comb_auc = combined_auc(rotation_error_degrees, translation_error_degrees)

        # Calculate Mean Average Accuracy
        rot_maa = mean_average_accuracy(rotation_error_degrees)
        trans_maa = mean_average_accuracy(translation_error_degrees)

        # Calculate Combined Mean Average Accuracy
        combined_mean_aa = combined_maa(rotation_error_degrees, translation_error_degrees)

        ### WHOLE SCENE
        ate = average_trajectory_error(
            self.rotations, self.translations, self.rotations_gt, self.translations_gt
        )

        # registration rate
        n_images = len(self.active_view_idxs)
        n_images_all = len(self.scenegraph.views)
        registration_rate = n_images / n_images_all * 100

        # number of tracks
        n_tracks = len(self.active_track_idxs)

        return dict(
            # Traditional metrics
            rra5=rra5,
            rta5=rta5,
            rra15=rra15,
            rta15=rta15,
            ate=ate,
            # AUC metrics
            rotation_auc=rot_auc,
            translation_auc=trans_auc,
            combined_auc=comb_auc,
            # Mean Average Accuracy
            rotation_maa=rot_maa,
            translation_maa=trans_maa,
            combined_maa=combined_mean_aa,
            # Other statistics
            n_tracks=n_tracks,
            n_images=n_images,
            registration_rate=registration_rate,
            n_images_provided=n_images_all,
            # for debugging
            rotation_error_degrees=rotation_error_degrees.mean().item(),
            translation_error_degrees=translation_error_degrees.mean().item(),
        )

    # PROPERTIES
    @property
    def rotations(self):
        rotations = list()
        for view_idx in self.active_view_idxs:
            rot = self.scenegraph.views[view_idx].camera.rotation
            rotations.append(rot)
        return torch.stack(rotations).type(torch.float32)

    @property
    def rotations_gt(self):
        rotations = list()
        for view_idx in self.active_view_idxs:
            rot = self.scenegraph.views[view_idx].camera_gt.rotation
            rotations.append(rot)
        return torch.stack(rotations).type(torch.float32)

    @property
    def translations(self):
        translations = list()
        for view_idx in self.active_view_idxs:
            trans = self.scenegraph.views[view_idx].camera.translation
            translations.append(trans)
        return torch.stack(translations).type(torch.float32)

    @property
    def translations_gt(self):
        translations = list()
        for view_idx in self.active_view_idxs:
            trans = self.scenegraph.views[view_idx].camera_gt.translation
            translations.append(trans)
        return torch.stack(translations).type(torch.float32)

    @property
    def camera_params(self):
        rotations = list()
        translations = list()
        for view_idx in self.active_view_idxs:
            rot = self.scenegraph.views[view_idx].camera.rotation
            trans = self.scenegraph.views[view_idx].camera.translation
            rotations.append(rot)
            translations.append(trans)
        rotations = torch.stack(rotations).type(torch.float32)
        translations = torch.stack(translations).type(torch.float32)
        return dict(rotations=rotations, translations=translations)

    @property
    def points(self):
        points = list()
        for track_idx in self.active_track_idxs:
            points.append(torch.from_numpy(self.scenegraph.tracks[track_idx].point))
        return torch.stack(points).type(torch.float32)

    # PLOTTING/VISUALIZATION FUNCTIONALITY
    def plot_reconstruction(self, show_gt=False, remove_outliers=False):
        if show_gt:
            return viz.plot_3d(
                self.points,
                self.rotations,
                self.translations,
                self.scenegraph.views[0].K,
                None,
                self.rotations_gt,
                self.translations_gt,
                remove_outliers=remove_outliers,
            )
        else:
            return viz.plot_3d(
                self.points,
                self.rotations,
                self.translations,
                self.scenegraph.views[0].K,
                remove_outliers=remove_outliers,
            )  # assume that all intrinsics are shared

    def plot_reprojection(self, view_idx):
        """Plot reprojection of tracks in a view."""
        if view_idx not in self.active_view_idxs:
            raise ValueError("View has not been added to the reconstruction yet")

        view = self.scenegraph.views[view_idx]
        keypoint_idxs = list()
        points = list()

        # go over all tracks for that specific view
        for track in self.scenegraph.view_to_tracks[view_idx]:
            # check if the track is active
            if track.state == TrackState.ACTIVE:
                keypoint = track.find_feature_for_view(view_idx)
                keypoint_idxs.append(keypoint)
                points.append(torch.from_numpy(track.point))
        # project points onto image plane
        points = torch.stack(points)
        projections = project(points, view.camera.rotation, view.camera.translation, view.K)

        return view.draw_keypoints(projections, keypoint_gt_idxs=keypoint_idxs)

    def plot_pnp_registration(self, show_gt=True):
        """Plot the new camera that is being registered along with the 3d points being used for
        registration."""
        info = self._find_best_next_view()
        next_view_idx, visible_tracks = info["best_view_idx"], info["visible_tracks"]

        info_pnp = self._register_view(next_view_idx, visible_tracks, update_scenegraph=False)

        fig = self.plot_reconstruction(show_gt=True)

        # color points being used for PnP
        fig.add_trace(viz.add_points(info_pnp["points_3d"], color="yellow", name="PnP Points"))

        # add new camera
        fig.add_trace(
            viz.add_pyramid(
                rotation=torch.from_numpy(info_pnp["rvec"]).squeeze(),
                translation=torch.from_numpy(info_pnp["tvec"]).squeeze(),
                K=self.scenegraph.views[0].K.numpy(),
                color="yellow",
                name="PnP Estimated",
            )
        )

        if show_gt:
            # add gt new camera
            fig.add_trace(
                viz.add_pyramid(
                    rotation=self.scenegraph.views[next_view_idx].camera_gt.rotation,
                    translation=self.scenegraph.views[next_view_idx].camera_gt.translation,
                    K=self.scenegraph.views[0].K.numpy(),
                    color="orange",
                    name="PnP GT",
                )
            )

        return fig

    def plot_triangulation(self, show_gt=True, remove_outliers=False):
        """Plot the triangulation after the new camera has been registered."""
        info = self._find_best_next_view()
        next_view_idx, visible_tracks = info["best_view_idx"], info["visible_tracks"]

        info_pnp = self._register_view(next_view_idx, visible_tracks, update_scenegraph=True)

        # compose P matrix
        R = roma.rotvec_to_rotmat(torch.from_numpy(info_pnp["rvec"]).squeeze())
        t = torch.from_numpy(info_pnp["tvec"])

        P = torch.hstack((R, t)).numpy()

        info_triag = self._triangulate_new_tracks(next_view_idx, P1=P, update_scenegraph=False)

        fig = self.plot_reconstruction(show_gt=True, remove_outliers=remove_outliers)

        # add new camera
        fig.add_trace(
            viz.add_pyramid(
                rotation=torch.from_numpy(info_pnp["rvec"]).squeeze(),
                translation=torch.from_numpy(info_pnp["tvec"]).squeeze(),
                K=self.scenegraph.views[0].K.numpy(),
                color="yellow",
                name="PnP Estimated",
            )
        )

        if show_gt:
            # add gt new camera
            fig.add_trace(
                viz.add_pyramid(
                    rotation=self.scenegraph.views[next_view_idx].camera_gt.rotation,
                    translation=self.scenegraph.views[next_view_idx].camera_gt.translation,
                    K=self.scenegraph.views[0].K.numpy(),
                    color="orange",
                    name="PnP GT",
                )
            )

        # Remove point outliers based on the standard deviation of the point statistics
        if remove_outliers:
            mean = np.mean(info_triag["points"], axis=0)
            std = np.std(info_triag["points"], axis=0)
            valid_points = info_triag["points"][
                np.linalg.norm(info_triag["points"] - mean, axis=1) < 2 * np.linalg.norm(std)
            ]
            points = valid_points
        else:
            points = info_triag["points"]

        # All triangulated points
        fig.add_trace(
            viz.add_points(
                points[~info_triag["chirality_mask"]],
                color="darkmagenta",
                name="Triangulated Points [Chirality Failed]",
            )
        )

        # triangulate points after filter
        fig.add_trace(
            viz.add_points(
                points[info_triag["chirality_mask"]],
                color="violet",
                name="Triangulated Points [Chirality Passed]",
            )
        )

        return fig

    # UTILS
    def _connect_rerun(self, name: str = "SfM"):
        connect_server(name=name)
        setup_blueprint()

    def _send_to_device_and_adjust_type(self, info, device="cpu", type=torch.float32):
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                info[key] = torch.from_numpy(value).to(device, dtype=type)
            elif isinstance(value, torch.Tensor):
                if torch.is_floating_point(value):
                    info[key] = value.to(device, dtype=type)
                else:
                    info[key] = value.to(device)
            elif isinstance(value, dict):
                self._send_to_device_and_adjust_type(value, device, type)

    def _write_back_params(self, params, n_cameras, n_points, n_poses):
        """Write back the optimized parameters to the scene graph.

        Args:
            params: Optimized parameters
            n_cameras: Number of cameras
            n_points: Number of points
            n_poses: Number of poses
        """
        camera_params = params[: n_cameras * 6]
        point_params = params[n_cameras * 6 : n_cameras * 6 + n_points * 3]
        pose_params = params[n_cameras * 6 + n_points * 3 :]

        # Reshape camera parameters
        if n_cameras > 0:
            camera_params = camera_params.reshape_as(
                torch.hstack((self.rotations, self.translations))
            )

            # Update rotations and translations in views
            for i, camera_param in enumerate(camera_params):
                self.scenegraph.views[self.active_view_idxs[i]].camera.rotation = camera_param[
                    :3
                ]  # rotvec
                self.scenegraph.views[self.active_view_idxs[i]].camera.translation = camera_param[
                    3:
                ]

        # Update points in tracks
        for i, point in enumerate(point_params.reshape_as(self.points)):
            self.scenegraph.tracks[self.active_track_idxs[i]].point = point.numpy()
        # TODO possibly save the poses (instead of recomputing them every step)

    def _log_points_to_wandb(self, params, info_data):
        """Log 3D points and camera positions to Wandb.

        Args:
            params: The parameter vector containing camera, track, and pose information
            info_data: Dictionary containing information about the reconstruction
        """
        if wandb.run is None:
            return

        point_tracks = (
            params[
                info_data["n_cameras"] * 6 : info_data["n_cameras"] * 6 + info_data["n_points"] * 3
            ]
            .reshape(-1, 3)
            .cpu()
            .numpy()
        )

        # Add color to point tracks (purple: 128, 0, 128)
        purple = np.repeat([[128, 0, 128]], point_tracks.shape[0], axis=0)
        colored_point_tracks = np.hstack([point_tracks, purple])

        # Initialize points with just the colored tracks
        points = colored_point_tracks

        # Add aligned pointmaps if they exist
        if params[info_data["n_cameras"] * 6 + info_data["n_points"] * 3 :].shape[0] > 0:
            params_poses = params[info_data["n_cameras"] * 6 + info_data["n_points"] * 3 :]
            pointmaps = info_data["pointmaps_dust3r"]
            pose_indices = info_data["pose_indices"]

            aligned_pointmaps = (
                dust3r_pointmaps_align(pointmaps, params_poses.reshape(-1, 7)[pose_indices])
                .cpu()
                .numpy()
            )

            # Add color to aligned pointmaps (orange: 238, 129, 99)
            orange = np.repeat([[238, 129, 99]], aligned_pointmaps.shape[0], axis=0)
            colored_aligned_pointmaps = np.hstack([aligned_pointmaps, orange])

            # Concatenate with point tracks
            points = np.concatenate([colored_point_tracks, colored_aligned_pointmaps], axis=0)

        # Add cameras
        blue = np.repeat([[0, 128, 255]], info_data["n_cameras"], axis=0)
        rotmats = roma.rotvec_to_rotmat(self.rotations).transpose(1, 2)
        translations = self.translations.unsqueeze(-1).cpu().numpy()
        centers = -rotmats @ translations
        cameras = np.hstack([centers.squeeze(-1), blue])

        # Combine all points
        all_points = np.concatenate([points, cameras], axis=0)

        # Log to wandb
        wandb.log({"3d/points": wandb.Object3D(all_points)})

    def _log_ransac_iteration_to_rerun(
        self, iter_idx, aligned_pts_np, tracks_np, inliers, idx=None, confidences=None
    ):
        """Log the current RANSAC iteration to rerun for visualization.

        Args:
            iter_idx: Current RANSAC iteration index
            aligned_pts_np: Numpy array of aligned points
            tracks_np: Numpy array of target points
            inliers: Boolean mask indicating inlier points
            idx: Optional indices of sampled points for current iteration
            confidences: Optional array of confidence values between 1 and 3
        """
        rr.set_time_sequence("/ransac", iter_idx)
        if not hasattr(self, "_logged_mesh"):
            self._log_mesh_to_rerun()
            self._logged_mesh = True

        # Log aligned points with confidence-based coloring if provided
        if confidences is not None:
            # Clip and normalize confidences to [0, 1] range
            conf = np.clip(confidences, 1, 3)
            conf_normalized = (conf - 1) / 2.0
            # Create colors using jet colormap
            colors = plt.cm.jet(conf_normalized)[:, :3]  # Only take RGB values, drop alpha
            # Convert to uint32 format for rerun
            colors_uint32 = (colors * 255).astype(np.uint8)
        else:
            colors_uint32 = None

        rr.log(
            "/ransac/aligned_pointmaps",
            rr.Points3D(
                aligned_pts_np,
                colors=colors_uint32
                if colors_uint32 is not None
                else RERUN_KWARGS["pointmaps"]["colors"],
                radii=RERUN_KWARGS["pointmaps"]["radii"],
            ),
        )

        # Log target tracks (purple)
        rr.log(
            "/ransac/tracks",
            rr.Points3D(
                tracks_np,
                **RERUN_KWARGS["points"],
            ),
        )

        # Log sample points used for current iteration (blue)
        if idx is not None:
            sample_aligned = aligned_pts_np[idx]
            rr.log(
                "/ransac/sample_pointmaps",
                rr.Points3D(
                    sample_aligned,
                    colors=np.array([0x0000FFFF], dtype=np.uint32),  # Blue color
                    radii=0.01,
                ),
            )

        # Create line strips connecting all aligned points to their targets
        all_strips = np.stack(
            [np.stack((aligned_pts_np[i], tracks_np[i])) for i in range(len(aligned_pts_np))],
            axis=0,
        )

        # Create colors for all strips based on inlier/outlier status
        all_strip_colors = np.array(
            [0x00FF00FF if i else 0xFF0000FF for i in inliers], dtype=np.uint32
        )  # Green for inliers, Red for outliers

        rr.log(
            "/ransac/all_strips",
            rr.LineStrips3D(all_strips, colors=all_strip_colors, radii=0.0002),
        )

        # # Log inliers/outliers status
        # inlier_colors = np.array(
        #     [0x00FF00FF if i else 0xFF0000FF for i in inliers], dtype=np.uint32
        # )  # Green for inliers, Red for outliers
        # rr.log(
        #     "/ransac/inlier_status", rr.Points3D(aligned_pts_np, colors=inlier_colors, radii=0.01)
        # )

    def _rigid_alignment_ransac(
        self, pointmaps, tracks, min_points=3, iterations=100, threshold=0.1, confmaps=None
    ):
        n_points = len(pointmaps)
        best_inliers = np.zeros(n_points, dtype=np.bool)
        best_error = float("inf")
        best_R = torch.eye(3)
        best_t = torch.zeros(3)
        best_s = 1.0
        max_inliers = min_points  # Initialize to minimum required points
        best_aligned_pts_np = np.array([])
        best_tracks_np = np.array([])

        for iter_idx in range(iterations):
            if confmaps is not None:
                # confidence-weighted sampling
                idx = torch.multinomial(confmaps, min_points, replacement=False)
            else:
                # random sampling
                idx = torch.randperm(n_points)[:min_points]

            sample_pointmaps = pointmaps[idx]
            sample_tracks = tracks[idx]
            sample_confmaps = confmaps[idx] if confmaps is not None else None
            # Fit transformation to minimal set
            try:
                R, t, s = roma.rigid_points_registration(
                    sample_pointmaps, sample_tracks, compute_scaling=True, weights=sample_confmaps
                )
            except Exception as e:
                print(f"Error in rigid alignment: {e}")
                continue

            # Transform all points
            aligned_pts = (s * R @ pointmaps.T).T + t

            # Calculate errors for all points
            errors = torch.norm(aligned_pts - tracks, dim=1)

            # Find inliers
            inliers = errors < threshold
            n_inliers = inliers.sum()

            # Log current iteration to rerun
            if self.cfg.rerun_log_pointmaps_ransac:
                aligned_pts_np = aligned_pts.cpu().numpy()
                tracks_np = tracks.cpu().numpy()
                self._log_ransac_iteration_to_rerun(
                    iter_idx, aligned_pts_np, tracks_np, inliers.numpy(), idx
                )

            if n_inliers >= max_inliers:  # Changed from > min_points
                # Refine transformation using all inliers
                R_refined, t_refined, s_refined = roma.rigid_points_registration(
                    pointmaps[inliers], tracks[inliers], compute_scaling=True
                )

                # Calculate error for inliers
                aligned_pts_refined = (s_refined * R_refined @ pointmaps[inliers].T).T + t_refined
                error = torch.norm(aligned_pts_refined - tracks[inliers], dim=1).mean()

                # Update best model if we have more inliers, or same inliers but lower error
                if n_inliers > max_inliers or (n_inliers == max_inliers and error < best_error):
                    max_inliers = n_inliers
                    best_error = error
                    best_inliers = inliers
                    best_R = R_refined
                    best_t = t_refined
                    best_s = s_refined

                    best_aligned_pts_np = aligned_pts_refined.cpu().numpy()
                    best_tracks_np = tracks[inliers].cpu().numpy()

                    if self.cfg.rerun_log_pointmaps_ransac:
                        # Log the refined alignment
                        aligned_pts_refined_np = aligned_pts_refined.cpu().numpy()
                        tracks_inliers_np = tracks[inliers].cpu().numpy()
                        self._log_ransac_iteration_to_rerun(
                            iter_idx,
                            aligned_pts_refined_np,
                            tracks_inliers_np,
                            torch.arange(len(aligned_pts_refined_np)),  # Use all points as samples
                            torch.ones(
                                len(aligned_pts_refined_np), dtype=torch.bool
                            ),  # All points are inliers
                        )

        # log the best alignment overall
        if self.cfg.rerun_log_pointmaps_ransac:
            self._log_ransac_iteration_to_rerun(
                iter_idx + 1,
                best_aligned_pts_np,
                best_tracks_np,
                torch.arange(len(best_aligned_pts_np)),  # Use all points as samples
                torch.ones(len(best_aligned_pts_np), dtype=torch.bool),  # All points are inliers
            )

        return best_R, best_t, best_s, best_inliers, best_error

    def _align_poses_with_gt(self):
        """Aligns the estimated camera poses with the ground truth camera poses using rigid
        registration.

        Returns:
            tuple: (aligned_rotations, aligned_translations) - the aligned rotation matrices and translation vectors
        """
        # Convert rotation matrices to camera centers
        rotations_rotmat = roma.rotvec_to_rotmat(self.rotations)
        rotations_rotmat_gt = roma.rotvec_to_rotmat(self.rotations_gt)

        # Calculate camera centers in world coordinates
        camera_centers_vggt = -rotations_rotmat.transpose(1, 2) @ self.translations.unsqueeze(-1)
        camera_centers_vggt = camera_centers_vggt.squeeze()
        camera_centers_gt = -rotations_rotmat_gt.transpose(1, 2) @ self.translations_gt.unsqueeze(
            -1
        )
        camera_centers_gt = camera_centers_gt.squeeze()

        # Perform rigid alignment with scaling
        R, t, s = roma.rigid_points_registration(
            camera_centers_vggt, camera_centers_gt, weights=None, compute_scaling=True
        )

        # Transform camera centers
        new_camera_centers = torch.stack([s * R @ x + t for x in camera_centers_vggt], dim=0)

        # Transform rotations (world-to-camera)
        updated_rots = torch.stack([x @ R.T for x in rotations_rotmat], dim=0)

        # Compute new translations (world-to-camera)
        new_translations = torch.bmm(updated_rots, -new_camera_centers.unsqueeze(-1)).squeeze(-1)

        return updated_rots, new_translations
