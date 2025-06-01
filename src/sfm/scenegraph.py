import random
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image  # Make sure to import the Image module
from prettytable import PrettyTable

import wandb
from data.colmap import read_cameras_text, read_images_text
from data.scannet_scene import ScannetppScene
from sfm import Pair, Track, View
from sfm.pointmap import Pointmap
from utils.camera import undistort_fisheye_intrinsics
from utils.visualization import plot_keypoints_connections, plot_views


class SceneGraph:
    """The SceneGraph holds information about the relationship between different images.

    It mainly contains views (images) and pairs (the relationship informoation between this image)
    It takes care of the feature matching pipeline.
    """

    def __init__(
        self,
        views: List[View],
        pairs_mat: List[List[Pair]],
        tracks: List[Track],
        feature_to_track: Dict[Tuple[int, int], Track],
        view_to_tracks: Dict[int, Track],
        cfg: DictConfig,
        scene: ScannetppScene = None,
        pointmaps: Pointmap = None,
    ):
        """
        pairs: (lower triangular) matrix filled with Pairs
        """
        self.views = views
        self.pairs_mat = pairs_mat
        self.tracks = tracks
        self.feature_to_track = feature_to_track
        self.view_to_tracks = view_to_tracks
        self.scene = scene
        self.cfg = cfg
        self.points_3d = None
        self.pointmaps = pointmaps

    @staticmethod
    def build_from_images_colmap(
        image_paths: List[PosixPath],
        scene: ScannetppScene,
        cfg: DictConfig = None,
        use_precomputed_matches: bool = False,
        pointmaps: Pointmap = None,
        **kwargs,
    ):
        """Essentially we read in the colmap images to have ground truth."""

        # load data
        camera_data = read_cameras_text(scene.dslr_colmap_dir / "cameras.txt")
        image_data = read_images_text(scene.dslr_colmap_dir / "images.txt")
        image_data = {v.name: v for _, v in image_data.items()}

        # undistort intrinsics
        K = undistort_fisheye_intrinsics(camera_data[1])

        # load images and extract features
        views = []
        for path in image_paths:
            view = View(
                scene.dslr_undistorted_images_dir / path,
                parameterization=cfg.view.parameterization,
                descriptor=cfg.view.descriptor,
                mask_path=scene.dslr_undistorted_anon_masks_dir / (path.name + ".png")
                if cfg.view.mask
                else None,
                mask_blur_iterations=cfg.view.mask_blur_iterations,
                keypoints_path=scene.dslr_dir / "keypoints" / cfg.name / (path.name + ".npy")
                if use_precomputed_matches
                else None,
                K=torch.tensor(K),
            )
            view.add_camera(kind="gt", K=K, R_t=image_data[path.name].world_to_camera)
            if "resize" in cfg.view and cfg.view.resize is not None:
                view.resize(cfg.view.resize[0])
            views.append(view)

        # # Initialize pairs matrix
        n_views = len(views)
        pairs_mat = [[None for _ in range(n_views)] for _ in range(n_views)]

        if use_precomputed_matches:
            for i in range(n_views):
                for j in range(i):
                    matches_path = (
                        scene.dslr_dir
                        / "matches"
                        / cfg.name
                        / (views[j].img_path.name + "_" + views[i].img_path.name + ".npy")
                    )
                    if matches_path.exists():
                        pair = Pair(views[j], views[i])
                        pair._deserialize_matches(matches_path)
                        if pair.n_matches >= cfg.pair.min_matches:
                            pair.geometric_verification(**cfg.pair.geometric_verification)
                            pair._remove_spurious_matches(
                                keep_one_to_one=cfg.pair.spurious_matches_keep_one_to_one
                            )
                            if cfg.pair.repeat_geometric_verification:
                                pair.geometric_verification(**cfg.pair.geometric_verification)
                            if pair.n_matches >= cfg.pair.min_matches_after_verification:
                                if pair.n_matches >= cfg.pair.max_verified_matches:
                                    print(
                                        f"Pair {j} {i} has {pair.n_matches} verified matches, max is {cfg.pair.max_verified_matches}; randomly sampling {cfg.pair.max_verified_matches} matches"
                                    )
                                    pair.matches = random.sample(
                                        pair.matches, cfg.pair.max_verified_matches
                                    )
                                pair.set_kp_active()
                                pair.create_dicts_from_matches()
                                pairs_mat[j][i] = pair
        else:
            # Time both approaches
            def run_parallel():
                pairs_mat = [[None for _ in range(n_views)] for _ in range(n_views)]

                def process_pair(i, j):
                    pair = Pair(views[j], views[i])
                    pair.find_matches(
                        ratio=cfg.pair.sift_lowe_ratio,
                        max_distance=cfg.pair.sift_max_distance,
                        distance_metric=cfg.pair.sift_distance_metric,
                    )
                    if pair.n_matches >= cfg.pair.min_matches:
                        pair.geometric_verification(**cfg.pair.geometric_verification)
                        pair._remove_spurious_matches(
                            keep_one_to_one=cfg.pair.spurious_matches_keep_one_to_one
                        )
                        if cfg.pair.repeat_geometric_verification:
                            pair.geometric_verification(**cfg.pair.geometric_verification)
                        if pair.n_matches >= cfg.pair.min_matches_after_verification:
                            pair.set_kp_active()
                            pair.create_dicts_from_matches()
                            return (j, i, pair)
                    return (j, i, None)

                pairs_to_process = [(i, j) for i in range(1, n_views) for j in range(i)]
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(lambda p: process_pair(*p), pairs_to_process))

                for j, i, pair in results:
                    if pair is not None:
                        pairs_mat[j][i] = pair

                return pairs_mat

            def run_sequential():
                pairs_mat = [[None for _ in range(n_views)] for _ in range(n_views)]
                for i in range(1, n_views):
                    for j in range(i):
                        pair = Pair(views[j], views[i])
                        pair.find_matches(
                            ratio=cfg.pair.sift_lowe_ratio,
                            max_distance=cfg.pair.sift_max_distance,
                            distance_metric=cfg.pair.sift_distance_metric,
                        )
                        if pair.n_matches >= cfg.pair.min_matches:
                            pair.geometric_verification(**cfg.pair.geometric_verification)
                            pair._remove_spurious_matches(
                                keep_one_to_one=cfg.pair.spurious_matches_keep_one_to_one
                            )
                            if cfg.pair.repeat_geometric_verification:
                                pair.geometric_verification(**cfg.pair.geometric_verification)
                            if pair.n_matches >= cfg.pair.min_matches_after_verification:
                                pair.set_kp_active()
                                pairs_mat[j][i] = pair
                                pair.create_dicts_from_matches()
                return pairs_mat

            if cfg.parallel:
                pairs_mat = run_parallel()  # run_sequential()
            else:
                pairs_mat = run_sequential()

        # keep track of which kp have been visited:
        visited_kp_list = list()
        for view in views:
            visited_kp_list_view = [not truuu for truuu in view.kp_active.copy()]
            visited_kp_list.append(visited_kp_list_view)

        tracks = list()
        # for all images
        for view_i, view in enumerate(views):
            # for all keypoints
            for kp_j in range(len(visited_kp_list[view_i])):
                # have we considered the feature/kp?
                if not visited_kp_list[view_i][kp_j]:
                    # now we will have considered the feature
                    # bfs
                    track = Track()
                    visited_kp_list[view_i][kp_j] = True
                    queue = deque([(view_i, kp_j)])
                    while queue:
                        current = queue.popleft()  # current: (view_idx, kp_idx)
                        track.add_view(*current)
                        # explore pairwise relation if that feature to other features
                        for k in range(len(views)):
                            if current[0] != k:
                                current_pair = (
                                    pairs_mat[current[0]][k]
                                    if (k > current[0])
                                    else pairs_mat[k][current[0]]
                                )
                                if current_pair is None:
                                    # current pair has not passed number of matches to be necessary to be considered
                                    continue
                                current_dict = (
                                    current_pair.query_to_database
                                    if (k > current[0])
                                    else current_pair.database_to_query
                                )
                                if current[1] in current_dict:
                                    kp_for_k = current_dict[current[1]]
                                    # check if kp has been visited, if not, add to the queue
                                    if not visited_kp_list[k][kp_for_k]:
                                        visited_kp_list[k][kp_for_k] = True
                                        queue.append((k, kp_for_k))
                            else:
                                # dont consider relationship to itself
                                continue

                    tracks.append(track)

        # remove spurious and potentially single-point tracks
        new_tracks = list()
        track_length_counts = {}
        removed_track_counts = {}

        for track in tracks:
            track_length = len(track.views)

            if track.check_consistent() and track_length >= cfg.min_track_length:
                new_tracks.append(track)

                # Count kept tracks by length
                if track_length in track_length_counts:
                    track_length_counts[track_length] += 1
                else:
                    track_length_counts[track_length] = 1
            else:
                # Count removed tracks by length
                if track_length in removed_track_counts:
                    removed_track_counts[track_length] += 1
                else:
                    removed_track_counts[track_length] = 1

        # Print summary of removed tracks
        print("\n--- Track Filtering Summary ---")
        print(f"Total tracks before filtering: {len(tracks)}")
        print(f"Total tracks after filtering: {len(new_tracks)}")
        print(f"Total tracks removed: {len(tracks) - len(new_tracks)}")

        print("\nRemoved tracks by length:")
        for length in sorted(removed_track_counts.keys()):
            count = removed_track_counts[length]
            print(f"  Length {length}: {count} tracks removed")

        print("\nKept tracks by length:")
        for length in sorted(track_length_counts.keys()):
            count = track_length_counts[length]
            print(f"  Length {length}: {count} tracks kept")

        # construct a feature to track mapping
        feature_to_track = dict()
        view_to_tracks = defaultdict(list)
        for track in new_tracks:
            for view_kp in track.views:
                feature_to_track[view_kp] = track
                view_to_tracks[view_kp[0]].append(track)

        return SceneGraph(
            views=views,
            pairs_mat=pairs_mat,
            scene=scene,
            tracks=new_tracks,
            feature_to_track=feature_to_track,
            view_to_tracks=view_to_tracks,
            cfg=cfg,
            pointmaps=pointmaps,
        )

    def compute_statistics(self, write_to_file=False, file_path="scenegraph_statistics.txt"):
        # Get component statistics
        self.get_largest_connected_component()  # This will set self.component_stats

        # if self.component_stats["largest_component_size"] == 1:
        #     raise ValueError("No connected component found, skipping statistics")

        # number of tracks
        stats_tracks = dict()
        stats_tracks["Amount"] = len(self.tracks)

        # average track length
        track_len = [len(track.views) for track in self.tracks]  # Calculate track lengths
        stats_tracks["Length"] = track_len  # Store lengths for further statistics

        # pairwise matching stats
        pairwise_data = defaultdict(list)
        for i in range(len(self.views)):
            for j in range(i + 1, len(self.views)):
                pair = self.pairs_mat[i][j]
                if pair is not None:
                    pairwise_data["matched_successful"].append(1)
                    pairwise_data["n_matches"].append(pair.n_matches)
                else:
                    pairwise_data["matched_successful"].append(0)

        # create tables
        basic_stats_table = PrettyTable()
        basic_stats_table.field_names = ["Basic Stats", "Value"]

        # Number of views
        basic_stats_table.add_row(["Number of Views", len(self.views)])

        # Number of geom verified matches
        total_matches = sum(pairwise_data["n_matches"]) if pairwise_data["n_matches"] else 0
        basic_stats_table.add_row(
            ["Total Matches (Verified)", sum(pairwise_data["matched_successful"])]
        )

        # Number of tracks
        basic_stats_table.add_row(["Number of Tracks", len(self.tracks)])

        tracks_table = PrettyTable()
        tracks_table.field_names = ["Tracks", "Value"]

        for key, value in stats_tracks.items():
            if isinstance(value, int):  # Check for integer
                tracks_table.add_row([key, f"{value}"])  # Display integer without decimal places
            elif isinstance(value, float):  # Check for float
                tracks_table.add_row([key, f"{value:.3f}"])  # Format float to 3 decimal places
            if isinstance(value, list):
                tracks_table.add_row(
                    [f"{key} Mean", f"{np.mean(value):.3f}"]
                )  # Format to 3 decimal places
                tracks_table.add_row(
                    [f"{key} Std", f"{np.std(value):.3f}"]
                )  # Format to 3 decimal places
                tracks_table.add_row(
                    [f"{key} Min", f"{np.min(value)}"]
                )  # Format to 3 decimal places
                tracks_table.add_row(
                    [f"{key} Max", f"{np.max(value)}"]
                )  # Format to 3 decimal places

        # Create a new table for Matches
        matches_table = PrettyTable()
        matches_table.field_names = ["Matches", "Value"]

        # Add pairwise statistics to the Matches table
        matches_table.add_row(["Total Pairs", len(pairwise_data["matched_successful"])])
        matches_table.add_row(
            ["Images Matched Successfully", sum(pairwise_data["matched_successful"])]
        )
        if pairwise_data["n_matches"]:
            matches_table.add_row(
                ["Average #Matches per Pair", int(np.mean(pairwise_data["n_matches"]))]
            )
            matches_table.add_row(
                ["Std #Matches per Pair", int(np.std(pairwise_data["n_matches"]))]
            )
            matches_table.add_row(
                ["Min #Matches per Pair", int(np.min(pairwise_data["n_matches"]))]
            )
            matches_table.add_row(
                ["Max #Matches per Pair", int(np.max(pairwise_data["n_matches"]))]
            )

        # Add connected component statistics
        component_table = PrettyTable()
        component_table.field_names = ["Component Stats", "Value"]
        component_table.add_row(["Number of Components", self.component_stats["n_components"]])
        component_table.add_row(
            ["Largest Component Size", self.component_stats["largest_component_size"]]
        )

        # Write statistics to a file if specified
        if write_to_file:
            with open(file_path, "w") as f:
                f.write("Basic Statistics:\n")
                f.write(str(basic_stats_table) + "\n\n")
                f.write("Track Statistics:\n")
                f.write(str(tracks_table) + "\n\n")
                f.write("Matches Statistics:\n")
                f.write(str(matches_table) + "\n")
                f.write("\nComponent Statistics:\n")
                f.write(str(component_table) + "\n")

        # Return all tables

        if wandb.run is not None:
            wandb.run.summary["#Images"] = len(self.views)
            wandb.run.summary["Largest Connected Component"] = self.component_stats[
                "largest_component_size"
            ]
            wandb.log(
                {
                    "Basic Statistics": wandb.Table(
                        data=basic_stats_table.rows, columns=basic_stats_table.field_names
                    )
                }
            )
            wandb.log(
                {
                    "Track Statistics": wandb.Table(
                        data=tracks_table.rows, columns=tracks_table.field_names
                    )
                }
            )
            wandb.log(
                {
                    "Matches Statistics": wandb.Table(
                        data=matches_table.rows, columns=matches_table.field_names
                    )
                }
            )
            wandb.log(
                {
                    "Component Statistics": wandb.Table(
                        data=component_table.rows, columns=component_table.field_names
                    )
                }
            )

        return [basic_stats_table, tracks_table, matches_table, component_table]

    def save_images(self, save_dir):
        for idx, view in enumerate(self.views):
            # Convert the NumPy array to a PIL Image
            img = Image.fromarray(view.img.astype("uint8"))  # Ensure the array is in uint8 format
            img.save(save_dir / f"{idx}_{view.name}.png")

    def save_verified_matches_png(self, save_dir):
        for i in range(len(self.views)):
            for j in range(i + 1, len(self.views)):
                pair = self.pairs_mat[i][j]
                if pair is not None:
                    pair.draw_matches().savefig(save_dir / f"{i}_{j}_{pair.n_matches}.png")

    def init_gt(self):
        """
        utility function to initialize the ground truth data
        1. select one image pair, use keypoints of the first image as "GT"
        2. unproject them into 3d space -> gt 3d points
        3. project them into other views (including intersection test)
        4. filter which gt points belong in which image based on this
        """
        gt_pair = self.pairs_mat[0][1]
        gt_pair.scenegraph = self
        v1, v2 = gt_pair.v1, gt_pair.v2
        mesh_path = str(self.scene.scans_dir / "mesh_aligned_0.05.ply")

        # set the keypoint idxs from the matches of the pair
        gt_pair.set_active_keypoint_idxs()  # don't like this function, could lead to some bugs??
        points_3d = v1.get_gt_points_from_keypoints(mesh_path)
        # filter points that did not hit anything
        filter_gt = np.linalg.norm(points_3d, axis=1) == np.inf

        points_3d = torch.tensor(points_3d[~filter_gt])
        v1.filter_active_keypoint_idxs(~filter_gt)

        # project those points into other views
        for i in range(len(self.views)):
            projections_2d = self.views[i].project_3d_points_onto_image_plane(points_3d)
            tmp = projections_2d.clone()
            # check if projections land on image plane
            H, W = self.views[i].shape
            # on_image_plane = ((projections_2d.T[0] >= 0) & (projections_2d.T[0] <= W)) & ((projections_2d.T[1] >= 0) & (projections_2d.T[1] <= H))
            # reproject points to check for consistency issue
            points_3d_other = self.views[i].get_gt_points_from_keypoints(mesh_path, projections_2d)

            same_3d_point = np.isclose(points_3d, points_3d_other).all(
                axis=1
            )  # similar points are marked as "True"

            # filter 3d points and/or 2d projections
            points_3d_filter = same_3d_point  # & on_image_plane # to filter out which points can be seen in the image

            self.views[i].projections_2d = projections_2d
            self.views[i].points_3d_filter = points_3d_filter

        self.points_3d = points_3d
        self._points_3d_gt = points_3d.clone()

    def init_noised_cameras(self, noise_level_rotation=0.0, noise_level_translation=0.0):
        """
        utility function for debugging:
        purpose:
        noise the data that we have to have a running bundje adjustment
        """
        import roma

        from opt.residuals_func import noise_input
        from utils.camera import Camera

        for view in self.views:
            # noising logic needs to be implemented here?
            rotation = view.camera_gt.rotation.clone()
            translation = view.camera_gt.translation.clone()
            K = view.camera_gt.K.clone()
            parameterization = view.camera_gt.parameterization

            # add noise
            rotation += roma.random_rotvec() * noise_level_rotation
            translation += noise_input(translation, noise_level=noise_level_translation)

            view.camera = Camera.from_params(rotation, translation, K, parameterization)

    def init_noised_points_3d(self, noise_level=0.0):
        """
        utility function for debugging:
        purpose:
        noise the data that we have to have a running bundje adjustment
        """
        from opt.residuals_func import noise_input

        self._points_3d_gt = self.points_3d.clone()
        self._points_3d += noise_input(self._points_3d, noise_level=noise_level)
        self._points_3d = torch.tensor(self._points_3d)

    # stuff for bundle adjustment
    @property
    def points_3d(self):
        return self._points_3d

    @points_3d.setter
    def points_3d(self, points_3d):
        self._points_3d = points_3d

    @property
    def points_3d_gt(self):
        return self._points_3d_gt

    @property
    def rotations(self):
        rotations = list()
        for view in self.views:
            rotations.append(view.camera.rotation)
        return torch.stack(rotations)

    @property
    def rotations_gt(self):
        rotations_gt = list()
        for view in self.views:
            rotations_gt.append(view.camera_gt.rotation)
        return torch.stack(rotations_gt)

    @property
    def translations(self):
        translations = list()
        for view in self.views:
            translations.append(view.camera.translation)
        return torch.stack(translations)

    @property
    def translations_gt(self):
        translations_gt = list()
        for view in self.views:
            translations_gt.append(view.camera_gt.translation)
        return torch.stack(translations_gt)

    @property
    def reprojections(self):
        reprojections = list()
        for view in self.views:
            reprojections.append(view.projections_2d.clone())
        # make homogeneous
        reprojections = torch.stack(reprojections)
        return reprojections.transpose(1, 2)

    @property
    def reprojections_filter(self):
        reprojections_filter = list()
        for view in self.views:
            reprojections_filter.append(view.points_3d_filter)
        return np.array(reprojections_filter)

    def get_ba_data_scipy(self, use_gt=False, fix_first_camera=False):
        if use_gt:
            camera_params = (
                torch.hstack((self.rotations_gt, self.translations_gt)).flatten().detach()
            )
            point_params = self.points_3d_gt.flatten().detach()
        else:
            camera_params = torch.hstack((self.rotations, self.translations)).flatten().detach()
            point_params = self.points_3d.flatten().detach()
        params_all = torch.concat((camera_params, point_params)).numpy()
        n_cameras = self.rotations.shape[0]
        n_points = self.points_3d.shape[0]
        point_ids = torch.arange(self.reprojections_filter.shape[1])
        point_ids = point_ids.repeat(self.reprojections_filter.shape[0])
        point_ids = point_ids[self.reprojections_filter.flatten()]

        camera_ids = self.reprojections_filter.astype(int)
        for i in torch.arange(self.reprojections_filter.shape[0]):
            camera_ids[i] = i
        camera_ids = camera_ids[self.reprojections_filter].flatten()

        if fix_first_camera:
            first_camera_params = camera_params[:6]
            camera_params_rest = camera_params[6:]
            params_tmp = params_all[6:]
            tmp = params_tmp[: (n_cameras - 1) * 6].reshape((n_cameras - 1, 6))
            np.concatenate((first_camera_params[None], tmp))
        else:
            return dict(
                params_all=params_all,
                n_cameras=n_cameras,
                n_points=n_points,
                point_ids=point_ids,
                camera_ids=camera_ids,
            )

    def get_ba_data_torch(self):
        # get GT params
        camera_params = torch.hstack((self.rotations_gt, self.translations_gt)).flatten().detach()
        point_params = self.points_3d_gt.flatten().detach()
        params_gt = torch.concat((camera_params, point_params))

        # get optim params
        camera_params = torch.hstack((self.rotations, self.translations)).flatten().detach()
        point_params = self.points_3d.flatten().detach()
        params = torch.concat((camera_params, point_params))

        n_cameras = self.rotations.shape[0]
        K = self.views[0].camera_gt.K
        n_points = self.points_3d.shape[0]
        point_ids = torch.arange(self.reprojections_filter.shape[1])
        point_ids = point_ids.repeat(self.reprojections_filter.shape[0])
        point_ids = point_ids[self.reprojections_filter.flatten()]

        camera_ids = self.reprojections_filter.astype(int)
        for i in torch.arange(self.reprojections_filter.shape[0]):
            camera_ids[i] = i
        camera_ids = camera_ids[self.reprojections_filter].flatten()

        points_2d = self.reprojections.transpose(1, 2)[self.reprojections_filter]
        return dict(
            params=params,
            params_gt=params_gt,
            n_cameras=n_cameras,
            n_points=n_points,
            point_indices=point_ids,
            camera_indices=camera_ids,
            K=K,
            points_2d=points_2d,
        )

    def plot_track(self, track_idx):
        track = self.tracks[track_idx]
        images = list()
        kps = list()

        for feature in track.views:
            view_idx, kp_idx = feature
            images.append(self.views[view_idx].img)
            kps.append(self.views[view_idx].kp[kp_idx].pt)

        return plot_keypoints_connections(images, kps)

    def plot_views(self):
        images = list()
        for view in self.views:
            images.append(view.img)

        return plot_views(images)

    def get_largest_connected_component(self):
        """Find the largest connected component of views in the scene graph.

        Returns:
            List[int]: List of view indices in the largest connected component
        """
        # Create adjacency matrix of view connections through tracks
        n_views = len(self.views)
        adjacency = torch.zeros((n_views, n_views), dtype=torch.bool)

        # Fill adjacency matrix based on tracks
        for track in self.tracks:
            view_indices = [v[0] for v in track.views]
            # Add edges between all pairs of views that share this track
            for i in range(len(view_indices)):
                for j in range(i + 1, len(view_indices)):
                    adjacency[view_indices[i], view_indices[j]] = True
                    adjacency[view_indices[j], view_indices[i]] = True

        # Find connected components using DFS
        components = []
        visited = set()

        def dfs(node, current_component):
            visited.add(node)
            current_component.append(node)
            for neighbor in torch.where(adjacency[node])[0]:
                if neighbor.item() not in visited:
                    dfs(neighbor.item(), current_component)

        # Find all connected components
        for node in range(n_views):
            if node not in visited:
                current_component = []
                dfs(node, current_component)
                components.append(current_component)

        # Return the largest component
        largest_component = max(components, key=len)

        # Add statistics about components to the statistics output
        self.component_stats = {
            "n_components": len(components),
            "component_sizes": [len(c) for c in components],
            "largest_component_size": len(largest_component),
            "largest_component_ratio": len(largest_component) / n_views,
        }

        return largest_component
