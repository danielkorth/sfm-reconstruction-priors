"""Utils for debugging on how to integrate DUSt3R into the Bundle Adjustment Energy Landscape /
Pipeline.

mainly used in combination with the notebook:
"""

import numpy as np
import roma
import torch

from utils.dust3r import (
    export_pts,
    get_idx_from_pair,
    get_output_from_idx,
    transform_coordinates_resize_crop,
)
from utils.metrics import relative_rotation_error, relative_translation_error


def preprocess_pointmaps_from_location(location0, location1, output, tracks, edge=(0, 1)):
    """Preprocess pointmaps from location0 and location1 using DUSt3R output.

    Args:
        location0: List of keypoints from location0
        location1: List of keypoints from location1
        output: DUSt3R output dictionary
        tracks: List of tracks
        edge: Tuple of (0, 1) for the edge to use for the pointmaps
    """

    def extract_pointmaps_from_location(
        location0, location1, output, input_res=(1168, 1752), edge=(0, 1)
    ):
        """Assign confidence values to keypoints in a pair of images using DUSt3R output.

        Args:
            pair: Pair object containing matched keypoints
            output: DUSt3R output dictionary
            match_idx: Index of the match to use for confidence assignment
            input_res: Tuple of (height, width) for input image resolution

        Returns:
            Tuple of (pts, conf) where:
                pts: List of point clouds for both images
                conf: List of confidence maps for both images
        """
        # ----- select pair -----
        idx = get_idx_from_pair(output, edge)
        stuff = get_output_from_idx(output, idx)

        # ----- get points and conf -----
        pts = [stuff["pts3d_1"], stuff["pts3d_2"]]
        conf = [stuff["conf_1"], stuff["conf_2"]]

        # Transform coordinates to DUSt3R resolution
        location0_cvt = transform_coordinates_resize_crop(location0, input_res, 224)
        location1_cvt = transform_coordinates_resize_crop(location1, input_res, 224)

        # Round to nearest integer
        location0_cvt = np.round(location0_cvt).astype(int)
        location1_cvt = np.round(location1_cvt).astype(int)

        zero = 0
        one = 1

        if edge[0] > edge[1]:
            zero, one = one, zero
        # pay close attention herer on the indexing of pts and conf. this can fuck up everything :)
        pts1 = pts[zero][location0_cvt[0][1], location0_cvt[0][0]]
        pts2 = pts[one][location1_cvt[0][1], location1_cvt[0][0]]

        conf1 = conf[zero][location0_cvt[0][1], location0_cvt[0][0]]
        conf2 = conf[one][location1_cvt[0][1], location1_cvt[0][0]]

        pts = torch.stack([pts1, pts2])
        conf = torch.stack([conf1, conf2])

        return {"pts": pts, "conf": conf}

    tracks = np.repeat(tracks, 2, axis=0)
    pointmaps = []
    for pts1, pts2 in zip(location0, location1):
        res = extract_pointmaps_from_location(
            pts1.unsqueeze(0), pts2.unsqueeze(0), output, edge=edge
        )
        pointmaps.append(res)

    pointmaps_pts = [x["pts"] for x in pointmaps]
    pointmaps_conf = [x["conf"] for x in pointmaps]

    pointmaps_pts = torch.cat(pointmaps_pts, dim=0)
    pointmaps_conf = torch.cat(pointmaps_conf, dim=0)

    pointmaps_pts = pointmaps_pts.double()
    pointmaps_conf = pointmaps_conf.double()

    R, t, s = roma.rigid_points_registration(
        pointmaps_pts.double(), tracks, pointmaps_conf, compute_scaling=True
    )
    aligned_pts = (s * R @ pointmaps_pts.T).T + t

    # set stuff to requires grad
    rotvec = roma.rotmat_to_rotvec(R)
    rotvec.requires_grad_(True)
    t.requires_grad_(True)
    s.requires_grad_(True)

    return {
        "aligned_pts": aligned_pts,
        "pointmaps_pts": pointmaps_pts,
        "pointmaps_conf": pointmaps_conf,
        "R": R,
        "t": t,
        "s": s,
        "rotvec": rotvec,
    }


def preprocess_pointmaps_from_matches(pair, output, match_idxs, tracks, edge=(0, 1)):
    def extract_pointmaps_from_match(pair, output, match_idx, input_res=(1168, 1752), edge=(0, 1)):
        """Assign confidence values to keypoints in a pair of images using DUSt3R output.

        Args:
            pair: Pair object containing matched keypoints
            output: DUSt3R output dictionary
            match_idx: Index of the match to use for confidence assignment
            input_res: Tuple of (height, width) for input image resolution

        Returns:
            Tuple of (pts, conf) where:
                pts: List of point clouds for both images
                conf: List of confidence maps for both images
        """
        # ----- select pair -----
        idx = get_idx_from_pair(output, edge)
        stuff = get_output_from_idx(output, idx)

        # ----- get points and conf -----
        pts = [stuff["pts3d_1"], stuff["pts3d_2"]]
        conf = [stuff["conf_1"], stuff["conf_2"]]

        # Get matched keypoints
        idx0 = pair.matches[match_idx].query_idx
        idx1 = pair.matches[match_idx].database_idx

        # Convert keypoints to numpy arrays
        location0 = np.array(pair.v1.kp[idx0].pt).reshape(1, 2)
        location1 = np.array(pair.v2.kp[idx1].pt).reshape(1, 2)

        # Transform coordinates to DUSt3R resolution
        # print(match_idx)
        location0_cvt = transform_coordinates_resize_crop(location0, input_res, 224)
        location1_cvt = transform_coordinates_resize_crop(location1, input_res, 224)

        # Round to nearest integer
        location0_cvt = np.round(location0_cvt).astype(int)
        location1_cvt = np.round(location1_cvt).astype(int)

        zero = 0
        one = 1

        if edge[0] > edge[1]:
            zero, one = one, zero
        # pay close attention herer on the indexing of pts and conf. this can fuck up everything :)
        pts1 = pts[zero][location0_cvt[0][1], location0_cvt[0][0]]
        pts2 = pts[one][location1_cvt[0][1], location1_cvt[0][0]]

        conf1 = conf[zero][location0_cvt[0][1], location0_cvt[0][0]]
        conf2 = conf[one][location1_cvt[0][1], location1_cvt[0][0]]

        pts = torch.stack([pts1, pts2])
        conf = torch.stack([conf1, conf2])

        return {"pts": pts, "conf": conf}

    tracks = np.repeat(tracks, 2, axis=0)
    pointmaps = []
    for match_idx in match_idxs:
        res = extract_pointmaps_from_match(pair, output, match_idx, edge=edge)
        pointmaps.append(res)

    pointmaps_pts = [x["pts"] for x in pointmaps]
    pointmaps_conf = [x["conf"] for x in pointmaps]

    pointmaps_pts = torch.cat(pointmaps_pts, dim=0)
    pointmaps_conf = torch.cat(pointmaps_conf, dim=0)

    pointmaps_pts = pointmaps_pts.double()
    pointmaps_conf = pointmaps_conf.double()

    R, t, s = roma.rigid_points_registration(
        pointmaps_pts.double(), tracks, pointmaps_conf, compute_scaling=True
    )
    aligned_pts = (s * R @ pointmaps_pts.T).T + t

    # set stuff to requires grad
    rotvec = roma.rotmat_to_rotvec(R)
    rotvec.requires_grad_(True)
    t.requires_grad_(True)
    s.requires_grad_(True)

    return {
        "aligned_pts": aligned_pts,
        "pointmaps_pts": pointmaps_pts,
        "pointmaps_conf": pointmaps_conf,
        "R": R,
        "t": t,
        "s": s,
        "rotvec": rotvec,
    }


def align_pointmaps_to_tracks(pointmaps_pts, pointmaps_conf, tracks):
    R, t, s = roma.rigid_points_registration(
        pointmaps_pts.double(), tracks, pointmaps_conf, compute_scaling=True
    )
    return (s * R @ pointmaps_pts.T).T + t


def extract_keypoints_and_tracks_from_twoview_recon(pair, rec, chosen_indices):
    """Extract ground truth 2D points, rotations, translations, and 3D tracks from reconstruction
    data.

    Args:
        pair: Pair object containing matched keypoints
        rec: Reconstruction dictionary containing R, t, and points
        chosen_indices: List of indices to extract matches for

    Returns:
        Dictionary containing:
            pts0_gt: Ground truth 2D points in view 0
            pts1_gt: Ground truth 2D points in view 1
            rotvec_0: Rotation vector for view 0
            rotvec_1: Rotation vector for view 1
            t_0: Translation for view 0
            t_1: Translation for view 1
            tracks: 3D points corresponding to chosen matches
    """
    pts1_gt = []
    pts0_gt = []

    for i in chosen_indices:
        # Get matched keypoints
        idx0 = pair.matches[i].query_idx
        idx1 = pair.matches[i].database_idx

        # Convert keypoints to numpy arrays
        location0 = np.array(pair.v1.kp[idx0].pt).reshape(1, 2)
        location1 = np.array(pair.v2.kp[idx1].pt).reshape(1, 2)
        pts0_gt.append(location0)
        pts1_gt.append(location1)

    pts0_gt = torch.tensor(pts0_gt).squeeze()
    pts1_gt = torch.tensor(pts1_gt).squeeze()

    rotvec_0 = roma.rotmat_to_rotvec(torch.from_numpy(rec["R1"]))
    rotvec_1 = roma.rotmat_to_rotvec(torch.from_numpy(rec["R2"]))
    t_0 = torch.from_numpy(rec["t1"])
    t_1 = torch.from_numpy(rec["t2"])

    tracks = torch.from_numpy(rec["points"][chosen_indices])

    rotvec_0_gt = pair.v1.camera_gt.rotation
    rotvec_1_gt = pair.v2.camera_gt.rotation
    t_0_gt = pair.v1.camera_gt.translation.unsqueeze(1)
    t_1_gt = pair.v2.camera_gt.translation.unsqueeze(1)
    K = pair.v1.camera_gt.K
    return {
        "pts0_gt": pts0_gt,
        "pts1_gt": pts1_gt,
        "rotvec_0": rotvec_0,
        "rotvec_1": rotvec_1,
        "t_0": t_0,
        "t_1": t_1,
        "tracks": tracks,
        "rotvec_0_gt": rotvec_0_gt,
        "rotvec_1_gt": rotvec_1_gt,
        "t_0_gt": t_0_gt,
        "t_1_gt": t_1_gt,
        "K": K,
    }


def extract_keypoints_and_tracks_by_reprojection_and_sampling(pair, rec, tracks=None, n_tracks=10):
    """Extract ground truth 2D points, rotations, translations, and 3D tracks from reconstruction
    data.

    Args:
        pair: Pair object containing matched keypoints
        rec: Reconstruction dictionary containing R, t, and points
        chosen_indices: List of indices to extract matches for

    Returns:
        Dictionary containing:
            pts0_gt: Ground truth 2D points in view 0
            pts1_gt: Ground truth 2D points in view 1
            rotvec_0: Rotation vector for view 0
            rotvec_1: Rotation vector for view 1
            t_0: Translation for view 0
            t_1: Translation for view 1
            tracks: 3D points corresponding to chosen matches
    """
    if tracks is None:
        from sklearn.mixture import GaussianMixture

        points = rec["points"]
        gmm = GaussianMixture(n_components=len(points), random_state=42)
        gmm.fit(points)

        tracks = gmm.sample(n_tracks)[0]
        tracks = torch.from_numpy(tracks)
    else:
        tracks = tracks

    K = pair.v1.camera_gt.K
    repro_v1 = (K @ (pair.v1.camera_gt.R @ tracks.T + pair.v1.camera_gt.t)).T
    repro_v1 = repro_v1 / repro_v1[:, 2:]
    repro_v2 = (K @ (pair.v2.camera_gt.R @ tracks.T + pair.v2.camera_gt.t)).T
    repro_v2 = repro_v2 / repro_v2[:, 2:]

    rotvec_0 = pair.v1.camera_gt.rotation.clone()
    rotvec_1 = pair.v2.camera_gt.rotation.clone()
    t_0 = pair.v1.camera_gt.translation.clone().unsqueeze(1)
    t_1 = pair.v2.camera_gt.translation.clone().unsqueeze(1)

    rotvec_0_gt = pair.v1.camera_gt.rotation
    rotvec_1_gt = pair.v2.camera_gt.rotation
    t_0_gt = pair.v1.camera_gt.translation.unsqueeze(1)
    t_1_gt = pair.v2.camera_gt.translation.unsqueeze(1)

    return {
        "pts0_gt": repro_v1[:, :2],
        "pts1_gt": repro_v2[:, :2],
        "rotvec_0": rotvec_0,
        "rotvec_1": rotvec_1,
        "t_0": t_0,
        "t_1": t_1,
        "tracks": tracks,
        "rotvec_0_gt": rotvec_0_gt,
        "rotvec_1_gt": rotvec_1_gt,
        "t_0_gt": t_0_gt,
        "t_1_gt": t_1_gt,
        "K": K,
    }


def optimize_loop(
    optimizer, forward_dust3r, forward_ba, params, use_dust3r=True, use_ba=True, n_iter=1000
):
    # ----- unpack params -----
    tracks = params["tracks"]
    rotvec_0 = params["rotvec_0"]
    rotvec_1 = params["rotvec_1"]
    t_0 = params["t_0"]
    t_1 = params["t_1"]
    pts0_gt = params["pts0_gt"]
    pts1_gt = params["pts1_gt"]
    K = params["K"]
    rotvec_0_gt = params["rotvec_0_gt"]
    rotvec_1_gt = params["rotvec_1_gt"]
    t_0_gt = params["t_0_gt"]
    t_1_gt = params["t_1_gt"]
    pointmaps_01 = params["pointmaps_01"]
    pointmaps_10 = params["pointmaps_10"]
    pts0_gt_normalized = params["pts0_gt_normalized"]
    pts1_gt_normalized = params["pts1_gt_normalized"]

    # ----- initialize lists -----
    loss_history = []
    loss_01_history = []
    loss_10_history = []
    loss_0_ba_history = []
    loss_1_ba_history = []
    rerror_history = []
    terror_history = []
    dust3r_energy_history = []
    ba_energy_history = []
    dist_01_history = []
    dist_10_history = []
    dist_0_ba_history = []
    dist_1_ba_history = []
    dist_0_ba_normalized_history = []
    dist_1_ba_normalized_history = []

    # ----- optimization loop -----
    for i in range(n_iter):
        optimizer.zero_grad()
        tracks_double = tracks.repeat_interleave(2, dim=0)
        error_01 = forward_dust3r(
            pointmaps_01["rotvec"],
            pointmaps_01["t"],
            pointmaps_01["s"],
            pointmaps_01["pointmaps_pts"],
            pointmaps_01["pointmaps_conf"],
            tracks_double,
        )
        loss_01 = error_01["cost"]
        dist_01 = error_01["avg_l2_dist"]
        error_10 = forward_dust3r(
            pointmaps_10["rotvec"],
            pointmaps_10["t"],
            pointmaps_10["s"],
            pointmaps_10["pointmaps_pts"],
            pointmaps_10["pointmaps_conf"],
            tracks_double,
        )
        loss_10 = error_10["cost"]
        dist_10 = error_10["avg_l2_dist"]

        out_0_ba = forward_ba(tracks, pts0_gt, pts0_gt_normalized, K, rotvec_0, t_0)
        loss_0_ba = out_0_ba["cost"]
        dist_0_ba = out_0_ba["avg_pixel_error"]
        dist_0_ba_normalized = out_0_ba["avg_error"]

        out_1_ba = forward_ba(tracks, pts1_gt, pts1_gt_normalized, K, rotvec_1, t_1)
        loss_1_ba = out_1_ba["cost"]
        dist_1_ba = out_1_ba["avg_pixel_error"]
        dist_1_ba_normalized = out_1_ba["avg_error"]

        if not use_dust3r:
            loss_01 = torch.tensor(0.0)
            loss_10 = torch.tensor(0.0)
        if not use_ba:
            loss_0_ba = torch.tensor(0.0)
            loss_1_ba = torch.tensor(0.0)

        loss = loss_01 + loss_10 + loss_0_ba + loss_1_ba
        loss.backward()
        optimizer.step()

        # Store and log loss
        loss_history.append(loss.item())
        loss_01_history.append(loss_01.item())
        loss_10_history.append(loss_10.item())
        loss_0_ba_history.append(loss_0_ba.item())
        loss_1_ba_history.append(loss_1_ba.item())
        dust3r_energy_history.append(loss_01.item() + loss_10.item())
        ba_energy_history.append(loss_0_ba.item() + loss_1_ba.item())
        dist_01_history.append(dist_01.item())
        dist_10_history.append(dist_10.item())
        dist_0_ba_history.append(dist_0_ba.item())
        dist_1_ba_history.append(dist_1_ba.item())
        dist_0_ba_normalized_history.append(dist_0_ba_normalized.item())
        dist_1_ba_normalized_history.append(dist_1_ba_normalized.item())

        with torch.no_grad():
            r0 = roma.rotvec_to_rotmat(rotvec_0)
            r1 = roma.rotvec_to_rotmat(rotvec_1)
            r0_gt = roma.rotvec_to_rotmat(rotvec_0_gt)
            r1_gt = roma.rotvec_to_rotmat(rotvec_1_gt)

            # Calculate and store errors
            rerror = relative_rotation_error(r0, r1, r0_gt, r1_gt)
            t0_w = r0.T @ -t_0
            t1_w = r1.T @ -t_1
            t0_gt_w = r0_gt.T @ -t_0_gt
            t1_gt_w = r1_gt.T @ -t_1_gt
            terror = relative_translation_error(t0_w.T, t1_w.T, t0_gt_w.T, t1_gt_w.T)
            # terror = relative_translation_error(t1_w.T, t0_w.T, t1_gt_w.T, t0_gt_w.T)

            rerror_history.append(rerror.mean().item())
            terror_history.append(terror.mean().item())

            if i % 1 == 0:
                print(
                    f"Step {i}: Loss = {loss.item():.4f} | Loss 01 = {loss_01.item():.4f} | Loss 10 = {loss_10.item():.4f} | Loss 0_ba = {loss_0_ba.item():.6f} | Loss 1_ba = {loss_1_ba.item():.6f}"
                )
                print(f"Relative rotation error: {rerror.mean():.4f}")
                print(f"Relative translation error: {terror.mean():.4f}")
                print(
                    f'Scaling factor 01: {params["pointmaps_01"]["s"]} | Scaling factor 10: {params["pointmaps_10"]["s"]}'
                )

    return {
        "loss_history": loss_history,
        "loss_01_history": loss_01_history,
        "loss_10_history": loss_10_history,
        "loss_0_ba_history": loss_0_ba_history,
        "loss_1_ba_history": loss_1_ba_history,
        "rerror_history": rerror_history,
        "terror_history": terror_history,
        "dust3r_energy_history": dust3r_energy_history,
        "ba_energy_history": ba_energy_history,
        "dist_01_history": dist_01_history,
        "dist_10_history": dist_10_history,
        "dist_0_ba_history": dist_0_ba_history,
        "dist_1_ba_history": dist_1_ba_history,
        "dist_0_ba_normalized_history": dist_0_ba_normalized_history,
        "dist_1_ba_normalized_history": dist_1_ba_normalized_history,
    }
