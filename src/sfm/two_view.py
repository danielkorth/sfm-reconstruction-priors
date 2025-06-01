import cv2
import numpy as np


def two_view_relative(image1, image2, K):
    # extract image features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # match image features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # ratio test to filter out points
    pts1 = list()
    pts2 = list()
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    # # ... existing code ...

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    # Normalize coordinates for better numerical stability
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None)

    # Compute Essential matrix directly with normalized coordinates
    E, mask = cv2.findEssentialMat(
        pts1_norm, pts2_norm, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=0.001
    )
    # E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=0.001)

    # select only inlier points
    pts1_norm = pts1_norm[mask.ravel() == 1]
    pts2_norm = pts2_norm[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # recover pose and triangulate with stricter threshold
    retval, R, t, mask, points_3d = cv2.recoverPose(
        E, pts1_norm, pts2_norm, np.eye(3), distanceThresh=0.1
    )
    # retval, R, t, mask, points_3d = cv2.recoverPose(E, pts1, pts2, K, distanceThresh=0.1)

    return dict(R=R, t=t, mask=mask, pts1=pts1, pts2=pts2, points_3d=points_3d)

    # # estimate fundamental/essential matrix
    # # V1
    # # 1. 7Point: cv2.FM_7POINT
    # # 2. 8Point: cv2.FM_8POINT
    # # 3. FM_RANSAC: cv2.FM_RANSAC
    # # 4. FM_LMEDS: cv2.FM_LMEDS
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # # select only inlier
    # pts1 = pts1[mask.ravel() == 1]
    # pts2 = pts2[mask.ravel() == 1]

    # E = K.T @ F @ K

    # # V2

    # # V3

    # # recover relative pose
    # # retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    # # return dict(R=R, t=t, mask=mask, pts1=pts1, pts2=pts2)

    # # recover pose and triangulate
    # retval, R, t, mask, points_3d = cv2.recoverPose(E, pts1, pts2, K, distanceThresh=0.5)
    # return dict(R=R, t=t, mask=mask, pts1=pts1, pts2=pts2, points_3d=points_3d)


def relative_to_global(R_glb, t_glb, R_rel, t_rel):
    """Takes a relative rotation and translation and aligns it into the "global" coordinate
    frame."""
    R_new = R_glb[:3, :3] @ R_rel[:3, :3]
    t_new = t_glb.squeeze() + R_glb[:3, :3] @ t_rel.squeeze()
    return R_new, t_new
