import roma
import torch
from jaxtyping import Float
from scipy.spatial import procrustes


def relative_rotation_error(
    R1: torch.Tensor, R2: torch.Tensor, R1_gt: torch.Tensor, R2_gt: torch.Tensor
) -> torch.Tensor:
    """Calculates the relative rotation error between two rotations and their ground truth values.
    The error is computed as the angle between R_rel_pred and R_rel_gt in degrees.

    Args:
        R1: Predicted rotation matrices (B, 3, 3)
        R2: Predicted rotation matrices (B, 3, 3)
        R1_gt: Ground truth rotation matrices (B, 3, 3)
        R2_gt: Ground truth rotation matrices (B, 3, 3)

    Returns:
        Tensor of rotation errors in degrees (B,)
    """
    if R1.ndim == 2:
        R1 = R1.unsqueeze(0)
        R2 = R2.unsqueeze(0)
        R1_gt = R1_gt.unsqueeze(0)
        R2_gt = R2_gt.unsqueeze(0)

    # Calculate relative rotations
    R_rel_pred = R2 @ R1.transpose(1, 2)  # (B, 3, 3)
    R_rel_gt = R2_gt @ R1_gt.transpose(1, 2)  # (B, 3, 3)

    # Calculate error angle
    R_error = R_rel_gt.transpose(1, 2) @ R_rel_pred  # (B, 3, 3)
    traces = torch.diagonal(R_error, dim1=1, dim2=2).sum(dim=1)  # (B,)
    angles = torch.arccos(((traces - 1) / 2).clamp(-1, 1))  # (B,)
    return torch.rad2deg(angles)  # (B,)


def relative_translation_error(
    t1_world: torch.Tensor,
    t2_world: torch.Tensor,
    t1_gt_world: torch.Tensor,
    t2_gt_world: torch.Tensor,
    eps: float = 1e-15,
) -> torch.Tensor:
    """Calculates the relative translation error between translations and their ground truth. The
    error is computed as the angle between normalized translation vectors.

    Args:
        t1_world, t2_world: Translation vectors in world space (B, 3)
        t1_gt_world, t2_gt_world: Ground truth translation vectors in world space (B, 3)
        eps: Small value to avoid division by zero

    Returns:
        Tensor of translation errors in degrees (B,)
    """

    if t1_world.ndim == 1:
        t1_world = t1_world.unsqueeze(0)
        t2_world = t2_world.unsqueeze(0)
        t1_gt_world = t1_gt_world.unsqueeze(0)
        t2_gt_world = t2_gt_world.unsqueeze(0)

    # Compute relative translations as differences in world space
    t_rel = t2_world - t1_world  # (B, 3)
    t_rel_gt = t2_gt_world - t1_gt_world  # (B, 3)

    # Normalize translations
    t_rel_norm = torch.norm(t_rel, dim=1, keepdim=True)  # (B, 1)
    t_rel = t_rel / (t_rel_norm + eps)  # (B, 3)

    t_rel_gt_norm = torch.norm(t_rel_gt, dim=1, keepdim=True)  # (B, 1)
    t_rel_gt = t_rel_gt / (t_rel_gt_norm + eps)  # (B, 3)

    # Calculate angle between normalized vectors
    cos_angle = torch.sum(t_rel * t_rel_gt, dim=1).clamp(-1, 1)  # (B,)
    angles = torch.arccos(cos_angle)  # (B,)

    return torch.rad2deg(angles)  # (B,)


def relative_accuracy(error: torch.Tensor, tau: float = 5.0) -> float:
    """Calculate the relative accuracy at threshold tau.

    Args:
        error: Tensor of errors in degrees (N,)
        tau: Threshold in degrees

    Returns:
        Percentage of pairs with error below threshold
    """
    return (error < tau).float().mean().item() * 100


def average_trajectory_error(
    rotations: Float[torch.Tensor, "n_cameras 3"],
    translations: Float[torch.Tensor, "n_cameras 3"],
    rotations_gt: Float[torch.Tensor, "n_cameras 3"],
    translations_gt: Float[torch.Tensor, "n_cameras 3"],
) -> float:
    """
    Calculate the average translation error (as calculated in FlowMap: https://arxiv.org/pdf/2404.15259).

    1. convert estimated and gt translations into world coordinate space
    2. rigidly align estimated translations to gt translations (or camera coordinates to be specific)
    3. calculate the average trajectory error

    Args:
        rotations: Estimated rotations (n_cameras, 3)
        translations: Estimated translations (n_cameras, 3)
        rotations_gt: Ground truth rotations (n_cameras, 3)
        translations_gt: Ground truth translations (n_cameras, 3)
    """
    # Convert to world coordinate space
    camera_center = roma.rotvec_to_rotmat(rotations).transpose(1, 2) @ -translations.reshape(
        -1, 3, 1
    )
    camera_center_gt = roma.rotvec_to_rotmat(rotations_gt).transpose(
        1, 2
    ) @ -translations_gt.reshape(-1, 3, 1)

    mtx1, mtx2, _ = procrustes(camera_center.squeeze(), camera_center_gt.squeeze())

    ate = ((mtx1 - mtx2) ** 2).mean() ** 0.5
    return ate.item()


def absolute_trajectory_error() -> float:
    pass


def registration_rate() -> float:
    pass


# UTILS
def read_metrics(file_path):
    metrics = {}
    with open(file_path) as f:
        for line in f:
            key, value = line.strip().split(": ")
            metrics[key] = float(value)
    return metrics


def write_metrics_to_file(
    file_path,
    average_metrics,
    stddev_metrics,
    average_metrics_colmap=None,
    stddev_metrics_colmap=None,
    average_metrics_dust3r=None,
    stddev_metrics_dust3r=None,
    failed_reads_mine=0,
    failed_reads_colmap=0,
    failed_reads_dust3r=0,
):
    with open(file_path, "w") as f:
        f.write(f"Failed Runs (Mine): {failed_reads_mine}\n")
        f.write(f"Failed Runs (COLMAP): {failed_reads_colmap}\n")
        f.write(f"Failed Runs (DUST3R): {failed_reads_dust3r}\n\n")

        f.write("Metrics (Mine)\n")
        f.write("Metric, Average, Standard Deviation\n")
        for key in average_metrics.keys():
            f.write(f"{key}, {average_metrics[key]:.3f}, {stddev_metrics[key]:.3f}\n")

        if average_metrics_colmap is not None and stddev_metrics_colmap is not None:
            f.write("\nMetrics (COLMAP)\n")
            f.write("Metric, Average, Standard Deviation\n")
            for key in average_metrics_colmap.keys():
                f.write(
                    f"{key}, {average_metrics_colmap[key]:.3f}, {stddev_metrics_colmap[key]:.3f}\n"
                )
        else:
            f.write("\nMetrics (COLMAP) not available.\n")

        if average_metrics_dust3r is not None and stddev_metrics_dust3r is not None:
            f.write("\nMetrics (DUST3R)\n")
            f.write("Metric, Average, Standard Deviation\n")
            for key in average_metrics_dust3r.keys():
                f.write(
                    f"{key}, {average_metrics_dust3r[key]:.3f}, {stddev_metrics_dust3r[key]:.3f}\n"
                )


THRESHOLDS = [x for x in range(1, 31, 1)]


def rotation_auc(R_pred: torch.Tensor, R_gt: torch.Tensor, thresholds: list = None) -> float:
    """Calculate the Area Under the Curve (AUC) for rotation errors.

    Args:
        R_pred: Predicted rotation matrices (N, 3, 3)
        R_gt: Ground truth rotation matrices (N, 3, 3)
        thresholds: List of thresholds in degrees to evaluate at.
                    Default is [5, 10, 15, 20, 25, 30]

    Returns:
        AUC value normalized between 0 and 1
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    # Calculate rotation errors
    R_error = R_gt.transpose(1, 2) @ R_pred  # (N, 3, 3)
    traces = torch.diagonal(R_error, dim1=1, dim2=2).sum(dim=1)  # (N,)
    angles = torch.rad2deg(torch.arccos(((traces - 1) / 2).clamp(-1, 1)))  # (N,)

    # Calculate accuracy at each threshold using relative_accuracy
    accuracies = []
    for tau in thresholds:
        acc = relative_accuracy(angles, tau)
        accuracies.append(acc)

    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(thresholds)):
        auc += (accuracies[i] + accuracies[i - 1]) * (thresholds[i] - thresholds[i - 1]) / 2

    # Normalize by the maximum possible area
    max_area = thresholds[-1] * 100
    return auc / max_area


def translation_auc(
    t_pred: torch.Tensor, t_gt: torch.Tensor, thresholds: list = None, eps: float = 1e-15
) -> float:
    """Calculate the Area Under the Curve (AUC) for translation errors.

    Args:
        t_pred: Predicted translation vectors (N, 3)
        t_gt: Ground truth translation vectors (N, 3)
        thresholds: List of thresholds in degrees to evaluate at.
                    Default is [5, 10, 15, 20, 25, 30]
        eps: Small value to avoid division by zero

    Returns:
        AUC value normalized between 0 and 1
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    # Normalize translations
    t_pred_norm = torch.norm(t_pred, dim=1, keepdim=True)  # (N, 1)
    t_pred_normalized = t_pred / (t_pred_norm + eps)  # (N, 3)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)  # (N, 1)
    t_gt_normalized = t_gt / (t_gt_norm + eps)  # (N, 3)

    # Calculate angle between normalized vectors
    cos_angle = torch.sum(t_pred_normalized * t_gt_normalized, dim=1).clamp(-1, 1)  # (N,)
    angles = torch.rad2deg(torch.arccos(cos_angle))  # (N,)

    # Calculate accuracy at each threshold using relative_accuracy
    accuracies = []
    for tau in thresholds:
        acc = relative_accuracy(angles, tau)
        accuracies.append(acc)

    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(thresholds)):
        auc += (accuracies[i] + accuracies[i - 1]) * (thresholds[i] - thresholds[i - 1]) / 2

    # Normalize by the maximum possible area
    max_area = thresholds[-1] * 100
    return auc / max_area


def relative_rotation_auc(
    R1: torch.Tensor,
    R2: torch.Tensor,
    R1_gt: torch.Tensor,
    R2_gt: torch.Tensor,
    thresholds: list = None,
) -> float:
    """Calculate the Area Under the Curve (AUC) for relative rotation errors.

    Args:
        R1: Predicted rotation matrices (B, 3, 3)
        R2: Predicted rotation matrices (B, 3, 3)
        R1_gt: Ground truth rotation matrices (B, 3, 3)
        R2_gt: Ground truth rotation matrices (B, 3, 3)
        thresholds: List of thresholds in degrees to evaluate at.
                    Default is [5, 10, 15, 20, 25, 30]

    Returns:
        AUC value normalized between 0 and 1
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    # Calculate errors
    errors = relative_rotation_error(R1, R2, R1_gt, R2_gt)

    # Calculate accuracy at each threshold using relative_accuracy
    accuracies = []
    for tau in thresholds:
        acc = relative_accuracy(errors, tau)
        accuracies.append(acc)

    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(thresholds)):
        auc += (accuracies[i] + accuracies[i - 1]) * (thresholds[i] - thresholds[i - 1]) / 2

    # Normalize by the maximum possible area
    max_area = thresholds[-1] * 100
    return auc / max_area


def relative_translation_auc(
    t1_world: torch.Tensor,
    t2_world: torch.Tensor,
    t1_gt_world: torch.Tensor,
    t2_gt_world: torch.Tensor,
    thresholds: list = None,
    eps: float = 1e-15,
) -> float:
    """Calculate the Area Under the Curve (AUC) for relative translation errors.

    Args:
        t1_world, t2_world: Translation vectors in world space (B, 3)
        t1_gt_world, t2_gt_world: Ground truth translation vectors in world space (B, 3)
        thresholds: List of thresholds in degrees to evaluate at.
                    Default is [5, 10, 15, 20, 25, 30]
        eps: Small value to avoid division by zero

    Returns:
        AUC value normalized between 0 and 1
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    # Calculate errors
    errors = relative_translation_error(t1_world, t2_world, t1_gt_world, t2_gt_world, eps)

    # Calculate accuracy at each threshold using relative_accuracy
    accuracies = []
    for tau in thresholds:
        acc = relative_accuracy(errors, tau)
        accuracies.append(acc)

    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(thresholds)):
        auc += (accuracies[i] + accuracies[i - 1]) * (thresholds[i] - thresholds[i - 1]) / 2

    # Normalize by the maximum possible area
    max_area = thresholds[-1] * 100
    return auc / max_area


def mean_average_accuracy(errors: torch.Tensor, thresholds: list = None) -> float:
    """Calculate the Mean Average Accuracy (mAA) across multiple thresholds.

    Args:
        errors: Tensor of errors in degrees (N,)
        thresholds: List of thresholds in degrees. Default is [5, 10, 15, 20, 25, 30]

    Returns:
        Mean Average Accuracy value (percentage)
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    accuracies = []
    for tau in thresholds:
        acc = relative_accuracy(errors, tau)
        accuracies.append(acc)

    return sum(accuracies) / len(accuracies)


def combined_accuracy(
    rotation_error: torch.Tensor, translation_error: torch.Tensor, tau: float = 5.0
) -> float:
    """Calculate the combined accuracy at threshold tau. A pose is considered correct only if both
    rotation and translation errors are below the threshold.

    Args:
        rotation_error: Tensor of rotation errors in degrees (N,)
        translation_error: Tensor of translation errors in degrees (N,)
        tau: Threshold in degrees

    Returns:
        Percentage of pairs with both errors below threshold
    """
    return ((rotation_error < tau) & (translation_error < tau)).float().mean().item() * 100


def combined_maa(
    rotation_error: torch.Tensor, translation_error: torch.Tensor, thresholds: list = None
) -> float:
    """Calculate the Mean Average Accuracy (mAA) for combined rotation and translation errors. A
    pose is considered correct only if both rotation and translation errors are below the
    threshold.

    Args:
        rotation_error: Tensor of rotation errors in degrees (N,)
        translation_error: Tensor of translation errors in degrees (N,)
        thresholds: List of thresholds in degrees. Default is [1, 5, 10, 15, 20, 25, 30]

    Returns:
        Mean Average Accuracy value (percentage)
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    accuracies = []
    for tau in thresholds:
        acc = combined_accuracy(rotation_error, translation_error, tau)
        accuracies.append(acc)

    return sum(accuracies) / len(accuracies)


def combined_auc(
    rotation_error: torch.Tensor, translation_error: torch.Tensor, thresholds: list = None
) -> float:
    """Calculate the Area Under the Curve (AUC) for combined rotation and translation errors.

    Following the Image Matching Benchmark, this calculates the area under the curve where
    the accuracy at each threshold is defined as min(RRA@τ, RTA@τ).

    Args:
        rotation_error: Tensor of rotation errors in degrees (N,)
        translation_error: Tensor of translation errors in degrees (N,)
        thresholds: List of thresholds in degrees. Default is [1, 5, 10, 15, 20, 25, 30]

    Returns:
        AUC value normalized between 0 and 1
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    # Calculate accuracy at each threshold
    accuracies = []
    for tau in thresholds:
        # Calculate individual accuracies
        rot_acc = relative_accuracy(rotation_error, tau)
        trans_acc = relative_accuracy(translation_error, tau)

        # Take the minimum of the two accuracies
        min_acc = min(rot_acc, trans_acc)
        accuracies.append(min_acc)

    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(thresholds)):
        auc += (accuracies[i] + accuracies[i - 1]) * (thresholds[i] - thresholds[i - 1]) / 2

    # Normalize by the maximum possible area
    max_area = thresholds[-1] * 100
    return auc / max_area
