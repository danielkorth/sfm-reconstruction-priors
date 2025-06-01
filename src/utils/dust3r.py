from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh


def show_dust3r_input(output):
    # Create a figure with 1 row and 2 columns
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Show the first image in the first subplot
    ax[0].imshow((output["view1"]["img"][0].permute(1, 2, 0) + 1) / 2)
    ax[0].set_title("View 1")  # Optional: set a title for the first image
    ax[0].axis("off")  # Optional: turn off the axis

    # Show the second image in the second subplot
    ax[1].imshow((output["view2"]["img"][0].permute(1, 2, 0) + 1) / 2)
    ax[1].set_title("View 2")  # Optional: set a title for the second image
    ax[1].axis("off")  # Optional: turn off the axis

    # Show the plot
    plt.show()


def export_pts(
    pts: torch.Tensor | List[torch.Tensor],
    conf: torch.Tensor | List[torch.Tensor] | None = None,
    name: str = "points.ply",
    save_image: bool = False,
    image_name: str = "points.png",
):
    # reformat pts
    if isinstance(pts, List):
        if isinstance(pts[0], torch.Tensor):
            pts = torch.cat(pts, dim=0).cpu().numpy()
            pts = pts.reshape(-1, 3)
        else:
            pts = np.concatenate(pts, axis=0)
            pts = pts.reshape(-1, 3)

    # Process confidence values if provided
    colors = None
    if conf is not None:
        if isinstance(conf, List):
            if isinstance(conf[0], torch.Tensor):
                conf = torch.cat(conf, dim=0).cpu().numpy().flatten()
            else:
                conf = np.concatenate(conf, axis=0).flatten()
        elif isinstance(conf, torch.Tensor):
            conf = conf.cpu().numpy().flatten()
        # Clip the confidence values to be within [1, 3]

        conf = np.clip(conf, 1, 3)
        # Normalize confidence values assuming they are in the range [1, 3]
        conf_normalized = (conf - 1) / 2.0
        # Create a colormap from blue (low confidence) to red (high confidence)
        colors = plt.cm.jet(conf_normalized)[:, :3]  # Only take RGB values, drop alpha

    pointcloud = trimesh.points.PointCloud(pts, colors=colors)
    pointcloud.export(name)


# best used in conjunction
def get_idx_from_pair(output, pair_tuple: Tuple[int, int]):
    mapping = {}
    for i, (idx1, idx2) in enumerate(zip(output["view1"]["idx"], output["view2"]["idx"])):
        mapping[(idx1, idx2)] = i
    return mapping[pair_tuple]


def get_output_from_idx(output, idx: int):
    return {
        "pts3d_1": output["pred1"]["pts3d"][idx],
        "pts3d_2": output["pred2"]["pts3d_in_other_view"][idx],
        "conf_1": output["pred1"]["conf"][idx],
        "conf_2": output["pred2"]["conf"][idx],
    }


def transform_coordinates_resize_crop(points, input_shape=(1168, 1752), output_size=224):
    """Transform coordinates considering the specific resize-then-crop logic.

    Args:
        points: Nx2 array of (x,y) - (width, height!) coordinates in input resolution
        input_shape: Tuple of (height, width) for input image
        output_size: Size of the output square image (default 224)

    Returns:
        Nx2 array of transformed (x,y) coordinates in output resolution
    """
    points = np.array(points)
    H1, W1 = input_shape

    # Calculate the scaling factor for the resize operation
    scale = output_size * max(W1 / H1, H1 / W1)

    # Calculate the dimensions after resizing
    if W1 > H1:
        resized_w = scale
        resized_h = output_size
    else:
        resized_w = output_size
        resized_h = scale

    # Transform coordinates for the resized image
    points_resized = points.copy()
    points_resized[:, 0] *= resized_w / W1  # x coordinates
    points_resized[:, 1] *= resized_h / H1  # y coordinates

    # Calculate the crop offsets
    if W1 > H1:
        crop_x = (resized_w - output_size) / 2
        crop_y = 0
    else:
        crop_x = 0
        crop_y = (resized_h - output_size) / 2

    # Apply the crop transformation
    points_cropped = points_resized.copy()
    points_cropped[:, 0] -= crop_x
    points_cropped[:, 1] -= crop_y

    return points_cropped


def transform_coordinates(points: torch.Tensor, input_shape=(1168, 1752), output_shape=(336, 512)):
    """Transform coordinates from input resolution to output resolution.

    Args:
        points: Nx2 array of (x,y) - (width, height!) coordinates in input resolution
        input_shape: Tuple of (height, width) for input image
        output_shape: Tuple of (height, width) for output image

    Returns:
        Nx2 array of transformed (x,y) coordinates in output resolution
    """
    # Scale factors
    scale_x = output_shape[1] / input_shape[1]
    scale_y = output_shape[0] / input_shape[0]

    # Transform coordinates
    points_transformed = points.clone().to(torch.float32)
    points_transformed[:, 0] *= scale_x  # x coordinates
    points_transformed[:, 1] *= scale_y  # y coordinates

    return points_transformed
