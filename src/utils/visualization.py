import matplotlib

matplotlib.use("Agg")  # Use the Agg backend for rendering
import os
import pathlib
from io import BytesIO
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import roma
import torch
from jaxtyping import Float
from matplotlib.patches import ConnectionPatch
from PIL import Image
from plotly import graph_objects as go
from plotly import io as pio


# WANDB UTILS
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it."""
    with BytesIO() as buf:
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img.copy()


def images_to_gif(image_fnames, fname, sort_key=None, save_mp4=False):
    image_fnames.sort(key=sort_key)  # sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(
        f"{fname}.gif", format="GIF", append_images=frames, save_all=True, duration=30, loop=0
    )

    if save_mp4:
        cmd = f"ffmpeg -loglevel error -i {f'{fname}.gif'} -vcodec libx264 -crf 25 -pix_fmt yuv420p {f'{fname}.mp4'} -y"
        os.system(cmd)
        if not os.path.exists(f"{fname}.mp4"):
            print("Failed to create mp4 file.")


# SFM MATCHER VISUALIZATIONS
def plot_matches(query_img, database_img, query_kp, database_kp, matches, linewidth=0.3):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(query_img)
    ax2.imshow(database_img)
    ax1.axis("off")
    ax2.axis("off")
    for match in matches:
        con = ConnectionPatch(
            xyA=query_kp[match.query_idx].pt,
            xyB=database_kp[match.database_idx].pt,
            coordsA="data",
            coordsB="data",
            axesA=ax1,
            axesB=ax2,
            color="green",
            linewidth=linewidth,  # Made the connection path thinner
        )
        ax2.add_artist(con)

    # Save the figure and then close it
    plt.close(fig)  # Close the figure after saving
    return fig


def plot_matches_from_points(img1, img2, kp1, kp2, linewidth=0.3):
    # Create a high-quality figure with higher DPI and larger size
    fig = plt.figure(figsize=(20, 10), dpi=300)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Use higher quality image display
    ax1.imshow(img1, interpolation="nearest")
    ax2.imshow(img2, interpolation="nearest")
    ax1.axis("off")
    ax2.axis("off")

    first = True
    for pt1, pt2 in zip(kp1, kp2):
        con = ConnectionPatch(
            xyA=pt1,
            xyB=pt2,
            coordsA="data",
            coordsB="data",
            axesA=ax1,
            axesB=ax2,
            color="green" if not first else "red",
            linewidth=linewidth,  # Made the connection path thinner
            alpha=0.8,  # Slight transparency for better visibility
            antialiased=True,  # Smoother lines
        )
        ax2.add_artist(con)
        first = False

    # Save the figure and then close it
    plt.close(fig)  # Close the figure after saving
    return fig


def plot_residuals(residuals):
    # Reuse existing figure if possible, or create new one
    fig = plt.gcf() if plt.get_fignums() else plt.figure()
    # Clear previous content
    fig.clear()
    ax = fig.add_subplot(111)

    ax.set_yscale("log")
    # hard coded
    ax.set_ylim(0.001, 1000)

    ax.fill_between(range(len(residuals)), abs(residuals), color="blue", alpha=1)
    ax.plot(abs(residuals), color="blue")
    ax.set_title("Residuals Plot")
    ax.set_xlabel("Residual Index")
    ax.set_ylabel("Residual Value (Absolute)")
    # fig.canvas.draw()
    return fig


def plot_matches_3views(query_img, database_img, query_kp, database_kp, matches):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax_database = [ax2, ax3]

    ax1.imshow(query_img)
    ax2.imshow(database_img[0])
    ax3.imshow(database_img[1])
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    for match in matches:
        con = ConnectionPatch(
            xyA=query_kp[match.query_idx].pt,
            xyB=database_kp[match.database_img][match.database_idx].pt,
            coordsA="data",
            coordsB="data",
            axesA=ax1,
            axesB=ax_database[match.database_img],
            color="red",
        )
        ax_database[match.database_img].add_artist(con)
    plt.show()


# BA VISUALIZATIONS
def visualize_cameras_and_points_gt_prediction(points_gt, cameras_gt, points_hat, cameras_hat):
    # gt stuff
    points_x_gt = points_gt[:, 0]
    points_y_gt = points_gt[:, 1]
    points_z_gt = points_gt[:, 2]

    cameras_x_gt = cameras_gt[:, 0]
    cameras_y_gt = cameras_gt[:, 1]
    cameras_z_gt = cameras_gt[:, 2]

    # hat stuff
    points_x_hat = points_hat[:, 0]
    points_y_hat = points_hat[:, 1]
    points_z_hat = points_hat[:, 2]

    cameras_x_hat = cameras_hat[:, 0]
    cameras_y_hat = cameras_hat[:, 1]
    cameras_z_hat = cameras_hat[:, 2]

    # color the first point blue, the others red
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points_x_gt,
                y=points_y_gt,
                z=points_z_gt,
                mode="markers",
                marker=dict(size=12, color="red"),
            ),
            go.Scatter3d(
                x=cameras_x_gt,
                y=cameras_y_gt,
                z=cameras_z_gt,
                mode="markers",
                marker=dict(size=12, color="blue"),
            ),
            go.Scatter3d(
                x=points_x_hat,
                y=points_y_hat,
                z=points_z_hat,
                mode="markers",
                marker=dict(size=12, color="orange"),
            ),
            go.Scatter3d(
                x=cameras_x_hat,
                y=cameras_y_hat,
                z=cameras_z_hat,
                mode="markers",
                marker=dict(size=12, color="green"),
            ),
        ]
    )
    fig.show()


def visualize_reprojections(
    track_hat=None, track_gt=None, homo_in=False, img=None, title=None, markersize=2
):
    # check if only single track
    if track_hat is not None and len(track_hat.shape) == 1:
        track_hat = track_hat.unsqueeze(0)
    if track_gt is not None and len(track_gt.shape) == 1:
        track_gt = track_gt.unsqueeze(0)

    # setup frames
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)

    ax1.set_title(title)
    ax1.set_aspect("equal")
    ax1.grid(False)

    # paint points with smaller size
    if track_hat is not None:
        for track_h in track_hat:
            ax1.plot(
                *track_h[:2], "ro", label="prediction", markersize=markersize
            )  # Reduced marker size

    if track_gt is not None:
        for track_g in track_gt:
            ax1.plot(*track_g[:2], "go", label="gt", markersize=markersize)  # Reduced marker size

    if type(img) is str or type(img) is pathlib.PosixPath:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for matplotlib
        ax1.imshow(img)
    elif type(img) is np.ndarray:
        ax1.imshow(img)
    elif type(img) is tuple:
        ax1.set_xlim(0, img[1])
        ax1.set_ylim(img[0])
    else:
        ax1.set_xlim(32)
        ax1.set_ylim(32)

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    plt.close()
    return fig


def plot_reprojections(graph, reprojections, data, views=[0]):
    figs = list()
    for view in views:
        reprojections_camera = reprojections[data["camera_indices"] == view]
        fig = graph.views[view].draw_active_keypoints(predictions=reprojections_camera)
        figs.append(fig)
    return figs


def save_reprojections(figs, path, step):
    for i, fig in enumerate(figs):
        view_path = path / f"view_{i}"
        view_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(view_path / f"step_{step}")


def visualize_reprojections_2(track_hat_list, track_gt_list=None, homo_in=False):
    # setup frames
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    axes = [ax1, ax2]

    for ax, track_hat, track_gt in zip(axes, track_hat_list, track_gt_list):
        ax.set_title("View")
        ax.set_xlim(right=32)
        ax.set_ylim(top=32)
        # ax.set_aspect("equal")

        # paint points
        for track_h in track_hat.T:
            ax.plot(*track_h[:2], "ro", label="track_hat")

        if track_gt is not None:
            for track_g in track_gt.T:
                ax.plot(*track_g[:2], "go", label="track_gt")

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    plt.close()
    return fig


def plot_keypoints_connections(images, keypoints, connections=False):
    n_images = len(images)  # Get the number of images
    height = int(np.ceil(np.sqrt(n_images)))  # Calculate the number of rows
    width = int(np.ceil(n_images / height))  # Calculate the number of columns
    fig = plt.figure(
        figsize=(5 * height, 5 * width)
    )  # Adjust figure size based on number of images
    ax_list = [
        fig.add_subplot(height, width, i + 1) for i in range(n_images)
    ]  # Subplots for each image

    # Plot each image and its keypoint
    for i, ax in enumerate(ax_list):
        ax.imshow(images[i])
        ax.axis("off")
        # Plot the single keypoint for the current image
        (x, y) = keypoints[i]  # Assuming keypoints is a list of tuples with one keypoint per image
        ax.plot(x, y, color="red", marker="x", markersize=10)  # Plot keypoint as a red dot

    # Connect keypoints between all pairs of images
    if connections:
        for i in range(n_images):
            for j in range(i + 1, n_images):
                (x1, y1) = keypoints[i]  # Keypoint from the first image
                (x2, y2) = keypoints[j]  # Keypoint from the second image
                con = ConnectionPatch(
                    xyA=(x1, y1),
                    xyB=(x2, y2),
                    coordsA="data",
                    coordsB="data",
                    axesA=ax_list[i],
                    axesB=ax_list[j],
                    color="red",  # Color of the connection lines
                    linewidth=1.0,
                )
                ax_list[j].add_artist(con)

    plt.show()


def plot_views(images):
    # TODO add if geometrically verified
    n_images = len(images)  # Get the number of images
    height = int(np.ceil(np.sqrt(n_images)))  # Calculate the number of rows
    width = int(np.ceil(n_images / height))  # Calculate the number of columns
    fig = plt.figure(
        figsize=(5 * height, 5 * width)
    )  # Adjust figure size based on number of images
    ax_list = [
        fig.add_subplot(height, width, i + 1) for i in range(n_images)
    ]  # Subplots for each image

    # Plot each image and its keypoint
    for i, ax in enumerate(ax_list):
        ax.imshow(images[i])
        ax.axis("off")
        # Plot the single keypoint for the current image
        # (x, y) = keypoints[i]  # Assuming keypoints is a list of tuples with one keypoint per image
        # ax.plot(x, y, color='red', marker='x', markersize=10)  # Plot keypoint as a red dot

    plt.show()


def visualize_points_3d(points, size=12):
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]
    # color the first point blue, the others red
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points_x,
                y=points_y,
                z=points_z,
                mode="markers",
                marker=dict(size=size, color="red"),
            ),
        ]
    )
    fig.show()


def visualize_cameras_and_points(points, cameras, size=4):
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]

    cameras_x = cameras[:, 0]
    cameras_y = cameras[:, 1]
    cameras_z = cameras[:, 2]

    # color the first point blue, the others red
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points_x,
                y=points_y,
                z=points_z,
                mode="markers",
                marker=dict(size=size, color="red"),
            ),
            go.Scatter3d(
                x=cameras_x,
                y=cameras_y,
                z=cameras_z,
                mode="markers",
                marker=dict(size=size, color="blue"),
            ),
        ]
    )
    fig.show()


# SfM (over time) visualizations
def add_pyramid(
    rotation: torch.tensor,
    translation: torch.tensor,
    K: np.ndarray,
    gt=True,
    showlegend=True,
    color=None,
    name=None,
):
    if name is None:
        name = "GT Camera" if gt else "Estimated Camera"
    if color is not None:
        color = color
    else:
        color = "blue" if gt else "cyan"

    R = roma.rotvec_to_rotmat(rotation).numpy()
    t = translation.numpy()

    W, H = K[0, 2] * 2, K[1, 2] * 2

    image_extent = max(W / 1024.0, H / 1024.0)
    world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
    scale = 0.5 * image_extent / world_extent

    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]])
    # make cornres homogeneous
    ones = np.ones((corners.shape[:-1] + (1,)), dtype=corners.dtype)
    corners = np.concatenate([corners, ones], axis=-1)

    corners_local_camera = np.linalg.inv(K) @ corners.T
    # scale corners
    corners_local_camera *= 0.5 * scale

    # get camera center in world
    camera_center = R.T @ -t
    corners_world = (R.T @ corners_local_camera) + camera_center[:, None]
    corners_world_all = np.concatenate((corners_world, camera_center[:, None]), axis=1)

    # define vertices
    i = [4, 4, 4, 4]
    j = [0, 1, 2, 3]
    k = [1, 2, 3, 0]

    triangles = np.vstack((i, j, k)).T
    triangle_points = np.array([corners_world_all.T[i] for i in triangles.reshape(-1)])

    x, y, z = triangle_points.T

    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color=color, width=3),
        showlegend=showlegend,
        name=name,
        legendgroup=name,
        # hovertemplate="testestesetest"
    )
    return pyramid


def add_points(points, gt=True, color="red", name=None):
    if name is None:
        name = "GT Points" if gt else "Estimated Points"
    x, y, z = points.T
    pp = go.Scatter3d(
        x=x, y=y, z=z, mode="markers", marker=dict(size=3, color=color), showlegend=True, name=name
    )
    return pp


def add_pyramids(rotations, translations, K, gt=True, color=None):
    pyramids = []
    first = True
    for rot, t in zip(rotations, translations):
        pyramids.append(add_pyramid(rot, t, K, gt=gt, showlegend=first, color=color))
        first = False
    return pyramids


# def add_image_plane(image_path, position, size=(1, 1)):
#     """Add an image plane to the 3D scene."""
#     # Load the image
#     img = Image.open(image_path)
#     img = img.resize((int(size[0] * 100), int(size[1] * 100)))  # Resize for better aspect ratio
#     img_array = np.array(img)

#     # Create a mesh for the image plane
#     x = np.array([-size[0]/2, size[0]/2, size[0]/2, -size[0]/2]) + position[0]
#     y = np.array([-size[1]/2, -size[1]/2, size[1]/2, size[1]/2]) + position[1]
#     z = np.array([0, 0, 0, 0]) + position[2]  # Position the plane at the desired z level

#     # Create a mesh for the image plane
#     image_plane = go.Mesh3d(
#         x=x,
#         y=y,
#         z=z,
#         i=[0, 0, 1, 1],
#         j=[1, 2, 2, 3],
#         k=[2, 3, 0, 1],
#         color='rgba(255, 255, 255, 0)',  # Transparent color
#         opacity=1,
#         hoverinfo='none',
#         showscale=False,
#     )

#     return image_plane


def ba_over_time(
    points: Float[torch.Tensor, "n_steps n_points 3"],
    rotations: Float[torch.Tensor, "n_steps n_cameras 3"],
    translations: Float[torch.Tensor, "n_steps n_cameras 3"],
    K,
    points_gt=None,
    rotations_gt=None,
    translations_gt=None,
    save_animation=False,
    images_path="",
    gif_path="animation_frames",
    buttons_enabled=False,
    image_planes=None,  # New parameter for image planes
):
    # points = torch.concat((points_gt[None], points_ot[0][None]), dim=0)
    # rotations = torch.concat((rotations_gt[None], rotations_ot[0][None]))
    # translations = torch.concat((translations_gt[None], translations_ot[0][None]))
    # camera_centers_all = roma.rotvec_to_rotmat(rotations).transpose(2,3) @ -translations[..., None]

    # # calculate bounds
    # xmax = torch.max(points[:,:, 0].max(), camera_centers_all[:, :, 0].max())
    # xmin = torch.min(points[:,:, 0].min(), camera_centers_all[:, :, 0].min())
    # ymax = torch.max(points[:,:, 1].max(), camera_centers_all[:, :, 1].max())
    # ymin = torch.min(points[:,:, 1].min(), camera_centers_all[:, :, 1].min())
    # zmax = torch.max(points[:,:, 2].max(), camera_centers_all[:, :, 2].max())
    # zmin = torch.min(points[:,:, 2].min(), camera_centers_all[:, :, 2].min())

    if isinstance(K, torch.Tensor):
        K = K.numpy()

    # Calculate center and range for each axis to maintain aspect ratio
    # center_x = (xmax + xmin) / 2
    # center_y = (ymax + ymin) / 2
    # center_z = (zmax + zmin) / 2

    # range_x = xmax - xmin
    # range_y = ymax - ymin
    # range_z = zmax - zmin

    # # Use the maximum range for all axes to maintain aspect ratio
    # max_range = max(range_x, range_y, range_z)

    # # Add some padding (e.g., 10%)
    # padding = max_range * 0.1

    # # Calculate consistent bounds for all axes
    # xmin, xmax = center_x - max_range/2 - padding, center_x + max_range/2 + padding
    # ymin, ymax = center_y - max_range/2 - padding, center_y + max_range/2 + padding
    # zmin, zmax = center_z - max_range/2 - padding, center_z + max_range/2 + padding

    # instantiate figure
    fig = go.Figure(
        data=[
            add_points(points[0], gt=False),
            *add_pyramids(rotations[0], translations[0], K, gt=False),
        ]
    )

    # add gt data
    if points_gt is not None:
        fig.add_trace(add_points(points_gt, gt=True))
    if rotations_gt is not None and translations_gt is not None:
        fig.add_traces(add_pyramids(rotations_gt, translations_gt, K, gt=True))

    # compute data for frames
    frames = []
    step_size = 1 if int(points.shape[0] / 100) == 0 else int(points.shape[0] / 100)
    for k in range(0, points.shape[0], step_size):
        frames.append(
            go.Frame(
                data=[
                    add_points(points[k], gt=False),
                    *add_pyramids(rotations[k], translations[k], K, gt=False),
                ],
                # traces=np.arange(rotations_gt.shape[0] + 1),
                name=f"frame_{k}",  # Add name for slider reference
            )
        )
    fig.update(frames=frames)

    # Add slider only if buttons_enabled is True
    sliders = []
    if buttons_enabled:
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "Frame: "},
                pad={"t": 50},
                steps=[
                    dict(
                        args=[
                            [f"frame_{k}"],
                            dict(
                                mode="immediate",
                                frame=dict(duration=100, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                        label=str(k),
                        method="animate",
                    )
                    for k in range(0, points.shape[0], step_size)
                ],
            )
        ]

    # add update menu with reset button
    updatemenus = [
        dict(
            buttons=[
                dict(
                    args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    label="Play",
                    method="animate",
                    visible=buttons_enabled,  # Control visibility based on the parameter
                ),
                dict(
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    label="Pause",
                    method="animate",
                    visible=buttons_enabled,  # Control visibility based on the parameter
                ),
            ],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            type="buttons",
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top",
        )
    ]

    # update layout to include buttons and conditionally sliders
    # fig.update_layout(
    #     width=800, height=600,
    # updatemenus=updatemenus,
    # sliders=sliders if buttons_enabled else [],  # Add sliders only if buttons_enabled is True
    #     scene = dict(
    #         xaxis=dict(visible=False, range=[xmin, xmax], autorange=False),
    #         yaxis=dict(visible=False, range=[ymin, ymax], autorange=False),
    #         zaxis=dict(visible=False, range=[zmin, zmax], autorange=False),
    #         bgcolor='black',
    #         # aspectmode='cube',
    #         aspectmode='data',
    #         camera = dict(
    #             up=dict(x=0, y=0, z=1),
    #             center=dict(x=0, y=0, z=0),
    #             eye=dict(x=1., y=-0.2, z=0.5)
    #         )
    #     ),
    # )

    fig.update_layout(
        template="plotly_dark",
        width=800,
        height=800,
        updatemenus=updatemenus,
        sliders=sliders if buttons_enabled else [],  # Add sliders only if buttons_enabled is True
        scene=dict(
            xaxis=dict(visible=False, autorange=True),
            yaxis=dict(visible=False, autorange=True),
            zaxis=dict(visible=False, autorange=True),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                # center=dict(x=float(center_of_rotation[0]), y=float(center_of_rotation[1]), z=float(center_of_rotation[2])),  # Center the camera around the centroid
                #     eye=dict(x=5, y=5, z=5),
            ),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, b=10, t=10),
        legend=dict(orientation="h", yanchor="top", y=0.95, xanchor="left", x=0.1),
    )

    # # Add image planes if provided
    # if image_planes:
    #     for image_path, position in image_planes:
    #         fig.add_trace(add_image_plane(image_path, position))

    if save_animation:
        # Disable buttons when saving animation
        # Create the gif file.
        import os

        if not os.path.exists(images_path):
            os.makedirs(images_path)
        for k, _ in enumerate(frames):
            for trace in fig.frames[0]["traces"]:
                for key in ["x", "y", "z"]:
                    fig.data[trace][key] = fig.frames[k].data[trace][key]
            pio.write_image(
                fig, f"{images_path}/step_{k+1:03d}.png", width=800, height=600, scale=1
            )
        # Convert PNGs to GIF
        images_to_gif(
            image_fnames=list(images_path.iterdir()),
            fname=gif_path / "3d_animation",
            sort_key=lambda x: int(x.name.split("_")[-1].split(".")[0]),
        )

    return fig


def sfm_over_time(
    points_list: List[Float[torch.Tensor, "n_steps n_points 3"]],
    rotations_list: List[Float[torch.Tensor, "n_steps n_cameras 3"]],
    translations_list: List[Float[torch.Tensor, "n_steps n_cameras 3"]],
    K,
    # points_gt=None,
    # rotations_gt=None,
    # translations_gt=None,
    save_animation=False,
    images_path="",
    gif_path="animation_frames",
    buttons_enabled=False,
    image_planes=None,  # New parameter for image planes
):
    if isinstance(K, torch.Tensor):
        K = K.numpy()

    # instantiate figure with the first frame
    fig = go.Figure(data=[add_points(points_list[0][0], gt=False)])
    # , *add_pyramids(rotations_list[2][0], translations_list[2][0], , K, gt=False)

    frames = []
    names = []
    for i, (points, rotations, translations) in enumerate(
        zip(points_list, rotations_list, translations_list)
    ):
        step_size = 1 if int(points.shape[0] / 100) == 0 else int(points.shape[0] / 100)
        for k in range(0, points.shape[0], step_size):
            # Add all pyramids for the current frame
            names.append(f"step_{i}_frame_{k}")

            # Create frame data
            frame_data = [
                add_points(points[k], gt=False),
                *add_pyramids(
                    rotations[k], translations[k], K, gt=False
                ),  # Ensure all pyramids are added
            ]

            # Debugging: Print frame data
            print(f"Frame {k}: {frame_data}")

            frames.append(
                go.Frame(
                    data=frame_data, name=f"step_{i}_frame_{k}"  # Add name for slider reference
                )
            )

    fig.update(frames=frames)

    # Add slider only if buttons_enabled is True
    sliders = []
    if buttons_enabled:
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "Frame: "},
                pad={"t": 50},
                steps=[
                    dict(
                        args=[
                            [name],
                            dict(
                                mode="immediate",
                                frame=dict(duration=100, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                        method="animate",
                        label=str(k),
                    )
                    for k, name in enumerate(names)
                ],
            )
        ]

    # Add update menu with reset button
    updatemenus = [
        dict(
            buttons=[
                dict(
                    args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    label="Play",
                    method="animate",
                    visible=buttons_enabled,
                ),
                dict(
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    label="Pause",
                    method="animate",
                    visible=buttons_enabled,
                ),
            ],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            type="buttons",
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top",
        )
    ]

    fig.update_layout(
        template="plotly_dark",
        width=800,
        height=800,
        updatemenus=updatemenus,
        sliders=sliders if buttons_enabled else [],
        scene=dict(
            xaxis=dict(visible=True, autorange=False, range=[0, 6]),
            yaxis=dict(visible=True, autorange=False, range=[0, 3]),
            zaxis=dict(visible=True, autorange=False, range=[0, 3]),
            camera=dict(up=dict(x=0, y=0, z=1)),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, b=10, t=10),
        legend=dict(orientation="h", yanchor="top", y=0.95, xanchor="left", x=0.1),
    )

    if save_animation:
        # Disable buttons when saving animation
        # Create the gif file.
        import os

        if not os.path.exists(images_path):
            os.makedirs(images_path)
        for k, _ in enumerate(frames):
            for trace in fig.frames[0]["traces"]:
                for key in ["x", "y", "z"]:
                    fig.data[trace][key] = fig.frames[k].data[trace][key]
            pio.write_image(
                fig, f"{images_path}/step_{k+1:03d}.png", width=800, height=600, scale=1
            )
        # Convert PNGs to GIF
        images_to_gif(
            image_fnames=list(images_path.iterdir()),
            fname=gif_path / "3d_animation",
            sort_key=lambda x: int(x.name.split("_")[-1].split(".")[0]),
        )

    return fig, frames


def plot_3d(
    points: Float[torch.Tensor, "n_points 3"],
    rotations: Float[torch.Tensor, "n_cameras 3"],
    translations: Float[torch.Tensor, "n_cameras 3"],
    K: Float[np.ndarray, "3 3"],
    points_gt: Float[torch.Tensor, "n_points 3"] = None,
    rotations_gt: Float[torch.Tensor, "n_cameras 3"] = None,
    translations_gt: Float[torch.Tensor, "n_cameras 3"] = None,
    image_planes=None,  # New parameter for image planes
    remove_outliers=False,
):
    if remove_outliers:
        # Calculate the centroid of the points
        center_of_rotation = torch.median(points, dim=0).values
        # Filter outlier points based on distance from the centroid
        distance_from_centroid = torch.sqrt(torch.sum((points - center_of_rotation) ** 2, dim=1))
        std_dev = torch.std(distance_from_centroid)
        points = points[
            distance_from_centroid < 4 * std_dev
        ]  # Filter points within 3 standard deviations
        removed_points_count = (~(distance_from_centroid < 4 * std_dev)).sum()
        print(
            f"ALERT: Removing {removed_points_count} points as outliers for visualization purposes!"
        )

    if isinstance(K, torch.Tensor):
        K = K.numpy()

    # instantiate figure
    fig = go.Figure(
        data=[add_points(points, gt=False), *add_pyramids(rotations, translations, K, gt=False)]
    )

    # Optionally add ground truth data
    if points_gt is not None:
        fig.add_trace(add_points(points_gt, gt=True))

    if rotations_gt is not None and translations_gt is not None:
        fig.add_traces(add_pyramids(rotations_gt, translations_gt, K, gt=True))

    fig.update_layout(
        template="plotly_dark",
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(visible=False, autorange=True),
            yaxis=dict(visible=False, autorange=True),
            zaxis=dict(visible=False, autorange=True),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                # center=dict(x=float(center_of_rotation[0]), y=float(center_of_rotation[1]), z=float(center_of_rotation[2])),  # Center the camera around the centroid
                #     eye=dict(x=5, y=5, z=5),
            ),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, b=10, t=10),
        legend=dict(orientation="h", yanchor="top", y=0.95, xanchor="left", x=0.1),
    )

    return fig


def create_image_grid(images_path: pathlib.Path, output_path: pathlib.Path):
    images = sorted(images_path.glob("*.png"), key=lambda x: int(x.name.split("_")[0]))

    # Load all images
    images = [Image.open(img) for img in images]

    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))  # Calculate the grid size
    image_width, image_height = images[0].size  # Assuming all images are the same size

    # Create a square layout
    new_image = Image.new(
        "RGB", (grid_size * image_width, grid_size * image_height), (255, 255, 255)
    )
    x_offset = 0
    y_offset = 0
    for index, img in enumerate(images):
        new_image.paste(img, (x_offset, y_offset))
        x_offset += image_width
        if (index + 1) % grid_size == 0:  # Move to the next row after filling the current one
            x_offset = 0
            y_offset += image_height

    # Save the new image instead of displaying it
    fig = plt.figure(
        figsize=(grid_size * 5, grid_size * 5)
    )  # Adjust figure size based on grid size
    plt.imshow(np.asarray(new_image))
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
