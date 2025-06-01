from data.colmap import read_images_text


def get_camera_centers(scene):
    """Get camera centers from scannet data."""
    images = read_images_text(scene.dslr_colmap_dir / "images.txt")
    image_data_scannet = {v.name: v for _, v in images.items()}
    camera_centers = {}

    for name, data in image_data_scannet.items():
        camera_center = data.world_to_camera[:3, :3].T @ -data.world_to_camera[:3, 3]
        camera_centers[name] = camera_center

    return image_data_scannet, camera_centers
