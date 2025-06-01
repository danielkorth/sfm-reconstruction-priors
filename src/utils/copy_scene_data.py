"""Copy Scene Data Utility.

This script helps copy specific data from scenes to a destination directory,
and provides guidance for transferring the data to a local machine.

Example usage:

1. Copy data from scenes with both keypoints and matches to a temporary directory:
   ```
   conda activate guided
   python src/utils/copy_scene_data.py --source data/scannetppv2/data --dest /tmp/scene_data_transfer --estimate-size
   ```

2. Copy data from specific scenes:
   ```
   conda activate guided
   python src/utils/copy_scene_data.py --source data/scannetppv2/data --dest /tmp/scene_data_transfer --scenes abb6639ceb e6372539da
   ```

3. Limit the number of scenes to copy:
   ```
   conda activate guided
   python src/utils/copy_scene_data.py --source data/scannetppv2/data --dest /tmp/scene_data_transfer --max-scenes 10
   ```

4. Generate SCP command for transferring to local machine:
   ```
   conda activate guided
   python src/utils/copy_scene_data.py --source data/scannetppv2/data --dest /tmp/scene_data_transfer --max-scenes 10 --generate-scp --server-address username@server
   ```

5. Transfer data to local machine (run this on your local machine):
   ```
   # Basic SCP transfer
   scp -r username@server:/tmp/scene_data_transfer ~/local_destination

   # SCP with compression for faster transfer over slow connections
   scp -r -C username@server:/tmp/scene_data_transfer ~/local_destination

   # Using rsync for resumable transfers
   rsync -avz --progress username@server:/tmp/scene_data_transfer ~/local_destination
   ```
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def check_directories(base_path):
    """Check all scene directories in base_path for the existence of 'keypoints' and 'matches'
    folders within the 'dslr' subdirectory.

    Args:
        base_path: Path to the base directory (data/scannetppv2/data)

    Returns:
        DataFrame with results
    """
    results = []

    # Get all scene directories
    try:
        scene_dirs = [
            f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))
        ]
    except FileNotFoundError:
        print(f"Error: Base path '{base_path}' not found.")
        return pd.DataFrame()

    # Check each scene directory
    for scene_dir in tqdm(scene_dirs, desc="Checking directories"):
        dslr_path = os.path.join(base_path, scene_dir, "dslr")

        # Skip if dslr directory doesn't exist
        if not os.path.isdir(dslr_path):
            continue

        has_keypoints = os.path.isdir(os.path.join(dslr_path, "keypoints"))
        has_matches = os.path.isdir(os.path.join(dslr_path, "matches"))
        has_colmap = os.path.isdir(os.path.join(dslr_path, "colmap"))
        has_undistorted = os.path.isdir(os.path.join(dslr_path, "undistorted_images"))

        results.append(
            {
                "directory": scene_dir,
                "has_keypoints": has_keypoints,
                "has_matches": has_matches,
                "has_colmap": has_colmap,
                "has_undistorted": has_undistorted,
                "complete": has_keypoints and has_matches,
            }
        )

    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(results)

    # Print summary
    total = len(df)
    if total == 0:
        print("No directories with 'dslr' subdirectory found.")
        return df

    complete = df["complete"].sum()

    print("\nSummary:")
    print(f"Total directories with 'dslr' subdirectory: {total}")
    print(
        f"Complete directories (has both keypoints and matches): {complete} ({complete/total*100:.1f}%)"
    )

    return df


def copy_scene_data(source_base_path, dest_base_path, scene_dir):
    """Copy the specified data from a scene to the destination.

    Args:
        source_base_path: Path to the source base directory
        dest_base_path: Path to the destination base directory
        scene_dir: Name of the scene directory to copy
    """
    source_dslr_path = os.path.join(source_base_path, scene_dir, "dslr")
    dest_scene_path = os.path.join(dest_base_path, scene_dir, "dslr")

    # Create destination directories if they don't exist
    os.makedirs(dest_scene_path, exist_ok=True)

    # 1. Copy colmap folder
    colmap_src = os.path.join(source_dslr_path, "colmap")
    colmap_dst = os.path.join(dest_scene_path, "colmap")
    if os.path.exists(colmap_src):
        print(f"Copying colmap folder for {scene_dir}...")
        shutil.copytree(colmap_src, colmap_dst, dirs_exist_ok=True)

    # 2. Copy keypoints folder
    keypoints_src = os.path.join(source_dslr_path, "keypoints")
    keypoints_dst = os.path.join(dest_scene_path, "keypoints")
    if os.path.exists(keypoints_src):
        print(f"Copying keypoints folder for {scene_dir}...")
        shutil.copytree(keypoints_src, keypoints_dst, dirs_exist_ok=True)

    # 3. Copy matches folder
    matches_src = os.path.join(source_dslr_path, "matches")
    matches_dst = os.path.join(dest_scene_path, "matches")
    if os.path.exists(matches_src):
        print(f"Copying matches folder for {scene_dir}...")
        shutil.copytree(matches_src, matches_dst, dirs_exist_ok=True)

    # 4. Copy only images in undistorted_images that match keypoints
    undistorted_src = os.path.join(source_dslr_path, "undistorted_images")
    undistorted_dst = os.path.join(dest_scene_path, "undistorted_images")

    if os.path.exists(undistorted_src) and os.path.exists(keypoints_src):
        print(f"Copying matching undistorted images for {scene_dir}...")
        os.makedirs(undistorted_dst, exist_ok=True)

        # Get list of keypoint files (to extract image names)
        keypoint_files = os.listdir(keypoints_src)
        keypoint_image_names = set()

        # Extract image names from keypoint files (assuming format like "image_name.JPG.npy")
        for kp_file in keypoint_files:
            # Remove the .npy extension to get the original image filename
            if kp_file.endswith(".npy"):
                image_name = kp_file[:-4]  # Remove .npy
                keypoint_image_names.add(image_name)

        # Copy only matching undistorted images
        copied_count = 0
        for image_name in os.listdir(undistorted_src):
            if image_name in keypoint_image_names:
                src_image = os.path.join(undistorted_src, image_name)
                dst_image = os.path.join(undistorted_dst, image_name)
                shutil.copy2(src_image, dst_image)
                copied_count += 1

        print(f"  - Copied {copied_count} matching undistorted images")


def estimate_scene_size(source_base_path, scene_dir):
    """Estimate the size of a scene's data to be copied.

    Args:
        source_base_path: Path to the source base directory
        scene_dir: Name of the scene directory

    Returns:
        Estimated size in bytes
    """
    source_dslr_path = os.path.join(source_base_path, scene_dir, "dslr")
    total_size = 0

    # Estimate colmap folder size
    colmap_path = os.path.join(source_dslr_path, "colmap")
    if os.path.exists(colmap_path):
        for dirpath, dirnames, filenames in os.walk(colmap_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

    # Estimate keypoints folder size
    keypoints_path = os.path.join(source_dslr_path, "keypoints")
    if os.path.exists(keypoints_path):
        for dirpath, dirnames, filenames in os.walk(keypoints_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

    # Estimate matches folder size
    matches_path = os.path.join(source_dslr_path, "matches")
    if os.path.exists(matches_path):
        for dirpath, dirnames, filenames in os.walk(matches_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

    # Estimate undistorted_images folder size (only for images that match keypoints)
    undistorted_path = os.path.join(source_dslr_path, "undistorted_images")
    if os.path.exists(undistorted_path) and os.path.exists(keypoints_path):
        keypoint_files = os.listdir(keypoints_path)
        keypoint_image_names = set()

        for kp_file in keypoint_files:
            if kp_file.endswith(".npy"):
                image_name = kp_file[:-4]  # Remove .npy
                keypoint_image_names.add(image_name)

        for image_name in os.listdir(undistorted_path):
            if image_name in keypoint_image_names:
                fp = os.path.join(undistorted_path, image_name)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)

    return total_size


def format_size(size_bytes):
    """Format size in bytes to human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def main():
    parser = argparse.ArgumentParser(description="Copy scene data to local device")
    parser.add_argument(
        "--source",
        type=str,
        default="data/scannetppv2/data",
        help="Source base path containing scene directories",
    )
    parser.add_argument("--dest", type=str, required=True, help="Destination path for copied data")
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        help="Specific scene directories to copy (if not specified, all complete scenes will be copied)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be copied without actually copying",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None, help="Maximum number of scenes to copy"
    )
    parser.add_argument(
        "--estimate-size", action="store_true", help="Estimate the total size before copying"
    )
    parser.add_argument(
        "--generate-scp",
        action="store_true",
        help="Generate SCP command for transferring data to local machine",
    )
    parser.add_argument(
        "--local-dest",
        type=str,
        default="~/scene_data",
        help="Local destination path for SCP transfer",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="",
        help="Server address for SCP transfer (e.g., username@server)",
    )

    args = parser.parse_args()

    # Check if the source path exists
    if not os.path.exists(args.source):
        print(f"Error: Source path '{args.source}' does not exist.")
        exit(1)

    # Create destination directory if it doesn't exist
    os.makedirs(args.dest, exist_ok=True)

    # If specific scenes are provided, use those
    if args.scenes:
        scene_dirs = args.scenes
        print(f"Will copy {len(scene_dirs)} specified scenes")
    else:
        # Otherwise, find all scenes with both keypoints and matches
        results_df = check_directories(args.source)

        if results_df.empty:
            print("No suitable directories found.")
            exit(0)

        complete_df = results_df[results_df["complete"]]
        if complete_df.empty:
            print("No directories with both keypoints and matches found.")
            exit(0)

        scene_dirs = complete_df["directory"].tolist()
        print(f"Found {len(scene_dirs)} scenes with both keypoints and matches")

    # Limit the number of scenes if requested
    if args.max_scenes is not None and args.max_scenes > 0 and args.max_scenes < len(scene_dirs):
        print(f"Limiting to {args.max_scenes} scenes (out of {len(scene_dirs)})")
        scene_dirs = scene_dirs[: args.max_scenes]

    # Estimate total size if requested
    if args.estimate_size:
        print("Estimating total size...")
        total_size = 0
        for scene_dir in tqdm(scene_dirs, desc="Estimating"):
            scene_size = estimate_scene_size(args.source, scene_dir)
            total_size += scene_size

        print(f"Estimated total size: {format_size(total_size)}")

        # Ask for confirmation
        if not args.dry_run and not args.generate_scp:
            confirm = input(f"Continue with copying {len(scene_dirs)} scenes? (y/n): ")
            if confirm.lower() != "y":
                print("Aborting.")
                exit(0)

    # If generating SCP command, copy data and then print SCP command
    if args.generate_scp:
        if not args.server_address:
            print("Error: --server-address is required when using --generate-scp")
            exit(1)

        print("\nCopying data to temporary location before SCP transfer...")

        # Copy data for each scene
        for scene_dir in tqdm(scene_dirs, desc="Copying scenes"):
            print(f"\nProcessing scene: {scene_dir}")
            copy_scene_data(args.source, args.dest, scene_dir)

        # Get absolute path of destination
        abs_dest = os.path.abspath(args.dest)

        # Generate SCP command
        scp_command = f"scp -r {args.server_address}:{abs_dest} {args.local_dest}"

        print("\nCopy operation completed!")
        print("\n" + "=" * 80)
        print(
            "To transfer the data to your local machine, run this command from your local terminal:"
        )
        print(f"\n{scp_command}")
        print("\nOr for better performance with compression:")
        print(f"scp -r -C {args.server_address}:{abs_dest} {args.local_dest}")
        print("=" * 80)

    else:
        # Copy data for each scene
        for scene_dir in tqdm(scene_dirs, desc="Copying scenes"):
            print(f"\nProcessing scene: {scene_dir}")

            if args.dry_run:
                print(f"[DRY RUN] Would copy data for scene {scene_dir}")
            else:
                copy_scene_data(args.source, args.dest, scene_dir)

        print("\nCopy operation completed!")


if __name__ == "__main__":
    main()
