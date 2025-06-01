import os
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

        results.append(
            {
                "directory": scene_dir,
                "has_keypoints": has_keypoints,
                "has_matches": has_matches,
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


if __name__ == "__main__":
    base_path = "data/scannetppv2/data"

    # Check if the path exists
    if not os.path.exists(base_path):
        print(f"Error: Path '{base_path}' does not exist.")
        exit(1)

    # Run the check
    results_df = check_directories(base_path)

    # Print only directories that contain both keypoints and matches
    if not results_df.empty:
        complete = results_df[results_df["complete"]]
        if not complete.empty:
            print("\nDirectories with both keypoints and matches in dslr subdirectory:")
            for directory in complete["directory"]:
                print(f"- {directory}")
