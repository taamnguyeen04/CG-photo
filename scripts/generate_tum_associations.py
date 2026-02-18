#!/usr/bin/env python3
"""
Generate association files for ALL TUM RGB-D sequences.
Matches rgb.txt and depth.txt by closest timestamp.

Usage:
    python scripts/generate_tum_associations.py

Output:
    Creates association files in cfg/ORB_SLAM3/RGB-D/TUM/associations/
    Also copies to each dataset folder as associations.txt
"""

import os
import sys

TUM_DATA_DIR = "/media/tam/DATA/data/TUM"
ASSOC_OUTPUT_DIR = "/media/tam/DATA/3D/CG-photo/cfg/ORB_SLAM3/RGB-D/TUM/associations"

# Map folder name → short association file name
FOLDER_TO_SHORT = {
    "rgbd_dataset_freiburg1_desk": "fr1_desk",
    "rgbd_dataset_freiburg1_room": "fr1_room",
    "rgbd_dataset_freiburg1_360": "fr1_360",
    "rgbd_dataset_freiburg1_rpy": "fr1_rpy",
    "rgbd_dataset_freiburg2_xyz": "fr2_xyz",
    "rgbd_dataset_freiburg2_desk": "fr2_desk",
    "rgbd_dataset_freiburg2_360_hemisphere": "fr2_360h",
    "rgbd_dataset_freiburg2_360_kidnap": "fr2_360k",
    "rgbd_dataset_freiburg2_large_no_loop": "fr2_large",
    "rgbd_dataset_freiburg2_pioneer_slam": "fr2_pioneer",
    "rgbd_dataset_freiburg3_long_office_household": "fr3_office",
    "rgbd_dataset_freiburg3_structure_texture_near": "fr3_str_tex_near",
    "rgbd_dataset_freiburg3_structure_texture_far": "fr3_str_tex_far",
    "rgbd_dataset_freiburg3_structure_notexture_near": "fr3_str_notex_near",
    "rgbd_dataset_freiburg3_structure_notexture_far": "fr3_str_notex_far",
    "rgbd_dataset_freiburg3_nostructure_texture_near_withloop": "fr3_nstr_tex_near",
    "rgbd_dataset_freiburg3_nostructure_texture_far": "fr3_nstr_tex_far",
    "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop": "fr3_nstr_notex_near",
    "rgbd_dataset_freiburg3_nostructure_notexture_far": "fr3_nstr_notex_far",
}


def read_file_list(filename):
    """Read a TUM-format file list (timestamp filename), skip comments."""
    data = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                timestamp = float(parts[0])
                path = parts[1]
                data.append((timestamp, path))
    return data


def associate(rgb_list, depth_list, max_diff=0.02):
    """Associate two lists by closest timestamps (within max_diff seconds)."""
    matches = []
    depth_idx = 0
    for rgb_ts, rgb_path in rgb_list:
        # Find closest depth timestamp
        best_diff = max_diff
        best_match = None
        for i in range(max(0, depth_idx - 5), len(depth_list)):
            d_ts, d_path = depth_list[i]
            diff = abs(rgb_ts - d_ts)
            if diff < best_diff:
                best_diff = diff
                best_match = (d_ts, d_path)
                depth_idx = i
            elif d_ts > rgb_ts + max_diff:
                break
        if best_match is not None:
            matches.append((rgb_ts, rgb_path, best_match[0], best_match[1]))
    return matches


def generate_for_folder(folder_path, folder_name):
    """Generate association file for one TUM sequence."""
    rgb_file = os.path.join(folder_path, "rgb.txt")
    depth_file = os.path.join(folder_path, "depth.txt")

    if not os.path.exists(rgb_file) or not os.path.exists(depth_file):
        return None, "missing rgb.txt or depth.txt"

    rgb_list = read_file_list(rgb_file)
    depth_list = read_file_list(depth_file)

    if not rgb_list or not depth_list:
        return None, "empty rgb.txt or depth.txt"

    matches = associate(rgb_list, depth_list)

    if not matches:
        return None, "no matches found"

    # Write to ORB associations dir
    short_name = FOLDER_TO_SHORT.get(folder_name, folder_name.replace("rgbd_dataset_", ""))
    assoc_path = os.path.join(ASSOC_OUTPUT_DIR, f"{short_name}.txt")

    with open(assoc_path, 'w') as f:
        for rgb_ts, rgb_path, d_ts, d_path in matches:
            f.write(f"{rgb_ts:.6f} {rgb_path} {d_ts:.6f} {d_path}\n")

    # Also save to dataset dir
    dataset_assoc = os.path.join(folder_path, "associations.txt")
    with open(dataset_assoc, 'w') as f:
        for rgb_ts, rgb_path, d_ts, d_path in matches:
            f.write(f"{rgb_ts:.6f} {rgb_path} {d_ts:.6f} {d_path}\n")

    return len(matches), None


def main():
    os.makedirs(ASSOC_OUTPUT_DIR, exist_ok=True)

    # Find all TUM dataset folders
    folders = sorted([
        d for d in os.listdir(TUM_DATA_DIR)
        if d.startswith("rgbd_dataset_") and os.path.isdir(os.path.join(TUM_DATA_DIR, d))
    ])

    print(f"Found {len(folders)} TUM sequences\n")

    total_ok = 0
    total_skip = 0
    for folder_name in folders:
        folder_path = os.path.join(TUM_DATA_DIR, folder_name)
        count, error = generate_for_folder(folder_path, folder_name)

        short = FOLDER_TO_SHORT.get(folder_name, "???")
        if error:
            print(f"  [X] {folder_name} ({short}): {error}")
            total_skip += 1
        else:
            print(f"  [✓] {folder_name} ({short}): {count} pairs")
            total_ok += 1

    print(f"\nDone: {total_ok} generated, {total_skip} skipped")
    print(f"Files saved to: {ASSOC_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
