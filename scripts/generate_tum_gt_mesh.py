#!/usr/bin/env python3
"""
Generate GT mesh for TUM RGB-D dataset using depth maps and GT trajectory.

Usage:
    python generate_tum_gt_mesh.py --scene freiburg1_desk
    python generate_tum_gt_mesh.py --scene freiburg2_xyz --voxel_size 0.02
    python generate_tum_gt_mesh.py --all  # Generate for all 3 scenes
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# TUM RGB-D dataset intrinsics (from official TUM website)
# freiburg1 and freiburg2: 640x480
# freiburg3: 640x480
TUM_INTRINSICS = {
    'freiburg1': {'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3, 'width': 640, 'height': 480},
    'freiburg2': {'fx': 520.9, 'fy': 521.0, 'cx': 325.1, 'cy': 249.7, 'width': 640, 'height': 480},
    'freiburg3': {'fx': 535.4, 'fy': 539.2, 'cx': 320.1, 'cy': 247.6, 'width': 640, 'height': 480},
}

# TUM scene names
TUM_SCENES = [
    'freiburg1_desk',
    'freiburg2_xyz',
    'freiburg3_long_office_household',
]


def load_groundtruth(gt_path: Path) -> dict:
    """
    Load TUM groundtruth.txt file.
    Format: timestamp tx ty tz qx qy qz qw
    Returns: dict[timestamp] = 4x4 pose matrix (camera-to-world)
    """
    poses = {}
    with open(gt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            
            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            # Build 4x4 transformation matrix
            rotation = Rotation.from_quat([qx, qy, qz, qw])
            R = rotation.as_matrix()
            
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = [tx, ty, tz]
            
            poses[timestamp] = pose
    
    return poses


def load_depth_list(depth_txt_path: Path) -> list:
    """
    Load depth.txt file.
    Format: timestamp depth/filename.png
    Returns: list of (timestamp, relative_depth_path)
    """
    depth_list = []
    with open(depth_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            timestamp = float(parts[0])
            depth_path = parts[1]
            depth_list.append((timestamp, depth_path))
    return depth_list


def find_closest_pose(query_timestamp: float, poses: dict, max_diff: float = 0.05) -> np.ndarray:
    """
    Find the closest pose to the query timestamp.
    Returns None if no pose is within max_diff seconds.
    """
    if not poses:
        return None
    
    best_ts = None
    best_diff = float('inf')
    
    for ts in poses.keys():
        diff = abs(ts - query_timestamp)
        if diff < best_diff:
            best_diff = diff
            best_ts = ts
    
    if best_diff > max_diff:
        return None
    
    return poses[best_ts]


def generate_gt_mesh(scene_name: str, data_dir: Path, output_dir: Path, 
                     voxel_size: float = 0.01, max_depth: float = 5.0, stride: int = 5):
    """
    Generate GT mesh for a TUM scene.
    
    Args:
        scene_name: e.g., 'freiburg1_desk'
        data_dir: Path to TUM data root (e.g., /media/tam/DATA/data/TUM)
        output_dir: Path to save GT meshes
        voxel_size: TSDF voxel size
        max_depth: Maximum depth truncation in meters
        stride: Process every Nth frame
    """
    scene_dir = data_dir / f"rgbd_dataset_{scene_name}"
    
    if not scene_dir.exists():
        print(f"[ERROR] Scene directory not found: {scene_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Generating GT mesh for: {scene_name}")
    print(f"Scene directory: {scene_dir}")
    print(f"{'='*60}")
    
    # Determine intrinsics based on scene prefix
    if 'freiburg1' in scene_name:
        intrinsics = TUM_INTRINSICS['freiburg1']
    elif 'freiburg2' in scene_name:
        intrinsics = TUM_INTRINSICS['freiburg2']
    else:
        intrinsics = TUM_INTRINSICS['freiburg3']
    
    print(f"Intrinsics: fx={intrinsics['fx']}, fy={intrinsics['fy']}, "
          f"cx={intrinsics['cx']}, cy={intrinsics['cy']}")
    
    # Create Open3D intrinsic object
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics['width'], intrinsics['height'],
        intrinsics['fx'], intrinsics['fy'],
        intrinsics['cx'], intrinsics['cy']
    )
    
    # Load groundtruth poses
    gt_path = scene_dir / 'groundtruth.txt'
    print(f"Loading groundtruth from: {gt_path}")
    poses = load_groundtruth(gt_path)
    print(f"Loaded {len(poses)} poses")
    
    # Load depth list
    depth_txt_path = scene_dir / 'depth.txt'
    print(f"Loading depth list from: {depth_txt_path}")
    depth_list = load_depth_list(depth_txt_path)
    print(f"Found {len(depth_list)} depth frames")
    
    # Initialize TSDF Volume
    print(f"Initializing TSDF Volume (voxel_size={voxel_size}, stride={stride})...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    # TUM depth scale: depth_value / 5000.0 = depth_in_meters
    depth_scale = 5000.0
    
    # Integrate frames
    integrated_count = 0
    skipped_count = 0
    
    for i, (timestamp, rel_depth_path) in enumerate(tqdm(depth_list, desc="Integrating frames")):
        # Skip frames based on stride
        if i % stride != 0:
            continue
        
        # Find corresponding pose
        pose = find_closest_pose(timestamp, poses)
        if pose is None:
            skipped_count += 1
            continue
        
        # Load depth image
        depth_path = scene_dir / rel_depth_path
        if not depth_path.exists():
            skipped_count += 1
            continue
        
        try:
            depth_img = np.array(Image.open(depth_path))
        except Exception as e:
            print(f"Error loading {depth_path}: {e}")
            skipped_count += 1
            continue
        
        # Load corresponding RGB image (optional, for colored mesh)
        # TUM depth timestamp ~ rgb timestamp, try to find matching RGB
        rgb_timestamp = timestamp
        rgb_path = None
        
        # Simple matching: look for RGB file with closest timestamp
        rgb_dir = scene_dir / 'rgb'
        if rgb_dir.exists():
            # Find RGB file with similar timestamp
            depth_ts_str = str(timestamp)
            for rgb_file in rgb_dir.iterdir():
                if rgb_file.suffix == '.png':
                    rgb_ts_str = rgb_file.stem
                    try:
                        rgb_ts = float(rgb_ts_str)
                        if abs(rgb_ts - timestamp) < 0.05:
                            rgb_path = rgb_file
                            break
                    except:
                        continue
        
        # Load or create color image
        if rgb_path and rgb_path.exists():
            color_img = np.array(Image.open(rgb_path))
        else:
            # Create dummy color
            h, w = depth_img.shape
            color_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert to Open3D format
        depth_o3d = o3d.geometry.Image(depth_img.astype(np.uint16))
        color_o3d = o3d.geometry.Image(color_img.astype(np.uint8))
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False
        )
        
        # Integrate (Open3D expects extrinsic = world-to-camera = inv(pose))
        Tcw = np.linalg.inv(pose)
        volume.integrate(rgbd, cam_intrinsic, Tcw)
        integrated_count += 1
    
    print(f"Integrated {integrated_count} frames (skipped {skipped_count})")
    
    if integrated_count == 0:
        print("[ERROR] No frames were integrated!")
        return False
    
    # Extract mesh
    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # Save mesh
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / f"{scene_name}_gt.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"Saved GT mesh to: {mesh_path}")
    
    # Also save as point cloud
    pcd = volume.extract_point_cloud()
    pcd_path = output_dir / f"{scene_name}_gt.pcd"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"Saved GT point cloud to: {pcd_path}")
    
    # Print mesh stats
    print(f"Mesh vertices: {len(mesh.vertices)}, triangles: {len(mesh.triangles)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate GT mesh for TUM RGB-D dataset')
    parser.add_argument('--scene', type=str, default=None, 
                        help='Scene name (e.g., freiburg1_desk, freiburg2_xyz, freiburg3_long_office_household)')
    parser.add_argument('--all', action='store_true', help='Generate for all 3 TUM scenes')
    parser.add_argument('--data_dir', type=str, default='/media/tam/DATA/data/TUM',
                        help='TUM dataset root directory')
    parser.add_argument('--output_dir', type=str, default='/media/tam/DATA/data/TUM/gt_meshes',
                        help='Output directory for GT meshes')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='TSDF voxel size (meters)')
    parser.add_argument('--max_depth', type=float, default=5.0, help='Maximum depth truncation (meters)')
    parser.add_argument('--stride', type=int, default=5, help='Process every Nth frame')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if args.all:
        scenes = TUM_SCENES
    elif args.scene:
        scenes = [args.scene]
    else:
        print("Error: Please specify --scene <name> or --all")
        print(f"Available scenes: {TUM_SCENES}")
        return
    
    print(f"TUM GT Mesh Generator")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scenes to process: {scenes}")
    print(f"Voxel size: {args.voxel_size}")
    print(f"Max depth: {args.max_depth}")
    print(f"Stride: {args.stride}")
    
    successful = []
    failed = []
    
    for scene in scenes:
        try:
            if generate_gt_mesh(scene, data_dir, output_dir, 
                               args.voxel_size, args.max_depth, args.stride):
                successful.append(scene)
            else:
                failed.append(scene)
        except Exception as e:
            print(f"[ERROR] Failed to process {scene}: {e}")
            failed.append(scene)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successful: {len(successful)} - {successful}")
    print(f"Failed: {len(failed)} - {failed}")
    print(f"GT meshes saved to: {output_dir}")


if __name__ == "__main__":
    main()
