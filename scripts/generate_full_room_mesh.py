#!/usr/bin/env python3
"""
Generate mesh for the entire room by accumulating depth maps from multiple frames.
Uses TSDF fusion for better mesh quality.

Example 1: Replica with GT depth (all frames)
    python scripts/generate_full_room_mesh.py \\
        --depth_dir /media/tam/DATA/data/Replica/office0/results \\
        --traj /media/tam/DATA/data/Replica/office0/traj.txt \\
        --output ./meshes/office0_gt_full.ply \\
        --dataset replica \\
        --depth_pattern "depth*.png" \\
        --stride 5

Example 2: Replica with Photo-SLAM rendered depth (keyframes only)
    python scripts/generate_full_room_mesh.py \\
        --depth_dir /media/tam/DATA/3D/CG-photo/results_replica_pa6_v2/replica_rgbd_0/office0/2881_shutdown/depth \\
        --traj /media/tam/DATA/3D/CG-photo/results_replica_pa6_v2/replica_rgbd_0/office0/KeyFrameTrajectory_TUM.txt \\
        --traj_format keyframe_tum \\
        --output ./meshes/office0_rendered_full.ply \\
        --dataset replica \\
        --depth_pattern "*_depth.png" \\
        --frame_id_pattern keyframe
"""

import numpy as np
import open3d as o3d
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm


def load_intrinsics(dataset='replica'):
    """Load camera intrinsics for different datasets."""
    intrinsics = {
        'replica': {'fx': 600.0, 'fy': 600.0, 'cx': 599.5, 'cy': 339.5, 'width': 1200, 'height': 680},
        'tum': {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5, 'width': 640, 'height': 480},
    }
    return intrinsics.get(dataset, intrinsics['replica'])


def load_poses(traj_path, format='replica'):
    """Load camera poses from trajectory file.
    
    Formats:
    - replica: 16 values per line (4x4 matrix row-major), line index = frame ID
    - tum: timestamp tx ty tz qx qy qz qw, timestamp = frame ID
    - keyframe_tum: Same as tum but specifically for keyframe trajectories
    """
    poses = {}
    with open(traj_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = [float(x) for x in line.split()]
            if format == 'replica' and len(values) == 16:
                poses[i] = np.array(values).reshape(4, 4)
            elif format in ['tum', 'keyframe_tum'] and len(values) == 8:
                # TUM format: timestamp tx ty tz qx qy qz qw
                from scipy.spatial.transform import Rotation as R
                frame_id = int(values[0])  # Use timestamp as frame ID
                t = np.array(values[1:4])
                q = np.array([values[4], values[5], values[6], values[7]])  # xyzw
                rot = R.from_quat(q).as_matrix()
                pose = np.eye(4)
                pose[:3, :3] = rot
                pose[:3, 3] = t
                poses[frame_id] = pose
    return poses


def create_tsdf_volume(voxel_size=0.01, sdf_trunc=0.04, volume_bounds=None):
    """Create TSDF volume for fusion."""
    if volume_bounds is not None:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    else:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    return volume


def integrate_frame(volume, depth_img, color_img, intrinsics, pose, depth_scale=1.0, max_depth=10.0):
    """Integrate a single frame into TSDF volume."""
    # Create Open3D intrinsic object
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics['width'], intrinsics['height'],
        intrinsics['fx'], intrinsics['fy'],
        intrinsics['cx'], intrinsics['cy']
    )
    
    # Create RGBD image
    depth_o3d = o3d.geometry.Image((depth_img * 1000).astype(np.uint16))  # Convert to mm
    
    if color_img is not None:
        color_o3d = o3d.geometry.Image(color_img)
    else:
        # Create dummy color (gray)
        color_o3d = o3d.geometry.Image(np.full((intrinsics['height'], intrinsics['width'], 3), 128, dtype=np.uint8))
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,  # We already converted to mm
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False
    )
    
    # Integrate
    volume.integrate(rgbd, o3d_intrinsic, np.linalg.inv(pose))


def collect_depth_files(depth_dir, pattern="*.png"):
    """Collect all depth files from directory."""
    depth_files = sorted(Path(depth_dir).glob(pattern))
    return depth_files


def build_keyframe_mapping(traj_path):
    """Build mapping from keyframe index to frame ID from TUM-format keyframe trajectory.
    
    Returns:
        dict: {keyframe_index: frame_id}
    """
    mapping = {}
    with open(traj_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = line.split()
            if len(values) >= 1:
                frame_id = int(float(values[0]))  # timestamp is frame ID
                mapping[i] = frame_id
    return mapping


def extract_frame_id(filename, pattern='replica'):
    """Extract frame ID from filename.
    
    Patterns:
    - replica: depth000000.png -> 0
    - rendered: 24_depth.png -> 24
    - keyframe: 2881_24_depth.png -> 24 (second number is keyframe index)
    """
    name = filename.stem
    if pattern == 'replica':
        # depth000000.png -> 0
        if name.startswith('depth'):
            return int(name.replace('depth', ''))
    elif pattern == 'keyframe':
        # Photo-SLAM keyframe pattern: 2881_24_depth.png -> 24
        # Format: {total_iter}_{keyframe_idx}_depth.png
        parts = name.replace('_depth', '').split('_')
        if len(parts) >= 2:
            return int(parts[1])  # Return keyframe index
        return int(parts[0])
    elif pattern == 'rendered':
        # Various patterns like 24_depth.png
        parts = name.replace('_depth', '').split('_')
        return int(parts[-1])
    # Fallback: try to extract any number
    import re
    nums = re.findall(r'\d+', name)
    if nums:
        return int(nums[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description='Generate mesh for entire room using TSDF fusion')
    parser.add_argument('--depth_dir', type=str, required=True, help='Directory containing depth images')
    parser.add_argument('--color_dir', type=str, default=None, help='Directory containing color images (optional)')
    parser.add_argument('--traj', type=str, required=True, help='Trajectory file (GT or keyframe trajectory)')
    parser.add_argument('--traj_format', type=str, default=None, choices=['replica', 'tum', 'keyframe_tum'], 
                        help='Trajectory file format. If not specified, uses dataset type.')
    parser.add_argument('--keyframe_traj', type=str, default=None, 
                        help='Optional: Keyframe trajectory for mapping keyframe index to frame ID (use with GT poses)')
    parser.add_argument('--output', type=str, required=True, help='Output mesh path (.ply)')
    parser.add_argument('--depth_scale', type=float, default=6553.5, help='Depth scale factor')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='TSDF voxel size (meters)')
    parser.add_argument('--sdf_trunc', type=float, default=0.04, help='TSDF truncation distance')
    parser.add_argument('--max_depth', type=float, default=10.0, help='Maximum depth to consider')
    parser.add_argument('--stride', type=int, default=1, help='Process every N-th frame')
    parser.add_argument('--dataset', type=str, default='replica', choices=['replica', 'tum'], help='Dataset type (for intrinsics)')
    parser.add_argument('--depth_pattern', type=str, default='depth*.png', help='Depth image filename pattern')
    parser.add_argument('--frame_id_pattern', type=str, default='replica', choices=['replica', 'rendered', 'keyframe'], help='How to extract frame ID from filename')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process')
    args = parser.parse_args()
    
    # Load intrinsics
    intrinsics = load_intrinsics(args.dataset)
    print(f"Using intrinsics for {args.dataset}: {intrinsics}")
    
    # Determine trajectory format
    traj_format = args.traj_format if args.traj_format else args.dataset
    
    # Load poses
    poses = load_poses(args.traj, format=traj_format)
    print(f"Loaded {len(poses)} poses from {args.traj}")
    
    # Build keyframe mapping if using keyframe pattern
    # If --keyframe_traj is provided, use it for mapping (allows using GT poses with keyframe depths)
    # Otherwise, use the main trajectory
    keyframe_mapping = None
    if args.frame_id_pattern == 'keyframe':
        if args.keyframe_traj:
            # Use separate keyframe trajectory for mapping
            keyframe_mapping = build_keyframe_mapping(args.keyframe_traj)
            print(f"Built keyframe mapping from {args.keyframe_traj} with {len(keyframe_mapping)} entries")
        elif traj_format in ['keyframe_tum', 'tum']:
            # Use main trajectory for mapping
            keyframe_mapping = build_keyframe_mapping(args.traj)
            print(f"Built keyframe mapping with {len(keyframe_mapping)} entries")
    
    # Collect depth files
    depth_files = collect_depth_files(args.depth_dir, args.depth_pattern)
    print(f"Found {len(depth_files)} depth images in {args.depth_dir}")
    
    if len(depth_files) == 0:
        print("No depth files found! Check the depth_dir and depth_pattern arguments.")
        return
    
    # Create TSDF volume
    volume = create_tsdf_volume(
        voxel_size=args.voxel_size,
        sdf_trunc=args.sdf_trunc
    )
    
    # Process frames
    processed = 0
    skipped = 0
    for i, depth_path in enumerate(tqdm(depth_files, desc="Integrating frames")):
        if i % args.stride != 0:
            continue
        
        if args.max_frames and processed >= args.max_frames:
            break
        
        # Extract frame ID (or keyframe index)
        extracted_id = extract_frame_id(depth_path, args.frame_id_pattern)
        if extracted_id is None:
            print(f"Warning: Could not extract ID from {depth_path}")
            continue
        
        # Determine the actual frame ID for pose lookup
        if args.frame_id_pattern == 'keyframe' and keyframe_mapping:
            # extracted_id is keyframe index, need to map to frame ID
            if extracted_id not in keyframe_mapping:
                skipped += 1
                continue
            frame_id = keyframe_mapping[extracted_id]
        else:
            frame_id = extracted_id
        
        # Get pose
        if frame_id not in poses:
            skipped += 1
            continue
        pose = poses[frame_id]
        
        # Load depth
        depth_img = np.array(Image.open(depth_path)).astype(np.float32) / args.depth_scale
        
        # Load color if available
        color_img = None
        if args.color_dir:
            # Try to find corresponding color image
            color_patterns = [
                f"rgb{frame_id:06d}.png",
                f"frame{frame_id:06d}.jpg",
                f"frame{frame_id:06d}.png",
                f"{frame_id:06d}.png",
                f"{frame_id:06d}.jpg",
            ]
            for cp in color_patterns:
                color_path = Path(args.color_dir) / cp
                if color_path.exists():
                    color_img = np.array(Image.open(color_path))
                    if len(color_img.shape) == 2:
                        color_img = np.stack([color_img]*3, axis=-1)
                    break
        
        # Integrate frame
        try:
            integrate_frame(volume, depth_img, color_img, intrinsics, pose, 
                          max_depth=args.max_depth)
            processed += 1
        except Exception as e:
            print(f"Warning: Failed to integrate frame {frame_id}: {e}")
            continue
    
    print(f"Integrated {processed} frames (skipped {skipped})")
    
    # Extract mesh
    print("Extracting mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # Save mesh
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    print(f"Saved mesh to {output_path}")
    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Also extract and save point cloud
    pcd = volume.extract_point_cloud()
    pcd_path = output_path.with_suffix('.pcd')
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"Saved point cloud to {pcd_path}")


if __name__ == '__main__':
    main()
