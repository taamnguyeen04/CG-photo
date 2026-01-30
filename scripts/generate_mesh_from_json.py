import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def load_cameras_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cameras = {}
    for cam in data:
        # Extract ID
        fid = cam['id']
        
        # Extract intrinsic parameters
        fx = cam['fx']
        fy = cam['fy']
        width = cam['width']
        height = cam['height']
        # Assume cx, cy are at the center if not specified (json usually doesn't have them explicitly for some reason in this format, 
        # or we calculate from width/height). 
        # Wait, the user's json snippet DOES NOT have cx/cy.
        # But `generate_full_room_mesh.py` used cx=599.5, cy=339.5 for Replica (1200x680).
        # We will deduce them:
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        
        # Extract Rotation (3x3) and Position (3)
        R = np.array(cam['rotation']) # 3x3 list
        t = np.array(cam['position']) # 3 list elements
        
        # Construct 4x4 Pose Matrix (Camera-to-World)
        # Note: The JSON usually stores R_wc and t_wc (Camera to World)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        
        cameras[fid] = {
            'intrinsic': intrinsic,
            'pose': pose
        }
    return cameras

def load_gt_traj(traj_path):
    """Load GT trajectory file (Replica format: 4x4 matrix per line)"""
    with open(traj_path, 'r') as f:
        line = f.readline().strip()
        values = [float(x) for x in line.split()]
        pose = np.array(values).reshape(4, 4)
    return pose

def main():
    parser = argparse.ArgumentParser(description='Generate mesh using cameras.json for precise alignment')
    parser.add_argument('--json_path', type=str, required=True, help='Path to cameras.json')
    parser.add_argument('--depth_dir', type=str, required=True, help='Directory containing depth images')
    parser.add_argument('--output', type=str, default='mesh.ply', help='Output mesh path')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel size for TSDF')
    parser.add_argument('--max_depth', type=float, default=10.0, help='Max depth truncation (meters)')
    parser.add_argument('--depth_scale', type=float, default=5000.0, help='Depth scale factor (default: 5000 for TUM/Replica in Photo-SLAM)')
    parser.add_argument('--gt_traj', type=str, default=None, help='GT trajectory file (Replica format) for coordinate alignment')
    
    args = parser.parse_args()
    
    # 1. Load Cameras (Poses + Intrinsics)
    print(f"Loading cameras from {args.json_path}...")
    cameras = load_cameras_json(args.json_path)
    print(f"Loaded {len(cameras)} cameras.")
    
    # 2. Initialize TSDF Volume
    print(f"Initializing TSDF Volume with voxel_size={args.voxel_size}...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.voxel_size * 5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )
    
    # 3. Integrate Frames
    depth_dir = Path(args.depth_dir)
    processed_count = 0
    
    # Sort keys to process in order (nice for progress bar)
    sorted_fids = sorted(cameras.keys())
    
    for fid in tqdm(sorted_fids, desc="Integrating frames"):
        cam_info = cameras[fid]
        
        # Construct depth filename patterns
        # User has files like: 6681_1_depth.png (iteration_ID_depth.png)
        # We look for files ending with _{fid}_depth.png or exact match logic
        # A simple glob is safer.
        
        # Pattern: *_{fid}_depth.png
        # This handles "6681_1_depth.png" where fid is 1.
        candidates = list(depth_dir.glob(f"*_{fid}_depth.png"))
        
        if not candidates:
            # Try just {fid}_depth.png
            candidates = list(depth_dir.glob(f"{fid}_depth.png"))
            
        if not candidates:
            # print(f"Warning: No depth file found for Frame ID {fid}. Skipping.")
            continue
            
        depth_path = candidates[0] # Take the first match
        
        # Load Depth Image
        try:
            depth_img_raw = np.array(Image.open(depth_path))
        except Exception as e:
            print(f"Error loading {depth_path}: {e}")
            continue

        # Convert to Open3D Image
        # NOTE: Open3D expects depth in uint16 (or float), but we must handle scaling.
        # We can pass the raw uint16 to Open3D and let it handle 5000.0 scale.
        depth_o3d = o3d.geometry.Image(depth_img_raw)
        
        # Create RGBD Image (Depth only)
        # We create a dummy color image because ScalableTSDFVolume often expects RGBD
        # But we set color_type=NoColor, so maybe it's fine.
        # Ideally, we use RGBDImage.create_from_color_and_depth with convert_rgb_to_intensity=False
        
        # Create dummy black color
        h, w = depth_img_raw.shape
        color_o3d = o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=args.depth_scale,
            depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False
        )
        
        # Integrate
        # Tcw (World-to-Camera) is needed? NO, Open3D integrate takes EXTRINSIC (World-to-Camera).
        # Our JSON provides Pose (Camera-to-World). So we need inverse.
        # Wait, let's verify ScalableTSDFVolume.integrate documentation.
        # "pose (numpy.ndarray[float64[4, 4]]) – extrinsic camera parameter"
        # Usually extrinsic means World-to-Camera (Tcw).
        # Let's check typical usage.
        # In `generate_full_room_mesh.py`, it calculates `np.linalg.inv(pose)` before passing.
        # Because `pose` loaded from TUM format is Camera-to-World (Twc).
        # So we also need to inverse it.
        
        Twc = cam_info['pose']
        Tcw = np.linalg.inv(Twc)
        
        volume.integrate(rgbd, cam_info['intrinsic'], Tcw)
        processed_count += 1
        
    print(f"Integrated {processed_count} frames.")
    
    # 4. Extract Mesh
    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # 5. Transform to GT coordinate system if gt_traj provided
    if args.gt_traj:
        print(f"Transforming mesh to GT coordinate system using {args.gt_traj}...")
        
        # Get first frame pose from cameras.json
        first_cam = cameras[sorted_fids[0]]
        json_pose = first_cam['pose']
        
        # Load GT first frame pose
        gt_pose = load_gt_traj(args.gt_traj)
        
        # Calculate transformation: GT = T @ JSON => T = GT @ inv(JSON)
        T = gt_pose @ np.linalg.inv(json_pose)
        
        print(f"Transformation matrix:\n{T}")
        
        # Apply transformation to mesh
        mesh.transform(T)
        print("Mesh transformed to GT coordinate system.")
    
    # 6. Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    print(f"Saved mesh to {output_path}")

    # Also save Point Cloud for verification
    pcd = volume.extract_point_cloud()
    if args.gt_traj:
        pcd.transform(T)
    pcd_path = output_path.with_suffix('.pcd')
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"Saved point cloud to {pcd_path}")

if __name__ == "__main__":
    main()
