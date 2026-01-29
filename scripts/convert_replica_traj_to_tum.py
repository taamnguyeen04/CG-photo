#!/usr/bin/env python3
"""
Convert Replica traj.txt (4x4 transformation matrices) to TUM format
TUM format: timestamp tx ty tz qx qy qz qw
"""
import numpy as np
from scipy.spatial.transform import Rotation
import os
import sys

def matrix_to_tum(matrix_4x4):
    """Convert 4x4 transformation matrix to TUM format (tx ty tz qx qy qz qw)"""
    # Extract translation
    tx, ty, tz = matrix_4x4[:3, 3]
    
    # Extract rotation matrix and convert to quaternion
    rotation_matrix = matrix_4x4[:3, :3]
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()  # Returns [qx, qy, qz, qw]
    
    return tx, ty, tz, quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]

def convert_replica_traj_to_tum(input_file, output_file):
    """Convert Replica traj.txt to TUM format"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    tum_data = []
    for i, line in enumerate(lines):
        values = list(map(float, line.strip().split()))
        if len(values) != 16:
            print(f"Warning: Line {i+1} has {len(values)} values, expected 16. Skipping.")
            continue
        
        # Reshape to 4x4 matrix (row-major order)
        matrix = np.array(values).reshape(4, 4)
        
        # Convert to TUM format
        tx, ty, tz, qx, qy, qz, qw = matrix_to_tum(matrix)
        
        # Use frame index as timestamp
        timestamp = i
        
        tum_data.append(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
    
    # Write TUM format file
    with open(output_file, 'w') as f:
        f.writelines(tum_data)
    
    print(f"Converted {len(tum_data)} poses from {input_file} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_replica_traj_to_tum.py <replica_dataset_path>")
        print("Example: python convert_replica_traj_to_tum.py /media/tam/DATA/data/Replica")
        sys.exit(1)
    
    replica_path = sys.argv[1]
    scenes = ['office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2']
    
    for scene in scenes:
        traj_file = os.path.join(replica_path, scene, 'traj.txt')
        tum_file = os.path.join(replica_path, scene, 'traj_TUM.txt')
        
        if os.path.exists(traj_file):
            print(f"Processing {scene}...")
            convert_replica_traj_to_tum(traj_file, tum_file)
        else:
            print(f"Warning: {traj_file} not found. Skipping.")
    
    print("\nAll conversions complete!")
