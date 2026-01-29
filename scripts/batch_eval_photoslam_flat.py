#!/usr/bin/env python3
"""
Batch evaluate individual mesh files for ORIGINAL Photo-SLAM model.
It handles flat .ply files in 'meshes_world' folder (e.g., depth_2582_0.ply).
Usage:
    python3 batch_eval_photoslam_flat.py \
        --meshes_dir /path/to/meshes_world \
        --gt_mesh /path/to/gt_mesh.ply \
        --eval_script eval_recon.py
"""

import argparse
import subprocess
import re
from pathlib import Path
import sys
import numpy as np

def parse_metrics(output_str):
    metrics = {}
    for line in output_str.split('\n'):
        if 'accuracy:' in line:
            metrics['accuracy'] = float(line.split(':')[1].strip())
        elif 'completion:' in line:
            metrics['completion'] = float(line.split(':')[1].strip())
        elif 'completion ratio:' in line:
            metrics['completion_ratio'] = float(line.split(':')[1].strip())
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Batch evaluate Photo-SLAM flat meshes')
    parser.add_argument('--meshes_dir', required=True, help='Directory containing .ply files (e.g. meshes_world)')
    parser.add_argument('--gt_mesh', required=True, help='Path to GT mesh')
    parser.add_argument('--eval_script', default='eval_recon.py', help='Path to eval script')
    args = parser.parse_args()

    meshes_dir = Path(args.meshes_dir)
    results = []

    # Find all ply files matching pattern depth_*.ply
    ply_files = sorted(list(meshes_dir.glob('depth_*.ply')))
    
    if not ply_files:
        # Fallback to all ply files if pattern doesn't match
        ply_files = sorted(list(meshes_dir.glob('*.ply')))

    if not ply_files:
        print(f"No .ply files found in {meshes_dir}")
        sys.exit(1)

    # Sort ply files by FID (last number in filename)
    # Example: depth_2582_10.ply -> 10
    def get_fid(p):
        match = re.search(r'_(\d+)\.ply$', p.name)
        return int(match.group(1)) if match else -1
    
    ply_files.sort(key=get_fid)

    print(f"Found {len(ply_files)} meshes to evaluate in {meshes_dir}")
    print("-" * 60)
    print(f"{'Mesh Name':<30} | {'Acc (cm)':<10} | {'Comp (cm)':<10} | {'Ratio':<10}")
    print("-" * 60)

    for mesh_path in ply_files:
        mesh_name = mesh_path.name
        fid = get_fid(mesh_path)
        
        # Run evaluation
        cmd = [
            sys.executable, args.eval_script,
            '--rec_mesh', str(mesh_path),
            '--gt_mesh', args.gt_mesh,
            '-3d'
        ]
        
        try:
            # Capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metrics = parse_metrics(result.stdout)
            
            if metrics:
                results.append(metrics)
                display_name = mesh_name
                if len(display_name) > 30:
                    display_name = "..." + display_name[-27:]
                print(f"{display_name:<30} | {metrics.get('accuracy', 0):<10.2f} | {metrics.get('completion', 0):<10.2f} | {metrics.get('completion_ratio', 0):<10.2f}")
            else:
                print(f"{mesh_name:<30} | {'Error parsing':<30}")
                
        except subprocess.CalledProcessError as e:
            print(f"{mesh_name:<30} | {'Error running':<30}")

    if results:
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_comp = np.mean([r['completion'] for r in results])
        print("-" * 60)
        print(f"{'AVERAGE':<30} | {avg_acc:<10.2f} | {avg_comp:<10.2f} |")
        print("-" * 60)

if __name__ == '__main__':
    main()
