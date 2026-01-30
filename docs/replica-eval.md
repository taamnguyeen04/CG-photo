---
description: Complete Replica RGBD evaluation workflow (training, photometric, mesh, geometric)
---

# Replica RGBD Evaluation Workflow

// turbo-all

## Prerequisites
- Dataset: `/media/tam/DATA/data/Replica`
- GT meshes: `/media/tam/DATA/data/Replica/{scene}/mesh.ply`

## Step 1: Training (Photo-SLAM)

Run training on Replica scenes:

```bash
cd /media/tam/DATA/3D/CG-photo

# Single scene
./bin/replica_rgbd \
    ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /media/tam/DATA/data/Replica/office0 \
    results_pa9/office0 \
    no_viewer

# All scenes (3 runs each)
bash scripts/replica_rgbd_pa6.sh
```

**Loss weights config:** `cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml`
```yaml
Optimization.lambda_geo: 0.5
Optimization.lambda_smooth: 0.01
Optimization.lambda_var: 0.0
```

## Step 2: Photometric Evaluation (PSNR, SSIM, LPIPS)

```bash
cd /media/tam/DATA/3D/CG-photo/Photo-SLAM-eval
source /media/tam/DATA/3D/Photo-SLAM/venv/bin/activate
# Single scene
python run.py ../results_pa9/office0 /media/tam/DATA/data/Replica/office0

# All scenes at once
python onekey.py --results ../results_replica_pa6_v2

# Or batch eval
python ../scripts/batch_eval_photoslam_flat.py \
    --results_dir ../results_replica_pa6_v2 \
    --dataset_dir /media/tam/DATA/data/Replica
```


echo "=== PSNR ===" && awk '{sum+=$1; count++} END {print "Mean:", sum/count}' /media/tam/DATA/3D/CG-photo/results_pa9/office0/psnr.txt && echo "=== SSIM ===" && awk '{sum+=$1; count++} END {print "Mean:", sum/count}' /media/tam/DATA/3D/CG-photo/results_pa9/office0/ssim.txt && echo "=== LPIPS ===" && awk '{sum+=$1; count++} END {print "Mean:", sum/count}' /media/tam/DATA/3D/CG-photo/results_pa9/office0/lpips.txt

**Output:** `metrics.json` with PSNR, SSIM, LPIPS, ATE

## Step 3: Mesh Generation (cameras.json method - RECOMMENDED)

Generate mesh from rendered depth using `cameras.json` which contains accurate camera poses from Photo-SLAM Gaussian mapper. This method ensures correct coordinate alignment.

### Single Scene

```bash
cd /media/tam/DATA/3D/CG-photo
source scripts/tsdf_env/bin/activate

# Find the shutdown directory containing cameras.json
# Format: results_paX/scene_runY/XXXX_shutdown/ply/cameras.json

python scripts/generate_mesh_from_json.py \
    --json_path results_pa11/room0_run1/3381_shutdown/ply/cameras.json \
    --depth_dir results_pa11/room0_run1/3381_shutdown/depth \
    --output results_pa11/room0_run1/meshes/room0_json_aligned.ply \
    --voxel_size 0.01 \
    --depth_scale 6553.5 \
    --max_depth 10.0 \
    --gt_traj /media/tam/DATA/data/Replica/room0/traj.txt
```

### All Scenes (Batch)

```bash
./scripts/batch_generate_mesh_from_json.sh results_pa11
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--json_path` | path | Path to `cameras.json` from Photo-SLAM output |
| `--depth_dir` | path | Directory with rendered depth images |
| `--gt_traj` | path | **IMPORTANT**: GT trajectory to transform mesh to correct coordinate system |
| `--voxel_size` | 0.01 | TSDF voxel size (m), smaller = more detail |
| `--depth_scale` | 6553.5 | Depth scale for Replica dataset |

> **Note**: The `--gt_traj` flag is essential! Without it, mesh will be in Photo-SLAM's coordinate system (identity at frame 0) instead of GT coordinate system, causing evaluation errors ~100cm.


## Step 4: Geometric Evaluation (Accuracy, Completeness, Chamfer)

### Full Room Mesh Evaluation

Đánh giá mesh toàn phòng so với GT mesh:

```bash
cd /media/tam/DATA/3D/CG-photo/neural_slam_eval-main
source ../scripts/tsdf_env/bin/activate

python eval_recon.py \
    --rec_mesh ../results_pa9/office0/meshes/office0_rendered_full.ply \
    --gt_mesh /media/tam/DATA/data/Replica/cull_replica_mesh/office0.ply \
    -3d
```

**Kết quả mẫu (PA6 office0):**
```
accuracy:  2.66 cm
completion:  2.66 cm
completion ratio:  88.16%
```

### Per-Keyframe Mesh Evaluation

```bash
cd /media/tam/DATA/3D/CG-photo/neural_slam_eval-main
source ../scripts/tsdf_env/bin/activate

# Batch evaluate all keyframe meshes
python3 ../scripts/batch_eval_keyframes.py \
    --meshes_dir ../results_pa7/office0/meshes \
    --gt_mesh /media/tam/DATA/data/Replica/office0_mesh_triangles.ply \
    --eval_script eval_recon.py
```

**Metrics:**
- Accuracy (cm): How close reconstructed mesh is to GT
- Completeness (cm): How much of GT is covered  
- Completion Ratio (%): % of GT covered within 5cm

## Summary of Key Files

| Purpose | File |
|---------|------|
| **Training** | `bin/replica_rgbd` or `scripts/run_full_pipeline.sh` |
| **Loss Config** | `cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml` |
| **Photometric Eval** | `Photo-SLAM-eval/run.py` |
| **Mesh Generation** | `scripts/generate_mesh_from_json.py` (RECOMMENDED) |
| **Batch Mesh Gen** | `scripts/batch_generate_mesh_from_json.sh` |
| **Geometric Eval** | `neural_slam_eval-main/eval_recon.py -3d` |
| **Full Pipeline** | `scripts/run_full_pipeline.sh` |

