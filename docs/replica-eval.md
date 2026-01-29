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
    results_pa7/office0 \
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
python run.py ../results_pa7/office0 /media/tam/DATA/data/Replica/office0

# All scenes at once
python onekey.py --results ../results_replica_pa6_v2

# Or batch eval
python ../scripts/batch_eval_photoslam_flat.py \
    --results_dir ../results_replica_pa6_v2 \
    --dataset_dir /media/tam/DATA/data/Replica
```

**Output:** `metrics.json` with PSNR, SSIM, LPIPS, ATE

## Step 3: Mesh Generation

Generate mesh from rendered depth using Poisson Reconstruction.

### Single Frame
```bash
cd /media/tam/DATA/3D/CG-photo
source scripts/tsdf_env/bin/activate

python scripts/compare_single_frame_mesh.py \
    --frame_id 0 \
    --result_dir results_pa7/office0 \
    --gt_depth_dir /media/tam/DATA/data/Replica/office0/results \
    --gt_traj /media/tam/DATA/data/Replica/office0/traj.txt \
    --output_dir results_pa7/office0/meshes
```

### All Keyframes (Batch)
```bash
python scripts/batch_compare_keyframes.py \
    --result_dir results_pa7/office0 \
    --gt_depth_dir /media/tam/DATA/data/Replica/office0/results \
    --gt_traj /media/tam/DATA/data/Replica/office0/traj.txt \
    --output_dir results_pa7/office0/meshes
```

**Output:**
- `frame{N}_rendered.ply` - Mesh từ rendered depth của Photo-SLAM
- `frame{N}_gt.ply` - Mesh từ GT depth (để so sánh)

### Full Room Mesh (TSDF Fusion)

Tạo mesh cho toàn bộ căn phòng bằng cách tích hợp depth maps từ nhiều keyframes:

```bash
cd /media/tam/DATA/3D/CG-photo
source scripts/tsdf_env/bin/activate

# Mesh từ Photo-SLAM rendered depth (dùng GT poses để align đúng trục)
python scripts/generate_full_room_mesh.py \
    --depth_dir results_replica_pa6_v2/replica_rgbd_0/office0/2881_shutdown/depth \
    --traj /media/tam/DATA/data/Replica/office0/traj.txt \
    --keyframe_traj results_replica_pa6_v2/replica_rgbd_0/office0/KeyFrameTrajectory_TUM.txt \
    --output meshes/office0_rendered_full.ply \
    --dataset replica \
    --depth_pattern "*_depth.png" \
    --frame_id_pattern keyframe \
    --voxel_size 0.01

# Mesh từ GT depth (để so sánh)
python scripts/generate_full_room_mesh.py \
    --depth_dir /media/tam/DATA/data/Replica/office0/results \
    --traj /media/tam/DATA/data/Replica/office0/traj.txt \
    --output meshes/office0_gt_full.ply \
    --dataset replica \
    --depth_pattern "depth*.png" \
    --stride 5 \
    --voxel_size 0.01
```

**Tham số quan trọng:**
| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `--voxel_size` | 0.01 | Kích thước voxel (m), nhỏ hơn = chi tiết hơn |
| `--stride` | 5 | Xử lý mỗi N frame (chỉ dùng cho GT depth) |
| `--keyframe_traj` | path | Trajectory để map keyframe index → frame ID |

## Step 4: Geometric Evaluation (Accuracy, Completeness, Chamfer)

### Full Room Mesh Evaluation

Đánh giá mesh toàn phòng so với GT mesh:

```bash
cd /media/tam/DATA/3D/CG-photo/neural_slam_eval-main
source ../scripts/tsdf_env/bin/activate

python eval_recon.py \
    --rec_mesh ../meshes/office0_rendered_full.ply \
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
| **Training** | `bin/replica_rgbd` or `scripts/replica_rgbd_pa6.sh` |
| **Loss Config** | `cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml` |
| **Photometric Eval** | `Photo-SLAM-eval/run.py` |
| **Mesh Generation** | `scripts/batch_compare_keyframes.py` |
| **Geometric Eval** | `scripts/batch_eval_keyframes.py` + `neural_slam_eval-main/eval_recon.py` |
