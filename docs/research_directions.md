# Hướng Phát Triển Monocular Gaussian SLAM

## Mục tiêu
Cải thiện Geometric Accuracy cho Monocular mode trong Photo-SLAM mà vẫn giữ **real-time online** capability.

## Vấn đề hiện tại
- Mono mode: Accuracy ~60-80cm (rất kém)
- RGB-D mode: Accuracy ~2-3cm (tốt)
- Nguyên nhân: Không có Ground Truth depth

---

## Hướng 1: Multi-View Stereo từ Keyframes

### Ý tưởng
ORB-SLAM đã track được keyframes với accurate poses → Dùng triangulation để estimate depth từ multiple views.

### Cách thực hiện
1. Với mỗi keyframe mới, lấy N keyframes nearby có sufficient baseline
2. Match features hoặc dense patches giữa các views
3. Triangulate để tính depth
4. Dùng depth này làm supervision cho Gaussians

### Ưu điểm
- Không cần neural network
- Real-time friendly (chỉ tính trên keyframes)
- Đã có poses từ ORB-SLAM

### Paper tham khảo
| Paper | Venue | Link | Mô tả |
|-------|-------|------|-------|
| DSO: Direct Sparse Odometry | PAMI 2018 | https://arxiv.org/abs/1607.02565 | Photometric BA với sparse depth |
| LDSO | IROS 2018 | https://arxiv.org/abs/1808.01111 | Loop closing cho DSO |
| DTAM | ICCV 2011 | https://ieeexplore.ieee.org/document/6126513 | Dense tracking and mapping |
| MGS-SLAM | arXiv 2024 | https://arxiv.org/abs/2407.19813 | MVS depth cho Gaussian SLAM |

---

## Hướng 2: Epipolar Depth Consistency

### Ý tưởng
Depth của 1 point phải consistent khi nhìn từ nhiều views khác nhau theo epipolar geometry.

### Loss function
```
L_epipolar = |depth_view1 - triangulated_depth_from_view2|
```

### Cách thực hiện
1. Với mỗi Gaussian, render depth từ current view
2. Project tới reference view theo epipolar line
3. Tính expected depth dựa trên fundamental matrix
4. Penalize difference

### Paper tham khảo
| Paper | Venue | Link | Mô tả |
|-------|-------|------|-------|
| MonoGS | CVPR 2024 | https://arxiv.org/abs/2312.00948 | Geometric verification cho Gaussian SLAM |
| Structure from Motion Revisited | CVPR 2016 | https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf | COLMAP SfM |

---

## Hướng 3: Depth Consistency qua Gaussian Observation

### Ý tưởng
Mỗi Gaussian được observe từ nhiều views → depth estimates phải consistent. Gaussians với high variance depth → unreliable → prune hoặc penalize.

### Metrics
- View count: Số views đã observe Gaussian
- Depth variance: Variance của depth từ các views
- Confidence = view_count / variance

### Loss function
```
L_consensus = Σ variance(depth_estimates) / observation_count
```

### Paper tham khảo
| Paper | Venue | Link | Mô tả |
|-------|-------|------|-------|
| NICE-SLAM | CVPR 2022 | https://arxiv.org/abs/2112.12130 | Neural implicit với uncertainty |
| Point-SLAM | CVPR 2023 | https://arxiv.org/abs/2304.04278 | Point-based neural SLAM |

---

## Hướng 4: Planar Regularization (đã implement)

### Ý tưởng
Indoor environments có nhiều planar surfaces (walls, floors, tables). Ràng buộc Gaussians flat và align với surfaces.

### Loss functions đã có
- `lambda_iso`: Isotropic Gaussians (chống needle)
- `lambda_align`: Normal alignment với depth gradient
- `lambda_reg`: Planar regularization (flatten z-scale)
- `lambda_g1`, `lambda_g2`: Gradient consistency

### Paper tham khảo
| Paper | Venue | Link | Mô tả |
|-------|-------|------|-------|
| 2D-3DGS | arXiv 2024 | https://arxiv.org/abs/2403.16927 | 2D Gaussian splatting |
| SuGaR | CVPR 2024 | https://arxiv.org/abs/2311.12775 | Surface Gaussians |
| MonoGS++ | BMVC 2024 | https://arxiv.org/abs/2404.16085 | Enhanced MonoGS |

---

## Hướng 5: Photometric Bundle Adjustment

### Ý tưởng
Hiện tại ORB-SLAM tracking → fixed poses → Gaussian mapping.
Cải tiến: Jointly optimize poses + Gaussians dựa trên photometric error.

### Cách thực hiện
1. Sau ORB-SLAM tracking, có initial pose
2. Render Gaussians từ pose
3. Compute photometric error với observation
4. Backprop gradient tới cả pose lẫn Gaussians

### Độ khó
⭐⭐⭐ (High) - Cần modify cả tracking pipeline

### Paper tham khảo
| Paper | Venue | Link | Mô tả |
|-------|-------|------|-------|
| DSO | PAMI 2018 | https://arxiv.org/abs/1607.02565 | Direct photometric optimization |
| SplaTAM | CVPR 2024 | https://arxiv.org/abs/2312.02126 | Joint tracking + mapping với 3DGS |
| Gaussian-SLAM | CVPR 2024 | https://arxiv.org/abs/2312.06741 | Dense Gaussian SLAM |

---

## Hướng 6: Scene Priors (Indoor)

### Ý tưởng
Indoor scenes có structural priors:
- Floor/ceiling planes
- Wall orientations (Manhattan world)
- Object sizes (doors ~2m, tables ~0.75m)

### Cách thực hiện
1. Detect floor plane từ Gaussians (RANSAC)
2. Constrain camera height relative to floor
3. Use floor normal để fix global orientation
4. Optionally detect walls (perpendicular to floor)

### Paper tham khảo
| Paper | Venue | Link | Mô tả |
|-------|-------|------|-------|
| Manhattan SLAM | CVPR 2021 | https://arxiv.org/abs/2103.15068 | Manhattan world assumptions |
| PlaneRCNN | CVPR 2019 | https://arxiv.org/abs/1812.04072 | Plane detection |

---

## Hướng 7: Lightweight Depth Prior (Nếu chấp nhận neural network)

### Ý tưởng
Nếu chấp nhận ~30-50ms latency, có thể dùng lightweight depth models.

### Models nhanh nhất
| Model | Resolution | Speed | Accuracy |
|-------|------------|-------|----------|
| MiDaS Small | 384×384 | ~50 FPS | Trung bình |
| Depth Anything Small | 518×518 | ~35 FPS | Tốt |
| FastDepth | 224×224 | ~100 FPS | Kém |

### Cách dùng để maintain novelty
- **Confidence-weighted loss**: Dùng uncertainty từ depth model
- **Scale alignment**: Train small network để align scale
- **Async processing**: Chạy depth estimation background thread

### Paper tham khảo
| Paper | Venue | Link | Mô tả |
|-------|-------|------|-------|
| Depth Anything V2 | arXiv 2024 | https://arxiv.org/abs/2406.09414 | Foundation depth model |
| Metric3D V2 | ICCV 2023 | https://arxiv.org/abs/2307.10984 | Metric depth estimation |
| DepthSplat | ECCV 2024 | https://arxiv.org/abs/2408.16789 | Depth + Gaussian splatting |

---

## Roadmap đề xuất

### Phase 1: Geometry-only (Hiện tại)
- [x] Implement các loss functions (iso, align, g1, g2, reg)
- [x] Ablation study trên Replica Mono
- [ ] Chọn best loss configuration

### Phase 2: Multi-View Stereo
- [ ] Implement sparse MVS từ ORB-SLAM keyframes
- [ ] Thêm loss supervision từ MVS depth
- [ ] Evaluate improvement

### Phase 3: Depth Consistency
- [ ] Implement epipolar consistency loss
- [ ] Implement Gaussian observation consistency
- [ ] Ablation study

### Phase 4: Paper Preparation
- [ ] Implement best methods combination
- [ ] Full evaluation (TUM, Replica, EuRoC)
- [ ] Comparison với baselines (MonoGS, Photo-SLAM, ORB-SLAM3)
- [ ] Write IROS paper

---

## Target conferences

| Conference | Deadline | Focus |
|------------|----------|-------|
| **IROS 2025** | ~March 2025 | Robotics, real-time systems |
| **ICRA 2025** | ~September 2024 (passed) | Robotics |
| **3DV 2025** | ~TBD | 3D vision |
| **CVPR 2025** | ~November 2024 | Computer vision (very competitive) |

---

## Kết luận

Với mục tiêu IROS A*, hướng phát triển tốt nhất là:
1. **Multi-View Stereo** (không neural network, real-time)
2. **Epipolar consistency** (novel contribution)
3. **Kết hợp với existing loss functions**

Contribution có thể claim:
- "First systematic study of geometric regularization for Monocular Gaussian SLAM"
- "Novel multi-view depth consistency for real-time Gaussian mapping"
- "Comprehensive ablation on loss functions for 3DGS SLAM"
