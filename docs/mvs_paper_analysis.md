# Phân Tích Papers MVS cho Monocular Depth Improvement

## Tổng quan
Tài liệu này tổng hợp các ý tưởng từ 3 papers quan trọng có thể áp dụng để cải thiện depth estimation trong Monocular Gaussian SLAM.

---

## Paper 1: DTAM - Dense Tracking and Mapping in Real-Time (ICCV 2011)

### Thông tin Paper
- **Tác giả:** Richard A. Newcombe, Steven J. Lovegrove, Andrew J. Davison
- **Venue:** ICCV 2011
- **Link:** https://ieeexplore.ieee.org/document/6126513

### Core Ideas

#### 1. Cost Volume cho Dense Depth Estimation
DTAM xây dựng **cost volume** cho mỗi keyframe bằng cách tính photometric error tại nhiều mức depth khác nhau.

```
Cost(u, ξ) = Σ ρ(I_r(u) - I_m(π(K T π^{-1}(u, ξ))))
```

Trong đó:
- `u`: pixel coordinates
- `ξ`: inverse depth
- `I_r, I_m`: reference và measurement images
- `ρ`: robust cost function

**Ý tưởng áp dụng:**
- Với mỗi Gaussian, xây dựng mini cost volume từ nearby keyframes
- Tìm optimal depth minimize photometric error

#### 2. Total Variation Regularization
DTAM sử dụng **TV regularization** để smooth depth map:

```
L_total = L_data + λ * L_TV
L_TV = Σ |∇ξ(u)|  (L1 norm của gradient)
```

**Đặc điểm:**
- Preserve edges (không blur qua object boundaries)
- Smooth trong regions có similar color
- Real-time với GPU implementation

**Ý tưởng áp dụng:**
- Thêm `lambda_tv` loss cho rendered depth
- Regularize depth gradients của nearby Gaussians

#### 3. Whole Image Alignment cho Tracking
Thay vì feature matching, DTAM dùng **direct photometric alignment**:

```
T_lv = argmin Σ (I_l(π(K T π^{-1}(u, ξ(u)))) - I_v(u))²
```

**Ưu điểm:**
- Accurate hơn feature-based (subpixel)
- Robust với motion blur

#### 4. Keyframe Selection Strategy
Keyframe mới được thêm khi số pixels không có visible surface vượt threshold.

**Ý tưởng áp dụng:**
- Adaptive keyframe selection cho Gaussian mapping
- Trigger new keyframe based on coverage

---

## Paper 2: SVO - Semi-Direct Visual Odometry (ICRA 2014)

### Thông tin Paper
- **Tác giả:** Christian Forster, Matia Pizzoli, Davide Scaramuzza
- **Venue:** ICRA 2014, TRO 2017
- **Link:** https://arxiv.org/abs/1408.2821

### Core Ideas

#### 1. Semi-Direct Approach
Kết hợp feature-based và direct methods:
- **Direct motion estimation** từ image alignment
- **Feature-based refinement** cho bundle adjustment

**Pipeline:**
```
1. Sparse Model-based Image Alignment → rough pose
2. Feature Alignment → refine 2D correspondences
3. Pose & Structure Refinement → BA optimization
```

#### 2. Probabilistic Depth Filter
SVO sử dụng **Bayesian depth filter** để estimate depth từ multiple views:

```
p(ξ | z_1, ..., z_n) ∝ p(ξ) Π p(z_i | ξ)
```

**Chi tiết:**
- Mỗi feature có depth distribution (Gaussian + Uniform)
- Update depth belief với mỗi observation mới
- Chỉ insert point vào map khi depth **converged**

**Công thức update:**
```
Initial: p(ξ) = Uniform(ξ_min, ξ_max)
Update:  p(ξ|z) = Gaussian(μ, σ²) * (1-π) + Uniform * π
```

Với:
- `π`: outlier probability
- `μ, σ`: mean và variance của depth estimate
- Convergence khi `σ < threshold`

**Ý tưởng áp dụng:**
- Mỗi Gaussian có depth uncertainty
- Update uncertainty với mỗi observation
- Prune Gaussians với high uncertainty

#### 3. REMODE - Regularized Monocular Depth Estimation
Extension của SVO depth filter với:
- Multi-resolution depth estimation
- Spatial regularization
- Dense depth output

**Ý tưởng áp dụng:**
- Multi-scale depth consistency
- Regularize neighboring Gaussians

#### 4. Sparse-to-Dense Strategy
```
Sparse features → Depth filters → Converged points → Dense reconstruction
```

**Ưu điểm:**
- Few outliers (chỉ insert khi converged)
- Reliable tracking (nhiều features)

---

## Paper 3: Multi-Scale Geometric Consistency Guided MVS (CVPR 2019)

### Thông tin Paper
- **Tác giả:** Qingshan Xu, Wenbing Tao
- **Venue:** CVPR 2019
- **Link:** https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Multi-Scale_Geometric_Consistency_Guided_Multi-View_Stereo_CVPR_2019_paper.pdf

### Core Ideas

#### 1. PatchMatch MVS với Hypothesis Propagation
Dựa trên Gipuma/COLMAP approach:
- Random initialization
- Spatial propagation của depth hypothesis
- Refinement step

**Key insight:** Good hypotheses propagate to neighbors → spatial consistency

#### 2. Adaptive Checkerboard Sampling (ACMH)
Giải quyết vấn đề dependency trong parallel propagation:
- Black-red checkerboard pattern
- 8-neighbor propagation

#### 3. Multi-Scale Geometric Consistency (ACMM)

**Core contribution:** Sử dụng geometric consistency ở multiple scales

```
L_total = L_photometric + λ * L_geometric
```

**Geometric Consistency:**
```
Cho pixel p trong view i, depth d_i(p):
1. Project p tới view j: p' = π(K_j T_ji π^{-1}(p, d_i(p)))
2. Lấy depth d_j(p') từ view j
3. Reproject về view i: p'' = π(K_i T_ij π^{-1}(p', d_j(p')))
4. Check: |p - p''| < threshold
         |d_i(p) - d_i(p'')| < threshold
```

**Multi-scale:**
1. Compute depth ở coarse resolution
2. Propagate tới finer resolution
3. Geometric consistency checks ở each scale
4. Detail restorer để restore fine details

**Ý tưởng áp dụng:**
- Gaussian depth phải consistent khi reproject qua views
- Multi-scale: coarse Gaussians guide fine Gaussians

#### 4. Pixelwise View Selection
Chọn source views tốt nhất cho mỗi pixel dựa trên:
- Baseline magnitude
- Viewing angle
- Photometric quality

**Ý tưởng áp dụng:**
- Selective view aggregation cho depth estimation
- Weight views theo reliability

#### 5. Low-Texture Region Handling
Vấn đề: Photometric matching fail trong low-texture areas

**Giải pháp:**
- Geometric consistency quan trọng hơn ở low-texture
- Propagate từ textured regions
- Spatial regularization

---

## Paper 4: REMODE - Regularized Monocular Depth Estimation (ICRA 2014)

### Thông tin Paper
- **Tác giả:** Matia Pizzoli, Christian Forster, Davide Scaramuzza
- **Venue:** ICRA 2014
- **Link:** https://rpg.ifi.uzh.ch/docs/ICRA14_Pizzoli.pdf

### Tại sao REMODE phù hợp nhất cho Photo-SLAM?

| Tiêu chí | **REMODE** | **ACMM** | **DTAM** | **SVO** |
|----------|------------|----------|----------|---------|
| **Real-time** | ✅ Yes (GPU) | ❌ Offline | ✅ Yes (GPU) | ✅ Yes |
| **Online/Sequential** | ✅ Yes | ❌ Batch | ✅ Yes | ✅ Yes |
| **Depth Uncertainty** | ✅ Bayesian | ❌ No | ❌ No | ✅ Bayesian |
| **Multi-view** | ✅ Sequential | ✅ All views | ✅ Keyframes | ✅ Sequential |
| **Regularization** | ✅ Huber TV | ✅ Spatial | ✅ TV | ❌ No |
| **Code complexity** | ⭐⭐ Medium | ⭐⭐⭐ High | ⭐⭐⭐ High | ⭐ Low |
| **Fit với Photo-SLAM** | ⭐⭐⭐ Tốt nhất | ⭐⭐ OK | ⭐⭐ OK | ⭐⭐ OK |

### Core Ideas

#### 1. Bayesian Depth Estimation với Mixture Model

REMODE model depth measurement như Gaussian + Uniform mixture:

```
p(d_k | d̂, ρ) = ρ · N(d_k | d̂, τ_k²) + (1-ρ) · U(d_k | d_min, d_max)
```

Trong đó:
- `d_k`: measured depth từ view k
- `d̂`: true depth
- `ρ`: inlier probability
- `τ_k²`: measurement variance
- `U`: uniform distribution cho outliers

**Posterior update (Eq. 3, 5):**
```
p(d̂, ρ | d_{r+1}, ..., d_k) ∝ p(d̂, ρ) · Π_k p(d_k | d̂, ρ)
```

Approximate bằng Beta × Gaussian:
```
q(d̂, ρ | a_k, b_k, μ_k, σ_k²) = Beta(ρ | a_k, b_k) · N(d̂ | μ_k, σ_k²)
```

**Ý tưởng áp dụng:**
- Mỗi Gaussian có `(μ, σ², a, b)` để track depth belief
- Update sequential khi có observation mới
- Converge khi `σ² < threshold` và `a/(a+b) > 0.5`

#### 2. Huber TV Regularization (Eq. 6, 7)

```
min_F ∫_Ω { G(u) · ||∇F(u)||_Huber + λ · ||F(u) - D(u)||₁ } du
```

**Huber norm (preserve edges):**
```
||∇F||_Huber = 
    ||∇F||₂² / (2ε)     if ||∇F||₂ ≤ ε
    ||∇F||₁ - ε/2       otherwise
```

**Đặc điểm:**
- L2 cho small gradients → smooth
- L1 cho large gradients → preserve edges
- Tốt hơn pure TV (L1) vì không tạo staircasing

**Ý tưởng áp dụng:**
- Thay TV loss bằng Huber TV loss
- Preserve depth discontinuities tại object boundaries

#### 3. Confidence-Weighted Regularization (Eq. 8)

```
G(u) = E[ρ] · (σ²(u) / σ²_max) + (1 - E[ρ])
```

Trong đó `E[ρ] = a / (a + b)` là expected inlier ratio.

**Logic:**
- High confidence (`E[ρ]` cao, `σ²` thấp) → weight = small → **ít regularization**
- Low confidence (`E[ρ]` thấp, `σ²` cao) → weight = large → **nhiều regularization**

**Ý tưởng áp dụng:**
- Gaussians với high uncertainty → bị regularized mạnh hơn
- Gaussians với converged depth → giữ nguyên

#### 4. Depth Measurement từ Triangulation

Với reference view `r` và current view `k`:
1. Match feature/patch tại pixel `u` trong view `r` với view `k`
2. Estimate correspondence `u'` trong view `k`
3. Triangulate sử dụng known poses `T_r` và `T_k`
4. Compute measurement variance `τ_k²` từ geometric uncertainty

**Công thức variance (phụ thuộc baseline và angle):**
```
τ² ∝ 1 / (baseline² · sin²(viewing_angle))
```

→ Large baseline + good viewing angle = low variance

### Implementation chi tiết cho Photo-SLAM

#### Depth Filter Structure
```cpp
struct REMODEDepthFilter {
    float mu;           // mean depth
    float sigma2;       // depth variance
    float a, b;         // Beta parameters for inlier ratio
    int num_observations;
    
    // Initialize with uniform prior
    void init(float d_min, float d_max) {
        mu = (d_min + d_max) / 2.0f;
        sigma2 = (d_max - d_min) * (d_max - d_min) / 12.0f;  // Uniform variance
        a = 10.0f;  // Initial Beta params
        b = 10.0f;  // E[ρ] = 0.5
        num_observations = 0;
    }
    
    // Update với measurement mới
    void update(float measurement, float tau2) {
        // Từ Vogiatzis et al. ISMAR 2011 (cited trong REMODE)
        float sigma2_new = 1.0f / (1.0f / sigma2 + 1.0f / tau2);
        float mu_new = sigma2_new * (mu / sigma2 + measurement / tau2);
        
        // Update Beta parameters
        float c1 = compute_c1(measurement, mu, sigma2, tau2);
        float c2 = compute_c2();  // Uniform probability
        
        float p_inlier = c1 * a / (a + b);
        float p_outlier = c2 * b / (a + b);
        float p_total = p_inlier + p_outlier;
        
        float f = p_inlier / p_total;  // Posterior inlier probability
        
        a = f * (a + 1) + (1 - f) * a;
        b = f * b + (1 - f) * (b + 1);
        
        mu = f * mu_new + (1 - f) * mu;
        sigma2 = f * sigma2_new + (1 - f) * sigma2;
        
        num_observations++;
    }
    
    // Check convergence
    bool is_converged(float sigma2_thresh = 0.01f, float inlier_thresh = 0.7f) {
        float expected_inlier = a / (a + b);
        return sigma2 < sigma2_thresh && expected_inlier > inlier_thresh;
    }
    
    // Get confidence weight for regularization
    float get_regularization_weight(float sigma2_max) {
        float expected_inlier = a / (a + b);
        return expected_inlier * (sigma2 / sigma2_max) + (1.0f - expected_inlier);
    }
    
private:
    float compute_c1(float z, float mu, float sigma2, float tau2) {
        float diff = z - mu;
        return exp(-0.5f * diff * diff / (sigma2 + tau2)) / sqrt(2 * M_PI * (sigma2 + tau2));
    }
    
    float compute_c2() {
        // 1 / (d_max - d_min)
        return 1.0f / depth_range;
    }
};
```

#### Huber TV Loss
```cpp
torch::Tensor huber_tv_loss(torch::Tensor depth, float epsilon = 0.01f) {
    auto dx = depth.slice(2, 1) - depth.slice(2, 0, -1);
    auto dy = depth.slice(1, 1) - depth.slice(1, 0, -1);
    
    // Huber norm
    auto huber = [epsilon](torch::Tensor x) {
        auto norm = x.abs();
        auto mask_l2 = norm <= epsilon;
        auto result = torch::where(
            mask_l2,
            norm.pow(2) / (2 * epsilon),
            norm - epsilon / 2
        );
        return result;
    };
    
    return huber(dx).mean() + huber(dy).mean();
}
```

#### Confidence-Weighted Loss
```cpp
torch::Tensor confidence_weighted_loss(
    torch::Tensor depth_error,
    torch::Tensor confidence_weights  // từ get_regularization_weight()
) {
    // Low confidence → high weight regularization
    // High confidence → low weight (trust the data)
    return (depth_error * (1.0f / (confidence_weights + 1e-6f))).mean();
}
```

---

## 📊 So sánh tổng hợp: Approach nào chọn?

| Approach | Ưu điểm | Nhược điểm | Khuyến nghị |
|----------|---------|------------|-------------|
| **REMODE** | Online, Bayesian uncertainty, Edge-preserving regularization | Cần implement depth filter | ⭐⭐⭐ **Ưu tiên cao nhất** |
| **ACMM** | Geometric consistency tốt | Offline, Complex | ⭐⭐ Dùng ý tưởng consistency |
| **DTAM** | Cost volume concept | Heavy computation | ⭐ Tham khảo TV |
| **SVO** | Simple depth filter | Không có regularization | ⭐ Base knowledge |

### Kết luận

**REMODE là approach phù hợp nhất** vì:
1. **Online sequential** - match với Photo-SLAM pipeline
2. **Probabilistic** - track uncertainty, prune unreliable Gaussians
3. **Huber TV** - smooth nhưng preserve edges
4. **Confidence-weighted** - adaptive regularization

---

## Tổng Hợp Ý Tưởng Áp Dụng cho Photo-SLAM

### 1. Probabilistic Depth Estimation (từ SVO)

**Implementation:**
```cpp
struct GaussianDepthFilter {
    float mu;           // mean depth
    float sigma;        // uncertainty
    int observations;   // observation count
    float outlier_prob; // π
    
    void update(float measured_depth, float measurement_sigma) {
        // Bayesian update
        // Similar to SVO depth filter
    }
    
    bool is_converged() {
        return sigma < threshold && observations > min_obs;
    }
};
```

**Loss:**
```
L_depth_uncertainty = Σ (σ_i / σ_max)  // Penalize high uncertainty
```

### 2. Cost Volume Integration (từ DTAM)

**Implementation:**
```cpp
// Cho mỗi Gaussian với uncertain depth:
float compute_depth_cost(Gaussian& g, std::vector<Keyframe>& nearby_kfs) {
    float cost = 0;
    for (int d_idx = 0; d_idx < num_depths; d_idx++) {
        float depth = depth_min + d_idx * depth_step;
        for (auto& kf : nearby_kfs) {
            // Project và compute photometric error
            cost += photometric_error(g, kf, depth);
        }
    }
    return argmin(cost);  // Best depth
}
```

### 3. Geometric Consistency Loss (từ ACMM)

**Implementation:**
```cpp
float geometric_consistency_loss(Gaussian& g, Keyframe& ref, Keyframe& src) {
    // 1. Render depth từ ref view
    float d_ref = render_depth(g, ref);
    
    // 2. Project tới src view
    Point2D p_src = project(g.position, src);
    float d_src = render_depth_at(p_src, src);
    
    // 3. Reproject về ref view
    Point3D p_3d_src = unproject(p_src, d_src, src);
    Point2D p_reproj = project(p_3d_src, ref);
    float d_reproj = p_3d_src.z;
    
    // 4. Compute consistency
    float reproj_error = distance(p_ref, p_reproj);
    float depth_error = abs(d_ref - d_reproj);
    
    return reproj_error + lambda * depth_error;
}
```

### 4. Multi-Scale Approach (từ ACMM)

**Implementation:**
```cpp
// Coarse-to-fine Gaussian optimization
void multi_scale_optimization() {
    // Scale 1: Large Gaussians (rough depth)
    optimize_gaussians(scale=4, iterations=100);
    
    // Scale 2: Medium Gaussians
    propagate_depth_from_neighbors();
    optimize_gaussians(scale=2, iterations=50);
    
    // Scale 3: Fine Gaussians
    propagate_depth_from_neighbors();
    optimize_gaussians(scale=1, iterations=50);
}
```

### 5. TV Regularization (từ DTAM)

**Implementation:**
```cpp
// Total Variation loss cho rendered depth
torch::Tensor tv_loss(torch::Tensor depth) {
    auto dx = depth.slice(2, 1) - depth.slice(2, 0, -1);
    auto dy = depth.slice(1, 1) - depth.slice(1, 0, -1);
    return dx.abs().mean() + dy.abs().mean();
}
```

---

## Proposed New Loss Functions

Dựa trên phân tích papers, đề xuất các loss mới:

### 1. L_geo_consistency - Geometric Consistency Loss
```
L_geo_consistency = Σ_i Σ_j |depth_i - reproject(depth_j)|
```

### 2. L_tv - Total Variation Loss
```
L_tv = |∇_x(depth)| + |∇_y(depth)|
```

### 3. L_depth_uncertainty - Uncertainty Regularization
```
L_depth_uncertainty = Σ σ_i / count_i
```

### 4. L_multi_view - Multi-View Photometric Loss
```
L_multi_view = Σ_views photometric_error(view_i)
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (1 tuần)
- [ ] Add TV regularization loss (`lambda_tv`)
- [ ] Implement depth variance tracking per Gaussian

### Phase 2: Depth Filter (2 tuần)
- [ ] Implement SVO-style depth filter
- [ ] Track observation count và uncertainty
- [ ] Prune Gaussians với high uncertainty

### Phase 3: Geometric Consistency (2 tuần)
- [ ] Implement multi-view geometric consistency check
- [ ] Add `L_geo_consistency` loss
- [ ] Pixelwise view selection

### Phase 4: Multi-Scale (2 tuần)
- [ ] Coarse-to-fine optimization
- [ ] Depth propagation between scales
- [ ] Detail restoration

---

## Comparison với Existing Methods

| Feature | Photo-SLAM | + DTAM ideas | + SVO ideas | + ACMM ideas |
|---------|------------|--------------|-------------|--------------|
| Depth source | ORB-SLAM | Cost volume | Depth filter | Geometric consistency |
| Regularization | ISO, Align | TV | Uncertainty | Multi-scale |
| View aggregation | Single | Multi-view | Probabilistic | Pixelwise selection |
| Low-texture handling | Poor | Better | Better | Best |

---

## References

1. **DTAM:** Newcombe, R. A., Lovegrove, S. J., & Davison, A. J. (2011). DTAM: Dense tracking and mapping in real-time. ICCV.

2. **SVO:** Forster, C., Pizzoli, M., & Scaramuzza, D. (2014). SVO: Fast semi-direct monocular visual odometry. ICRA.

3. **REMODE:** Pizzoli, M., Forster, C., & Scaramuzza, D. (2014). REMODE: Probabilistic, monocular dense reconstruction in real time. ICRA.

4. **ACMM:** Xu, Q., & Tao, W. (2019). Multi-scale geometric consistency guided multi-view stereo. CVPR.

5. **COLMAP:** Schönberger, J. L., et al. (2016). Pixelwise view selection for unstructured multi-view stereo. ECCV.

6. **DSO:** Engel, J., Koltun, V., & Cremers, D. (2018). Direct sparse odometry. PAMI.

---

## Next Steps

1. **Đọc thêm:** REMODE paper (extension của SVO depth filter)
2. **Implement:** Bắt đầu với TV loss (đơn giản nhất)
3. **Test:** So sánh với baseline trên Replica Mono
4. **Iterate:** Thêm các features phức tạp hơn dần dần
