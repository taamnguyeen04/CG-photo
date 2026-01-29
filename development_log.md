# Depth-Photo-SLAM Development Log

> File này ghi lại quá trình thay đổi code sau mỗi bước phát triển.
> Mục tiêu: Tích hợp mô hình Depth Uncertainty từ CG-SLAM vào Photo-SLAM.

---

## Mục tiêu dự án

**Vấn đề với Photo-SLAM gốc:**
- Có thể tạo ra "floaters" (Gaussian lơ lửng trong không khí)
- Có thể tạo ra "needles" (Gaussian dài như cây kim)
- Không biết Gaussian nào đáng tin cậy

**Giải pháp Depth-Photo-SLAM:**
- Thêm mô hình **Depth Uncertainty** để đánh giá độ tin cậy mỗi Gaussian
- Thêm **Loss Functions** để loại bỏ floaters và needles
- Thêm **Pruning** để xóa Gaussian không đáng tin

---

## [2026-01-10 21:16] Step 1.1: Tạo depth_uncertainty.h

### Mục đích
Tạo file header mới định nghĩa các cấu trúc dữ liệu cho hệ thống uncertainty.

### File được tạo
📄 **`include/depth_uncertainty.h`** (MỚI)

### Nội dung chi tiết

#### 1. UncertaintyConfig - Cấu hình hệ thống
```cpp
struct UncertaintyConfig {
    // Ngưỡng để cắt tỉa
    float uncertainty_threshold = 0.1f;   // v_i > 0.1 → không ổn định
    float fisher_prune_threshold = 0.001f; // FIM < 0.001 → vô dụng
    int prune_interval = 100;              // Cắt tỉa mỗi 100 iterations
    
    // Trọng số cho các loss functions
    float alignment_lambda = 0.1f;   // L_align: chống floaters
    float isotropy_lambda = 0.01f;   // L_iso: chống needles
    float variance_lambda = 0.001f;  // L_var: giảm uncertainty
    float depth_lambda = 1.0f;       // L_geo: depth từ sensor
    
    // Tham số khác
    float isotropy_epsilon = 1.5f;   // Cho phép dài tối đa 1.5 lần
    int max_gaussians = 2000000;     // Giới hạn số lượng max
};
```

#### 2. GaussianUncertainty - Lưu trữ uncertainty cho mỗi Gaussian
```cpp
struct GaussianUncertainty {
    torch::Tensor depth_uncertainty;   // [N] - độ bất định v_i
    torch::Tensor fisher_info_score;   // [N] - điểm Fisher Information
    torch::Tensor observation_count;   // [N] - số lần quan sát
    torch::Tensor is_stable;           // [N] - Gaussian có ổn định?
    torch::Tensor creation_timestamp;  // [N] - tạo ở iteration nào
    
    // Các phương thức:
    void initialize(int64_t num_gaussians, ...);  // Khởi tạo
    void extend(int64_t new_count, ...);          // Thêm Gaussian mới
    void prune(const torch::Tensor& mask);        // Xóa theo mask
    void updateStability(float threshold);        // Cập nhật is_stable
};
```

#### 3. DepthRenderOutput - Kết quả render depth (dùng sau này)
```cpp
struct DepthRenderOutput {
    torch::Tensor alpha_depth;     // [H,W] - depth từ alpha-blending
    torch::Tensor alpha_depth_sq;  // [H,W] - depth^2 cho variance
    torch::Tensor median_depth;    // [H,W] - depth tại T=0.5
    torch::Tensor depth_variance;  // [H,W] - U_pix = E[d²]-E[d]²
    
    void computeVariance();  // Tính variance từ depth và depth_sq
};
```

### Verification
- ✅ Build successful

---

## [2026-01-10 21:22] Step 1.2: Mở rộng GaussianModel

### Mục đích
Thêm uncertainty vào class GaussianModel (class chính quản lý tất cả Gaussians).

### Files được sửa

#### 📄 `include/gaussian_model.h` (SỬA)

**Thêm include:**
```cpp
#include "depth_uncertainty.h"  // THÊM MỚI
```

**Thêm fields (dòng ~183):**
```cpp
// Depth-Photo-SLAM: Uncertainty tracking
depth_uncertainty::GaussianUncertainty uncertainty_;      // Lưu uncertainty
depth_uncertainty::UncertaintyConfig uncertainty_config_; // Cấu hình
```

**Thêm methods (dòng ~147):**
```cpp
// Depth-Photo-SLAM: Uncertainty management methods
void initializeUncertainty();                    // Khởi tạo khi tạo model
void updateUncertainty(tensor, iteration);       // Cập nhật mỗi frame
void uncertaintyPrune(float threshold);          // Xóa Gaussian uncertainty cao
void fisherPrune(float threshold);               // Xóa Gaussian ít đóng góp
GaussianUncertainty& getUncertainty();          // Getter
const UncertaintyConfig& getUncertaintyConfig(); // Getter
void setUncertaintyConfig(config);              // Setter
```

---

#### 📄 `src/gaussian_model.cpp` (SỬA)

**Thêm implementation ở cuối file (dòng ~1133-1220):**

```cpp
// ============================================================================
// Depth-Photo-SLAM: Uncertainty Management Methods
// ============================================================================

void GaussianModel::initializeUncertainty()
{
    int64_t num_gaussians = xyz_.size(0);
    uncertainty_.initialize(num_gaussians, device_type_);
}

void GaussianModel::updateUncertainty(const torch::Tensor& per_gaussian_uncertainty, int iteration)
{
    // Exponential moving average: v_i = 0.9*v_i + 0.1*new
    const float alpha = 0.9f;
    
    // Xử lý khi số lượng Gaussian thay đổi
    if (uncertainty_.depth_uncertainty.size(0) != per_gaussian_uncertainty.size(0)) {
        initializeUncertainty();
    }
    
    // Cập nhật uncertainty
    uncertainty_.depth_uncertainty = alpha * uncertainty_.depth_uncertainty + 
                                     (1.0f - alpha) * per_gaussian_uncertainty;
    uncertainty_.observation_count += 1;
    uncertainty_.updateStability(uncertainty_config_.uncertainty_threshold);
}

void GaussianModel::uncertaintyPrune(float threshold)
{
    // Tạo mask: True = cần xóa
    auto prune_mask = uncertainty_.depth_uncertainty > threshold;
    
    // Giới hạn xóa tối đa 10% mỗi lần
    int num_to_prune = prune_mask.sum().item<int>();
    int max_prune = static_cast<int>(xyz_.size(0) * 0.1f);
    
    if (num_to_prune > max_prune) {
        // Chỉ xóa những cái tệ nhất
        auto [sorted, indices] = torch::sort(uncertainty_.depth_uncertainty, true);
        auto top_indices = indices.slice(0, 0, max_prune);
        prune_mask = torch::zeros_like(prune_mask);
        prune_mask.index_put_({top_indices}, true);
    }
    
    if (prune_mask.any().item<bool>()) {
        prunePoints(prune_mask);           // Xóa từ GaussianModel
        uncertainty_.prune(prune_mask);    // Xóa từ uncertainty tensors
    }
}

void GaussianModel::fisherPrune(float threshold)
{
    // Xóa Gaussian có Fisher Information thấp (không đóng góp vào rendering)
    auto prune_mask = uncertainty_.fisher_info_score < threshold;
    
    // Không xóa Gaussian mới (chưa đủ 50 iterations)
    auto young_mask = uncertainty_.observation_count < 50;
    prune_mask = prune_mask & ~young_mask;
    
    if (prune_mask.any().item<bool>()) {
        prunePoints(prune_mask);
        uncertainty_.prune(prune_mask);
    }
}

// Getters và Setters...
```

### Verification
- ✅ Build successful
- ✅ All executables linked correctly

---

## [2026-01-10 21:24] Step 2.1-2.3: Thêm Loss Functions

### Mục đích
Thêm các hàm tính loss để huấn luyện model loại bỏ floaters và needles.

### File được sửa
📄 **`include/loss_utils.h`** (SỬA)

### Nội dung chi tiết

#### 1. depth_alignment_loss() - Chống Floaters

**Vấn đề:** Floaters xảy ra khi có nhiều Gaussian rải rác dọc theo tia nhìn → depth bị "mờ"

**Giải pháp:** Buộc alpha_depth ≈ median_depth

```cpp
/**
 * L_align = mean(|D_alpha - D_median|)
 * 
 * D_alpha: Depth tính bằng alpha-blending (trung bình có trọng số)
 * D_median: Depth tại điểm transmittance T = 0.5 (điểm "giữa")
 * 
 * Nếu bề mặt rắn chắc → D_alpha ≈ D_median
 * Nếu có floaters → D_alpha ≠ D_median (sương mù)
 */
inline torch::Tensor depth_alignment_loss(
    torch::Tensor& alpha_depth,
    torch::Tensor& median_depth,
    torch::Tensor valid_mask = torch::Tensor())
{
    auto diff = torch::abs(alpha_depth - median_depth);
    
    if (valid_mask.defined() && valid_mask.numel() > 0) {
        diff = diff * valid_mask.to(diff.dtype());
        auto num_valid = valid_mask.sum().clamp_min(1.0f);
        return diff.sum() / num_valid;
    }
    
    return diff.mean();
}
```

---

#### 2. isotropy_loss() - Chống Needles

**Vấn đề:** Gaussian có thể kéo dài vô hạn theo một hướng → tạo artifacts khi xem từ góc khác

**Giải pháp:** Phạt tỷ lệ max_scale/min_scale quá lớn

```cpp
/**
 * L_iso = mean(max(ratio - ε, 0))
 * 
 * ratio = max(sx,sy,sz) / min(sx,sy,sz)
 * ε = 1.5 (cho phép dài tối đa 1.5 lần)
 * 
 * Ví dụ:
 * - Hình cầu: ratio = 1.0 → penalty = 0 ✓
 * - Hình elip: ratio = 1.3 → penalty = 0 ✓
 * - Hình kim: ratio = 10.0 → penalty = 8.5 ✗
 */
inline torch::Tensor isotropy_loss(
    torch::Tensor& scales,
    float epsilon = 1.5f,
    bool use_log_scales = true)
{
    // GaussianModel lưu scales ở dạng log → cần exp()
    auto s = use_log_scales ? torch::exp(scales) : scales;
    
    // Tính ratio
    auto s_max = std::get<0>(torch::max(s, 1));
    auto s_min = std::get<0>(torch::min(s, 1));
    s_min = torch::clamp_min(s_min, 1e-7f);  // Tránh chia 0
    
    auto ratio = s_max / s_min;
    auto penalty = torch::clamp_min(ratio - epsilon, 0.0f);
    
    return penalty.mean();
}
```

---

#### 3. uncertainty_variance_loss() - Giảm Uncertainty

**Mục đích:** Khuyến khích hệ thống tạo ra depth chắc chắn (variance thấp)

```cpp
/**
 * L_var = mean(U_pix)
 * 
 * U_pix = E[d²] - E[d]² = depth_variance
 * 
 * Variance cao → nhiều Gaussian ở các depth khác nhau → không chắc chắn
 * Variance thấp → Gaussian tập trung → bề mặt rõ ràng
 */
inline torch::Tensor uncertainty_variance_loss(
    torch::Tensor& depth_variance,
    torch::Tensor valid_mask = torch::Tensor())
{
    if (valid_mask.defined() && valid_mask.numel() > 0) {
        auto masked = depth_variance * valid_mask.to(depth_variance.dtype());
        auto num_valid = valid_mask.sum().clamp_min(1.0f);
        return masked.sum() / num_valid;
    }
    
    return depth_variance.mean();
}
```

---

#### 4. sensor_depth_loss() - So sánh với Ground Truth

**Mục đích:** Supervision từ depth camera (RGB-D)

```cpp
/**
 * L_geo = mean(|D_rendered - D_sensor|)
 * 
 * D_rendered: Depth render từ Gaussian
 * D_sensor: Depth từ camera (Kinect, RealSense, v.v.)
 */
inline torch::Tensor sensor_depth_loss(
    torch::Tensor& rendered_depth,
    torch::Tensor& sensor_depth,
    torch::Tensor& valid_mask)
{
    auto diff = torch::abs(rendered_depth - sensor_depth);
    auto masked = diff * valid_mask.to(diff.dtype());
    auto num_valid = valid_mask.sum().clamp_min(1.0f);
    return masked.sum() / num_valid;
}
```

### Verification
- ✅ Build successful
- ✅ All executables linked correctly

---

## ⚠️ QUAN TRỌNG: Những gì CHƯA làm

### Phase 3: CUDA Kernels (Chưa làm!)

Các loss functions ở trên CẦN dữ liệu đầu vào:
- `alpha_depth` - CHƯA CÓ
- `median_depth` - CHƯA CÓ
- `depth_variance` - CHƯA CÓ

Những tensor này cần được **render từ CUDA kernel** trong file `cuda_rasterizer/forward.cu`.

**Hiện tại Photo-SLAM chỉ render:**
- ✅ Màu sắc (RGB)
- ❌ Depth
- ❌ Depth variance
- ❌ Median depth

**Cần sửa thêm ở Phase 3:**
```cpp
// Trong forward.cu, vòng lặp render:
// HIỆN TẠI:
C[ch] += color * alpha * T;  // Chỉ tích lũy màu

// CẦN THÊM:
D += depth * alpha * T;      // Tích lũy depth
D_sq += depth*depth * alpha*T; // Tích lũy depth²
if (T > 0.5 && test_T <= 0.5) {
    D_median = depth;        // Ghi nhận median
}
```

---

## Bước tiếp theo

- [ ] **Phase 3**: Sửa `cuda_rasterizer/forward.cu` để output depth tensors
- [ ] **Phase 4**: Cập nhật interface rasterizer
- [ ] **Phase 5**: Tích hợp vào training loop
- [ ] **Phase 6**: Test và đánh giá

---

## [2026-01-10 21:33] Step 3.1-3.3: CUDA Kernels Depth Rendering

### Mục đích
Sửa CUDA kernel để render depth, depth², và median depth trong quá trình alpha-blending.

### Files được sửa

#### 📄 `cuda_rasterizer/forward.h` (SỬA)

Thêm parameters vào hàm `render()`:
```cpp
void render(
    // ... existing params ...
    const float* depths,              // NEW: per-Gaussian depth
    // ... existing params ...
    float* out_depth,                 // NEW: output alpha-blended depth
    float* out_depth_sq,              // NEW: output depth^2 for variance
    float* out_median_depth);         // NEW: output median depth
```

---

#### 📄 `cuda_rasterizer/forward.cu` (SỬA)

**1. Thêm shared memory cho depth:**
```cpp
__shared__ float collected_depths[BLOCK_SIZE];  // NEW
```

**2. Thêm local accumulators:**
```cpp
float D = 0.0f;           // Alpha-blended depth: E[d]
float D_sq = 0.0f;        // Alpha-blended depth²: E[d²]
float D_median = 0.0f;    // Median depth at T=0.5
bool median_found = false;
```

**3. Tích lũy depth trong vòng lặp alpha-blending:**
```cpp
// Fetch depth vào shared memory
collected_depths[block.thread_rank()] = depths[coll_id];

// Trong inner loop:
float gaussian_depth = collected_depths[j];
float weight = alpha * T;
D += gaussian_depth * weight;
D_sq += gaussian_depth * gaussian_depth * weight;

// Track median: T lần đầu giảm xuống dưới 0.5
if (!median_found && T > 0.5f && test_T <= 0.5f) {
    D_median = gaussian_depth;
    median_found = true;
}
```

**4. Write depth outputs:**
```cpp
out_depth[pix_id] = D;
out_depth_sq[pix_id] = D_sq;
out_median_depth[pix_id] = median_found ? D_median : D;
```

---

#### 📄 `cuda_rasterizer/rasterizer_impl.h` (SỬA)

Thêm depth buffers vào `ImageState`:
```cpp
struct ImageState {
    // ... existing ...
    float* out_depth;        // Depth-Photo-SLAM
    float* out_depth_sq;     // Depth-Photo-SLAM
    float* out_median_depth; // Depth-Photo-SLAM
};
```

---

#### 📄 `cuda_rasterizer/rasterizer_impl.cu` (SỬA)

**1. Allocate depth buffers trong `ImageState::fromChunk()`:**
```cpp
obtain(chunk, img.out_depth, N, 128);
obtain(chunk, img.out_depth_sq, N, 128);
obtain(chunk, img.out_median_depth, N, 128);
```

**2. Gọi `FORWARD::render()` với depth buffers:**
```cpp
FORWARD::render(
    ...,
    geomState.depths,           // per-Gaussian depths
    ...,
    imgState.out_depth,         // output alpha-blended depth
    imgState.out_depth_sq,      // output depth²
    imgState.out_median_depth); // output median depth
```

### Cách tính Depth Variance (sau khi render)

Sau khi render, có thể tính uncertainty như sau:
```cpp
// U_pix = E[d²] - E[d]²
depth_variance = out_depth_sq - out_depth * out_depth;
```

### Verification Checklist
- [x] forward.h: Thêm 4 parameters mới
- [x] forward.cu: Sửa kernel renderCUDA
- [x] rasterizer_impl.h: Thêm depth buffers vào ImageState
- [x] rasterizer_impl.cu: Allocate và pass depth buffers
- [x] Build successful với nvcc và g++

### Tiếp theo
- Phase 4: Expose depth outputs qua rasterize_points interface
- Phase 5: Tích hợp loss functions vào training loop

---

## [2026-01-10 21:39] Step 4.1-4.3: Rasterizer Interface

### Mục đích
Expose depth outputs từ CUDA kernel ra interface C++/PyTorch.

### Files được sửa

#### 📄 `cuda_rasterizer/rasterizer.h` (SỬA)
Thêm 3 depth output params vào `Rasterizer::forward()`:
```cpp
static int forward(
    ...,
    float* out_depth = nullptr,
    float* out_depth_sq = nullptr,
    float* out_median_depth = nullptr,
    int* radii = nullptr);
```

#### 📄 `cuda_rasterizer/rasterizer_impl.cu` (SỬA)
- Cập nhật function signature
- Thêm `cudaMemcpy` để copy depth từ `imgState` ra user buffers:
```cpp
if (out_depth != nullptr) {
    cudaMemcpy(out_depth, imgState.out_depth, img_size, cudaMemcpyDeviceToDevice);
}
```

#### 📄 `include/rasterize_points.h` (SỬA)
Return type thay đổi từ 6 → 9 tensors:
```cpp
std::tuple<int, torch::Tensor, ..., torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(...);
// Returns: rendered, color, radii, geom, binning, img, depth, depth_sq, median_depth
```

#### 📄 `src/rasterize_points.cu` (SỬA)
- Tạo depth output tensors: `out_depth`, `out_depth_sq`, `out_median_depth`
- Pass vào `Rasterizer::forward()`
- Return 9 tensors

#### 📄 `src/gaussian_rasterizer.cpp` (SỬA)
Extract 9 values từ `RasterizeGaussiansCUDA()`:
```cpp
auto out_depth = std::get<6>(rasterization_result);
auto out_depth_sq = std::get<7>(rasterization_result);
auto out_median_depth = std::get<8>(rasterization_result);
```

### Verification
- [x] Build successful
- [x] All executables linked

### Kết quả Phase 4
- Depth outputs bây giờ được tính trong CUDA kernel
- Được copy ra torch::Tensor
- Sẵn sàng để sử dụng trong training loop

---

## ⏳ Ghi chú: Phase 5 chưa hoàn thành

Phase 5 cần thêm các thay đổi sau:
1. Sửa `GaussianRasterizerFunction::forward()` để trả về depth tensors
2. Sửa `GaussianRenderer::render()` để expose depth
3. Cập nhật training loop trong `gaussian_mapper.cpp`
4. Tích hợp loss functions từ `loss_utils.h`

Điều này có thể gây breaking changes cho code hiện tại.

---

## [2026-01-13 08:17] Phase 5 Completed: L_geo Sensor Depth Loss Integration

### Mục đích
Tích hợp **L_geo (Sensor Depth Loss)** từ CG-SLAM vào training loop để cung cấp supervision trực tiếp từ RGBD sensor. Thay thế L_align bằng L_geo với trọng số cao (lambda_geo=1.0) để ưu tiên độ chính xác hình học.

### Động lực
**Vấn đề với L_align:**
- L_align chỉ ràng buộc depth_alpha ≈ depth_median (tính nhất quán nội bộ)
- Không supervision từ ground truth → có thể sai về scale/offset tuyệt đối
- Không tận dụng được depth sensor của RGB-D camera

**Giải pháp L_geo:**
- So sánh **trực tiếp** rendered depth với sensor depth
- Supervision mạnh từ data thật → hình học chính xác
- Phù hợp với RGBD SLAM (TUM dataset có ground truth depth)

---

### Files được sửa

#### 📄 `src/gaussian_mapper.cpp` - Training Loop Integration

**1. Ground Truth Depth Preparation (Lines 648-697)**

```cpp
// Depth-Photo-SLAM: Prepare ground truth depth for L_geo (RGBD only)
torch::Tensor gt_depth;
bool has_gt_depth = false;

if (this->sensor_type_ == RGBD && viewpoint_cam->img_auxiliary_undist_.data) {
    // Load depth từ img_auxiliary_undist_ (depth map từ RGBD sensor)
    cv::Mat depth_undist = viewpoint_cam->img_auxiliary_undist_;
    
    // Convert sang torch tensor
    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat depth_gpu;
        depth_gpu.upload(depth_undist);
        gt_depth = tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu);
    } else {
        gt_depth = tensor_utils::cvMat2TorchTensor_Float32(depth_undist, device_type_);
    }
    
    // Resize nếu dùng Gaussian pyramid training
    if (training_level != num_gaus_pyramid_sub_levels_) {
        gt_depth = torch::nn::functional::interpolate(...).squeeze();
    }
    
    has_gt_depth = true;
}
```

**Data flow:**
```
TUM dataset depth/*.png → ORB-SLAM3 KeyFrame → img_auxiliary_undist_ → gt_depth tensor
```

---

**2. L_geo Loss Computation (Lines 759-782)**

```cpp
// DISABLED: L_align (depth alignment loss)
// auto L_align = loss_utils::depth_alignment_loss(depth, median_depth);

// NEW: Geometric sensor depth loss (L_geo)
torch::Tensor L_geo = torch::zeros(1, torch::TensorOptions().device(device_type_));

if (has_gt_depth) {
    // Valid mask: loại bỏ depth không hợp lệ
    auto depth_valid_mask = (gt_depth > RGBD_min_depth_) &   // > 0.1m
                            (gt_depth < RGBD_max_depth_) &   // < 10m
                            mask.squeeze().to(torch::kBool); // undistort mask
    
    auto depth_squeezed = depth.squeeze();  // [H, W]
    
    // L_geo = mean(|D_rendered - D_sensor|) over valid pixels
    L_geo = loss_utils::sensor_depth_loss(
        depth_squeezed,    // Rendered depth từ Gaussian Splatting
        gt_depth,          // Ground truth depth từ sensor
        depth_valid_mask   // Valid pixels only
    );
}
```

**Valid Mask:**
- `gt_depth > 0.1m`: loại bỏ invalid sensor readings
- `gt_depth < 10m`: ngoài range đáng tin của sensor
- `mask`: loại bỏ vùng bị undistortion crop

---

**3. Loss Weights Update (Lines 784-792)**

```cpp
// Loss weights
float lambda_geo = 1.0f;       // HIGH weight cho geometric accuracy
// float lambda_align = 0.1f;  // DISABLED - thay bằng L_geo
float lambda_var = 0.0f;       // Keep disabled
float lambda_iso = 0.0f;       // Keep disabled

// Total loss
auto loss = (1.0 - lambda_dssim) * Ll1
          + lambda_dssim * (1.0 - DSSIM)
          + lambda_geo * L_geo         // NEW
          + lambda_var * L_var;
```

**Lambda_geo = 1.0 justification:**
- L1 photometric ≈ 0.02-0.05 per pixel
- L_geo depth ≈ 0.05-0.1 meters per pixel
- λ=1.0 → depth error **cùng quan trọng** với color error
- Trade-off: Better geometry ↔ Potentially lower PSNR

---

**4. Logging Updates (Lines 797-812)**

```cpp
// CSV header
loss_log_file << "iteration,L1,DSSIM,L_geo,L_var,total" << std::endl;

// Values
float geo_val = L_geo.item<float>();
loss_log_file << iteration << "," 
             << l1_val << "," << dssim_val << "," 
             << geo_val << "," << var_val << ","
             << total_val << std::endl;
```

**Output:** `results/loss_components.csv` giờ track L_geo thay vì L_align.

---

### Backward Pass - Gradient Flow Chi Tiết

**L_geo đã có backward implementation sẵn** trong `cuda_rasterizer/backward_depth.cuh`.

#### 1. Loss Gradient (PyTorch Autograd)

```cpp
L_geo = mean(|D_rendered - D_sensor|)

// Gradient
∂L_geo/∂D_rendered = sign(D_rendered - D_sensor) / num_valid_pixels
```

Gradient này được PyTorch autograd tự động tính và truyền vào CUDA kernels qua `dL_ddepth`.

---

#### 2. Depth Forward Pass (forward.cu:370-375)

```cpp
// Alpha-blending trong rendering
float gaussian_depth = collected_depths[j];  // d_i (depth của Gaussian i)
float weight = alpha * T;                     // w_i = α_i × T_i
D += gaussian_depth * weight;                 // D = Σ w_i × d_i
D_sq += gaussian_depth * gaussian_depth * weight;
```

**Công thức:**
```
D_rendered = Σ w_i × d_i
           = Σ (α_i × T_i) × d_i

Trong đó:
- d_i: depth của Gaussian i (p_view.z)
- α_i: opacity của Gaussian i
- T_i: transmittance = Π(1 - α_j) for j < i
```

---

#### 3. Depth Gradient Backward (backward_depth.cuh:121-129)

```cpp
// Gradient w.r.t. mỗi Gaussian's depth
if (dL_ddepths != nullptr) {
    // Chain rule: ∂L/∂d_i = ∂L/∂D × ∂D/∂d_i = ∂L/∂D × w_i
    atomicAdd(&(dL_ddepths[global_id]), weight * dL_dpixel_depth);
}

// Depth cũng ảnh hưởng alpha gradient (qua transmittance)
accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
dL_dalpha += (gaussian_depth - accum_depth_rec) * dL_dpixel_depth * T;
```

**Ý nghĩa:**
- Gradient trực tiếp: `∂L/∂d_i = w_i × ∂L/∂D`
- Gradient gián tiếp qua alpha: depth càng sai → alpha cần điều chỉnh → transmittance thay đổi

---

#### 4. Position Gradient (backward_depth.cuh:198-204)

```cpp
// Depth được tính từ camera-space z-coordinate
// depth = viewmatrix[2]*x + viewmatrix[6]*y + viewmatrix[10]*z

if (dL_ddepths != nullptr) {
    float dL_ddepth = dL_ddepths[idx];
    dL_dmean.x += dL_ddepth * viewmatrix[2];   // ∂depth/∂x
    dL_dmean.y += dL_ddepth * viewmatrix[6];   // ∂depth/∂y
    dL_dmean.z += dL_ddepth * viewmatrix[10];  // ∂depth/∂z
}

dL_dmeans[idx] += dL_dmean;  // Accumulate gradient
```

**Gradient đầy đủ đến position:**
```
∂L_geo/∂μ = ∂L_geo/∂D × ∂D/∂d_i × ∂d_i/∂μ

Trong đó:
∂d_i/∂μ = viewmatrix (camera transformation)
```

---

#### Gradient Flow Diagram

```
[L_geo Loss]
    ↓ ∂L/∂D_rendered (autograd)
[Rendered Depth D]
    ↓ ∂D/∂d_i = w_i (alpha-blending weight)
[Gaussian Depths d_i]
    ↓ ∂d_i/∂μ = viewmatrix (camera space)
[Gaussian Positions μ_xyz]
    ↓ Adam optimizer
[Updated Positions]

Indirect path:
[L_geo] → [D] → [α_i] → [cov2D] → [μ] (via transmittance)
```

**Kết quả:** Gaussians di chuyển để khớp depth với sensor → hình học chính xác!

---

### Verification Checklist

- [x] **Build successful:** No compilation errors
- [x] **Backward pass:** Đã có sẵn trong `backward_depth.cuh`
- [x] **Ground truth depth:** Load từ `img_auxiliary_undist_` (RGBD)
- [x] **Valid mask:** Filter invalid sensor readings
- [x] **Loss logging:** CSV header updated to L_geo
- [ ] **Runtime test:** Chưa chạy full benchmark
- [ ] **Metrics comparison:** Chưa so sánh ATE/PSNR với L_align

---

### Build Output

```bash
cd build && make -j$(nproc)
# [100%] Built target train_colmap
# Exit code: 0
```

**Fixed issues:**
- ✅ Lvalue reference error: `depth.squeeze()` trả về rvalue
  - **Fix:** Store in variable `depth_squeezed` trước khi pass vào function

---

### Testing Plan

#### 1. Quick Smoke Test
```bash
./bin/tum_rgbd \
    Vocabulary/ORBvoc.bin \
    configs/TUM-RGBD/fr1_desk.yaml \
    configs/gaussian_mapper/TUM-RGBD/rgbd.yaml \
    dataset/rgbd_dataset_freiburg1_desk \
    no_viewer \
    results_lgeo_test
```

**Expected:** `loss_components.csv` có L_geo values giảm dần.

#### 2. Full Benchmark
```bash
bash scripts/tum_rgbd_pa2.sh  # Run toàn bộ TUM datasets
python Photo-SLAM-eval/onekey.py --results results_lgeo
```

**Metrics to compare:**
- **ATE** (geometric): Should ↓ (better)
- **PSNR/SSIM** (photometric): May ↓ slightly (trade-off)
- **L_geo convergence**: Should decrease steadily

---

### Kết luận Phase 5

✅ **Hoàn thành:**
- L_geo sensor depth loss đã được tích hợp đầy đủ
- Backward pass có sẵn với gradient flow chính xác
- Lambda_geo = 1.0 cho strong geometric supervision
- L_align bị disable như yêu cầu

📊 **Kết quả mong đợi:**
- Hình học chính xác hơn (ATE thấp hơn)
- Gaussians khớp với sensor depth
- Trade-off nhỏ về PSNR/SSIM do prioritize geometry

🔬 **Next steps:**
- Chạy benchmark đầy đủ trên TUM datasets
- So sánh metrics với PA3 (L_align) và baseline
- Tune lambda_geo nếu cần (0.5–2.0 range)

---

## [2026-01-29 15:20] Phase 6: Edge-aware Smoothness Loss (L_smooth)

### Mục đích
Thêm **Edge-aware Smoothness Loss (L_smooth)** để làm giảm hiện tượng mesh bị lồi lõm (bumpy surfaces). Loss này ép depth map phải trơn tru ở những vùng màu sắc đồng nhất, nhưng vẫn giữ được cạnh sắc nét ở nơi ảnh RGB có sự thay đổi lớn.

### Công thức toán học

$$L_{smooth} = \frac{1}{N} \sum_{p} \left( \lambda_x(p) \cdot |\partial_x D(p)| + \lambda_y(p) \cdot |\partial_y D(p)| \right)$$

**Trong đó:**
- $|\partial_x D(x, y)| = |D(x+1, y) - D(x, y)|$ — Gradient depth theo trục x (kernel 1x2)
- $|\partial_y D(x, y)| = |D(x, y+1) - D(x, y)|$ — Gradient depth theo trục y (kernel 2x1)
- $\lambda_x(p) = e^{-|\partial_x I(p)|}$ — Trọng số dựa trên gradient ảnh RGB
- $\lambda_y(p) = e^{-|\partial_y I(p)|}$

**Cơ chế:**
- Vùng tường phẳng (màu đồng nhất): $|\partial I| \approx 0 \Rightarrow \lambda \approx 1$ → Ép depth phẳng mạnh.
- Vùng cạnh vật thể (màu thay đổi): $|\partial I|$ lớn $\Rightarrow \lambda \approx 0$ → Cho phép depth thay đổi.

---

### Files được sửa

#### 📄 `include/loss_utils.h` (SỬA)
Thêm hàm `smoothness_loss()`:

```cpp
inline torch::Tensor smoothness_loss(torch::Tensor depth, torch::Tensor image)
{
    // Compute depth gradients using 1x2 kernel
    auto d_dx = torch::abs(depth[..., :, 1:] - depth[..., :, :-1]);
    auto d_dy = torch::abs(depth[..., 1:, :] - depth[..., :-1, :]);
    
    // Compute image gradients (mean across RGB channels)
    auto i_dx = torch::mean(torch::abs(image[..., :, 1:] - image[..., :, :-1]), 0, true);
    auto i_dy = torch::mean(torch::abs(image[..., 1:, :] - image[..., :-1, :]), 0, true);
    
    // Edge-aware weights: w = exp(-|∂I|)
    auto w_x = torch::exp(-i_dx);
    auto w_y = torch::exp(-i_dy);
    
    return (d_dx * w_x).mean() + (d_dy * w_y).mean();
}
```

---

#### 📄 `src/gaussian_mapper.cpp` (SỬA)

**1. Compute L_smooth (dòng 793-795):**
```cpp
// Edge-aware Smoothness Loss (L_smooth)
auto L_smooth = loss_utils::smoothness_loss(depth, gt_image);
```

**2. Thêm lambda_smooth và tích hợp vào total loss (dòng 800-810):**
```cpp
float lambda_smooth = 0.01f;   // Edge-aware smoothness

auto loss = (1.0 - lambda_dssim) * Ll1
          + lambda_dssim * (1.0 - DSSIM)
          + lambda_geo * L_geo
          + lambda_var * L_var
          + lambda_smooth * L_smooth;  // NEW
```

**3. Cập nhật logging CSV (dòng 820-840):**
- Header: `"iteration,L1,DSSIM,L_geo,L_var,L_smooth,total"`
- Thêm `float smooth_val = L_smooth.item<float>();`

---

### Weight tuning guide

| lambda_smooth | Hiệu ứng |
|---------------|----------|
| 0.001 | Rất nhẹ, gần như không ảnh hưởng |
| 0.01 | **Mặc định** - Cân bằng smooth và chi tiết |
| 0.05 | Trung bình mạnh - Mesh mượt hơn |
| 0.1+ | Rất mạnh - Có thể làm mờ chi tiết |

---

### Verification
- ✅ Build successful (100%)
- ✅ All executables linked
- ⏳ Chưa chạy benchmark

### Next steps
- Chạy training trên Replica/TUM để đánh giá hiệu quả
- Nếu mesh vẫn lồi lõm, tăng `lambda_smooth` lên 0.05 hoặc 0.1
- Nếu mất chi tiết, giảm xuống 0.005

---

## [2026-01-29 15:35] Phase 6.1: Configurable Loss Weights via YAML

### Mục đích
Cho phép thay đổi trọng số các loss function từ file config YAML mà **không cần build lại**.

### Files được sửa

#### 📄 `include/gaussian_parameters.h`
Thêm các member variables vào `GaussianOptimizationParams`:
```cpp
float lambda_geo_ = 0.5f;
float lambda_smooth_ = 0.01f;
float lambda_var_ = 0.0f;
float lambda_iso_ = 0.0f;
float lambda_align_ = 0.0f;
```

#### 📄 `src/gaussian_mapper.cpp` - readConfigFromFile()
Thêm code đọc loss weights từ config:
```cpp
if (!settings_file["Optimization.lambda_geo"].empty())
    opt_params_.lambda_geo_ = settings_file["Optimization.lambda_geo"].operator float();
// ... similar for other lambdas
```

#### 📄 `src/gaussian_mapper.cpp` - trainForOneIteration()
Thay thế hardcoded values bằng `opt_params_`:
```cpp
float lambda_geo = opt_params_.lambda_geo_;
float lambda_smooth = opt_params_.lambda_smooth_;
// ...
```

#### 📄 `cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml`
#### 📄 `cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml`
Thêm các parameters mới:
```yaml
# Loss Weights (Depth-Photo-SLAM)
Optimization.lambda_geo: 0.5
Optimization.lambda_smooth: 0.01
Optimization.lambda_var: 0.0
Optimization.lambda_iso: 0.0
Optimization.lambda_align: 0.0
```

### Verification
- ✅ Build successful (100%)
- ✅ All executables linked

### Cách sử dụng
Chỉ cần sửa file YAML và chạy lại chương trình:
```bash
# Ví dụ: tăng lambda_smooth lên 0.05
nano cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml
./bin/replica_rgbd ...
```

