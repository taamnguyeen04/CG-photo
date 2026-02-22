/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <vector>

#include <torch/torch.h>

namespace loss_utils
{

inline torch::Tensor l1_loss(torch::Tensor &network_output, torch::Tensor &gt)
{
    return torch::abs(network_output - gt).mean();
}

inline torch::Tensor psnr(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).mean();
    return 10.0f * torch::log10(1.0f / mse);
}

/** def psnr(img1, img2):
 *     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
 *     return 20 * torch.log10(1.0 / torch.sqrt(mse))
 */
inline torch::Tensor psnr_gaussian_splatting(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).view({img1.size(0) , -1}).mean(1, /*keepdim=*/true);
    return 20.0f * torch::log10(1.0f / torch::sqrt(mse)).mean();
}

inline torch::Tensor gaussian(
    int window_size,
    float sigma,
    torch::DeviceType device_type = torch::kCUDA)
{
    std::vector<float> gauss_values(window_size);
    for (int x = 0; x < window_size; ++x) {
        int temp = x - window_size / 2;
        gauss_values[x] = std::exp(-temp * temp / (2.0f * sigma * sigma));
    }
    torch::Tensor gauss = torch::tensor(
        gauss_values,
        torch::TensorOptions().device(device_type));
    return gauss / gauss.sum();
}

inline torch::autograd::Variable create_window(
    int window_size,
    int64_t channel,
    torch::DeviceType device_type = torch::kCUDA)
{
    auto _1D_window = gaussian(window_size, 1.5f, device_type).unsqueeze(1);
    auto _2D_window = _1D_window.mm(_1D_window.t()).to(torch::kFloat).unsqueeze(0).unsqueeze(0);
    auto window = torch::autograd::Variable(_2D_window.expand({channel, 1, window_size, window_size}).contiguous());
    return window;
}

inline torch::Tensor _ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::autograd::Variable &window,
    int window_size,
    int64_t channel,
    bool size_average = true)
{
    int window_size_half = window_size / 2;
    auto mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));
    auto mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));

    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1 * mu2;

    auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_sq;
    auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu2_sq;
    auto sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_mu2;

    auto C1 = 0.01 * 0.01;
    auto C2 = 0.03 * 0.03;

    auto ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

    if (size_average)
        return ssim_map.mean();
    else
        return ssim_map.mean(1).mean(1).mean(1);
}

inline torch::Tensor ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::DeviceType device_type = torch::kCUDA,
    int window_size = 11,
    bool size_average = true)
{
    auto channel = img1.size(-3);
    auto window = create_window(window_size, channel, device_type);

    // window = window.to(img1.device());
    window = window.type_as(img1);

    return _ssim(img1, img2, window, window_size, channel, size_average);
}

// ============================================================================
// Depth-Photo-SLAM: Geometric Consistency Loss Functions
// ============================================================================

/**
 * @brief Depth Alignment Loss (L_align)
 * 
 * Forces alpha-blended depth to match median depth, eliminating floaters.
 * When D_alpha ≈ D_median, the depth distribution is unimodal (solid surface).
 * Divergence indicates multimodal distribution (fog/floaters).
 * 
 * L_align = (1/|Ω|) * Σ |D_alpha(u) - D_median(u)|
 * 
 * @param alpha_depth Depth from alpha-blending: D = Σ w_i * d_i
 * @param median_depth Depth at transmittance T = 0.5
 * @param valid_mask Optional mask for valid pixels
 * @return Scalar alignment loss
 */
inline torch::Tensor depth_alignment_loss(
    torch::Tensor& alpha_depth,
    torch::Tensor& median_depth,
    torch::Tensor valid_mask = torch::Tensor())
{
    auto diff = torch::abs(alpha_depth - median_depth);
    
    if (valid_mask.defined() && valid_mask.numel() > 0) {
        // Apply mask - only consider valid pixels
        diff = diff * valid_mask.to(diff.dtype());
        auto num_valid = valid_mask.sum().clamp_min(1.0f);
        return diff.sum() / num_valid;
    }
    
    return diff.mean();
}

/**
 * @brief Isotropy Loss (L_iso)
 * 
 * Penalizes highly anisotropic Gaussians (needles) by constraining the ratio
 * between max and min scales. Forces Gaussians to be more spherical.
 * 
 * L_iso = Σ max(s_max/s_min - ε, 0)
 * 
 * @param scales Scaling vectors [N, 3] (in log space from model)
 * @param epsilon Allowed anisotropy ratio before penalty (default 1.5)
 * @param use_log_scales True if scales are in log space (from model.scaling_)
 * @return Scalar isotropy loss
 */
inline torch::Tensor isotropy_loss(
    torch::Tensor& scales,
    float epsilon = 1.5f,
    bool use_log_scales = true)
{
    // Convert from log space if needed
    auto s = use_log_scales ? torch::exp(scales) : scales;
    
    // Get max and min scales per Gaussian
    auto s_max = std::get<0>(torch::max(s, /*dim=*/1));
    auto s_min = std::get<0>(torch::min(s, /*dim=*/1));
    
    // Avoid division by zero
    s_min = torch::clamp_min(s_min, 1e-7f);
    
    // Compute ratio and penalize if above epsilon
    auto ratio = s_max / s_min;
    auto penalty = torch::clamp_min(ratio - epsilon, 0.0f);
    
    return penalty.mean();
}

/**
 * @brief Uncertainty Variance Loss (L_var)
 * 
 * Minimizes depth variance (uncertainty) across the image.
 * Lower variance = more confident depth estimates = better surface.
 * 
 * U_pix = E[d^2] - E[d]^2
 * L_var = mean(U_pix)
 * 
 * @param depth_variance Per-pixel depth variance map [H, W]
 * @param valid_mask Optional mask for valid pixels
 * @return Scalar variance loss
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

/**
 * @brief Sensor Depth Loss (L_geo)
 * 
 * L1 distance between rendered depth and sensor/GT depth.
 * Standard geometric supervision for RGB-D SLAM.
 * 
 * @param rendered_depth Rendered depth from Gaussian splatting
 * @param sensor_depth Ground truth depth from sensor
 * @param valid_mask Mask for valid sensor readings
 * @return Scalar depth loss
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

/**
 * @brief Edge-aware Smoothness Loss (L_smooth)
 * 
 * Encourages depth to be locally smooth, weighted by image gradients.
 * This preserves edges where image changes sharply while smoothing
 * uniform regions.
 * 
 * L_smooth = mean(|∂D/∂x| * exp(-|∂I/∂x|) + |∂D/∂y| * exp(-|∂I/∂y|))
 * 
 * @param depth Rendered depth [1, H, W] or [H, W]
 * @param image RGB image [C, H, W] normalized to 0-1
 * @return Scalar smoothness loss
 */
inline torch::Tensor smoothness_loss(
    torch::Tensor depth,
    torch::Tensor image)
{
    // Ensure depth is [1, H, W]
    if (depth.dim() == 2) {
        depth = depth.unsqueeze(0);
    }
    if (depth.dim() == 3 && depth.size(0) != 1) {
        depth = depth.index({0}).unsqueeze(0);  // Take first channel if multiple
    }
    
    // Ensure image is [C, H, W]
    if (image.dim() == 4) {
        image = image.squeeze(0);  // Remove batch dimension if present
    }
    
    // Compute depth gradients using 1x2 kernel (neighbor difference)
    // ∂D/∂x = D(x+1, y) - D(x, y)
    auto d_dx = torch::abs(
        depth.index({"...", torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}) - 
        depth.index({"...", torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1)})
    );
    // ∂D/∂y = D(x, y+1) - D(x, y)
    auto d_dy = torch::abs(
        depth.index({"...", torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}) - 
        depth.index({"...", torch::indexing::Slice(torch::indexing::None, -1), torch::indexing::Slice()})
    );
    
    // Compute image gradients (mean across color channels)
    // ∂I/∂x
    auto i_dx = torch::mean(torch::abs(
        image.index({"...", torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}) - 
        image.index({"...", torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1)})
    ), /*dim=*/0, /*keepdim=*/true);
    // ∂I/∂y
    auto i_dy = torch::mean(torch::abs(
        image.index({"...", torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}) - 
        image.index({"...", torch::indexing::Slice(torch::indexing::None, -1), torch::indexing::Slice()})
    ), /*dim=*/0, /*keepdim=*/true);
    
    // Compute edge-aware weights: w = exp(-|∂I|)
    // Where image changes sharply, weight is low (allow depth discontinuity)
    // Where image is uniform, weight is high (enforce smooth depth)
    auto w_x = torch::exp(-i_dx);
    auto w_y = torch::exp(-i_dy);
    
    // Weighted smoothness loss
    auto loss_x = (d_dx * w_x).mean();
    auto loss_y = (d_dy * w_y).mean();
    
    return loss_x + loss_y;
}

/**
 * @brief Planar Regularization Loss (L_reg) - MonoGS++
 * 
 * Encourages Gaussians to become flat disks by minimizing their smallest scale.
 * This is particularly effective for planar surfaces like walls, tables, and floors.
 * 
 * Formula: L_reg = mean(max(floor, min(s)) - floor)
 * 
 * When min(s) → floor, the Gaussian becomes a flat 2D disk.
 * The floor value (default 0.01) prevents complete degeneration while
 * encouraging extreme flatness.
 * 
 * Reference: MonoGS++ - Equation 5
 * 
 * @param scales Scaling vectors [N, 3] (in log space from model.scaling_)
 * @param min_scale_floor Minimum scale threshold (default 0.01)
 * @param use_log_scales True if scales are in log space (from model)
 * @return Scalar regularization loss
 */
inline torch::Tensor planar_regularization_loss(
    torch::Tensor& scales,
    float min_scale_floor = 0.01f,
    bool use_log_scales = true)
{
    // Convert from log space if needed
    auto s = use_log_scales ? torch::exp(scales) : scales;
    
    // Get min scale per Gaussian: [N, 3] → [N]
    auto s_min = std::get<0>(torch::min(s, /*dim=*/1));
    
    // Compute penalty: max(floor, s_min) - floor
    // When s_min > floor: penalty = s_min - floor (pushes s_min toward floor)
    // When s_min <= floor: penalty = 0 (already flat enough)
    auto penalty = torch::clamp_min(s_min, min_scale_floor) - min_scale_floor;
    
    return penalty.mean();
}

/**
 * @brief First-Order Image Gradient Loss (L_g1) - 2D-3DGS Paper
 * 
 * Penalizes differences in first-order image gradients between rendered and GT.
 * This helps preserve sharp edges in the reconstruction.
 * 
 * Formula: L_g1 = ||∇I - ∇Î||₁
 * 
 * Uses Sobel operator for gradient computation.
 * 
 * Reference: 2D-3DGS Paper - Equation 10
 * 
 * @param rendered Rendered image [B, C, H, W] or [C, H, W]
 * @param gt Ground truth image [B, C, H, W] or [C, H, W]
 * @param device_type Device type (CUDA/CPU)
 * @return Scalar L1 gradient loss
 */
inline torch::Tensor first_order_gradient_loss(
    torch::Tensor& rendered,
    torch::Tensor& gt,
    torch::DeviceType device_type = torch::kCUDA)
{
    // Ensure 4D input [B, C, H, W]
    auto img1 = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
    auto img2 = gt.dim() == 3 ? gt.unsqueeze(0) : gt;
    
    int64_t channels = img1.size(1);
    
    // Sobel kernels for gradient computation
    // Sobel X: detects vertical edges
    auto sobel_x = torch::tensor({
        {-1.0f, 0.0f, 1.0f},
        {-2.0f, 0.0f, 2.0f},
        {-1.0f, 0.0f, 1.0f}
    }, torch::TensorOptions().device(device_type).dtype(torch::kFloat32));
    
    // Sobel Y: detects horizontal edges
    auto sobel_y = torch::tensor({
        {-1.0f, -2.0f, -1.0f},
        { 0.0f,  0.0f,  0.0f},
        { 1.0f,  2.0f,  1.0f}
    }, torch::TensorOptions().device(device_type).dtype(torch::kFloat32));
    
    // Reshape for conv2d: [out_channels, in_channels/groups, kH, kW]
    // For depthwise conv with groups=channels, we need [channels, 1, kH, kW]
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat({channels, 1, 1, 1});
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat({channels, 1, 1, 1});
    
    // Compute gradients for rendered image
    auto grad_x1 = torch::nn::functional::conv2d(img1, sobel_x, 
        torch::nn::functional::Conv2dFuncOptions().padding(1).groups(channels));
    auto grad_y1 = torch::nn::functional::conv2d(img1, sobel_y, 
        torch::nn::functional::Conv2dFuncOptions().padding(1).groups(channels));
    
    // Compute gradients for GT image
    auto grad_x2 = torch::nn::functional::conv2d(img2, sobel_x, 
        torch::nn::functional::Conv2dFuncOptions().padding(1).groups(channels));
    auto grad_y2 = torch::nn::functional::conv2d(img2, sobel_y, 
        torch::nn::functional::Conv2dFuncOptions().padding(1).groups(channels));
    
    // L1 loss on gradient differences
    auto loss_x = torch::abs(grad_x1 - grad_x2).mean();
    auto loss_y = torch::abs(grad_y1 - grad_y2).mean();
    
    return loss_x + loss_y;
}

/**
 * @brief Second-Order Image Gradient Loss (L_g2) - 2D-3DGS Paper
 * 
 * Penalizes differences in second-order image gradients (curvature) between 
 * rendered and GT. This helps preserve corners and fine geometric details.
 * 
 * Formula: L_g2 = ||∇²I - ∇²Î||₁
 * 
 * Uses Laplacian operator for second-order gradient computation.
 * 
 * Reference: 2D-3DGS Paper - Equation 10
 * 
 * @param rendered Rendered image [B, C, H, W] or [C, H, W]
 * @param gt Ground truth image [B, C, H, W] or [C, H, W]
 * @param device_type Device type (CUDA/CPU)
 * @return Scalar L1 Laplacian loss
 */
inline torch::Tensor second_order_gradient_loss(
    torch::Tensor& rendered,
    torch::Tensor& gt,
    torch::DeviceType device_type = torch::kCUDA)
{
    // Ensure 4D input [B, C, H, W]
    auto img1 = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
    auto img2 = gt.dim() == 3 ? gt.unsqueeze(0) : gt;
    
    int64_t channels = img1.size(1);
    
    // Laplacian kernel for second-order gradient
    // Standard discrete Laplacian: detects curvature and corners
    auto laplacian = torch::tensor({
        {0.0f,  1.0f, 0.0f},
        {1.0f, -4.0f, 1.0f},
        {0.0f,  1.0f, 0.0f}
    }, torch::TensorOptions().device(device_type).dtype(torch::kFloat32));
    
    // Reshape for depthwise conv: [channels, 1, kH, kW]
    laplacian = laplacian.unsqueeze(0).unsqueeze(0).repeat({channels, 1, 1, 1});
    
    // Compute Laplacian for rendered image
    auto lap1 = torch::nn::functional::conv2d(img1, laplacian, 
        torch::nn::functional::Conv2dFuncOptions().padding(1).groups(channels));
    
    // Compute Laplacian for GT image
    auto lap2 = torch::nn::functional::conv2d(img2, laplacian, 
        torch::nn::functional::Conv2dFuncOptions().padding(1).groups(channels));
    
    // L1 loss on Laplacian differences
    return torch::abs(lap1 - lap2).mean();
}

/**
 * @brief Analytical Epipolar Scale Consistency (ESC) Loss
 * 
 * Computes Σ2D analytically from Gaussian params + camera poses in pure PyTorch.
 * FULLY DIFFERENTIABLE — gradients flow back to scales, rotations, and xyz.
 * 
 * Unlike the old ESC which used cov2D from CUDA (detached, no gradients),
 * this version projects Σ3D → Σ2D analytically:
 *   Σ3D = R(q) @ diag(s²) @ R(q)ᵀ
 *   t = W @ [μ, 1]   (view-space position, z = t[2])
 *   J = [[fx/z, 0, -fx*tx/z²], [0, fy/z, -fy*ty/z²]]
 *   T = J @ W[:3,:3]
 *   Σ2D = T @ Σ3D @ Tᵀ
 *   L_esc = mean(|log(z_A² · det(Σ2D_A)) - log(z_B² · det(Σ2D_B))|)
 *
 * @param xyz            Gaussian positions [N, 3] (requires_grad)
 * @param scales         Activated scales [N, 3] (requires_grad)
 * @param rotations      Normalized quaternions [N, 4] (w,x,y,z) (requires_grad)
 * @param viewmatrix_A   View matrix of camera A [4, 4]
 * @param viewmatrix_B   View matrix of camera B [4, 4]
 * @param fx_A, fy_A     Focal lengths of camera A
 * @param fx_B, fy_B     Focal lengths of camera B
 * @param radii_A        Visibility radii from camera A [N] (int)
 * @param radii_B        Visibility radii from camera B [N] (int)
 * @return Scalar ESC loss
 */
inline torch::Tensor analytical_esc_loss(
    const torch::Tensor& xyz,          // [N, 3]
    const torch::Tensor& scales,       // [N, 3] activated
    const torch::Tensor& rotations,    // [N, 4] normalized quaternions (w, x, y, z)
    const torch::Tensor& viewmatrix_A, // [4, 4]
    const torch::Tensor& viewmatrix_B, // [4, 4]
    float fx_A, float fy_A,
    float fx_B, float fy_B,
    const torch::Tensor& radii_A,      // [N] int
    const torch::Tensor& radii_B)      // [N] int
{
    // --- Step 0: Find co-visible Gaussians ---
    auto covisible = (radii_A > 0) & (radii_B > 0);
    int K = covisible.sum().item<int>();
    
    if (K < 100) {
        return torch::zeros(1, xyz.options());
    }
    
    // Select co-visible subset
    auto pos = xyz.index({covisible});        // [K, 3]
    auto s = scales.index({covisible});       // [K, 3]
    auto q = rotations.index({covisible});    // [K, 4]
    
    // --- Step 1: Quaternion → Rotation matrix [K, 3, 3] ---
    auto qw = q.index({"...", 0});
    auto qx = q.index({"...", 1});
    auto qy = q.index({"...", 2});
    auto qz = q.index({"...", 3});
    
    auto r00 = 1.f - 2.f * (qy*qy + qz*qz);
    auto r01 = 2.f * (qx*qy - qw*qz);
    auto r02 = 2.f * (qx*qz + qw*qy);
    auto r10 = 2.f * (qx*qy + qw*qz);
    auto r11 = 1.f - 2.f * (qx*qx + qz*qz);
    auto r12 = 2.f * (qy*qz - qw*qx);
    auto r20 = 2.f * (qx*qz - qw*qy);
    auto r21 = 2.f * (qy*qz + qw*qx);
    auto r22 = 1.f - 2.f * (qx*qx + qy*qy);
    
    auto R = torch::stack({
        torch::stack({r00, r01, r02}, -1),
        torch::stack({r10, r11, r12}, -1),
        torch::stack({r20, r21, r22}, -1)
    }, -2); // [K, 3, 3]
    
    // --- Step 2: 3D covariance Σ3D = R @ diag(s²) @ Rᵀ ---
    auto S_sq = s.square(); // [K, 3]
    auto RS = R * S_sq.unsqueeze(-2); // [K, 3, 3] scale columns
    auto cov3D = torch::bmm(RS, R.transpose(1, 2)); // [K, 3, 3]
    
    // --- Helper: compute log(z² · det(Σ2D)) for a given camera ---
    auto compute_log_scale = [&](const torch::Tensor& viewmat, float fx, float fy) {
        // View matrix decomposition
        auto W = viewmat.slice(0, 0, 3).slice(1, 0, 3);     // [3,3] rotation
        auto t_off = viewmat.slice(0, 0, 3).index({"...", 3}); // [3] translation
        
        // Transform to view space: t = pos @ Wᵀ + t_off
        auto t = torch::mm(pos, W.t()) + t_off.unsqueeze(0); // [K, 3]
        auto tz = torch::clamp_min(t.index({"...", 2}), 0.01f);
        auto tx = t.index({"...", 0});
        auto ty = t.index({"...", 1});
        
        auto tz_inv = 1.0f / tz;
        auto tz_inv2 = tz_inv * tz_inv;
        auto zeros_K = torch::zeros_like(tz);
        
        // J [K, 2, 3]
        auto J = torch::stack({
            torch::stack({fx * tz_inv, zeros_K, -fx * tx * tz_inv2}, -1),
            torch::stack({zeros_K, fy * tz_inv, -fy * ty * tz_inv2}, -1)
        }, -2); // [K, 2, 3]
        
        // T = J @ W  [K, 2, 3]
        auto T = torch::matmul(J, W.unsqueeze(0).expand({K, 3, 3}));
        
        // Σ2D = T @ Σ3D @ Tᵀ  [K, 2, 2]
        auto cov2D = torch::bmm(torch::bmm(T, cov3D), T.transpose(1, 2));
        
        // det(Σ2D) = cov2D[0,0]*cov2D[1,1] - cov2D[0,1]²
        auto det = cov2D.index({"...", 0, 0}) * cov2D.index({"...", 1, 1})
                 - cov2D.index({"...", 0, 1}).square();
        det = torch::clamp_min(det, 1e-8f);
        tz = torch::clamp_min(tz, 1e-4f);
        
        // log(z² · det(Σ2D)) = 2·log(z) + log(det)
        return 2.0f * torch::log(tz) + torch::log(det);
    };
    
    // --- Step 3: Compute for both cameras ---
    auto log_scale_A = compute_log_scale(viewmatrix_A, fx_A, fy_A);
    auto log_scale_B = compute_log_scale(viewmatrix_B, fx_B, fy_B);
    
    // --- Step 4: L_esc = mean(|diff|) ---
    return torch::abs(log_scale_A - log_scale_B).mean();
}

// ============================================================================
// Molding-GS: TSDF-Anchored Gaussian Splatting Loss Functions
// ============================================================================

/**
 * @brief SDF Anchor Loss (L_sdf)
 * 
 * Pulls Gaussian centers toward the TSDF zero-level set (the actual surface).
 * Weighted by opacity so that high-opacity (important) Gaussians are more
 * strongly constrained to lie on the surface.
 * 
 * L_sdf = mean( |SDF(μ_i)| × sigmoid(opacity_i) × 1[weight_i > min_weight] )
 * 
 * Only applies to Gaussians in well-observed regions (weight > min_weight).
 * 
 * @param sdf_values  [N] TSDF value at each Gaussian center
 * @param opacity     [N, 1] raw opacity (pre-sigmoid logit)
 * @param weights     [N] TSDF observation weight at each Gaussian center
 * @param min_weight  Minimum weight to trust SDF (skip unobserved regions)
 * @return Scalar SDF anchor loss
 */
inline torch::Tensor sdf_anchor_loss(
    torch::Tensor& sdf_values,
    torch::Tensor& opacity,
    torch::Tensor& weights,
    float min_weight = 3.0f)
{
    // Mask: only apply where TSDF has been sufficiently observed
    auto observed_mask = (weights > min_weight).to(torch::kFloat32);  // [N]
    
    int64_t num_observed = observed_mask.sum().item<int64_t>();
    if (num_observed == 0) {
        return torch::zeros(1, sdf_values.options());
    }
    
    // Opacity weighting: high-opacity Gaussians should obey SDF more strictly
    auto opacity_weight = torch::sigmoid(opacity.squeeze(-1));  // [N] in [0, 1]
    
    // L_sdf = |SDF(μ)| × σ(opacity) × observed
    auto loss = sdf_values.abs() * opacity_weight * observed_mask;
    
    return loss.sum() / std::max<int64_t>(num_observed, 1);
}

/**
 * @brief Normal Alignment Loss (L_normal)
 * 
 * Aligns the shortest axis (Z-axis) of each Gaussian's rotation to the 
 * TSDF surface normal. This effectively "flattens" Gaussians into surfels
 * that lie tangent to the surface.
 * 
 * L_normal = mean( (1 - (R_z · n)²) × 1[weight_i > min_weight] )
 * 
 * Where R_z is the third column of the rotation matrix (local Z-axis)
 * and n is the TSDF gradient (surface normal). The loss is zero when
 * R_z is perfectly aligned with n.
 * 
 * @param rotations  [N, 4] quaternions (w, x, y, z) from Gaussian model
 * @param normals    [N, 3] surface normals from ∇TSDF (normalized)
 * @param weights    [N] TSDF observation weight
 * @param min_weight Minimum weight to trust normals
 * @return Scalar normal alignment loss
 */
inline torch::Tensor normal_alignment_loss(
    torch::Tensor& rotations,
    torch::Tensor& normals,
    torch::Tensor& weights,
    float min_weight = 3.0f)
{
    auto observed_mask = (weights > min_weight).to(torch::kFloat32);  // [N]
    
    int64_t num_observed = observed_mask.sum().item<int64_t>();
    if (num_observed == 0) {
        return torch::zeros(1, rotations.options());
    }
    
    // Extract quaternion components
    auto qw = rotations.index({"...", 0});
    auto qx = rotations.index({"...", 1}); 
    auto qy = rotations.index({"...", 2});
    auto qz = rotations.index({"...", 3});
    
    // Third column of rotation matrix (local Z-axis):
    // R_z = [2(qx*qz - qw*qy), 2(qy*qz + qw*qx), 1 - 2(qx² + qy²)]
    auto rz_x = 2.0f * (qx * qz - qw * qy);
    auto rz_y = 2.0f * (qy * qz + qw * qx);
    auto rz_z = 1.0f - 2.0f * (qx * qx + qy * qy);
    
    // Dot product: R_z · n
    auto nx = normals.index({"...", 0});
    auto ny = normals.index({"...", 1});
    auto nz = normals.index({"...", 2});
    
    auto dot = rz_x * nx + rz_y * ny + rz_z * nz;  // [N]
    
    // Loss: 1 - (R_z · n)² — zero when perfectly aligned (either direction)
    auto alignment_error = (1.0f - dot * dot) * observed_mask;
    
    return alignment_error.sum() / std::max<int64_t>(num_observed, 1);
}

} // namespace loss_utils
