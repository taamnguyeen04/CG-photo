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

}

