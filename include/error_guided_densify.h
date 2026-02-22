/**
 * ConeGS Integration for CG-photo (3D Gaussian Splatting SLAM)
 *
 * Replaces 3 core 3DGS mechanisms:
 * 1. Clone/Split → Error-Guided Insertion (error-weighted multinomial sampling)
 * 2. k-NN Scaling → Cone-based Scaling (pixel cone footprint at depth)
 * 3. Opacity Reset → Pre-activation Opacity Penalty (continuous penalty loss)
 *
 * Reference: ConeGS (ECCV 2024) - adapted for RGB-D SLAM (sensor depth instead of iNGP)
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace conegs {

// ============================================================================
// Config structs
// ============================================================================

/**
 * Cone-based scale initialization config.
 * Replaces k-NN distance heuristic (distCUDA2) with pixel cone footprint.
 * scale = depth / focal_length — so Gaussian size ≈ 1 pixel at that depth.
 */
struct ConeScaleConfig {
    bool enabled = false;
    float scale_multiplier = 1.0f;   ///< Tunable multiplier on cone radius
    float min_scale = -7.0f;         ///< Min log-scale (≈0.001m in world units)
    float max_scale = 0.0f;          ///< Max log-scale (≈1.0m in world units)
};

/**
 * Error-guided point insertion config.
 * Replaces gradient-based Clone/Split with error-weighted pixel sampling.
 */
struct ConeDensifyConfig {
    bool enabled = false;
    int budget_per_iter = 200;       ///< Max new points per densification step
    int densify_interval = 100;      ///< Run error insertion every N iterations
    int prune_interval = 100;        ///< Prune low-opacity Gaussians every N iterations
    float prune_opacity_threshold = 0.005f;  ///< Prune if sigmoid(opacity) < this
    float min_error_threshold = 0.01f;       ///< Skip pixels with L1 error below this
    float flatten_ratio = 0.3f;      ///< Z-axis squash ratio (0.05=20x thin, 0.3=3.3x, 1.0=sphere)
};

// ============================================================================
// Thay thế 2: Cone-based Scaling (replaces k-NN / distCUDA2)
// ============================================================================

/**
 * Compute isotropic Gaussian scale from pixel cone footprint.
 *
 * ConeGS formula (adapted for RGB-D):
 *   r_cone = depth / min(fx, fy) * multiplier
 *   scale = log(clamp(r_cone, min, max))   — log-space for GaussianModel
 *
 * Returns [N, 3] tensor in log-space (ready for GaussianModel::scaling_).
 *
 * @param depths    [N] sensor depths of candidate points
 * @param fx        camera focal length x
 * @param fy        camera focal length y
 * @param config    cone scale configuration
 * @return          [N, 3] log-space isotropic scales
 */
inline torch::Tensor computeConeScale(
    const torch::Tensor& depths,
    float fx, float fy,
    const ConeScaleConfig& config)
{
    float f_min = std::min(fx, fy);
    auto r_cone = depths.abs() / f_min * config.scale_multiplier;
    // Clamp in log-space: config.min_scale and max_scale are log-space values
    // e.g. min_scale=-7.0 (≈0.001m), max_scale=0.0 (≈1.0m)
    auto log_scale = torch::clamp(torch::log(r_cone), config.min_scale, config.max_scale);
    // Return [N, 3] in log-space (isotropic: same scale on all 3 axes)
    return log_scale.unsqueeze(1).expand({-1, 3}).contiguous();
}

// ============================================================================
// Advanced ConeGS: Normal-Aligned Surfels (NAS)
// ============================================================================

/**
 * Compute pure tensor-based surface normals and quaternions for sampled pixels.
 *
 * 1. Computes depth gradients using Pytorch conv2d (Sobel).
 * 2. Extracts normals in Camera Space at sampled (u, v) coordinates.
 * 3. Transforms normals to World Space using Twc.
 * 4. Computes quaternions that align the local Z-axis [0, 0, 1] to the normal.
 *
 * @param depth_map     [H, W] or [1, H, W] sensor depth map
 * @param u_valid       [M] valid u coordinates
 * @param v_valid       [M] valid v coordinates 
 * @param d_valid       [M] valid depth values at (u, v)
 * @param fx, fy        camera focal lengths
 * @param Twc           [4, 4] Camera-to-World transform matrix
 * @return              [M, 4] quaternions (w, x, y, z) for Gaussians
 */
inline torch::Tensor computeNormalsAndRotations(
    const torch::Tensor& depth_map,
    const torch::Tensor& u_valid,
    const torch::Tensor& v_valid,
    const torch::Tensor& d_valid,
    float fx, float fy,
    const torch::Tensor& Twc)
{
    int M = u_valid.size(0);
    if (M == 0) return torch::zeros({0, 4}, depth_map.options());

    auto d = depth_map.dim() == 2 ? depth_map.unsqueeze(0).unsqueeze(0) : depth_map.unsqueeze(0);
    
    // Sobel filters
    auto opts = torch::TensorOptions().device(depth_map.device()).dtype(torch::kFloat32);
    auto sobel_x = torch::tensor({{{{-1.0f, 0.0f, 1.0f},
                                    {-2.0f, 0.0f, 2.0f},
                                    {-1.0f, 0.0f, 1.0f}}}}, opts) / 8.0f;
    auto sobel_y = torch::tensor({{{{-1.0f, -2.0f, -1.0f},
                                    { 0.0f,  0.0f,  0.0f},
                                    { 1.0f,  2.0f,  1.0f}}}}, opts) / 8.0f;
    
    // Pad depth and apply Sobel
    auto d_pad = torch::reflection_pad2d(d, {1, 1, 1, 1});
    auto ddu = torch::conv2d(d_pad, sobel_x).squeeze(); // [H, W]
    auto ddv = torch::conv2d(d_pad, sobel_y).squeeze(); // [H, W]

    // Extract gradients at valid pixels
    auto u_idx = u_valid.to(torch::kLong);
    auto v_idx = v_valid.to(torch::kLong);
    auto ddu_sampled = ddu.index({v_idx, u_idx}); // [M]
    auto ddv_sampled = ddv.index({v_idx, u_idx}); // [M]

    // Normal in Camera Space: n = (-fx * ddu, -fy * ddv, D)
    auto nx_cam = -fx * ddu_sampled;
    auto ny_cam = -fy * ddv_sampled;
    auto nz_cam = d_valid;

    auto n_cam_raw = torch::stack({nx_cam, ny_cam, nz_cam}, 1); // [M, 3]
    auto n_cam = torch::nn::functional::normalize(
        n_cam_raw, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));

    // Transform Normal to World Space: n_world = Rwc * n_cam
    auto Rwc = Twc.slice(0, 0, 3).slice(1, 0, 3); // [3, 3] rotation matrix
    auto n_world = n_cam.matmul(Rwc.t()); // [M, 3]
    n_world = torch::nn::functional::normalize(
        n_world, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));

    // Convert n_world to Quaternion that rotates [0, 0, 1] to n_world
    // q_raw = (1 + nz, -ny, nx, 0)
    auto nx_w = n_world.select(1, 0);
    auto ny_w = n_world.select(1, 1);
    auto nz_w = n_world.select(1, 2);

    auto qw = 1.0f + nz_w;
    auto qx = -ny_w;
    auto qy = nx_w;
    auto qz = torch::zeros_like(nx_w);

    auto q_raw = torch::stack({qw, qx, qy, qz}, 1); // [M, 4]
    
    // Normalize quaternion
    auto q_norm = torch::nn::functional::normalize(
        q_raw, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));

    // Handle edge case where nz_w == -1 (normal points exactly opposite to Z)
    auto degenerate = nz_w < -0.9999f;
    if (degenerate.any().item<bool>()) {
        auto alt_q = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f}, opts).unsqueeze(0).expand({M, 4});
        q_norm = torch::where(degenerate.unsqueeze(1), alt_q, q_norm);
    }

    return q_norm.contiguous();
}

// ============================================================================
// Thay thế 1: Error-Guided Insertion (replaces Clone/Split)
// ============================================================================

/**
 * Error-weighted multinomial pixel sampling.
 *
 * ConeGS approach: sample pixels proportional to rendering error.
 * P(pixel) ∝ L1_error(pixel) — high-error pixels get sampled more.
 *
 * Returns [N, 2] pixel coordinates (u, v) for backprojection.
 *
 * @param rendered      [C, H, W] rendered image
 * @param gt            [C, H, W] ground truth image
 * @param depth         [H, W] or [1, H, W] ground truth depth map for variance calculation
 * @param num_samples   maximum number of pixels to sample
 * @param min_error     minimum L1 error to consider (pixels below this are excluded)
 * @return              [N, 2] int tensor of (u, v) pixel coordinates
 */
inline torch::Tensor errorWeightedSample(
    const torch::Tensor& rendered,
    const torch::Tensor& gt,
    const torch::Tensor& depth,
    int num_samples,
    float min_error = 0.01f)
{
    // Per-pixel L1 error (mean across channels)
    auto l1_error = (rendered - gt).abs().mean(0);  // [H, W]

    // GVS: Geometry-Variance Guided Sampling
    // Compute depth gradient magnitude to suppress sampling on flat surfaces
    auto d = depth.dim() == 2 ? depth.unsqueeze(0).unsqueeze(0) : depth.unsqueeze(0);
    auto opts = torch::TensorOptions().device(depth.device()).dtype(torch::kFloat32);
    auto sobel_x = torch::tensor({{{{-1.0f, 0.0f, 1.0f},
                                    {-2.0f, 0.0f, 2.0f},
                                    {-1.0f, 0.0f, 1.0f}}}}, opts) / 8.0f;
    auto sobel_y = torch::tensor({{{{-1.0f, -2.0f, -1.0f},
                                    { 0.0f,  0.0f,  0.0f},
                                    { 1.0f,  2.0f,  1.0f}}}}, opts) / 8.0f;
    
    // Pad depth and apply Sobel
    auto d_pad = torch::reflection_pad2d(d, {1, 1, 1, 1});
    auto ddu = torch::conv2d(d_pad, sobel_x).squeeze(); // [H, W]
    auto ddv = torch::conv2d(d_pad, sobel_y).squeeze(); // [H, W]
    
    // Gradient magnitude as a measure of structural complexity/variance
    auto nabla_d = torch::sqrt(ddu * ddu + ddv * ddv);
    
    // Normalize nabla_d by depth to get scale-invariant relative variance, clamp to avoid noise
    // A flat wall, even if far away, has nabla_d ~ 0. An edge has high nabla_d.
    auto d_squeeze = d.squeeze();
    auto valid_depth = d_squeeze > 0.01f;
    auto relative_var = torch::where(valid_depth, nabla_d / d_squeeze, torch::zeros_like(nabla_d));
    
    // Modulate the L1 error by the geometric complexity
    // We add a tiny epsilon (0.01) so that perfectly flat walls can still spawn *some* points if color error is massive,
    // but heavily penalize them compared to structural edges.
    auto gvs_weight = torch::clamp(relative_var, 0.0f, 1.0f) + 0.01f;
    auto modulated_error = l1_error * gvs_weight;

    // Threshold: zero out low-error pixels using the modulated error
    auto weights = torch::where(modulated_error > min_error, modulated_error, 
                                torch::zeros_like(modulated_error));
    auto flat_weights = weights.reshape(-1);  // [H*W]
    
    // Check if any pixel has error
    float total_weight = flat_weights.sum().item<float>();
    if (total_weight < 1e-8f) {
        // No significant error — return empty
        std::vector<int64_t> empty_size = {0, 2};
        return torch::zeros(
            at::IntArrayRef(empty_size),
            torch::TensorOptions().dtype(torch::kInt32).device(rendered.device()));
    }

    // Multinomial sampling ∝ error
    int64_t actual_samples = std::min((int64_t)num_samples, flat_weights.size(0));
    auto selected = torch::multinomial(flat_weights, actual_samples, /*replacement=*/false);
    
    // Convert flat indices to (u, v)
    int W = gt.size(2);
    auto pixels_u = (selected % W).to(torch::kInt32);
    auto pixels_v = (selected / W).to(torch::kInt32);
    
    return torch::stack({pixels_u, pixels_v}, 1);  // [N, 2]
}

/**
 * Backproject sampled pixels to 3D points using sensor depth.
 *
 * Replaces iNGP ray casting in original ConeGS — direct backprojection
 * since CG-photo has RGB-D sensor depth.
 *
 * @param pixels    [N, 2] int (u, v) pixel coordinates
 * @param depth     [H, W] or [1, H, W] sensor depth map
 * @param fx, fy, cx, cy   camera intrinsics
 * @param Tcw       [4, 4] world-to-camera transform
 * @return tuple of (points_3d [M, 3], colors [M, 3], depths [M], valid_indices [M])
 *         where M <= N (invalid depths are filtered out)
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backprojectPixels(
    const torch::Tensor& pixels,     // [N, 2] int (u, v)
    const torch::Tensor& depth_map,  // [H, W] or [1, H, W]
    const torch::Tensor& gt_image,   // [C, H, W] for extracting colors
    float fx, float fy, float cx, float cy,
    const torch::Tensor& Twc)        // [4, 4] camera-to-world transform
{
    if (pixels.size(0) == 0) {
        auto opts = torch::TensorOptions().device(pixels.device()).dtype(torch::kFloat32);
        std::vector<int64_t> empty1 = {0};
        std::vector<int64_t> empty3 = {0, 3};
        return {torch::zeros(at::IntArrayRef(empty3), opts),
                torch::zeros(at::IntArrayRef(empty3), opts),
                torch::zeros(at::IntArrayRef(empty1), opts),
                torch::zeros(at::IntArrayRef(empty1), opts.dtype(torch::kLong))};
    }

    auto depth_2d = depth_map.dim() == 3 ? depth_map.squeeze(0) : depth_map;  // [H, W]

    auto u = pixels.index({torch::indexing::Slice(), 0}).to(torch::kLong);
    auto v = pixels.index({torch::indexing::Slice(), 1}).to(torch::kLong);
    
    // Sample depths at pixel locations
    auto depths = depth_2d.index({v, u});  // [N]
    
    // Filter out invalid depths (zero or negative)
    auto valid = depths > 0.0f;
    auto valid_indices = torch::nonzero(valid).squeeze(1);
    
    if (valid_indices.size(0) == 0) {
        auto opts = torch::TensorOptions().device(pixels.device()).dtype(torch::kFloat32);
        std::vector<int64_t> empty1 = {0};
        std::vector<int64_t> empty3 = {0, 3};
        return {torch::zeros(at::IntArrayRef(empty3), opts),
                torch::zeros(at::IntArrayRef(empty3), opts),
                torch::zeros(at::IntArrayRef(empty1), opts),
                torch::zeros(at::IntArrayRef(empty1), opts.dtype(torch::kLong))};
    }

    auto u_valid = u.index({valid_indices}).to(torch::kFloat32);
    auto v_valid = v.index({valid_indices}).to(torch::kFloat32);
    auto d_valid = depths.index({valid_indices});
    
    // Backproject to camera space: X = (u - cx) * d / fx, Y = (v - cy) * d / fy, Z = d
    auto X_cam = (u_valid - cx) * d_valid / fx;
    auto Y_cam = (v_valid - cy) * d_valid / fy;
    auto Z_cam = d_valid;
    
    // [M, 3] camera-space points
    auto pts_cam = torch::stack({X_cam, Y_cam, Z_cam}, 1);  // [M, 3]
    
    // Transform to world space: P_world = Twc * P_cam
    auto ones = torch::ones({pts_cam.size(0), 1}, pts_cam.options());
    auto pts_cam_h = torch::cat({pts_cam, ones}, 1);  // [M, 4]
    auto pts_world_h = (Twc.matmul(pts_cam_h.t())).t();  // [M, 4]
    auto pts_world = pts_world_h.slice(1, 0, 3);  // [M, 3]
    
    // Extract colors from GT image at valid pixel locations
    auto u_color = u.index({valid_indices}).to(torch::kLong);
    auto v_color = v.index({valid_indices}).to(torch::kLong);
    auto colors = gt_image.index({torch::indexing::Slice(), v_color, u_color}).t();  // [M, 3]
    
    return {pts_world.contiguous(), colors.contiguous(), d_valid.contiguous(), valid_indices.contiguous()};
}

}  // namespace conegs
