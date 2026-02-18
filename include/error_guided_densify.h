/**
 * Error-Guided Densification (SIG-Densify) for 3D Gaussian Splatting SLAM
 *
 * Two core contributions:
 * 1. Error Frequency Decomposition (EFD): Separates rendering error into
 *    low-freq (geometry wrong) vs high-freq (detail missing) using GaussianBlur.
 *    Only densifies where high-freq error is dominant.
 * 2. Marginal Information Gain (MIG): Analytically predicts whether adding
 *    a candidate Gaussian will reduce rendering error, using the alpha-blending
 *    formula WITHOUT re-rendering.
 *
 * Risk mitigations:
 * - MIG Transmittance: Uses front/behind heuristic with rendered depth to
 *   approximate T_c. Candidate in front → T_c=1, behind → T_c≈(1-alpha_acc).
 *   Falls back to conservative T_c=0.5 when depth is unavailable.
 * - Redundancy: Uses O(1) 2D pixel-grid occupancy instead of O(N·logN) kNN
 *   on the full Gaussian cloud.
 */

#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace error_guided {

/**
 * Configuration for Error-Guided Densification.
 */
struct EGDConfig {
    bool enabled = false;

    // ---- EFD (Error Frequency Decomposition) ----
    int blur_kernel = 5;              ///< GaussianBlur kernel size for E_low
    float e_high_threshold = 0.02f;   ///< Min |E_high| to consider densifying
    float e_low_reject_ratio = 3.0f;  ///< Reject if |E_low|/|E_high| > this ratio
    
    // ---- MIG (Marginal Information Gain) ----
    float mig_threshold = 0.001f;     ///< Min MIG score to accept candidate
    float default_candidate_opacity = 0.1f; ///< Default α for MIG calculation
    float t_fallback = 0.5f;          ///< Fallback transmittance when depth unavailable
    
    // ---- Redundancy (2D pixel-grid) ----
    int redundancy_grid_cell = 4;     ///< Grid cell size in pixels for redundancy check
    int max_per_cell = 2;             ///< Max candidates accepted per grid cell
    
    // ---- Budget ----
    int budget_per_keyframe = 1500;   ///< Max candidates accepted per keyframe
    
    // ---- Confidence-weighted initialization ----
    float high_confidence_mig = 0.05f; ///< MIG threshold for high-confidence init
    float mid_confidence_mig = 0.01f;  ///< MIG threshold for mid-confidence init
    float opacity_high = 0.3f;         ///< Init opacity for high-confidence points
    float opacity_mid = 0.15f;         ///< Init opacity for mid-confidence points
    float opacity_low = 0.05f;         ///< Init opacity for low-confidence points
    float scale_penalty_mid = 0.7f;    ///< Scale multiplier for mid-confidence
    float scale_penalty_low = 0.5f;    ///< Scale multiplier for low-confidence
};

/**
 * Error Frequency Decomposition result per pixel.
 */
struct ErrorProfile {
    torch::Tensor e_low;    ///< Low-frequency error [C, H, W]
    torch::Tensor e_high;   ///< High-frequency error [C, H, W]
    torch::Tensor e_low_mag;  ///< |E_low| magnitude per pixel [H, W]
    torch::Tensor e_high_mag; ///< |E_high| magnitude per pixel [H, W]
};

/**
 * Result of error-guided selection.
 */
struct EGDResult {
    torch::Tensor accepted_points;   ///< [M, 3] accepted 3D points
    torch::Tensor accepted_colors;   ///< [M, 3] accepted colors
    torch::Tensor accepted_mask;     ///< [N] bool mask on original candidates
    torch::Tensor mig_scores;        ///< [N] MIG scores for all candidates
    torch::Tensor confidence_levels; ///< [M] 0=low, 1=mid, 2=high
    int n_total = 0;
    int n_efd_rejected = 0;
    int n_mig_rejected = 0;
    int n_redundancy_rejected = 0;
    int n_budget_rejected = 0;
    int n_accepted = 0;
};


// ============================================================================
// Error Frequency Decomposition (EFD)
// ============================================================================

/**
 * Decompose rendering error into low-freq and high-freq components.
 * E_low = GaussianBlur(E), E_high = E - E_low.
 *
 * @param rendered Rendered image [C, H, W] on CUDA
 * @param gt       Ground truth image [C, H, W] on CUDA
 * @param blur_kernel Kernel size for GaussianBlur (odd number)
 * @return ErrorProfile with E_low, E_high, and their magnitudes
 */
inline ErrorProfile decomposeRenderingError(
    const torch::Tensor& rendered,
    const torch::Tensor& gt,
    int blur_kernel = 5)
{
    ErrorProfile result;
    
    // E = rendered - gt, [C, H, W]
    auto E = rendered - gt;
    
    // GaussianBlur in PyTorch: use depthwise conv2d with Gaussian kernel
    int C = E.size(0);
    int H = E.size(1);
    int W = E.size(2);
    int pad = blur_kernel / 2;
    
    // Create 1D Gaussian kernel
    float sigma = blur_kernel / 4.0f;  // Rule of thumb
    auto coords = torch::arange(blur_kernel, E.options()) - pad;
    auto kernel_1d = torch::exp(-coords * coords / (2.0f * sigma * sigma));
    kernel_1d = kernel_1d / kernel_1d.sum();
    
    // 2D Gaussian kernel via outer product
    auto kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0);  // [k, k]
    // Shape for depthwise conv: [C, 1, k, k]
    auto weight = kernel_2d.unsqueeze(0).unsqueeze(0).repeat({C, 1, 1, 1});
    
    // Apply blur: E_low = GaussianBlur(E)
    auto E_batch = E.unsqueeze(0);  // [1, C, H, W]
    result.e_low = torch::nn::functional::conv2d(
        E_batch, weight,
        torch::nn::functional::Conv2dFuncOptions()
            .padding(pad)
            .groups(C)
    ).squeeze(0);  // [C, H, W]
    
    // E_high = E - E_low
    result.e_high = E - result.e_low;
    
    // Magnitude: L2 norm across channels → [H, W]
    result.e_low_mag = result.e_low.norm(2, /*dim=*/0);   // [H, W]
    result.e_high_mag = result.e_high.norm(2, /*dim=*/0);  // [H, W]
    
    return result;
}


// ============================================================================
// Marginal Information Gain (MIG)
// ============================================================================

/**
 * Compute Marginal Information Gain for each candidate point.
 *
 * MIG(G_c) = |E(u,v)|² − |E(u,v) − α_c · T_c · (c_c − C(u,v))|²
 *
 * Risk mitigation for Transmittance (T_c):
 * - If candidate depth < rendered depth: candidate is in front → T_c = 1.0
 * - If candidate depth > rendered depth: candidate is behind → T_c ≈ (1 - alpha_acc)
 * - alpha_acc is estimated from rendered image vs background (if bg=black, alpha ≈ rendered brightness)
 * - If depth unavailable: use conservative T_c = t_fallback (default 0.5)
 *
 * @param candidate_pixels  [N, 2] pixel coordinates (u, v) of candidates
 * @param candidate_colors  [N, 3] RGB colors of candidates (0-1 range)
 * @param candidate_depths  [N] depth of candidates in camera frame
 * @param rendered           [C, H, W] current rendered image
 * @param gt                 [C, H, W] ground truth image
 * @param rendered_depth     [1, H, W] or [H, W] rendered depth map
 * @param config             EGDConfig
 * @return [N] MIG scores (positive = helpful, negative = harmful)
 */
inline torch::Tensor computeMIG(
    const torch::Tensor& candidate_pixels,   // [N, 2] int (u, v)
    const torch::Tensor& candidate_colors,   // [N, 3] float
    const torch::Tensor& candidate_depths,   // [N] float
    const torch::Tensor& rendered,           // [C, H, W]
    const torch::Tensor& gt,                 // [C, H, W]
    const torch::Tensor& rendered_depth,     // [1, H, W] or [H, W]
    const EGDConfig& config)
{
    int N = candidate_pixels.size(0);
    int H = rendered.size(1);
    int W = rendered.size(2);
    
    auto device = rendered.device();
    auto mig_scores = torch::zeros({N}, torch::TensorOptions().device(device));
    
    if (N == 0) return mig_scores;
    
    // Clamp pixel coordinates to image bounds
    auto u = candidate_pixels.index({torch::indexing::Slice(), 0}).clamp(0, W - 1).to(torch::kLong);
    auto v = candidate_pixels.index({torch::indexing::Slice(), 1}).clamp(0, H - 1).to(torch::kLong);
    
    // Sample rendered colors and gt colors at candidate pixels: [N, C]
    // rendered is [C, H, W], we need rendered[:, v, u] → [C, N] → transpose → [N, C]
    auto C_current = rendered.index({torch::indexing::Slice(), v, u}).t();  // [N, C]
    auto C_gt = gt.index({torch::indexing::Slice(), v, u}).t();            // [N, C]
    
    // Current error at candidate pixels
    auto E = C_current - C_gt;  // [N, C]
    auto E_sq = (E * E).sum(1);  // [N] per-pixel error squared
    
    // Estimate transmittance T_c based on depth comparison
    // rendered_depth: squeeze to [H, W]
    auto rd = rendered_depth.squeeze();  // [H, W]
    auto rd_at_pixels = rd.index({v, u});  // [N] rendered depth at candidate pixels
    
    // Estimate accumulated alpha from rendered image
    // For black background (bg=0): accumulated alpha ≈ max channel value at pixel
    // This is an approximation but works well in practice
    auto alpha_acc = std::get<0>(C_current.max(1)).clamp(0.0f, 1.0f);  // [N]
    
    // Transmittance: T_c
    // Front of existing surface: T_c = 1 (candidate fully visible)
    // Behind existing surface: T_c = max(1 - alpha_acc, 0.01) (remaining transparency)
    auto is_front = candidate_depths < rd_at_pixels;
    auto T_c_behind = (1.0f - alpha_acc).clamp_min(0.01f);
    auto T_c = torch::where(is_front, 
                            torch::ones({N}, torch::TensorOptions().device(device)),
                            T_c_behind);
    
    // Handle invalid depth (rendered depth = 0 or very small): use fallback
    auto depth_invalid = rd_at_pixels < 0.001f;
    T_c = torch::where(depth_invalid,
                       torch::full({N}, config.t_fallback, torch::TensorOptions().device(device)),
                       T_c);
    
    // MIG computation:
    // C_new = C_current + α_c * T_c * (c_c - C_current)
    // MIG = |E_old|² - |E_new|²
    float alpha_c = config.default_candidate_opacity;
    auto delta = alpha_c * T_c.unsqueeze(1) * (candidate_colors - C_current);  // [N, C]
    auto E_new = E - delta;  // [N, C]
    auto E_new_sq = (E_new * E_new).sum(1);  // [N]
    
    mig_scores = E_sq - E_new_sq;  // Positive = improvement
    
    return mig_scores;
}


// ============================================================================
// 2D Grid-based Redundancy Check
// ============================================================================

/**
 * Check redundancy using 2D pixel-grid occupancy.
 * O(N) instead of O(N·logN) kNN on the full Gaussian cloud.
 *
 * @param candidate_pixels [N, 2] pixel coordinates (u, v)
 * @param image_width  Width of image
 * @param image_height Height of image
 * @param cell_size    Grid cell size in pixels
 * @param max_per_cell Max candidates per cell
 * @return [N] bool mask: true = not redundant (can proceed)
 */
inline torch::Tensor checkRedundancy2D(
    const torch::Tensor& candidate_pixels,  // [N, 2] int
    int image_width,
    int image_height,
    int cell_size = 4,
    int max_per_cell = 2)
{
    int N = candidate_pixels.size(0);
    auto device = candidate_pixels.device();
    
    if (N == 0) {
        return torch::zeros({0}, torch::TensorOptions().dtype(torch::kBool).device(device));
    }
    
    // Move to CPU for grid operations (small data, fast)
    auto pixels_cpu = candidate_pixels.cpu().to(torch::kInt32);
    auto u_data = pixels_cpu.index({torch::indexing::Slice(), 0}).data_ptr<int32_t>();
    auto v_data = pixels_cpu.index({torch::indexing::Slice(), 1}).data_ptr<int32_t>();
    
    int grid_w = (image_width + cell_size - 1) / cell_size;
    int grid_h = (image_height + cell_size - 1) / cell_size;
    
    // Occupancy grid: count per cell
    std::vector<int> grid(grid_w * grid_h, 0);
    std::vector<bool> not_redundant(N, false);
    
    for (int i = 0; i < N; ++i) {
        int cx = std::min(u_data[i] / cell_size, grid_w - 1);
        int cy = std::min(v_data[i] / cell_size, grid_h - 1);
        cx = std::max(cx, 0);
        cy = std::max(cy, 0);
        int cell_idx = cy * grid_w + cx;
        
        if (grid[cell_idx] < max_per_cell) {
            not_redundant[i] = true;
            grid[cell_idx]++;
        }
    }
    
    auto result = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool));
    auto result_data = result.data_ptr<bool>();
    for (int i = 0; i < N; ++i) {
        result_data[i] = not_redundant[i];
    }
    
    return result.to(device);
}


// ============================================================================
// Full Pipeline: Error-Guided Selection
// ============================================================================

/**
 * Detect scene edges in GT image using Sobel.
 * Points at scene edges are geometrically important for tracking.
 * 
 * @param gt [C, H, W] ground truth image
 * @return [H, W] bool mask of edge pixels
 */
inline torch::Tensor detectSceneEdges(
    const torch::Tensor& gt,
    float edge_threshold = 0.1f)
{
    // Convert to grayscale: mean across channels
    auto gray = gt.mean(0);  // [H, W]
    int H = gray.size(0);
    int W = gray.size(1);
    
    // Sobel via conv2d
    auto sobel_x = torch::tensor({{-1.f, 0.f, 1.f}, {-2.f, 0.f, 2.f}, {-1.f, 0.f, 1.f}},
                                 gt.options()).unsqueeze(0).unsqueeze(0);
    auto sobel_y = torch::tensor({{-1.f, -2.f, -1.f}, {0.f, 0.f, 0.f}, {1.f, 2.f, 1.f}},
                                 gt.options()).unsqueeze(0).unsqueeze(0);
    
    auto gray_batch = gray.unsqueeze(0).unsqueeze(0);  // [1, 1, H, W]
    auto gx = torch::nn::functional::conv2d(gray_batch, sobel_x,
        torch::nn::functional::Conv2dFuncOptions().padding(1)).squeeze();
    auto gy = torch::nn::functional::conv2d(gray_batch, sobel_y,
        torch::nn::functional::Conv2dFuncOptions().padding(1)).squeeze();
    
    auto edge_mag = torch::sqrt(gx * gx + gy * gy);  // [H, W]
    return edge_mag > edge_threshold;
}

/**
 * Full error-guided candidate selection pipeline with SLAM stability protections.
 *
 * Three critical protections for tracking stability:
 * 1. ORB Bypass: Candidates matching ORB keypoints always accepted (tracking anchors)
 * 2. Scene Edge Bypass: Candidates at GT scene edges bypass EFD (structural importance)
 * 3. Min Density Floor: Guarantee at least 1 point per 16x16 cell (prevent holes)
 *
 * Flow:
 * 1. Bypass: ORB keypoints + scene edge candidates → always accepted
 * 2. EFD: Decompose rendering error → reject low-freq-dominant non-bypass candidates
 * 3. MIG: Marginal information gain → reject candidates with MIG ≤ threshold
 * 4. Redundancy: 2D grid check → reject spatially redundant candidates
 * 5. Min Density: Fill empty cells from rejected candidates
 * 6. Budget: Sort by MIG, keep top-k
 * 7. Confidence: Assign opacity/scale based on MIG level
 */
inline EGDResult errorGuidedSelect(
    const torch::Tensor& candidate_points_3d,  // [N, 3]
    const torch::Tensor& candidate_colors,     // [N, 3]
    const torch::Tensor& candidate_pixels,     // [N, 2] (u, v)
    const torch::Tensor& candidate_depths,     // [N]
    const torch::Tensor& rendered,             // [C, H, W]
    const torch::Tensor& gt,                   // [C, H, W]
    const torch::Tensor& rendered_depth,       // [1, H, W] or [H, W]
    int image_width,
    int image_height,
    const EGDConfig& config,
    const torch::Tensor& bypass_mask = {})     // [N] bool: true = always accept (ORB)
{
    EGDResult result;
    int N = candidate_points_3d.size(0);
    result.n_total = N;
    auto device = candidate_points_3d.device();
    
    if (N == 0) {
        result.accepted_points = torch::zeros({0, 3}, torch::TensorOptions().device(device));
        result.accepted_colors = torch::zeros({0, 3}, torch::TensorOptions().device(device));
        result.accepted_mask = torch::zeros({0}, torch::TensorOptions().dtype(torch::kBool).device(device));
        result.mig_scores = torch::zeros({0}, torch::TensorOptions().device(device));
        result.confidence_levels = torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        return result;
    }
    
    // Pixel coordinates
    auto u = candidate_pixels.index({torch::indexing::Slice(), 0}).clamp(0, image_width - 1).to(torch::kLong);
    auto v = candidate_pixels.index({torch::indexing::Slice(), 1}).clamp(0, image_height - 1).to(torch::kLong);
    
    // ========== Protection 1: ORB Bypass ==========
    // ORB keypoint candidates are ALWAYS accepted (tracking anchors)
    auto is_bypassed = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(device));
    if (bypass_mask.defined() && bypass_mask.numel() == N) {
        is_bypassed = bypass_mask.to(device);
    }
    
    // ========== Protection 2: Scene Edge Bypass ==========
    // Detect edges in GT image — candidates at scene edges bypass EFD
    auto scene_edge_mask = detectSceneEdges(gt, 0.1f);  // [H, W]
    auto is_at_scene_edge = scene_edge_mask.index({v, u});  // [N]
    is_bypassed = is_bypassed | is_at_scene_edge;
    
    int n_bypassed = is_bypassed.sum().item<int>();
    
    // ========== Step 1: EFD Filter (non-bypass only) ==========
    auto error_profile = decomposeRenderingError(rendered, gt, config.blur_kernel);
    auto e_low_at = error_profile.e_low_mag.index({v, u});
    auto e_high_at = error_profile.e_high_mag.index({v, u});
    
    // Adaptive threshold based on median
    float e_high_median = e_high_at.median().item<float>();
    float adaptive_e_high_th = std::max(e_high_median * config.e_high_threshold, 0.001f);
    
    auto efd_pass = (e_high_at > adaptive_e_high_th);
    auto ratio = e_low_at / (e_high_at + 1e-8f);
    efd_pass = efd_pass & (ratio < config.e_low_reject_ratio);
    
    // Bypass candidates always pass EFD
    efd_pass = efd_pass | is_bypassed;
    result.n_efd_rejected = N - efd_pass.sum().item<int>();
    
    // ========== Step 2: MIG Filter ==========
    result.mig_scores = computeMIG(
        candidate_pixels, candidate_colors, candidate_depths,
        rendered, gt, rendered_depth, config);
    
    // Bypass candidates: set MIG to a positive value so they pass
    auto bypassed_mig_fill = torch::where(is_bypassed,
        torch::full({N}, config.high_confidence_mig, torch::TensorOptions().device(device)),
        result.mig_scores);
    
    auto mig_pass = bypassed_mig_fill > config.mig_threshold;
    result.n_mig_rejected = (efd_pass & ~mig_pass).sum().item<int>();
    
    auto pass_mask = efd_pass & mig_pass;
    
    // ========== Step 3: Redundancy Filter ==========
    auto pass_indices = torch::nonzero(pass_mask).squeeze(1);
    
    if (pass_indices.numel() == 0) {
        // ========== Protection 3: Min Density Floor ==========
        // Even if everything is rejected, keep a minimum set from bypassed candidates
        auto bypass_indices = torch::nonzero(is_bypassed).squeeze(1);
        if (bypass_indices.numel() > 0) {
            int n_keep = std::min((int)bypass_indices.numel(), config.budget_per_keyframe);
            auto keep_indices = bypass_indices.slice(0, 0, n_keep);
            result.accepted_points = candidate_points_3d.index({keep_indices});
            result.accepted_colors = candidate_colors.index({keep_indices});
            result.accepted_mask = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(device));
            result.accepted_mask.index_put_({keep_indices}, true);
            result.confidence_levels = torch::ones({n_keep}, torch::TensorOptions().dtype(torch::kInt32).device(device));
            result.n_accepted = n_keep;
        } else {
            result.accepted_points = torch::zeros({0, 3}, torch::TensorOptions().device(device));
            result.accepted_colors = torch::zeros({0, 3}, torch::TensorOptions().device(device));
            result.accepted_mask = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(device));
            result.confidence_levels = torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        }
        return result;
    }
    
    auto pass_pixels = candidate_pixels.index({pass_indices});
    auto pass_mig = bypassed_mig_fill.index({pass_indices});
    
    // Sort by MIG descending
    auto sort_result = pass_mig.sort(0, true);
    auto sorted_indices = std::get<1>(sort_result);
    auto pass_pixels_sorted = pass_pixels.index({sorted_indices});
    auto pass_indices_sorted = pass_indices.index({sorted_indices});
    
    auto redundancy_mask = checkRedundancy2D(
        pass_pixels_sorted, image_width, image_height,
        config.redundancy_grid_cell, config.max_per_cell);
    
    result.n_redundancy_rejected = (~redundancy_mask).sum().item<int>();
    auto after_redund_indices = pass_indices_sorted.index({redundancy_mask});
    
    // ========== Protection 3: Min Density Floor ==========
    // Check for empty cells and fill from rejected candidates
    int floor_cell = 16;  // Larger cell for density floor
    int floor_grid_w = (image_width + floor_cell - 1) / floor_cell;
    int floor_grid_h = (image_height + floor_cell - 1) / floor_cell;
    
    // Mark cells that have accepted candidates
    auto accepted_pixels_cpu = candidate_pixels.index({after_redund_indices}).cpu().to(torch::kInt32);
    std::vector<bool> cell_has_point(floor_grid_w * floor_grid_h, false);
    auto acc_u_data = accepted_pixels_cpu.index({torch::indexing::Slice(), 0}).data_ptr<int32_t>();
    auto acc_v_data = accepted_pixels_cpu.index({torch::indexing::Slice(), 1}).data_ptr<int32_t>();
    for (int i = 0; i < after_redund_indices.numel(); ++i) {
        int cx = std::clamp(acc_u_data[i] / floor_cell, 0, floor_grid_w - 1);
        int cy = std::clamp(acc_v_data[i] / floor_cell, 0, floor_grid_h - 1);
        cell_has_point[cy * floor_grid_w + cx] = true;
    }
    
    // Find rejected candidates that can fill empty cells
    auto rejected_mask = ~pass_mask;
    auto rejected_indices = torch::nonzero(rejected_mask).squeeze(1);
    std::vector<int64_t> floor_fill_indices;
    
    if (rejected_indices.numel() > 0) {
        auto rej_pixels_cpu = candidate_pixels.index({rejected_indices}).cpu().to(torch::kInt32);
        auto rej_u = rej_pixels_cpu.index({torch::indexing::Slice(), 0}).data_ptr<int32_t>();
        auto rej_v = rej_pixels_cpu.index({torch::indexing::Slice(), 1}).data_ptr<int32_t>();
        auto rej_idx_cpu = rejected_indices.cpu();
        auto rej_idx_data = rej_idx_cpu.data_ptr<int64_t>();
        
        for (int i = 0; i < rejected_indices.numel() && (int)floor_fill_indices.size() < config.budget_per_keyframe / 4; ++i) {
            int cx = std::clamp(rej_u[i] / floor_cell, 0, floor_grid_w - 1);
            int cy = std::clamp(rej_v[i] / floor_cell, 0, floor_grid_h - 1);
            int cell_idx = cy * floor_grid_w + cx;
            if (!cell_has_point[cell_idx]) {
                floor_fill_indices.push_back(rej_idx_data[i]);
                cell_has_point[cell_idx] = true;
            }
        }
    }
    
    // Merge accepted + floor-fill
    torch::Tensor all_accepted_indices;
    if (!floor_fill_indices.empty()) {
        auto floor_tensor = torch::tensor(floor_fill_indices, torch::TensorOptions().dtype(torch::kLong).device(device));
        all_accepted_indices = torch::cat({after_redund_indices, floor_tensor});
    } else {
        all_accepted_indices = after_redund_indices;
    }
    
    // ========== Step 4: Budget Cap ==========
    int n_after_all = all_accepted_indices.numel();
    int n_to_keep = std::min(n_after_all, config.budget_per_keyframe);
    result.n_budget_rejected = n_after_all - n_to_keep;
    auto final_indices = all_accepted_indices.slice(0, 0, n_to_keep);
    
    // ========== Step 5: Confidence levels ==========
    auto final_mig = bypassed_mig_fill.index({final_indices});
    auto conf = torch::zeros({n_to_keep}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    conf = torch::where(final_mig >= config.high_confidence_mig,
                        torch::full_like(conf, 2), conf);
    conf = torch::where((final_mig >= config.mid_confidence_mig) & (final_mig < config.high_confidence_mig),
                        torch::full_like(conf, 1), conf);
    
    // ========== Assemble results ==========
    result.accepted_points = candidate_points_3d.index({final_indices});
    result.accepted_colors = candidate_colors.index({final_indices});
    result.confidence_levels = conf;
    result.n_accepted = n_to_keep;
    result.accepted_mask = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(device));
    result.accepted_mask.index_put_({final_indices}, true);
    
    return result;
}


/**
 * Helper: Render from a keyframe to get rendered image + depth for EGD.
 * This is a lightweight forward pass (no gradient computation needed).
 *
 * Returns true if rendering was successful, false if insufficient data.
 */

/**
 * Helper: Compute candidate pixel coordinates and depths from 3D points.
 *
 * @param points_3d_world  [N, 3] world-space points
 * @param Tcw              [4, 4] camera-to-world transform (transposed for torch)
 * @param fx, fy, cx, cy   Camera intrinsics
 * @param image_width, image_height  Image dimensions
 * @return tuple of (pixels [N, 2], depths [N], valid_mask [N] bool)
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
computeCandidatePixelsAndDepths(
    const torch::Tensor& points_3d_world,  // [N, 3]
    const torch::Tensor& Tcw,              // [4, 4] world-to-camera
    float fx, float fy, float cx, float cy,
    int image_width, int image_height)
{
    int N = points_3d_world.size(0);
    auto device = points_3d_world.device();
    
    // Transform to camera coordinates: P_cam = Tcw * P_world
    // Tcw is [4, 4], points are [N, 3]
    auto R = Tcw.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)});  // [3, 3]
    auto t = Tcw.index({torch::indexing::Slice(0, 3), 3});  // [3]
    
    auto points_cam = torch::mm(points_3d_world, R.t()) + t.unsqueeze(0);  // [N, 3]
    
    // Depth = z coordinate in camera frame
    auto depths = points_cam.index({torch::indexing::Slice(), 2});  // [N]
    
    // Project to pixel: u = fx * x/z + cx, v = fy * y/z + cy
    auto x = points_cam.index({torch::indexing::Slice(), 0});
    auto y = points_cam.index({torch::indexing::Slice(), 1});
    auto z = depths.clamp_min(0.001f);
    
    auto u_px = fx * x / z + cx;
    auto v_px = fy * y / z + cy;
    
    auto pixels = torch::stack({u_px, v_px}, 1);  // [N, 2]
    
    // Valid mask: in bounds and positive depth
    auto valid = (u_px >= 0) & (u_px < image_width) &
                 (v_px >= 0) & (v_px < image_height) &
                 (depths > 0.001f);
    
    return {pixels, depths, valid};
}

}  // namespace error_guided
