/**
 * This file is part of Depth-Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Depth-Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This file contains Depth Uncertainty modeling components inspired by CG-SLAM.
 */

#pragma once

#include <torch/torch.h>

namespace depth_uncertainty
{

/**
 * @brief Configuration parameters for depth uncertainty modeling
 * 
 * These parameters control the behavior of:
 * - Uncertainty-based pruning
 * - Geometric loss functions (L_align, L_iso, L_var)
 * - Fisher Information-based densification
 */
struct UncertaintyConfig
{
    // CG-SLAM Paper Parameters
    float uncertainty_tau = 0.025f;           ///< Paper threshold τ for pruning (Eq. 13)
    int prune_interval = 100;                 ///< Prune every N iterations
    float ema_alpha = 0.1f;                   ///< EMA smoothing for uncertainty accumulation
    
    // Legacy compatibility (renamed)
    float uncertainty_threshold = 0.025f;     ///< Alias for uncertainty_tau
    
    // Loss function weights (lambdas)
    float alignment_lambda = 0.1f;           ///< Weight for L_align (median-alpha depth)
    float isotropy_lambda = 0.01f;           ///< Weight for L_iso (scale regularization)
    float variance_lambda = 0.001f;          ///< Weight for L_var (depth variance)
    float depth_lambda = 1.0f;               ///< Weight for L_geo (sensor depth L1)
    
    // Isotropy loss parameters
    float isotropy_epsilon = 1.5f;           ///< Max allowed scale ratio before penalty
    
    // Densification parameters
    float min_uncertainty_for_densify = 0.05f;  ///< Only densify where uncertainty > this
    float max_uncertainty_for_densify = 0.5f;   ///< Don't densify where uncertainty > this
    
    // Memory management
    int max_gaussians = 2000000;             ///< Hard limit on number of Gaussians
    
    // Dynamic object masking
    float dynamic_mask_threshold = 0.3f;     ///< High uncertainty = dynamic object
    bool enable_dynamic_masking = true;
};

/**
 * @brief Per-Gaussian uncertainty data
 * 
 * Tracks the accumulated uncertainty for each Gaussian primitive
 * across multiple observations from different keyframes.
 */
struct GaussianUncertainty
{
    torch::Tensor depth_uncertainty;     ///< v_i: accumulated depth error variance [N]
    torch::Tensor fisher_info_score;     ///< FIM diagonal trace score [N]
    torch::Tensor observation_count;     ///< Number of frames this Gaussian was observed [N]
    torch::Tensor is_stable;             ///< Bool mask: uncertainty < threshold [N]
    torch::Tensor creation_timestamp;    ///< Iteration when Gaussian was created [N]
    
    /**
     * @brief Initialize tensors for N Gaussians
     */
    void initialize(int64_t num_gaussians, torch::DeviceType device = torch::kCUDA)
    {
        auto options = torch::TensorOptions().device(device);
        depth_uncertainty = torch::zeros({num_gaussians}, options);
        fisher_info_score = torch::zeros({num_gaussians}, options);
        observation_count = torch::zeros({num_gaussians}, options.dtype(torch::kInt32));
        is_stable = torch::ones({num_gaussians}, options.dtype(torch::kBool));
        creation_timestamp = torch::zeros({num_gaussians}, options.dtype(torch::kInt32));
    }
    
    /**
     * @brief Extend tensors when new Gaussians are added
     */
    void extend(int64_t new_count, int current_iteration, torch::DeviceType device = torch::kCUDA)
    {
        auto options = torch::TensorOptions().device(device);
        depth_uncertainty = torch::cat({depth_uncertainty, torch::zeros({new_count}, options)});
        fisher_info_score = torch::cat({fisher_info_score, torch::zeros({new_count}, options)});
        observation_count = torch::cat({observation_count, torch::zeros({new_count}, options.dtype(torch::kInt32))});
        is_stable = torch::cat({is_stable, torch::ones({new_count}, options.dtype(torch::kBool))});
        creation_timestamp = torch::cat({creation_timestamp, 
            torch::full({new_count}, current_iteration, options.dtype(torch::kInt32))});
    }
    
    /**
     * @brief Prune Gaussians by mask (keep where mask is false)
     */
    void prune(const torch::Tensor& prune_mask)
    {
        auto keep_mask = ~prune_mask;
        depth_uncertainty = depth_uncertainty.index({keep_mask});
        fisher_info_score = fisher_info_score.index({keep_mask});
        observation_count = observation_count.index({keep_mask});
        is_stable = is_stable.index({keep_mask});
        creation_timestamp = creation_timestamp.index({keep_mask});
    }
    
    /**
     * @brief Update stability flags based on current uncertainty values
     */
    void updateStability(float threshold)
    {
        is_stable = depth_uncertainty < threshold;
    }
    
    /**
     * @brief Get number of Gaussians
     */
    int64_t size() const
    {
        return depth_uncertainty.size(0);
    }
};

/**
 * @brief Per-pixel depth render outputs
 * 
 * Extended rasterization outputs for uncertainty computation
 */
struct DepthRenderOutput
{
    torch::Tensor alpha_depth;       ///< Expected depth via alpha-blending [H, W]
    torch::Tensor alpha_depth_sq;    ///< Expected depth^2 for variance [H, W]
    torch::Tensor median_depth;      ///< Depth at T=0.5 transmittance [H, W]
    torch::Tensor depth_variance;    ///< U_pix = E[d^2] - E[d]^2 [H, W]
    
    /**
     * @brief Compute depth variance from alpha-blending outputs
     * U_pix = E[d^2] - E[d]^2 = alpha_depth_sq - alpha_depth^2
     */
    void computeVariance()
    {
        depth_variance = alpha_depth_sq - torch::pow(alpha_depth, 2);
        depth_variance = torch::clamp_min(depth_variance, 0.0f);  // Numerical stability
    }
};

} // namespace depth_uncertainty
