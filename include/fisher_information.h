/**
 * Fisher Information-based Uncertainty for 3D Gaussian Splatting SLAM
 * 
 * This file implements principled uncertainty quantification using Fisher Information.
 * Fisher Information measures how much information each Gaussian provides about the scene.
 * 
 * Key idea:
 *   F_i = E[(∂L/∂θ_i)²]  -- Expected squared gradient
 *   High F_i → Gaussian is important for rendering
 *   Low F_i  → Gaussian contributes little (candidate for pruning)
 * 
 * Copyright (C) 2024
 */

#pragma once

#include <torch/torch.h>

namespace fisher_info {

/**
 * @brief Configuration for Fisher Information-based uncertainty
 */
struct FisherConfig
{
    bool enabled = false;               ///< Enable Fisher-based uncertainty
    float prune_threshold = 0.01f;      ///< Prune Gaussians with score < threshold
    int update_interval = 50;           ///< Compute Fisher score every N iterations
    int prune_interval = 500;           ///< Apply pruning every N iterations
    float ema_alpha = 0.9f;             ///< EMA smoothing factor for score accumulation
    float min_observation_count = 10;   ///< Minimum observations before pruning
    
    // Gradient component weights (for combined score)
    float weight_position = 1.0f;       ///< Weight for position gradient
    float weight_scale = 0.5f;          ///< Weight for scale gradient
    float weight_rotation = 0.3f;       ///< Weight for rotation gradient
    float weight_opacity = 0.2f;        ///< Weight for opacity gradient
};

/**
 * @brief Compute per-Gaussian Fisher Information Score
 * 
 * Fisher Information approximated by gradient magnitude:
 *   F_i ≈ ||∂L/∂xyz_i||² + λ_s||∂L/∂scale_i||² + λ_r||∂L/∂rot_i||² + λ_o||∂L/∂opacity_i||²
 * 
 * This function aggregates gradient information from the backward pass
 * to estimate how much each Gaussian contributes to the rendering quality.
 * 
 * @param xyz Position tensor with gradients [N, 3]
 * @param scales Scale tensor with gradients [N, 3]
 * @param rotations Rotation quaternion tensor with gradients [N, 4]
 * @param opacities Opacity tensor with gradients [N, 1]
 * @param visibility_mask Boolean mask of visible Gaussians [N]
 * @param config Fisher configuration parameters
 * @return Per-Gaussian Fisher scores [N]
 */
inline torch::Tensor computeFisherScore(
    const torch::Tensor& xyz,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& visibility_mask,
    const FisherConfig& config = FisherConfig())
{
    const int64_t num_gaussians = xyz.size(0);
    auto options = torch::TensorOptions().device(xyz.device()).dtype(torch::kFloat32);
    auto fisher_score = torch::zeros({num_gaussians}, options);
    
    // Check if gradients are available
    bool has_any_grad = false;
    
    // Position gradient contribution
    if (xyz.grad().defined() && xyz.grad().numel() > 0) {
        // ||∂L/∂xyz||² = sum of squared gradients per Gaussian
        auto position_contribution = torch::sum(xyz.grad() * xyz.grad(), /*dim=*/1);
        fisher_score += config.weight_position * position_contribution;
        has_any_grad = true;
    }
    
    // Scale gradient contribution
    if (scales.grad().defined() && scales.grad().numel() > 0) {
        auto scale_contribution = torch::sum(scales.grad() * scales.grad(), /*dim=*/1);
        fisher_score += config.weight_scale * scale_contribution;
        has_any_grad = true;
    }
    
    // Rotation gradient contribution
    if (rotations.grad().defined() && rotations.grad().numel() > 0) {
        auto rotation_contribution = torch::sum(rotations.grad() * rotations.grad(), /*dim=*/1);
        fisher_score += config.weight_rotation * rotation_contribution;
        has_any_grad = true;
    }
    
    // Opacity gradient contribution
    if (opacities.grad().defined() && opacities.grad().numel() > 0) {
        auto opacity_contribution = opacities.grad().squeeze() * opacities.grad().squeeze();
        fisher_score += config.weight_opacity * opacity_contribution;
        has_any_grad = true;
    }
    
    if (!has_any_grad) {
        // No gradients available - return zeros
        return fisher_score;
    }
    
    // Mask by visibility - only visible Gaussians contribute
    if (visibility_mask.defined() && visibility_mask.numel() == num_gaussians) {
        fisher_score = fisher_score * visibility_mask.to(torch::kFloat32);
    }
    
    // Normalize by maximum to get relative importance [0, 1]
    auto max_score = fisher_score.max();
    if (max_score.item<float>() > 1e-8f) {
        fisher_score = fisher_score / max_score;
    }
    
    return fisher_score;
}

/**
 * @brief Compute information gain from potential Gaussian split
 * 
 * Information gain = expected increase in Fisher Information from splitting
 * Higher gain → splitting this Gaussian would improve reconstruction
 * 
 * @param current_fisher Current Fisher score [N]
 * @param gradient_accum Accumulated viewspace gradients [N]
 * @param scales Current Gaussian scales [N, 3]
 * @param grad_threshold Densification gradient threshold
 * @return Per-Gaussian information gain scores [N]
 */
inline torch::Tensor computeInformationGain(
    const torch::Tensor& current_fisher,
    const torch::Tensor& gradient_accum,
    const torch::Tensor& scales,
    float grad_threshold = 0.0002f)
{
    // Information gain is high when:
    // 1. Current Fisher is moderate (Gaussian is somewhat important)
    // 2. Gradient is high (optimization wants to change it)
    // 3. Scale is large (could benefit from splitting)
    
    // Normalize gradient accumulation
    auto grad_norm = gradient_accum.squeeze();
    auto grad_max = grad_norm.max();
    if (grad_max.item<float>() > 1e-8f) {
        grad_norm = grad_norm / grad_max;
    }
    
    // Scale factor (larger Gaussians have more potential for improvement)
    auto scale_factor = std::get<0>(scales.max(/*dim=*/1));
    auto scale_max = scale_factor.max();
    if (scale_max.item<float>() > 1e-8f) {
        scale_factor = scale_factor / scale_max;
    }
    
    // Information gain = gradient × (1 - fisher) × scale
    // High gradient + low fisher + large scale → high gain
    auto info_gain = grad_norm * (1.0f - current_fisher) * scale_factor;
    
    return info_gain;
}

/**
 * @brief Update Fisher Information scores with exponential moving average
 * 
 * @param accumulated_fisher Existing accumulated Fisher scores [N]
 * @param new_fisher Newly computed Fisher scores [N]
 * @param alpha EMA smoothing factor (higher = more weight on existing)
 * @return Updated Fisher scores [N]
 */
inline torch::Tensor updateFisherEMA(
    const torch::Tensor& accumulated_fisher,
    const torch::Tensor& new_fisher,
    float alpha = 0.9f)
{
    if (accumulated_fisher.size(0) != new_fisher.size(0)) {
        // Size mismatch - handle gracefully
        if (new_fisher.size(0) > accumulated_fisher.size(0)) {
            // More Gaussians now - pad with zeros
            auto padding = torch::zeros({new_fisher.size(0) - accumulated_fisher.size(0)}, 
                                         accumulated_fisher.options());
            auto padded_accum = torch::cat({accumulated_fisher, padding});
            return alpha * padded_accum + (1.0f - alpha) * new_fisher;
        } else {
            // Fewer Gaussians now - truncate
            auto truncated_accum = accumulated_fisher.slice(0, 0, new_fisher.size(0));
            return alpha * truncated_accum + (1.0f - alpha) * new_fisher;
        }
    }
    
    return alpha * accumulated_fisher + (1.0f - alpha) * new_fisher;
}

/**
 * @brief Generate prune mask based on Fisher Information
 * 
 * @param fisher_score Per-Gaussian Fisher scores [N]
 * @param observation_count Per-Gaussian observation counts [N]
 * @param config Fisher configuration
 * @return Boolean prune mask (true = should prune) [N]
 */
inline torch::Tensor generateFisherPruneMask(
    const torch::Tensor& fisher_score,
    const torch::Tensor& observation_count,
    const FisherConfig& config)
{
    // Prune if:
    // 1. Fisher score is below threshold (not contributing)
    // 2. AND observation count is above minimum (had enough chances)
    auto low_fisher_mask = fisher_score < config.prune_threshold;
    auto sufficient_obs_mask = observation_count >= config.min_observation_count;
    
    return low_fisher_mask & sufficient_obs_mask;
}

/**
 * @brief Get densification candidates based on information gain
 * 
 * @param info_gain Per-Gaussian information gain scores [N]
 * @param threshold Minimum gain to trigger densification
 * @return Boolean mask of densification candidates [N]
 */
inline torch::Tensor getDensifyCandidates(
    const torch::Tensor& info_gain,
    float threshold = 0.5f)
{
    return info_gain > threshold;
}

}  // namespace fisher_info
