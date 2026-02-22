/**
 * Wavelet Pyramid for Photo-SLAM
 * 
 * Implements Haar and Daubechies-4 wavelet decomposition for multi-scale
 * image representation. Provides better edge preservation compared to
 * Gaussian pyramid (bilinear downsampling).
 * 
 * Reference: Mallat, S. (1989). A Theory for Multiresolution Signal Decomposition
 */

#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <vector>
#include <cmath>

namespace wavelet {

/**
 * Wavelet decomposition result containing 4 subbands
 */
struct WaveletDecomposition {
    torch::Tensor LL;  ///< Low-Low (approximation) - downsampled smooth version
    torch::Tensor LH;  ///< Low-High (horizontal detail) - horizontal edges
    torch::Tensor HL;  ///< High-Low (vertical detail) - vertical edges  
    torch::Tensor HH;  ///< High-High (diagonal detail) - diagonal edges
    
    int height;  ///< Original height
    int width;   ///< Original width
};

/**
 * Wavelet type enumeration
 */
enum class WaveletType {
    HAAR,      ///< Haar wavelet (simplest, fastest)
    DB4,       ///< Daubechies-4 (better smoothness)
    DB2        ///< Daubechies-2 (balance)
};

/**
 * Configuration for wavelet pyramid
 */
struct WaveletPyramidConfig {
    bool enabled = false;           ///< Use adaptive densification
    bool use_sobel = true;          ///< true=Sobel edge detection, false=Wavelet
    WaveletType wavelet_type = WaveletType::HAAR;
    int num_levels = 3;             ///< Number of decomposition levels
    bool use_high_freq_loss = true; ///< Add high-freq subbands to loss
    float high_freq_weight = 0.1f;  ///< Weight for LH, HL, HH in loss
    
    // Wavelet-guided Gaussian Initialization
    bool init_enabled = false;      ///< Add extra Gaussians at edge regions during init
    float init_edge_threshold = 0.3f; ///< Min edge strength to add extra Gaussians
    int init_max_extra_points = 5000; ///< Max extra points per keyframe
};

/**
 * EFD (Error Frequency Decomposition) configuration
 * Decomposes rendering error into low-freq (geometry) and high-freq (detail)
 * components to guide densification gradient boosting.
 */
struct EFDConfig {
    bool enabled = false;           ///< Enable EFD gradient boosting
    float low_freq_weight = 1.0f;   ///< Weight for low-freq (geometry) error
    float high_freq_weight = 2.0f;  ///< Weight for high-freq (detail) error
    float boost_strength = 3.0f;    ///< Max gradient boost multiplier (1.0 = no boost)
    float min_error_threshold = 0.01f; ///< Min error to apply boost (ignore noise)
};

/**
 * Get wavelet filter coefficients
 * 
 * @param type Wavelet type
 * @return Pair of (low-pass, high-pass) filter coefficients
 */
inline std::pair<std::vector<float>, std::vector<float>> getWaveletFilters(WaveletType type) {
    std::vector<float> lo, hi;
    
    switch (type) {
        case WaveletType::HAAR:
            // Haar wavelet: simplest orthogonal wavelet
            lo = {1.0f / std::sqrt(2.0f), 1.0f / std::sqrt(2.0f)};
            hi = {1.0f / std::sqrt(2.0f), -1.0f / std::sqrt(2.0f)};
            break;
            
        case WaveletType::DB2:
            // Daubechies-2 coefficients
            {
                float c = 1.0f / (4.0f * std::sqrt(2.0f));
                float sq3 = std::sqrt(3.0f);
                lo = {c * (1 + sq3), c * (3 + sq3), c * (3 - sq3), c * (1 - sq3)};
                hi = {c * (1 - sq3), -c * (3 - sq3), c * (3 + sq3), -c * (1 + sq3)};
            }
            break;
            
        case WaveletType::DB4:
            // Daubechies-4 coefficients (8 taps)
            lo = {
                0.32580343f, 1.01094572f, 0.89220014f, -0.03957503f,
                -0.26450717f, 0.0436163f, 0.0465036f, -0.01498699f
            };
            // Normalize
            float sum = 0.0f;
            for (float v : lo) sum += v;
            for (float& v : lo) v /= sum;
            
            // High-pass from low-pass (QMF)
            hi.resize(lo.size());
            for (size_t i = 0; i < lo.size(); ++i) {
                hi[i] = ((i % 2 == 0) ? 1.0f : -1.0f) * lo[lo.size() - 1 - i];
            }
            break;
    }
    
    return {lo, hi};
}

/**
 * Create 1D convolution kernel from filter coefficients
 */
inline torch::Tensor createFilter1D(const std::vector<float>& coeffs, torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    return torch::from_blob((void*)coeffs.data(), {(int64_t)coeffs.size()}, 
                            torch::TensorOptions().dtype(torch::kFloat32))
           .clone().to(device);
}

/**
 * Perform 2D Haar wavelet decomposition (optimized version)
 * 
 * @param image Input image tensor [C, H, W] or [H, W]
 * @return WaveletDecomposition containing LL, LH, HL, HH subbands
 */
inline WaveletDecomposition haarDecompose2D(const torch::Tensor& image) {
    WaveletDecomposition result;
    
    // Handle different input dimensions
    torch::Tensor img = image;
    bool was_2d = false;
    if (img.dim() == 2) {
        img = img.unsqueeze(0);  // [H, W] -> [1, H, W]
        was_2d = true;
    }
    
    int C = img.size(0);
    int H = img.size(1);
    int W = img.size(2);
    
    result.height = H;
    result.width = W;
    
    // Ensure even dimensions for decomposition
    int H2 = H / 2;
    int W2 = W / 2;
    
    // Haar decomposition using averaging and differencing
    // This is more efficient than convolution for Haar
    
    // Reshape for pairwise operations
    auto img_reshaped = img.index({
        torch::indexing::Slice(), 
        torch::indexing::Slice(torch::indexing::None, H2 * 2),
        torch::indexing::Slice(torch::indexing::None, W2 * 2)
    });
    
    // Row-wise decomposition: average and difference of adjacent pixels
    auto even_cols = img_reshaped.index({
        torch::indexing::Slice(), 
        torch::indexing::Slice(),
        torch::indexing::Slice(0, torch::indexing::None, 2)
    });  // [C, H, W/2]
    
    auto odd_cols = img_reshaped.index({
        torch::indexing::Slice(), 
        torch::indexing::Slice(),
        torch::indexing::Slice(1, torch::indexing::None, 2)
    });  // [C, H, W/2]
    
    auto L_rows = (even_cols + odd_cols) * 0.5f;  // Low-pass (average)
    auto H_rows = (even_cols - odd_cols) * 0.5f;  // High-pass (difference)
    
    // Column-wise decomposition on L_rows
    auto L_even_rows = L_rows.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(0, torch::indexing::None, 2),
        torch::indexing::Slice()
    });
    auto L_odd_rows = L_rows.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(1, torch::indexing::None, 2),
        torch::indexing::Slice()
    });
    
    result.LL = (L_even_rows + L_odd_rows) * 0.5f;  // [C, H/2, W/2]
    result.LH = (L_even_rows - L_odd_rows) * 0.5f;  // Horizontal edges
    
    // Column-wise decomposition on H_rows
    auto H_even_rows = H_rows.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(0, torch::indexing::None, 2),
        torch::indexing::Slice()
    });
    auto H_odd_rows = H_rows.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(1, torch::indexing::None, 2),
        torch::indexing::Slice()
    });
    
    result.HL = (H_even_rows + H_odd_rows) * 0.5f;  // Vertical edges
    result.HH = (H_even_rows - H_odd_rows) * 0.5f;  // Diagonal edges
    
    // Squeeze back if input was 2D
    if (was_2d) {
        result.LL = result.LL.squeeze(0);
        result.LH = result.LH.squeeze(0);
        result.HL = result.HL.squeeze(0);
        result.HH = result.HH.squeeze(0);
    }
    
    return result;
}

/**
 * Reconstruct image from Haar wavelet decomposition
 */
inline torch::Tensor haarReconstruct2D(const WaveletDecomposition& decomp) {
    torch::Tensor LL = decomp.LL;
    torch::Tensor LH = decomp.LH;
    torch::Tensor HL = decomp.HL;
    torch::Tensor HH = decomp.HH;
    
    bool was_2d = false;
    if (LL.dim() == 2) {
        LL = LL.unsqueeze(0);
        LH = LH.unsqueeze(0);
        HL = HL.unsqueeze(0);
        HH = HH.unsqueeze(0);
        was_2d = true;
    }
    
    int C = LL.size(0);
    int H2 = LL.size(1);
    int W2 = LL.size(2);
    int H = H2 * 2;
    int W = W2 * 2;
    
    // Inverse transform
    // Reconstruct L_rows and H_rows
    auto L_even_rows = LL + LH;
    auto L_odd_rows = LL - LH;
    auto H_even_rows = HL + HH;
    auto H_odd_rows = HL - HH;
    
    // Interleave rows
    auto L_rows = torch::zeros({C, H, W2}, LL.options());
    L_rows.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2), torch::indexing::Slice()}, L_even_rows);
    L_rows.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2), torch::indexing::Slice()}, L_odd_rows);
    
    auto H_rows = torch::zeros({C, H, W2}, HL.options());
    H_rows.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2), torch::indexing::Slice()}, H_even_rows);
    H_rows.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2), torch::indexing::Slice()}, H_odd_rows);
    
    // Reconstruct columns
    auto even_cols = L_rows + H_rows;
    auto odd_cols = L_rows - H_rows;
    
    // Interleave columns
    auto result = torch::zeros({C, H, W}, LL.options());
    result.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)}, even_cols);
    result.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)}, odd_cols);
    
    if (was_2d) {
        result = result.squeeze(0);
    }
    
    return result;
}

/**
 * Build multi-level wavelet pyramid
 * 
 * @param image Input image tensor [C, H, W]
 * @param num_levels Number of decomposition levels
 * @return Vector of LL subbands at each level (coarse to fine)
 */
inline std::vector<torch::Tensor> buildWaveletPyramid(
    const torch::Tensor& image, 
    int num_levels,
    WaveletType type = WaveletType::HAAR) 
{
    std::vector<torch::Tensor> pyramid;
    torch::Tensor current = image;
    
    for (int l = 0; l < num_levels; ++l) {
        // For now, only Haar is optimized
        auto decomp = haarDecompose2D(current);
        pyramid.push_back(decomp.LL);
        current = decomp.LL;
    }
    
    // Reverse so index 0 = coarsest level
    std::reverse(pyramid.begin(), pyramid.end());
    
    return pyramid;
}

/**
 * Build wavelet pyramid preserving high-frequency details
 * Returns both LL subbands and high-freq components for each level
 */
struct WaveletPyramidLevel {
    torch::Tensor LL;   ///< Low-frequency (approximation)
    torch::Tensor LH;   ///< Horizontal detail
    torch::Tensor HL;   ///< Vertical detail
    torch::Tensor HH;   ///< Diagonal detail
    int orig_height;
    int orig_width;
};

inline std::vector<WaveletPyramidLevel> buildFullWaveletPyramid(
    const torch::Tensor& image,
    int num_levels)
{
    std::vector<WaveletPyramidLevel> pyramid;
    torch::Tensor current = image;
    
    for (int l = 0; l < num_levels; ++l) {
        auto decomp = haarDecompose2D(current);
        
        WaveletPyramidLevel level;
        level.LL = decomp.LL;
        level.LH = decomp.LH;
        level.HL = decomp.HL;
        level.HH = decomp.HH;
        level.orig_height = decomp.height;
        level.orig_width = decomp.width;
        
        pyramid.push_back(level);
        current = decomp.LL;
    }
    
    return pyramid;
}

/**
 * Convert OpenCV Mat to torch Tensor and apply wavelet decomposition
 * This replaces cv::resize for pyramid construction
 * 
 * @param img_gpu OpenCV GPU Mat
 * @param target_height Target height for LL subband  
 * @param target_width Target width for LL subband
 * @param device Torch device
 * @return LL subband tensor at target resolution
 */
inline torch::Tensor waveletDownsample(
    const cv::cuda::GpuMat& img_gpu,
    int target_height,
    int target_width,
    torch::Device device = torch::kCUDA)
{
    // Convert to tensor
    cv::Mat img_cpu;
    img_gpu.download(img_cpu);
    img_cpu.convertTo(img_cpu, CV_32FC3, 1.0f / 255.0f);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto tensor = torch::from_blob(img_cpu.data, 
                                   {img_cpu.rows, img_cpu.cols, img_cpu.channels()}, 
                                   options).clone();
    tensor = tensor.permute({2, 0, 1}).to(device);  // [H, W, C] -> [C, H, W]
    
    // Calculate number of wavelet levels needed
    int current_h = tensor.size(1);
    int current_w = tensor.size(2);
    
    // Iteratively decompose until reaching target size
    while (current_h / 2 >= target_height && current_w / 2 >= target_width) {
        auto decomp = haarDecompose2D(tensor);
        tensor = decomp.LL;
        current_h = tensor.size(1);
        current_w = tensor.size(2);
    }
    
    // Final resize to exact target size if needed
    if (current_h != target_height || current_w != target_width) {
        tensor = torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{target_height, target_width})
                .mode(torch::kBilinear)
                .align_corners(false)
        ).squeeze(0);
    }
    
    return tensor;
}

/**
 * CPU version of wavelet downsample
 */
inline torch::Tensor waveletDownsampleCPU(
    const cv::Mat& img,
    int target_height,
    int target_width,
    torch::Device device = torch::kCPU)
{
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3, 1.0f / 255.0f);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto tensor = torch::from_blob(img_float.data,
                                   {img_float.rows, img_float.cols, img_float.channels()},
                                   options).clone();
    tensor = tensor.permute({2, 0, 1}).to(device);
    
    int current_h = tensor.size(1);
    int current_w = tensor.size(2);
    
    while (current_h / 2 >= target_height && current_w / 2 >= target_width) {
        auto decomp = haarDecompose2D(tensor);
        tensor = decomp.LL;
        current_h = tensor.size(1);
        current_w = tensor.size(2);
    }
    
    if (current_h != target_height || current_w != target_width) {
        tensor = torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{target_height, target_width})
                .mode(torch::kBilinear)
                .align_corners(false)
        ).squeeze(0);
    }
    
    return tensor;
}

/**
 * Compute high-frequency edge loss from wavelet decomposition
 * This encourages rendered images to match ground truth edges
 * 
 * @param rendered Rendered image tensor [C, H, W]
 * @param gt Ground truth image tensor [C, H, W]
 * @return L1 loss on high-frequency subbands
 */
inline torch::Tensor waveletEdgeLoss(
    const torch::Tensor& rendered,
    const torch::Tensor& gt,
    float lh_weight = 0.1f,
    float hl_weight = 0.1f, 
    float hh_weight = 0.05f)
{
    auto decomp_rendered = haarDecompose2D(rendered);
    auto decomp_gt = haarDecompose2D(gt);
    
    // L1 loss on each high-frequency subband
    auto loss_lh = torch::abs(decomp_rendered.LH - decomp_gt.LH).mean() * lh_weight;
    auto loss_hl = torch::abs(decomp_rendered.HL - decomp_gt.HL).mean() * hl_weight;
    auto loss_hh = torch::abs(decomp_rendered.HH - decomp_gt.HH).mean() * hh_weight;
    
    return loss_lh + loss_hl + loss_hh;
}

/**
 * EFD: Compute per-pixel gradient boost weight map from rendering error.
 *
 * Decomposes |rendered - gt| via Haar wavelet:
 *   LL  = low-freq error  (geometry/structure missing)
 *   LH+HL+HH = high-freq error (fine detail missing)
 *
 * Returns a [H, W] weight map in [1.0, boost_strength] where high values
 * indicate pixels where Gaussians should receive boosted gradient accumulation.
 *
 * @param rendered  Rendered image [C, H, W]
 * @param gt        Ground truth image [C, H, W]
 * @param config    EFD configuration
 * @return          Weight map [H, W] on same device as input
 */
inline torch::Tensor efdComputeWeightMap(
    const torch::Tensor& rendered,
    const torch::Tensor& gt,
    const EFDConfig& config)
{
    int H = rendered.size(1);
    int W = rendered.size(2);

    // Error map: |rendered - gt|, averaged over channels → [H, W]
    auto error_map = (rendered - gt).abs().mean(0);  // [H, W]

    // Threshold: ignore noise below min_error_threshold
    auto valid_mask = error_map > config.min_error_threshold;

    // Haar wavelet decomposition of error map
    auto decomp = haarDecompose2D(error_map);  // LL, LH, HL, HH at [H/2, W/2]

    // Low-freq: geometry error (LL subband)
    auto low_freq = decomp.LL.abs();   // [H/2, W/2]

    // High-freq: detail error (LH + HL + HH)
    auto high_freq = decomp.LH.abs() + decomp.HL.abs() + decomp.HH.abs();  // [H/2, W/2]

    // Combine with weights
    auto combined = low_freq * config.low_freq_weight + high_freq * config.high_freq_weight;  // [H/2, W/2]

    // Upsample back to full resolution [H, W]
    auto weight_map = torch::nn::functional::interpolate(
        combined.unsqueeze(0).unsqueeze(0),  // [1, 1, H/2, W/2]
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{H, W})
            .mode(torch::kBilinear)
            .align_corners(false)
    ).squeeze(0).squeeze(0);  // [H, W]

    // Normalize to [0, 1]
    auto w_min = weight_map.min();
    auto w_max = weight_map.max();
    float range = (w_max - w_min).item<float>();
    if (range > 1e-6f) {
        weight_map = (weight_map - w_min) / (range + 1e-6f);
    } else {
        weight_map = torch::zeros_like(weight_map);
    }

    // Apply valid mask (zero out noise regions)
    weight_map = weight_map * valid_mask.to(torch::kFloat32);

    // Scale to [1.0, boost_strength]: 1.0 = no boost, boost_strength = max boost
    weight_map = 1.0f + weight_map * (config.boost_strength - 1.0f);

    return weight_map.contiguous();
}

}  // namespace wavelet
