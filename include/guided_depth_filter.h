/**
 * Guided Filter Dense Depth Initialization for 3DGS SLAM
 *
 * For RGBD: Selects semi-dense pixels from full sensor depth using
 * edge-aware sampling guided by RGB gradients + depth validity.
 * Key insight: RGBD sensor already has depth everywhere — no need to
 * interpolate. Instead, select which pixels to init Gaussians at.
 *
 * For Mono (future): Could propagate sparse ORB depth using Normalized
 * Convolution with Guided Filter.
 *
 * References:
 *   - He et al., "Guided Image Filtering", ECCV 2010 (TPAMI 2013)
 *   - Normalized Convolution: Knutsson & Westin, CVPR 1993
 *
 * Part of CG-Photo (Photo-SLAM derivative)
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>

namespace guided_depth {

/**
 * Configuration for Guided Depth Dense Initialization.
 */
struct GuidedDepthConfig
{
    bool enabled = false;

    // Sampling strategy
    int max_points_per_keyframe = 20000;     // Max total points per keyframe (0 = unlimited)
    float edge_sample_ratio = 0.6f;          // Ratio of points from edge regions (vs uniform)

    // Edge detection
    int sobel_ksize = 3;                     // Sobel kernel size for edge detection
    float edge_threshold = 0.1f;             // Normalized edge strength threshold [0,1]

    // Depth quality filtering
    float depth_gradient_threshold = 0.5f;   // Max depth gradient (m/pixel) to reject flying pixels
    float min_valid_depth_ratio = 0.01f;     // Min ratio of valid depth pixels needed

    // Grid-based uniform sampling
    int grid_cell_size = 8;                  // Pixel grid cell size for uniform subsampling
};

/**
 * Select semi-dense pixel locations from full RGBD depth using edge-aware
 * sampling. Instead of using only ORB keypoint locations, this function
 * selects a much larger set of pixels by:
 *
 * 1. Validating ALL sensor depth pixels (range + gradient check)
 * 2. Computing RGB edge strength map (Sobel)
 * 3. Splitting budget: edge_ratio -> edge regions, (1-edge_ratio) -> uniform grid
 * 4. Edge-prioritized sampling: pick pixels with highest edge strength
 * 5. Grid-based uniform sampling: 1 pixel per grid cell for coverage
 * 6. Combine both sets → expanded point_valid_flags
 *
 * @param img_rgb_gpu    RGB image on GPU (CV_32FC3)
 * @param img_depth_gpu  Sensor depth on GPU
 * @param kps_pixel      ORB keypoint pixel coords (guaranteed included)
 * @param image_width    Image width
 * @param image_height   Image height
 * @param min_depth      Min valid depth (meters, after depth_save_scale)
 * @param max_depth      Max valid depth (meters, after depth_save_scale)
 * @param config         Configuration
 * @param device_type    Torch device
 * @return               Boolean valid flags tensor [H*W] on device, empty on failure
 */
inline torch::Tensor selectDenseDepthPixels(
    const cv::cuda::GpuMat& img_rgb_gpu,
    const cv::cuda::GpuMat& img_depth_gpu,
    const std::vector<float>& kps_pixel,
    int image_width,
    int image_height,
    float min_depth,
    float max_depth,
    const GuidedDepthConfig& config,
    torch::DeviceType device_type)
{
    auto start_timing = std::chrono::steady_clock::now();

    int total_pixels = image_height * image_width;

    // --- Step 1: Download images to CPU ---
    cv::Mat img_rgb_cpu, img_depth_cpu;
    img_rgb_gpu.download(img_rgb_cpu);
    img_depth_gpu.download(img_depth_cpu);

    // Ensure depth is float32 single-channel
    cv::Mat depth_float;
    if (img_depth_cpu.type() == CV_16UC1) {
        img_depth_cpu.convertTo(depth_float, CV_32FC1);
    } else if (img_depth_cpu.channels() > 1) {
        std::vector<cv::Mat> channels;
        cv::split(img_depth_cpu, channels);
        depth_float = channels[0].clone();
        if (depth_float.type() != CV_32FC1)
            depth_float.convertTo(depth_float, CV_32FC1);
    } else {
        img_depth_cpu.convertTo(depth_float, CV_32FC1);
    }

    // --- Step 2: Create valid depth mask (range check) ---
    cv::Mat depth_valid = cv::Mat::zeros(image_height, image_width, CV_8UC1);
    int num_valid_depth = 0;
    for (int r = 0; r < image_height; ++r) {
        const float* drow = depth_float.ptr<float>(r);
        uchar* vrow = depth_valid.ptr<uchar>(r);
        for (int c = 0; c < image_width; ++c) {
            if (drow[c] > min_depth && drow[c] < max_depth) {
                vrow[c] = 255;
                ++num_valid_depth;
            }
        }
    }

    float valid_ratio = static_cast<float>(num_valid_depth) / total_pixels;
    if (valid_ratio < config.min_valid_depth_ratio) {
        std::cout << "[Guided Depth] Too few valid depth pixels ("
                  << num_valid_depth << " / " << total_pixels
                  << "), falling back" << std::endl;
        return torch::Tensor();
    }

    // --- Step 3: Depth gradient filtering (remove flying pixels) ---
    if (config.depth_gradient_threshold > 0.0f) {
        cv::Mat grad_x, grad_y, grad_mag;
        cv::Sobel(depth_float, grad_x, CV_32F, 1, 0, config.sobel_ksize);
        cv::Sobel(depth_float, grad_y, CV_32F, 0, 1, config.sobel_ksize);
        cv::magnitude(grad_x, grad_y, grad_mag);
        // Normalize by Sobel scale factor
        float sobel_scale = (config.sobel_ksize == 3) ? 8.0f : 1.0f;
        grad_mag /= sobel_scale;

        for (int r = 0; r < image_height; ++r) {
            const float* grow = grad_mag.ptr<float>(r);
            uchar* vrow = depth_valid.ptr<uchar>(r);
            for (int c = 0; c < image_width; ++c) {
                if (grow[c] > config.depth_gradient_threshold) {
                    vrow[c] = 0;  // Reject flying pixels
                }
            }
        }
    }

    // --- Step 4: Compute RGB edge strength map ---
    cv::Mat gray;
    if (img_rgb_cpu.type() == CV_32FC3) {
        cv::Mat rgb8;
        img_rgb_cpu.convertTo(rgb8, CV_8UC3, 255.0);
        cv::cvtColor(rgb8, gray, cv::COLOR_RGB2GRAY);
    } else {
        cv::cvtColor(img_rgb_cpu, gray, cv::COLOR_RGB2GRAY);
    }

    cv::Mat edge_x, edge_y, edge_mag;
    cv::Sobel(gray, edge_x, CV_32F, 1, 0, config.sobel_ksize);
    cv::Sobel(gray, edge_y, CV_32F, 0, 1, config.sobel_ksize);
    cv::magnitude(edge_x, edge_y, edge_mag);

    // Normalize edge magnitude to [0, 1]
    double max_edge;
    cv::minMaxLoc(edge_mag, nullptr, &max_edge);
    if (max_edge > 0)
        edge_mag /= static_cast<float>(max_edge);

    // --- Step 5: Categorize pixels ---
    // Collect valid pixel indices, split into edge vs non-edge
    std::vector<int> edge_pixels;
    std::vector<int> uniform_pixels;
    edge_pixels.reserve(total_pixels / 4);
    uniform_pixels.reserve(total_pixels / 4);

    for (int r = 0; r < image_height; ++r) {
        const uchar* vrow = depth_valid.ptr<uchar>(r);
        const float* erow = edge_mag.ptr<float>(r);
        for (int c = 0; c < image_width; ++c) {
            if (vrow[c] == 0) continue;
            int idx = r * image_width + c;
            if (erow[c] > config.edge_threshold) {
                edge_pixels.push_back(idx);
            } else {
                uniform_pixels.push_back(idx);
            }
        }
    }

    // --- Step 6: Budget allocation ---
    int total_budget = config.max_points_per_keyframe;
    if (total_budget <= 0)
        total_budget = static_cast<int>(edge_pixels.size() + uniform_pixels.size());

    int edge_budget = static_cast<int>(total_budget * config.edge_sample_ratio);
    int uniform_budget = total_budget - edge_budget;

    // Clamp to available
    edge_budget = std::min(edge_budget, static_cast<int>(edge_pixels.size()));
    uniform_budget = std::min(uniform_budget, static_cast<int>(uniform_pixels.size()));
    // Redistribute leftover
    if (edge_budget < static_cast<int>(total_budget * config.edge_sample_ratio)) {
        uniform_budget = std::min(total_budget - edge_budget,
                                   static_cast<int>(uniform_pixels.size()));
    }
    if (uniform_budget < total_budget - edge_budget) {
        edge_budget = std::min(total_budget - uniform_budget,
                                static_cast<int>(edge_pixels.size()));
    }

    // --- Step 7: Sample from each category ---
    std::mt19937 rng(42);  // Fixed seed for reproducibility

    // For edge pixels: sort by edge strength (descending) and pick top-K
    // This is more deterministic and picks the strongest edges
    if (edge_budget < static_cast<int>(edge_pixels.size())) {
        // Sort by edge magnitude (descending)
        std::sort(edge_pixels.begin(), edge_pixels.end(),
            [&edge_mag, image_width](int a, int b) {
                int ra = a / image_width, ca = a % image_width;
                int rb = b / image_width, cb = b % image_width;
                return edge_mag.at<float>(ra, ca) > edge_mag.at<float>(rb, cb);
            });
        edge_pixels.resize(edge_budget);
    }

    // For uniform pixels: grid-based subsampling for spatial coverage
    if (uniform_budget < static_cast<int>(uniform_pixels.size())) {
        // Grid-based approach: pick one random pixel per grid cell
        int grid_h = (image_height + config.grid_cell_size - 1) / config.grid_cell_size;
        int grid_w = (image_width + config.grid_cell_size - 1) / config.grid_cell_size;
        std::vector<std::vector<int>> grid_cells(grid_h * grid_w);

        for (int idx : uniform_pixels) {
            int r = idx / image_width;
            int c = idx % image_width;
            int gr = r / config.grid_cell_size;
            int gc = c / config.grid_cell_size;
            grid_cells[gr * grid_w + gc].push_back(idx);
        }

        uniform_pixels.clear();
        for (auto& cell : grid_cells) {
            if (cell.empty()) continue;
            // Pick one random pixel from each cell
            std::uniform_int_distribution<int> dist(0, cell.size() - 1);
            uniform_pixels.push_back(cell[dist(rng)]);
        }

        // If still too many, random subsample
        if (static_cast<int>(uniform_pixels.size()) > uniform_budget) {
            std::shuffle(uniform_pixels.begin(), uniform_pixels.end(), rng);
            uniform_pixels.resize(uniform_budget);
        }
    }

    // --- Step 8: Combine and include ORB keypoints ---
    cv::Mat final_mask = cv::Mat::zeros(image_height, image_width, CV_8UC1);

    // Include ORB keypoints (always present for consistency)
    int num_kps = static_cast<int>(kps_pixel.size()) / 2;
    int orb_included = 0;
    for (int kpidx = 0; kpidx < num_kps; ++kpidx) {
        int u = static_cast<int>(kps_pixel[kpidx * 2]);
        int v = static_cast<int>(kps_pixel[kpidx * 2 + 1]);
        if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
            if (depth_valid.at<uchar>(v, u) != 0) {
                final_mask.at<uchar>(v, u) = 255;
                ++orb_included;
            }
        }
    }

    // Include edge samples
    for (int idx : edge_pixels) {
        int r = idx / image_width;
        int c = idx % image_width;
        final_mask.at<uchar>(r, c) = 255;
    }

    // Include uniform samples
    for (int idx : uniform_pixels) {
        int r = idx / image_width;
        int c = idx % image_width;
        final_mask.at<uchar>(r, c) = 255;
    }

    int total_selected = cv::countNonZero(final_mask);

    // --- Step 9: Convert to torch tensor ---
    torch::Tensor flags = torch::zeros(
        {total_pixels},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    auto flags_accessor = flags.accessor<bool, 1>();
    for (int r = 0; r < image_height; ++r) {
        const uchar* mrow = final_mask.ptr<uchar>(r);
        for (int c = 0; c < image_width; ++c) {
            if (mrow[c] != 0) {
                flags_accessor[r * image_width + c] = true;
            }
        }
    }
    flags = flags.to(device_type);

    auto end_timing = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_timing - start_timing).count();

    std::cout << "[Guided Depth] KF: ORB=" << orb_included
              << " edge=" << edge_pixels.size()
              << " uniform=" << uniform_pixels.size()
              << " total=" << total_selected
              << " (" << std::fixed << std::setprecision(1)
              << (100.0f * total_selected / total_pixels) << "% coverage, "
              << elapsed_ms << "ms)" << std::endl;

    return flags;
}

} // namespace guided_depth
