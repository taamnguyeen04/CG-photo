/**
 * Depth Back-Projection Initialization for 3DGS SLAM
 *
 * Two modes of stride-based sampling from GT depth maps:
 *
 * 1. Uniform stride: Fixed stride across entire image.
 *    Simple but effective baseline.
 *
 * 2. Adaptive stride: Block-based edge-aware stride selection.
 *    - Divide image into blocks
 *    - Compute edge strength per block (Sobel on RGB)
 *    - High-edge blocks → small stride (dense, captures detail)
 *    - Low-edge blocks  → large stride (sparse, saves budget)
 *    - Produces content-adaptive spatial resolution
 */

#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>

namespace depth_backproject {

/**
 * Configuration for depth back-projection initialization.
 */
struct DepthBackprojectConfig {
    bool enabled = false;

    // Mode: 0 = uniform stride, 1 = adaptive stride
    int mode = 0;

    // Uniform mode params
    int stride = 4;             // Fixed stride for uniform mode

    // Adaptive mode params
    int stride_min = 2;         // Dense stride for edge regions
    int stride_max = 8;         // Sparse stride for flat regions
    int block_size = 16;        // Block size for edge strength computation
    float edge_threshold = 0.3f; // Edge strength threshold (0-1) for stride transition
    float edge_ratio = 0.7f;     // Budget split: edge_ratio for edges, (1-edge_ratio) for flat

    // Shared params
    int max_points_per_keyframe = 5000; // Budget cap (0 = unlimited)
    float min_depth = 0.001f;   // Min valid depth (meters)
    float max_depth = 10.0f;    // Max valid depth (meters)
};

/**
 * Select pixels using uniform stride.
 */
inline torch::Tensor selectStrideDepthPixels(
    const torch::Tensor& depth,
    int width,
    int height,
    const DepthBackprojectConfig& config,
    torch::DeviceType device_type)
{
    int stride = config.stride;

    auto row_indices = torch::arange(height, torch::kInt32);
    auto col_indices = torch::arange(width, torch::kInt32);

    auto row_mask = (row_indices % stride == 0);
    auto col_mask = (col_indices % stride == 0);

    auto stride_mask = (row_mask.unsqueeze(1) & col_mask.unsqueeze(0))
                        .reshape({-1}).to(device_type);

    auto depth_valid = (depth > config.min_depth) & (depth < config.max_depth);

    return stride_mask & depth_valid;
}

/**
 * Select pixels using adaptive stride based on local edge strength.
 *
 * Algorithm:
 *   1. Compute Sobel edge map from RGB image
 *   2. Divide image into blocks of block_size × block_size
 *   3. Compute max edge strength per block
 *   4. Assign stride per block:
 *      - edge >= threshold → stride_min (dense)
 *      - edge <  threshold → stride_max (sparse)
 *   5. Within each block, sample pixels at assigned stride
 *   6. Filter by valid depth range
 *
 * @param depth          Flattened depth tensor [H*W], float32, on device
 * @param img_rgb_gpu    RGB image on GPU (CV_32FC3)
 * @param width          Image width
 * @param height         Image height
 * @param config         Adaptive stride configuration
 * @param device_type    Torch device
 * @return Boolean mask tensor [H*W] on device
 */
inline torch::Tensor selectAdaptiveStridePixels(
    const torch::Tensor& depth,
    const cv::cuda::GpuMat& img_rgb_gpu,
    int width,
    int height,
    const DepthBackprojectConfig& config,
    torch::DeviceType device_type)
{
    auto start_timing = std::chrono::steady_clock::now();

    int stride_min = config.stride_min;
    int stride_max = config.stride_max;
    int block_size = config.block_size;
    float edge_th = config.edge_threshold;

    // --- Step 1: Compute edge map from RGB ---
    cv::Mat img_rgb_cpu;
    img_rgb_gpu.download(img_rgb_cpu);

    cv::Mat gray;
    if (img_rgb_cpu.type() == CV_32FC3) {
        cv::Mat rgb8;
        img_rgb_cpu.convertTo(rgb8, CV_8UC3, 255.0);
        cv::cvtColor(rgb8, gray, cv::COLOR_RGB2GRAY);
    } else {
        cv::cvtColor(img_rgb_cpu, gray, cv::COLOR_RGB2GRAY);
    }

    cv::Mat edge_x, edge_y, edge_mag;
    cv::Sobel(gray, edge_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, edge_y, CV_32F, 0, 1, 3);
    cv::magnitude(edge_x, edge_y, edge_mag);

    // Normalize to [0, 1]
    double max_edge;
    cv::minMaxLoc(edge_mag, nullptr, &max_edge);
    if (max_edge > 0)
        edge_mag /= static_cast<float>(max_edge);

    // --- Step 2: Compute per-block max edge strength ---
    int grid_h = (height + block_size - 1) / block_size;
    int grid_w = (width + block_size - 1) / block_size;

    // --- Step 3: Build separate edge and flat pixel lists ---
    int total_pixels = height * width;
    std::vector<int> edge_pixels;
    std::vector<int> flat_pixels;
    edge_pixels.reserve(total_pixels / 8);
    flat_pixels.reserve(total_pixels / 16);
    int n_dense_blocks = 0;
    int n_sparse_blocks = 0;

    for (int brow = 0; brow < grid_h; ++brow) {
        for (int bcol = 0; bcol < grid_w; ++bcol) {
            int r_start = brow * block_size;
            int c_start = bcol * block_size;
            int r_end = std::min(r_start + block_size, height);
            int c_end = std::min(c_start + block_size, width);

            // Compute max edge strength in this block
            float block_max_edge = 0.0f;
            for (int r = r_start; r < r_end; ++r) {
                const float* erow = edge_mag.ptr<float>(r);
                for (int c = c_start; c < c_end; ++c) {
                    if (erow[c] > block_max_edge)
                        block_max_edge = erow[c];
                }
            }

            // Classify block and collect pixels at appropriate stride
            bool is_edge_block = (block_max_edge >= edge_th);
            int local_stride = is_edge_block ? stride_min : stride_max;
            auto& target_list = is_edge_block ? edge_pixels : flat_pixels;

            if (is_edge_block) ++n_dense_blocks;
            else ++n_sparse_blocks;

            for (int r = r_start; r < r_end; r += local_stride) {
                for (int c = c_start; c < c_end; c += local_stride) {
                    target_list.push_back(r * width + c);
                }
            }
        }
    }

    // --- Step 4: Priority-based budget allocation ---
    // Edge pixels have priority; flat pixels fill remaining budget
    int budget = config.max_points_per_keyframe;
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

    int edge_keep = static_cast<int>(edge_pixels.size());
    int flat_keep = static_cast<int>(flat_pixels.size());

    if (budget > 0 && (edge_keep + flat_keep) > budget) {
        // Ratio-based allocation: guaranteed coverage for both types
        int edge_budget = static_cast<int>(budget * config.edge_ratio);
        int flat_budget = budget - edge_budget;

        // Clamp to available, redistribute unused
        if (edge_keep <= edge_budget) {
            // Edge has fewer than allocated → give surplus to flat
            flat_budget = budget - edge_keep;
        } else if (flat_keep <= flat_budget) {
            // Flat has fewer than allocated → give surplus to edge
            edge_budget = budget - flat_keep;
        }

        // Subsample if needed
        if (edge_keep > edge_budget) {
            std::shuffle(edge_pixels.begin(), edge_pixels.end(), rng);
            edge_keep = edge_budget;
        }
        if (flat_keep > flat_budget) {
            std::shuffle(flat_pixels.begin(), flat_pixels.end(), rng);
            flat_keep = flat_budget;
        }
    }

    // --- Step 5: Build final mask tensor with depth validation ---
    torch::Tensor flags = torch::zeros(
        {total_pixels},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    auto flags_data = flags.data_ptr<bool>();
    for (int i = 0; i < edge_keep; ++i)
        flags_data[edge_pixels[i]] = true;
    for (int i = 0; i < flat_keep; ++i)
        flags_data[flat_pixels[i]] = true;

    flags = flags.to(device_type);

    // Apply depth validity filter
    auto depth_valid = (depth > config.min_depth) & (depth < config.max_depth);
    flags = flags & depth_valid;

    int total_selected = flags.sum().item<int>();

    auto end_timing = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_timing - start_timing).count();

    std::cout << "[DepthBackproject Adaptive] "
              << "stride=[" << stride_min << "," << stride_max << "]"
              << " | blocks: dense=" << n_dense_blocks
              << " sparse=" << n_sparse_blocks
              << " | edge=" << edge_keep << " flat=" << flat_keep
              << " | selected=" << total_selected << " pts"
              << " (" << std::fixed << std::setprecision(1)
              << (100.0f * total_selected / total_pixels) << "% coverage, "
              << elapsed_ms << "ms)" << std::endl;

    return flags;
}

} // namespace depth_backproject
