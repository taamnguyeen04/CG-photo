/**
 * Geometry-Aware Gaussian Initialization
 *
 * Computes better initial rotation, scale, and opacity for 3D Gaussians
 * based on local surface geometry and image texture.
 *
 * Instead of default identity rotation / isotropic scale / uniform opacity:
 *   - Rotation: aligned to estimated surface normal (from depth gradient)
 *   - Scale: anisotropic flat disk (thin along surface normal)
 *   - Opacity: higher at textured/edge regions, lower at smooth regions
 *
 * v2: Adaptive confidence — noisy normals (high depth gradient) fall back
 *     to identity rotation and isotropic scale to avoid hurting complex scenes.
 *
 * Part of CG-Photo (Photo-SLAM derivative)
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <iostream>
#include <cmath>
#include <algorithm>

namespace geo_aware {

/**
 * Configuration for Geometry-Aware Gaussian Initialization
 */
struct GeoAwareConfig
{
    bool enabled = false;

    float flatten_ratio = 0.3f;              // sz/sx ratio (0.1=very flat, 1.0=sphere)
    float opacity_min = 0.05f;               // Opacity at smooth regions
    float opacity_max = 0.3f;                // Opacity at edge/textured regions
    float edge_opacity_threshold = 0.1f;     // Normalized edge strength for max opacity
    int sobel_ksize = 3;                     // Sobel kernel size for gradient computation
    float normal_confidence_threshold = 0.3f; // Max depth gradient magnitude for confident normal
};

/**
 * Compute surface normals AND per-pixel confidence for valid pixels.
 *
 * Confidence = 1.0 if depth gradient is small (smooth planar surface)
 *            = 0.0 if depth gradient > threshold (noisy / complex geometry)
 *
 * @param[out] out_normals     Surface normals [N, 3]
 * @param[out] out_confidence  Per-pixel confidence [N] in [0, 1]
 * @return number of valid pixels
 */
inline int computeSurfaceNormalsWithConfidence(
    const cv::Mat& depth_map,
    const torch::Tensor& valid_flags,
    float fx, float fy, float cx, float cy,
    int width, int height,
    int ksize,
    float confidence_threshold,
    torch::DeviceType device,
    torch::Tensor& out_normals,
    torch::Tensor& out_confidence)
{
    // Compute depth gradients
    cv::Mat dD_du, dD_dv;
    cv::Sobel(depth_map, dD_du, CV_32F, 1, 0, ksize);
    cv::Sobel(depth_map, dD_dv, CV_32F, 0, 1, ksize);

    // Normalize Sobel output
    float sobel_scale = (ksize == 3) ? 8.0f : 1.0f;
    dD_du /= sobel_scale;
    dD_dv /= sobel_scale;

    int N = valid_flags.sum().item<int>();
    if (N == 0) {
        out_normals = torch::zeros({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        out_confidence = torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        return 0;
    }

    // Compute normals and confidence on CPU
    out_normals = torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    out_confidence = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    auto n_acc = out_normals.accessor<float, 2>();
    auto c_acc = out_confidence.accessor<float, 1>();
    auto flags_cpu = valid_flags.to(torch::kCPU);
    auto flags_acc = flags_cpu.accessor<bool, 1>();

    int valid_idx = 0;
    int confident_count = 0;
    for (int v = 0; v < height; ++v) {
        const float* depth_row = depth_map.ptr<float>(v);
        const float* du_row = dD_du.ptr<float>(v);
        const float* dv_row = dD_dv.ptr<float>(v);

        for (int u = 0; u < width; ++u) {
            int flat_idx = v * width + u;
            if (!flags_acc[flat_idx]) continue;

            float D = depth_row[u];
            float ddu = du_row[u];
            float ddv = dv_row[u];

            // Depth gradient magnitude (normalized by depth to be scale-invariant)
            float grad_mag = std::sqrt(ddu*ddu + ddv*ddv);
            float relative_grad = (D > 0.01f) ? grad_mag / D : 1.0f;

            // Confidence: smooth falloff from 1.0 (flat) to 0.0 (noisy)
            float conf = std::clamp(1.0f - relative_grad / confidence_threshold, 0.0f, 1.0f);
            c_acc[valid_idx] = conf;
            if (conf > 0.1f) ++confident_count;

            // Surface normal
            float nx = -fx * ddu;
            float ny = -fy * ddv;
            float nz = D;

            float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len > 1e-8f) {
                n_acc[valid_idx][0] = nx / len;
                n_acc[valid_idx][1] = ny / len;
                n_acc[valid_idx][2] = nz / len;
            } else {
                n_acc[valid_idx][0] = 0.0f;
                n_acc[valid_idx][1] = 0.0f;
                n_acc[valid_idx][2] = 1.0f;
                c_acc[valid_idx] = 0.0f;  // no confidence for degenerate normal
            }
            ++valid_idx;
        }
    }

    out_normals = out_normals.to(device);
    out_confidence = out_confidence.to(device);
    return confident_count;
}

/**
 * Convert surface normals to quaternions, blending with identity based on confidence.
 *
 * confidence=1: full normal-aligned rotation
 * confidence=0: identity rotation [1,0,0,0]
 *
 * Uses SLERP: q_out = slerp(q_identity, q_normal, confidence)
 */
inline torch::Tensor normalToQuaternionAdaptive(
    const torch::Tensor& normals,
    const torch::Tensor& confidence)
{
    int N = normals.size(0);

    auto normals_cpu = normals.to(torch::kCPU);
    auto conf_cpu = confidence.to(torch::kCPU);
    auto quats_cpu = torch::zeros({N, 4}, torch::TensorOptions().dtype(torch::kFloat32));

    auto n_acc = normals_cpu.accessor<float, 2>();
    auto c_acc = conf_cpu.accessor<float, 1>();
    auto q_acc = quats_cpu.accessor<float, 2>();

    for (int i = 0; i < N; ++i) {
        float conf = c_acc[i];

        // Low confidence → identity quaternion
        if (conf < 0.05f) {
            q_acc[i][0] = 1.0f;
            q_acc[i][1] = 0.0f;
            q_acc[i][2] = 0.0f;
            q_acc[i][3] = 0.0f;
            continue;
        }

        float nx = n_acc[i][0];
        float ny = n_acc[i][1];
        float nz = n_acc[i][2];

        float cos_angle = std::clamp(nz, -1.0f, 1.0f);

        float qw, qx, qy, qz;
        if (cos_angle > 0.9999f) {
            qw = 1.0f; qx = 0.0f; qy = 0.0f; qz = 0.0f;
        } else if (cos_angle < -0.9999f) {
            qw = 0.0f; qx = 1.0f; qy = 0.0f; qz = 0.0f;
        } else {
            float ax = -ny;
            float ay = nx;
            float axis_len = std::sqrt(ax*ax + ay*ay);
            ax /= axis_len;
            ay /= axis_len;

            // Scale angle by confidence: partial rotation
            float full_angle = std::acos(cos_angle);
            float scaled_angle = full_angle * conf;
            float half = scaled_angle * 0.5f;

            qw = std::cos(half);
            qx = ax * std::sin(half);
            qy = ay * std::sin(half);
            qz = 0.0f;
        }

        q_acc[i][0] = qw;
        q_acc[i][1] = qx;
        q_acc[i][2] = qy;
        q_acc[i][3] = qz;
    }

    return quats_cpu.to(normals.device());
}

/**
 * Compute texture-aware opacity for each valid pixel based on RGB edge strength.
 */
inline torch::Tensor computeTextureAwareOpacity(
    const cv::Mat& rgb_image,
    const torch::Tensor& valid_flags,
    int width, int height,
    const GeoAwareConfig& config,
    torch::DeviceType device)
{
    cv::Mat gray;
    if (rgb_image.type() == CV_32FC3) {
        cv::Mat rgb8;
        rgb_image.convertTo(rgb8, CV_8UC3, 255.0);
        cv::cvtColor(rgb8, gray, cv::COLOR_RGB2GRAY);
    } else {
        cv::cvtColor(rgb_image, gray, cv::COLOR_RGB2GRAY);
    }

    cv::Mat edge_x, edge_y, edge_mag;
    cv::Sobel(gray, edge_x, CV_32F, 1, 0, config.sobel_ksize);
    cv::Sobel(gray, edge_y, CV_32F, 0, 1, config.sobel_ksize);
    cv::magnitude(edge_x, edge_y, edge_mag);

    double max_edge;
    cv::minMaxLoc(edge_mag, nullptr, &max_edge);
    if (max_edge > 0)
        edge_mag /= static_cast<float>(max_edge);

    int N = valid_flags.sum().item<int>();
    torch::Tensor opacities = torch::zeros(
        {N, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    auto flags_cpu = valid_flags.to(torch::kCPU);
    auto flags_acc = flags_cpu.accessor<bool, 1>();
    auto opa_acc = opacities.accessor<float, 2>();

    int valid_idx = 0;
    for (int v = 0; v < height; ++v) {
        const float* erow = edge_mag.ptr<float>(v);
        for (int u = 0; u < width; ++u) {
            if (!flags_acc[v * width + u]) continue;

            float edge = erow[u];
            float t = std::clamp(edge / config.edge_opacity_threshold, 0.0f, 1.0f);
            float opacity = config.opacity_min + t * (config.opacity_max - config.opacity_min);

            opacity = std::clamp(opacity, 1e-5f, 1.0f - 1e-5f);
            float inv_sig = std::log(opacity / (1.0f - opacity));
            opa_acc[valid_idx][0] = inv_sig;
            ++valid_idx;
        }
    }

    return opacities.to(device);
}

/**
 * Compute geometry-aware initial parameters for Gaussians (v2: adaptive).
 *
 * Key difference from v1: uses per-pixel normal confidence to blend between
 * geometry-aware init and default init, avoiding noisy normals on complex geometry.
 */
inline bool computeGeoAwareParams(
    const cv::Mat& depth_map,
    const cv::Mat& rgb_image,
    const torch::Tensor& valid_flags,
    const torch::Tensor& new_point_cloud,
    float fx, float fy, float cx, float cy,
    int width, int height,
    const GeoAwareConfig& config,
    torch::DeviceType device,
    torch::Tensor& out_rotations,
    torch::Tensor& out_scales,
    torch::Tensor& out_opacities)
{
    int N = new_point_cloud.size(0);
    if (N == 0) return false;

    // 1. Compute surface normals WITH confidence
    torch::Tensor normals, confidence;
    int confident_count = computeSurfaceNormalsWithConfidence(
        depth_map, valid_flags, fx, fy, cx, cy,
        width, height, config.sobel_ksize,
        config.normal_confidence_threshold,
        device, normals, confidence);

    if (normals.size(0) != N) {
        std::cerr << "[GeoAware] Normal count mismatch: " << normals.size(0)
                  << " vs " << N << " points" << std::endl;
        return false;
    }

    // 2. Adaptive rotation: blend with identity based on confidence
    out_rotations = normalToQuaternionAdaptive(normals, confidence);

    // 3. Adaptive anisotropic scaling: flatten only where confident
    // scale_mod[i] = [0, 0, log(lerp(1.0, flatten_ratio, confidence[i]))]
    // confidence=1 → log(flatten_ratio), confidence=0 → log(1.0)=0
    {
        auto conf_cpu = confidence.to(torch::kCPU);
        torch::Tensor scale_mod = torch::zeros(
            {N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto s_acc = scale_mod.accessor<float, 2>();
        auto c_acc = conf_cpu.accessor<float, 1>();

        for (int i = 0; i < N; ++i) {
            float c = c_acc[i];
            // Lerp between 1.0 (isotropic) and flatten_ratio
            float ratio = 1.0f - c * (1.0f - config.flatten_ratio);
            s_acc[i][2] = std::log(ratio);  // only modify z
        }
        out_scales = scale_mod.to(device);
    }

    // 4. Texture-aware opacity (unchanged — always beneficial)
    out_opacities = computeTextureAwareOpacity(
        rgb_image, valid_flags, width, height, config, device);

    float conf_pct = (N > 0) ? 100.0f * confident_count / N : 0.0f;
    std::cout << "[GeoAware] " << N << " pts | confident=" 
              << confident_count << " (" << std::fixed << std::setprecision(0) 
              << conf_pct << std::defaultfloat << std::setprecision(6)
              << "%) | flatten=" << config.flatten_ratio
              << " opacity=[" << config.opacity_min << "," << config.opacity_max << "]"
              << std::endl;

    return true;
}

} // namespace geo_aware
