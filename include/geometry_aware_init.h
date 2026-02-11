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

    float flatten_ratio = 0.3f;          // sz/sx ratio (0.1=very flat, 1.0=sphere)
    float opacity_min = 0.05f;           // Opacity at smooth regions
    float opacity_max = 0.3f;            // Opacity at edge/textured regions
    float edge_opacity_threshold = 0.1f; // Normalized edge strength for max opacity
    int sobel_ksize = 3;                 // Sobel kernel size for gradient computation
};

/**
 * Compute surface normal for each valid pixel from depth map using pinhole model.
 *
 * Given depth D(u,v) and pinhole intrinsics (fx, fy, cx, cy):
 *   X = (u - cx) * D / fx
 *   Y = (v - cy) * D / fy
 *   Z = D
 *
 * The surface normal is computed via cross product of tangent vectors:
 *   dP/du × dP/dv
 *
 * @param depth_map    Depth image as cv::Mat (CV_32FC1), [H x W]
 * @param valid_flags  Boolean tensor [H*W] indicating valid pixels
 * @param fx, fy       Focal lengths
 * @param cx, cy       Principal point
 * @param width        Image width
 * @param height       Image height
 * @param ksize        Sobel kernel size
 * @param device       Torch device
 * @return             Surface normals [N, 3] for valid pixels, in camera frame
 */
inline torch::Tensor computeSurfaceNormals(
    const cv::Mat& depth_map,
    const torch::Tensor& valid_flags,
    float fx, float fy, float cx, float cy,
    int width, int height,
    int ksize,
    torch::DeviceType device)
{
    // Compute depth gradients
    cv::Mat dD_du, dD_dv;
    cv::Sobel(depth_map, dD_du, CV_32F, 1, 0, ksize);
    cv::Sobel(depth_map, dD_dv, CV_32F, 0, 1, ksize);

    // Normalize Sobel output
    float sobel_scale = (ksize == 3) ? 8.0f : 1.0f;
    dD_du /= sobel_scale;
    dD_dv /= sobel_scale;

    // Count valid pixels
    int N = valid_flags.sum().item<int>();
    if (N == 0)
        return torch::zeros({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Compute normals on CPU for valid pixels
    torch::Tensor normals = torch::zeros(
        {N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    auto normals_acc = normals.accessor<float, 2>();
    auto flags_cpu = valid_flags.to(torch::kCPU);
    auto flags_acc = flags_cpu.accessor<bool, 1>();

    int valid_idx = 0;
    for (int v = 0; v < height; ++v) {
        const float* depth_row = depth_map.ptr<float>(v);
        const float* du_row = dD_du.ptr<float>(v);
        const float* dv_row = dD_dv.ptr<float>(v);

        for (int u = 0; u < width; ++u) {
            int flat_idx = v * width + u;
            if (!flags_acc[flat_idx]) continue;

            float D = depth_row[u];
            float ddu = du_row[u];  // ∂D/∂u
            float ddv = dv_row[u];  // ∂D/∂v

            // Tangent vectors (from pinhole projection):
            // dP/du = (D/fx + (u-cx)*ddu/fx, (v-cy)*ddu/fy, ddu)
            // dP/dv = ((u-cx)*ddv/fx, D/fy + (v-cy)*ddv/fy, ddv)
            // Normal = dP/du × dP/dv (simplification for common case):

            // Simplified normal (valid when depth gradients are small):
            float nx = -fx * ddu;
            float ny = -fy * ddv;
            float nz = D;

            // Normalize
            float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len > 1e-8f) {
                normals_acc[valid_idx][0] = nx / len;
                normals_acc[valid_idx][1] = ny / len;
                normals_acc[valid_idx][2] = nz / len;
            } else {
                // Default: pointing towards camera
                normals_acc[valid_idx][0] = 0.0f;
                normals_acc[valid_idx][1] = 0.0f;
                normals_acc[valid_idx][2] = 1.0f;
            }
            ++valid_idx;
        }
    }

    return normals.to(device);
}

/**
 * Convert surface normals to quaternions that align local Z-axis to the normal.
 *
 * Uses Rodrigues' rotation: find rotation from [0,0,1] to normal.
 *   axis = normalize(cross([0,0,1], normal))
 *   angle = acos(dot([0,0,1], normal))
 *   quaternion = [cos(angle/2), axis * sin(angle/2)]
 *
 * @param normals  Surface normals [N, 3], unit vectors
 * @return         Quaternions [N, 4] in (w, x, y, z) format
 */
inline torch::Tensor normalToQuaternion(const torch::Tensor& normals)
{
    int N = normals.size(0);
    torch::Tensor quats = torch::zeros(
        {N, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(normals.device()));

    auto normals_cpu = normals.to(torch::kCPU);
    auto quats_cpu = torch::zeros({N, 4}, torch::TensorOptions().dtype(torch::kFloat32));

    auto n_acc = normals_cpu.accessor<float, 2>();
    auto q_acc = quats_cpu.accessor<float, 2>();

    for (int i = 0; i < N; ++i) {
        float nx = n_acc[i][0];
        float ny = n_acc[i][1];
        float nz = n_acc[i][2];

        // dot([0,0,1], normal) = nz
        float cos_angle = std::clamp(nz, -1.0f, 1.0f);

        if (cos_angle > 0.9999f) {
            // Nearly aligned with Z: identity quaternion
            q_acc[i][0] = 1.0f;
            q_acc[i][1] = 0.0f;
            q_acc[i][2] = 0.0f;
            q_acc[i][3] = 0.0f;
        } else if (cos_angle < -0.9999f) {
            // Opposite to Z: 180-degree rotation around X
            q_acc[i][0] = 0.0f;
            q_acc[i][1] = 1.0f;
            q_acc[i][2] = 0.0f;
            q_acc[i][3] = 0.0f;
        } else {
            // cross([0,0,1], [nx,ny,nz]) = [-ny, nx, 0]
            float ax = -ny;
            float ay = nx;
            // az = 0

            float axis_len = std::sqrt(ax*ax + ay*ay);
            ax /= axis_len;
            ay /= axis_len;

            float half_angle = std::acos(cos_angle) * 0.5f;
            float sin_half = std::sin(half_angle);
            float cos_half = std::cos(half_angle);

            q_acc[i][0] = cos_half;       // w
            q_acc[i][1] = ax * sin_half;   // x
            q_acc[i][2] = ay * sin_half;   // y
            q_acc[i][3] = 0.0f;            // z (axis.z = 0)
        }
    }

    return quats_cpu.to(normals.device());
}

/**
 * Compute texture-aware opacity for each valid pixel based on RGB edge strength.
 *
 * @param rgb_image     RGB image as cv::Mat (CV_32FC3 or CV_8UC3)
 * @param valid_flags   Boolean tensor [H*W]
 * @param width, height Image dimensions
 * @param config        GeoAwareConfig
 * @param device        Torch device
 * @return              Opacity values [N, 1] (pre-sigmoid, i.e. inverse_sigmoid applied)
 */
inline torch::Tensor computeTextureAwareOpacity(
    const cv::Mat& rgb_image,
    const torch::Tensor& valid_flags,
    int width, int height,
    const GeoAwareConfig& config,
    torch::DeviceType device)
{
    // Convert to grayscale
    cv::Mat gray;
    if (rgb_image.type() == CV_32FC3) {
        cv::Mat rgb8;
        rgb_image.convertTo(rgb8, CV_8UC3, 255.0);
        cv::cvtColor(rgb8, gray, cv::COLOR_RGB2GRAY);
    } else {
        cv::cvtColor(rgb_image, gray, cv::COLOR_RGB2GRAY);
    }

    // Compute edge magnitude
    cv::Mat edge_x, edge_y, edge_mag;
    cv::Sobel(gray, edge_x, CV_32F, 1, 0, config.sobel_ksize);
    cv::Sobel(gray, edge_y, CV_32F, 0, 1, config.sobel_ksize);
    cv::magnitude(edge_x, edge_y, edge_mag);

    // Normalize
    double max_edge;
    cv::minMaxLoc(edge_mag, nullptr, &max_edge);
    if (max_edge > 0)
        edge_mag /= static_cast<float>(max_edge);

    // Compute opacity for valid pixels
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
            // Lerp between opacity_min and opacity_max based on edge strength
            float t = std::clamp(edge / config.edge_opacity_threshold, 0.0f, 1.0f);
            float opacity = config.opacity_min + t * (config.opacity_max - config.opacity_min);

            // Apply inverse sigmoid (since GaussianModel stores pre-sigmoid values)
            opacity = std::clamp(opacity, 1e-5f, 1.0f - 1e-5f);
            float inv_sig = std::log(opacity / (1.0f - opacity));
            opa_acc[valid_idx][0] = inv_sig;

            ++valid_idx;
        }
    }

    return opacities.to(device);
}

/**
 * Compute geometry-aware initial parameters for Gaussians.
 *
 * Combines: normal-aligned rotation, anisotropic scaling, texture-aware opacity.
 *
 * @param depth_map        Depth image (CV_32FC1) [H x W]
 * @param rgb_image        RGB image [H x W x 3]  
 * @param valid_flags      Boolean tensor [H*W] indicating valid pixels
 * @param new_point_cloud  3D points [N, 3] (already computed)
 * @param fx, fy, cx, cy   Camera intrinsics
 * @param width, height    Image dimensions
 * @param config           GeoAwareConfig
 * @param device           Torch device
 * @param[out] out_rotations   Quaternions [N, 4]
 * @param[out] out_scales      Log-scales [N, 3]
 * @param[out] out_opacities   Pre-sigmoid opacities [N, 1]
 * @return true on success
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

    // 1. Compute surface normals from depth
    torch::Tensor normals = computeSurfaceNormals(
        depth_map, valid_flags, fx, fy, cx, cy,
        width, height, config.sobel_ksize, device);

    if (normals.size(0) != N) {
        std::cerr << "[GeoAware] Normal count mismatch: " << normals.size(0)
                  << " vs " << N << " points" << std::endl;
        return false;
    }

    // 2. Normal → rotation quaternion
    out_rotations = normalToQuaternion(normals);

    // 3. Anisotropic scaling: keep KNN-based tangent scale, flatten normal direction
    // Note: distCUDA2 will be called by the caller (in increasePcd).
    // Here we just prepare a scale modifier tensor.
    // We'll pass the flatten_ratio and let increasePcd apply it.
    // Actually, better to compute full scales here to avoid modifying distCUDA2 flow.
    // We compute scales = log(sqrt(distCUDA2)) then modify z-component.
    // But distCUDA2 requires the extern function... so let the caller handle base scales.
    // We return a scale_modifier [N, 3] = [1, 1, flatten_ratio]
    torch::Tensor scale_modifier = torch::ones(
        {N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    // Apply log(flatten_ratio) as additive modifier to the z-scale (which is in log-space)
    scale_modifier.index({torch::indexing::Slice(), 2}) = config.flatten_ratio;
    // Store as log-scale modifier: log(flatten_ratio) to be ADDED to log-scale
    out_scales = torch::log(scale_modifier);  // [0, 0, log(flatten_ratio)]

    // 4. Texture-aware opacity
    out_opacities = computeTextureAwareOpacity(
        rgb_image, valid_flags, width, height, config, device);

    std::cout << "[GeoAware] Computed: " << N << " pts | "
              << "flatten=" << config.flatten_ratio
              << " opacity=[" << config.opacity_min << "," << config.opacity_max << "]"
              << std::endl;

    return true;
}

} // namespace geo_aware
