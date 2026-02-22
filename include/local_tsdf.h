/**
 * Local Hash-TSDF Volume for Molding-GS
 * 
 * Lightweight Truncated Signed Distance Field that acts as a geometric "mold"
 * for anchoring 3D Gaussians to true surfaces. Uses CPU spatial hashing for
 * fast insert/delete and batched GPU queries via PyTorch tensors.
 * 
 * Key features:
 *   - Spatial hashing: O(1) voxel access, memory-efficient (only allocate observed regions)
 *   - Trilinear interpolation: smooth SDF & gradient queries (no blocky artifacts)
 *   - Sliding window: evicts distant voxels to bound memory usage
 *   - Batched GPU query: querySDF/queryGradient return CUDA tensors for loss computation
 * 
 * Reference:
 *   Curless & Levoy, "A Volumetric Method for Building Complex Models from Range Images", 1996
 *   Niessner et al., "Real-time 3D Reconstruction at Scale using Voxel Hashing", TOG 2013
 */

#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <mutex>
#include <Eigen/Core>

namespace local_tsdf {

// ============================================================================
// Voxel data structures
// ============================================================================

struct VoxelKey {
    int x, y, z;
    
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const {
        // Large primes for spatial hashing (Niessner et al. 2013)
        return static_cast<std::size_t>(
            k.x * 73856093 ^ k.y * 19349669 ^ k.z * 83492791);
    }
};

struct Voxel {
    float sdf = 0.0f;      // Truncated signed distance
    float weight = 0.0f;    // Observation count (running average weight)
    // float color[3] = {0};   // Optional: running average color (not used yet)
};

// ============================================================================
// Configuration
// ============================================================================

struct TSDFConfig {
    bool enabled = false;
    float voxel_size = 0.02f;           // Voxel resolution in meters (2cm)
    float truncation = 0.06f;           // TSDF truncation distance (3× voxel_size)
    float max_radius = 3.0f;            // Local volume radius around camera (meters)
    float min_weight = 3.0f;            // Min observation weight to trust SDF
    float lambda_sdf = 0.1f;            // Weight for SDF anchor loss
    float lambda_normal = 0.05f;        // Weight for normal alignment loss
    float strain_threshold = 0.02f;     // Min strain for physics-based split (meters)
    float prune_sdf_threshold = 0.05f;  // Prune Gaussians with |SDF| > this (meters)
    int prune_interval = 200;           // SDF-based pruning every N iterations
    int fusion_interval = 1;            // Fuse depth every N keyframes
};

// ============================================================================
// Local Hash-TSDF Volume
// ============================================================================

class LocalTSDF {
public:
    LocalTSDF(float voxel_size, float truncation)
        : voxel_size_(voxel_size)
        , truncation_(truncation)
        , inv_voxel_size_(1.0f / voxel_size)
    {}

    // ========================================================================
    // Depth Fusion: integrate one depth frame into the TSDF
    // ========================================================================
    
    /**
     * @brief Integrate a depth frame into the local TSDF volume.
     * 
     * For each pixel with valid depth, compute the 3D point in world space,
     * then update all voxels along the viewing ray within the truncation band.
     * 
     * @param depth_cpu  Depth image (CV_32FC1, meters) on CPU
     * @param Tcw        Camera-to-world pose (4×4 Eigen matrix, world frame)
     * @param fx, fy     Focal lengths
     * @param cx, cy     Principal point
     * @param min_depth  Min valid depth
     * @param max_depth  Max valid depth
     */
    void integrateDepthFrame(
        const cv::Mat& depth_cpu,
        const Eigen::Matrix4f& Twc,
        float fx, float fy, float cx, float cy,
        float min_depth = 0.001f, float max_depth = 10.0f)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        int height = depth_cpu.rows;
        int width = depth_cpu.cols;
        
        // Camera position in world frame
        Eigen::Vector3f cam_pos = Twc.block<3,1>(0, 3);
        
        // Rotation matrix (world ← camera)
        Eigen::Matrix3f Rwc = Twc.block<3,3>(0, 0);
        
        // Stride for efficiency: don't process every pixel
        // Process every 2nd pixel for speed (still dense enough for 2cm voxels)
        const int stride = 2;
        
        for (int v = 0; v < height; v += stride) {
            const float* depth_row = depth_cpu.ptr<float>(v);
            for (int u = 0; u < width; u += stride) {
                float d = depth_row[u];
                if (d < min_depth || d > max_depth || !std::isfinite(d))
                    continue;
                
                // Unproject pixel to camera frame
                Eigen::Vector3f p_cam(
                    (u - cx) * d / fx,
                    (v - cy) * d / fy,
                    d);
                
                // Transform to world frame
                Eigen::Vector3f p_world = Rwc * p_cam + cam_pos;
                
                // Ray direction (normalized) from camera to point
                Eigen::Vector3f ray_dir = (p_world - cam_pos).normalized();
                
                // Update voxels along the ray within truncation band
                // Sample from (d - truncation) to (d + truncation)
                float t_start = std::max(d - truncation_, min_depth);
                float t_end = d + truncation_;
                
                // Step size = half voxel for adequate sampling
                float step = voxel_size_ * 0.5f;
                
                for (float t = t_start; t <= t_end; t += step) {
                    Eigen::Vector3f voxel_pos = cam_pos + ray_dir * t;
                    
                    // Signed distance: positive = in front of surface, negative = behind
                    float sdf = d - t;
                    
                    // Truncate
                    sdf = std::max(-truncation_, std::min(truncation_, sdf));
                    
                    // Get voxel key
                    VoxelKey key = worldToVoxel(voxel_pos);
                    
                    // Weighted running average update
                    auto& voxel = voxels_[key];
                    float old_w = voxel.weight;
                    float new_w = old_w + 1.0f;
                    voxel.sdf = (voxel.sdf * old_w + sdf) / new_w;
                    voxel.weight = std::min(new_w, 50.0f);  // Cap weight to avoid overflow
                }
            }
        }
        
        num_integrations_++;
    }
    
    // ========================================================================
    // Sliding Window: evict distant voxels
    // ========================================================================
    
    /**
     * @brief Remove voxels that are too far from the current camera position.
     * 
     * @param camera_pos  Current camera position in world frame
     * @param max_radius  Maximum distance to keep (meters)
     */
    void evictDistant(const Eigen::Vector3f& camera_pos, float max_radius) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        float max_radius_sq = max_radius * max_radius;
        
        auto it = voxels_.begin();
        while (it != voxels_.end()) {
            Eigen::Vector3f voxel_center = voxelToWorld(it->first);
            float dist_sq = (voxel_center - camera_pos).squaredNorm();
            
            if (dist_sq > max_radius_sq) {
                it = voxels_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    // ========================================================================
    // Batched GPU Queries (return PyTorch CUDA tensors)
    // ========================================================================
    
    /**
     * @brief Query SDF values at Gaussian center positions.
     *        Uses trilinear interpolation for smooth values.
     * 
     * @param xyz  [N, 3] Gaussian positions (CUDA tensor)
     * @return     [N] SDF values (CUDA tensor, differentiable-friendly)
     */
    torch::Tensor querySDF(const torch::Tensor& xyz) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        int64_t N = xyz.size(0);
        auto xyz_cpu = xyz.detach().cpu().contiguous();
        auto xyz_accessor = xyz_cpu.accessor<float, 2>();
        
        auto sdf_out = torch::zeros({N}, torch::kFloat32);
        auto sdf_accessor = sdf_out.accessor<float, 1>();
        
        for (int64_t i = 0; i < N; ++i) {
            float x = xyz_accessor[i][0];
            float y = xyz_accessor[i][1];
            float z = xyz_accessor[i][2];
            sdf_accessor[i] = trilinearQuery(x, y, z);
        }
        
        return sdf_out.to(xyz.device());
    }
    
    /**
     * @brief Query surface normals (∇TSDF) at Gaussian center positions.
     *        Uses central difference on trilinear-interpolated SDF.
     * 
     * @param xyz  [N, 3] Gaussian positions (CUDA tensor)
     * @return     [N, 3] Surface normals (CUDA tensor), normalized
     */
    torch::Tensor queryGradient(const torch::Tensor& xyz) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        int64_t N = xyz.size(0);
        auto xyz_cpu = xyz.detach().cpu().contiguous();
        auto xyz_accessor = xyz_cpu.accessor<float, 2>();
        
        auto grad_out = torch::zeros({N, 3}, torch::kFloat32);
        auto grad_accessor = grad_out.accessor<float, 2>();
        
        float eps = voxel_size_ * 0.5f;  // Half-voxel offset for central difference
        
        for (int64_t i = 0; i < N; ++i) {
            float x = xyz_accessor[i][0];
            float y = xyz_accessor[i][1];
            float z = xyz_accessor[i][2];
            
            // Central difference: ∂SDF/∂x ≈ (SDF(x+ε) - SDF(x-ε)) / (2ε)
            float gx = (trilinearQuery(x + eps, y, z) - trilinearQuery(x - eps, y, z)) / (2.0f * eps);
            float gy = (trilinearQuery(x, y + eps, z) - trilinearQuery(x, y - eps, z)) / (2.0f * eps);
            float gz = (trilinearQuery(x, y, z + eps) - trilinearQuery(x, y, z - eps)) / (2.0f * eps);
            
            // Normalize
            float norm = std::sqrt(gx*gx + gy*gy + gz*gz);
            if (norm > 1e-8f) {
                grad_accessor[i][0] = gx / norm;
                grad_accessor[i][1] = gy / norm;
                grad_accessor[i][2] = gz / norm;
            }
            // else: zero normal (unobserved region)
        }
        
        return grad_out.to(xyz.device());
    }
    
    /**
     * @brief Query observation weights at Gaussian center positions.
     * 
     * @param xyz  [N, 3] Gaussian positions (CUDA tensor)
     * @return     [N] Weights (CUDA tensor)
     */
    torch::Tensor queryWeight(const torch::Tensor& xyz) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        int64_t N = xyz.size(0);
        auto xyz_cpu = xyz.detach().cpu().contiguous();
        auto xyz_accessor = xyz_cpu.accessor<float, 2>();
        
        auto weight_out = torch::zeros({N}, torch::kFloat32);
        auto weight_accessor = weight_out.accessor<float, 1>();
        
        for (int64_t i = 0; i < N; ++i) {
            float x = xyz_accessor[i][0];
            float y = xyz_accessor[i][1];
            float z = xyz_accessor[i][2];
            weight_accessor[i] = trilinearWeightQuery(x, y, z);
        }
        
        return weight_out.to(xyz.device());
    }
    
    // ========================================================================
    // Statistics
    // ========================================================================
    
    size_t numVoxels() const { return voxels_.size(); }
    int numIntegrations() const { return num_integrations_; }
    
    /**
     * @brief Estimate VRAM usage in bytes.
     */
    size_t memoryUsageBytes() const {
        // Each voxel: key(12 bytes) + value(8 bytes) + hash overhead (~40 bytes)
        return voxels_.size() * 60;
    }

private:
    // ========================================================================
    // Coordinate conversion
    // ========================================================================
    
    VoxelKey worldToVoxel(const Eigen::Vector3f& pos) const {
        return {
            static_cast<int>(std::floor(pos.x() * inv_voxel_size_)),
            static_cast<int>(std::floor(pos.y() * inv_voxel_size_)),
            static_cast<int>(std::floor(pos.z() * inv_voxel_size_))
        };
    }
    
    Eigen::Vector3f voxelToWorld(const VoxelKey& key) const {
        return Eigen::Vector3f(
            (key.x + 0.5f) * voxel_size_,
            (key.y + 0.5f) * voxel_size_,
            (key.z + 0.5f) * voxel_size_);
    }
    
    // ========================================================================
    // Trilinear interpolation
    // ========================================================================
    
    /**
     * @brief Get SDF value at a voxel key. Returns 0 if not observed.
     */
    float getVoxelSDF(int ix, int iy, int iz) const {
        VoxelKey key{ix, iy, iz};
        auto it = voxels_.find(key);
        if (it != voxels_.end() && it->second.weight > 0.0f) {
            return it->second.sdf;
        }
        return 0.0f;  // Unknown region: assume zero (on surface) — safe default
    }
    
    /**
     * @brief Get weight at a voxel key. Returns 0 if not observed.
     */
    float getVoxelWeight(int ix, int iy, int iz) const {
        VoxelKey key{ix, iy, iz};
        auto it = voxels_.find(key);
        if (it != voxels_.end()) {
            return it->second.weight;
        }
        return 0.0f;
    }
    
    /**
     * @brief Trilinear interpolation of SDF at continuous world coordinate.
     * 
     * Smooth interpolation between the 8 neighboring voxels.
     * This prevents the "blocky normal" artifact from nearest-neighbor lookup.
     */
    float trilinearQuery(float wx, float wy, float wz) const {
        // Convert to voxel-space continuous coordinates
        float vx = wx * inv_voxel_size_ - 0.5f;
        float vy = wy * inv_voxel_size_ - 0.5f;
        float vz = wz * inv_voxel_size_ - 0.5f;
        
        int ix = static_cast<int>(std::floor(vx));
        int iy = static_cast<int>(std::floor(vy));
        int iz = static_cast<int>(std::floor(vz));
        
        float fx = vx - ix;
        float fy = vy - iy;
        float fz = vz - iz;
        
        // 8-corner lookup
        float c000 = getVoxelSDF(ix,     iy,     iz);
        float c100 = getVoxelSDF(ix + 1, iy,     iz);
        float c010 = getVoxelSDF(ix,     iy + 1, iz);
        float c110 = getVoxelSDF(ix + 1, iy + 1, iz);
        float c001 = getVoxelSDF(ix,     iy,     iz + 1);
        float c101 = getVoxelSDF(ix + 1, iy,     iz + 1);
        float c011 = getVoxelSDF(ix,     iy + 1, iz + 1);
        float c111 = getVoxelSDF(ix + 1, iy + 1, iz + 1);
        
        // Trilinear blend
        float c00 = c000 * (1 - fx) + c100 * fx;
        float c10 = c010 * (1 - fx) + c110 * fx;
        float c01 = c001 * (1 - fx) + c101 * fx;
        float c11 = c011 * (1 - fx) + c111 * fx;
        
        float c0 = c00 * (1 - fy) + c10 * fy;
        float c1 = c01 * (1 - fy) + c11 * fy;
        
        return c0 * (1 - fz) + c1 * fz;
    }
    
    /**
     * @brief Trilinear interpolation of weight at continuous world coordinate.
     */
    float trilinearWeightQuery(float wx, float wy, float wz) const {
        float vx = wx * inv_voxel_size_ - 0.5f;
        float vy = wy * inv_voxel_size_ - 0.5f;
        float vz = wz * inv_voxel_size_ - 0.5f;
        
        int ix = static_cast<int>(std::floor(vx));
        int iy = static_cast<int>(std::floor(vy));
        int iz = static_cast<int>(std::floor(vz));
        
        float fx = vx - ix;
        float fy = vy - iy;
        float fz = vz - iz;
        
        float c000 = getVoxelWeight(ix,     iy,     iz);
        float c100 = getVoxelWeight(ix + 1, iy,     iz);
        float c010 = getVoxelWeight(ix,     iy + 1, iz);
        float c110 = getVoxelWeight(ix + 1, iy + 1, iz);
        float c001 = getVoxelWeight(ix,     iy,     iz + 1);
        float c101 = getVoxelWeight(ix + 1, iy,     iz + 1);
        float c011 = getVoxelWeight(ix,     iy + 1, iz + 1);
        float c111 = getVoxelWeight(ix + 1, iy + 1, iz + 1);
        
        float c00 = c000 * (1 - fx) + c100 * fx;
        float c10 = c010 * (1 - fx) + c110 * fx;
        float c01 = c001 * (1 - fx) + c101 * fx;
        float c11 = c011 * (1 - fx) + c111 * fx;
        
        float c0 = c00 * (1 - fy) + c10 * fy;
        float c1 = c01 * (1 - fy) + c11 * fy;
        
        return c0 * (1 - fz) + c1 * fz;
    }
    
    // ========================================================================
    // Member variables
    // ========================================================================
    
    float voxel_size_;
    float truncation_;
    float inv_voxel_size_;
    
    std::unordered_map<VoxelKey, Voxel, VoxelKeyHash> voxels_;
    std::mutex mutex_;
    
    int num_integrations_ = 0;
};

} // namespace local_tsdf
