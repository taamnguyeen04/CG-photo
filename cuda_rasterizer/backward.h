/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * Depth-Photo-SLAM: Extended with depth backward pass
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	// Original render backward (for color only)
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors);

	// Depth-Photo-SLAM: Render backward with depth gradients
	// This propagates gradients from depth loss back to Gaussian parameters
	void renderWithDepth(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,           // Per-Gaussian depth (from forward)
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,       // Gradient from color loss [C, H, W]
		const float* dL_ddepth,        // Gradient from depth loss [H, W]
		const float* dL_ddepth_sq,     // Gradient from depth^2 loss [H, W]
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_ddepths);            // Output: gradient w.r.t. per-Gaussian depth

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		const float* dL_dcov2Ds, // ESC
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);

	// Depth-Photo-SLAM: Preprocess backward with depth gradient
	// Propagates gradient from per-Gaussian depth back to 3D mean
	void preprocessWithDepth(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		const float* dL_dcov2Ds, // ESC
		const float* dL_ddepths,       // Gradient w.r.t. per-Gaussian depth
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);
}

#endif