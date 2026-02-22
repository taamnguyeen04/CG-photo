/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/torch.h>
#include <cstdio>
#include <tuple>
#include <string>
	
// Depth-Photo-SLAM: Updated return type to include depth outputs
// CG-SLAM: Added gt_depth input and uncertainty output for L_var
// Returns: (rendered_count, out_color, radii, geomBuffer, binningBuffer, imgBuffer,
//           out_depth, out_depth_sq, out_median_depth, out_uncertainty, out_T,
//           out_cov2D [P,3], out_view_depths [P])
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const torch::Tensor& gt_depth);  // CG-SLAM: ground truth depth for L_var

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);
// Depth-Photo-SLAM: Backward with depth gradients
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardWithDepthCUDA(
 const torch::Tensor& background,
const torch::Tensor& means3D,
const torch::Tensor& radii,
    const torch::Tensor& colors,
const torch::Tensor& scales,
const torch::Tensor& rotations,
const float scale_modifier,
const torch::Tensor& cov3D_precomp,
const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
const float tan_fovx, 
const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_depth_sq,
const torch::Tensor& sh,
const int degree,
const torch::Tensor& campos,
const torch::Tensor& geomBuffer,
const int R,
const torch::Tensor& binningBuffer,
const torch::Tensor& imageBuffer);
