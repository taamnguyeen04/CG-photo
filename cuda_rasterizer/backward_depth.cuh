/*
 * Depth-Photo-SLAM: Backward pass with depth gradients
 * This file is included at the end of backward.cu
 */

// Depth-Photo-SLAM: Backward render kernel WITH depth gradient support
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDAWithDepth(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_ddepth,
	const float* __restrict__ dL_ddepth_sq,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float dL_dpixel_depth = 0.0f;
	
	if (inside) {
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		if (dL_ddepth != nullptr)
			dL_dpixel_depth = dL_ddepth[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float accum_depth_rec = 0.0f;
	float last_depth = 0.0f;

	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int ii = 0; ii < C; ii++)
				collected_colors[ii * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + ii];
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor--;
			if (contributor >= last_contributor)
				continue;

			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f) continue;

			T = T / (1.f - alpha);
			const float weight = alpha * T;
			const float gaussian_depth = collected_depths[j];
			const int global_id = collected_id[j];

			// Color gradients
			float dL_dalpha = 0.0f;
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;
				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				atomicAdd(&(dL_dcolors[global_id * C + ch]), weight * dL_dchannel);
			}

			// Depth gradient: dL/d_depth_i = dL/dD * weight
			if (dL_ddepths != nullptr) {
				atomicAdd(&(dL_ddepths[global_id]), weight * dL_dpixel_depth);
			}
			
			// Depth contribution to alpha gradient
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = gaussian_depth;
			dL_dalpha += (gaussian_depth - accum_depth_rec) * dL_dpixel_depth * T;

			dL_dalpha *= T;
			last_alpha = alpha;

			float bg_dot_dpixel = 0;
			for (int ii = 0; ii < C; ii++)
				bg_dot_dpixel += bg_color[ii] * dL_dpixel[ii];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;

			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * (-gdx * con_o.x - gdy * con_o.y) * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * (-gdy * con_o.z - gdx * con_o.y) * ddely_dy);
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::renderWithDepth(
	const dim3 grid, const dim3 block,
	const uint2* ranges, const uint32_t* point_list,
	int W, int H, const float* bg_color,
	const float2* means2D, const float4* conic_opacity,
	const float* colors, const float* depths,
	const float* final_Ts, const uint32_t* n_contrib,
	const float* dL_dpixels, const float* dL_ddepth, const float* dL_ddepth_sq,
	float3* dL_dmean2D, float4* dL_dconic2D,
	float* dL_dopacity, float* dL_dcolors, float* dL_ddepths)
{
	renderCUDAWithDepth<NUM_CHANNELS><<<grid, block>>>(
		ranges, point_list, W, H, bg_color, means2D, conic_opacity,
		colors, depths, final_Ts, n_contrib, dL_dpixels, dL_ddepth, dL_ddepth_sq,
		dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors, dL_ddepths);
}

// Preprocess backward with depth gradient
template<int C>
__global__ void preprocessCUDAWithDepth(
	int P, int D, int M,
	const float3* means, const int* radii,
	const float* shs, const bool* clamped,
	const glm::vec3* scales, const glm::vec4* rotations,
	const float scale_modifier, const float* proj,
	const glm::vec3* campos, const float* viewmatrix,
	const float3* dL_dmean2D, const float* dL_ddepths,
	glm::vec3* dL_dmeans, float* dL_dcolor, float* dL_dcov3D,
	float* dL_dsh, glm::vec3* dL_dscale, glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0)) return;

	float3 m = means[idx];
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	
	glm::vec3 dL_dmean;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// Depth gradient: depth = viewmatrix[2]*x + viewmatrix[6]*y + viewmatrix[10]*z
	if (dL_ddepths != nullptr) {
		float dL_ddepth = dL_ddepths[idx];
		dL_dmean.x += dL_ddepth * viewmatrix[2];
		dL_dmean.y += dL_ddepth * viewmatrix[6];
		dL_dmean.z += dL_ddepth * viewmatrix[10];
	}

	dL_dmeans[idx] += dL_dmean;

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

void BACKWARD::preprocessWithDepth(
	int P, int D, int M,
	const float3* means, const int* radii,
	const float* shs, const bool* clamped,
	const glm::vec3* scales, const glm::vec4* rotations,
	const float scale_modifier, const float* cov3Ds,
	const float* viewmatrix, const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos, const float3* dL_dmean2D,
	const float* dL_dconics, const float* dL_dcov2Ds, const float* dL_ddepths, // ESC
	glm::vec3* dL_dmeans, float* dL_dcolor, float* dL_dcov3D,
	float* dL_dsh, glm::vec3* dL_dscale, glm::vec4* dL_drot)
{
	computeCov2DCUDA<<<(P + 255) / 256, 256>>>(
		P, means, radii, cov3Ds, focal_x, focal_y, tan_fovx, tan_fovy,
		viewmatrix, dL_dconics, dL_dcov2Ds, (float3*)dL_dmeans, dL_dcov3D);

	preprocessCUDAWithDepth<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M, (float3*)means, radii, shs, clamped,
		(glm::vec3*)scales, (glm::vec4*)rotations, scale_modifier,
		projmatrix, campos, viewmatrix, (float3*)dL_dmean2D, dL_ddepths,
		(glm::vec3*)dL_dmeans, dL_dcolor, dL_dcov3D, dL_dsh, dL_dscale, dL_drot);
}
