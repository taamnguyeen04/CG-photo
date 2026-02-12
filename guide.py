"""
Guided Filter Demo — Hiểu cách Guided Filter hoạt động

Guided Filter sử dụng một ảnh hướng dẫn (guidance) để lọc ảnh input.
Đặc điểm: giữ nguyên edge từ guidance image, smooth noise ở vùng phẳng.

Ứng dụng trong SLAM: dùng RGB làm guidance để lọc depth map,
giữ edge depth tại biên vật thể (trùng edge RGB) mà smooth noise depth.

Demo này cho thấy 4 ví dụ:
1. Lọc depth bằng guided filter (RGB guidance) → depth mượt hơn, edge giữ nguyên
2. Sparse depth → dense depth (Normalized Convolution + Guided Filter)
3. So sánh guided filter vs gaussian blur vs bilateral filter
4. Trực quan hóa edge-aware property
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ================================
# Tạo dữ liệu mẫu (synthetic)
# ================================

def create_synthetic_scene():
    """Tạo ảnh RGB và depth map giả lập scene đơn giản."""
    H, W = 240, 320
    
    # --- RGB image: 3 vùng màu khác nhau ---
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    # Background xanh nhạt
    rgb[:, :] = [200, 220, 240]
    # Hình chữ nhật đỏ (vật thể 1: gần camera)
    rgb[40:120, 50:150] = [220, 80, 60]
    # Hình tròn xanh (vật thể 2: xa camera) 
    cv2.circle(rgb, (240, 140), 60, (60, 150, 60), -1)
    # Thêm texture gradient cho background
    for i in range(H):
        noise = int(20 * np.sin(i * 0.1))
        rgb[i, :, 0] = np.clip(rgb[i, :, 0].astype(int) + noise, 0, 255)
    
    # --- Depth map: mỗi vật thể ở depth khác nhau ---
    depth = np.ones((H, W), dtype=np.float32) * 3.0  # background = 3m
    depth[40:120, 50:150] = 1.0  # rectangle = 1m (gần)
    cv2.circle(depth, (240, 140), 60, 2.0, -1)  # circle = 2m (trung bình)
    
    # Thêm noise vào depth (giả lập sensor noise)
    noise = np.random.normal(0, 0.05, depth.shape).astype(np.float32)
    depth_noisy = depth + noise
    
    return rgb, depth, depth_noisy


def create_sparse_depth(depth_clean, sparsity=0.02):
    """Tạo sparse depth map (chỉ có depth tại 2% pixels — giả lập ORB keypoints)."""
    H, W = depth_clean.shape
    mask = np.zeros((H, W), dtype=np.float32)
    n_points = int(H * W * sparsity)
    
    ys = np.random.randint(0, H, n_points)
    xs = np.random.randint(0, W, n_points)
    mask[ys, xs] = 1.0
    
    sparse_depth = depth_clean * mask
    return sparse_depth, mask


# ================================
# Guided Filter Implementation
# ================================

def guided_filter(guidance, input_img, radius=8, eps=0.01):
    """
    Guided Filter (He et al., 2013)
    
    Công thức:
        output(i) = a_k * I_guidance(i) + b_k,  ∀i ∈ ω_k
        
    Với:
        a_k = (mean(I*p) - mean(I)*mean(p)) / (var(I) + eps)
        b_k = mean(p) - a_k * mean(I)
    
    Parameters:
        guidance:  Ảnh hướng dẫn I (grayscale float32) — quyết định MỞ/TẮT smoothing
        input_img: Ảnh input p (float32) — ảnh cần lọc
        radius:    Kích thước cửa sổ r
        eps:       Regularization ε — càng lớn càng smooth, càng nhỏ càng giữ detail
    
    Returns:
        output: Ảnh đã lọc q
    """
    # Đảm bảo float32
    I = guidance.astype(np.float32)
    p = input_img.astype(np.float32)
    
    ksize = 2 * radius + 1
    
    # Bước 1: Tính mean trong window
    mean_I = cv2.boxFilter(I, -1, (ksize, ksize))      # μ_k
    mean_p = cv2.boxFilter(p, -1, (ksize, ksize))      # p̄_k
    mean_Ip = cv2.boxFilter(I * p, -1, (ksize, ksize))  # mean(I·p)
    mean_II = cv2.boxFilter(I * I, -1, (ksize, ksize))  # mean(I²)
    
    # Bước 2: Tính a_k, b_k
    var_I = mean_II - mean_I * mean_I                    # σ²_k
    cov_Ip = mean_Ip - mean_I * mean_p                   # cov(I, p)
    
    a = cov_Ip / (var_I + eps)  # a_k = cov(I,p) / (σ²_k + ε)
    b = mean_p - a * mean_I     # b_k = p̄_k - a_k · μ_k
    
    # Bước 3: Output = mean(a) * I + mean(b)
    # Lấy mean của a, b qua tất cả windows chứa pixel i
    mean_a = cv2.boxFilter(a, -1, (ksize, ksize))
    mean_b = cv2.boxFilter(b, -1, (ksize, ksize))
    
    output = mean_a * I + mean_b
    return output


def guided_filter_sparse(guidance, sparse_input, mask, radius=16, eps=0.001):
    """
    Guided Filter cho sparse input (Normalized Convolution).
    
    Công thức:
        dense_output(i) = GF(I, sparse * mask)(i) / (GF(I, mask)(i) + δ)
    
    Ý nghĩa: 
    - Tử: guided filter trên sparse depth (0 ở chỗ không có depth)
    - Mẫu: guided filter trên mask (normalize bởi "lượng data có sẵn")
    
    Kết quả: depth được propagate từ sparse points sang neighbor pixels,
             nhưng KHÔNG LEAK QUA RGB EDGES (vì guided filter giữ edge).
    """
    I = guidance.astype(np.float32)
    p = sparse_input.astype(np.float32)
    m = mask.astype(np.float32)
    
    # Guided filter trên (sparse_depth * mask)
    filtered_numerator = guided_filter(I, p * m, radius=radius, eps=eps)
    
    # Guided filter trên mask (cho biết mỗi pixel có bao nhiêu data gần đó)
    filtered_denominator = guided_filter(I, m, radius=radius, eps=eps)
    
    # Normalized convolution
    delta = 1e-6
    dense_output = filtered_numerator / (filtered_denominator + delta)
    
    # Chỉ tin tưởng output ở nơi có đủ support
    confidence = filtered_denominator
    dense_output[confidence < 0.01] = 0
    
    return dense_output, confidence


# ================================
# Demo 1: Smoothing depth với guided filter
# ================================

def demo1_depth_smoothing(rgb, depth_clean, depth_noisy, output_dir):
    """
    Demo: Dùng guided filter để smooth depth noise, giữ nguyên edge.
    
    So sánh:
    - Noisy depth (input)
    - Gaussian blur (smooth nhưng MẤT edge)
    - Bilateral filter (giữ edge, nhưng dùng edge CỦA DEPTH)
    - Guided filter (giữ edge TỪ RGB → chính xác hơn)
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Gaussian blur
    gaussian = cv2.GaussianBlur(depth_noisy, (17, 17), 3.0)
    
    # Bilateral filter (edge từ chính depth)
    bilateral = cv2.bilateralFilter(depth_noisy, 9, 0.1, 17)
    
    # Guided filter (edge từ RGB!)
    guided = guided_filter(gray, depth_noisy, radius=8, eps=0.01)
    
    # OpenCV built-in guided filter (để verify)
    guided_cv = cv2.ximgproc.guidedFilter(
        guide=gray, src=depth_noisy, radius=8, eps=0.01
    ) if hasattr(cv2, 'ximgproc') else guided
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Demo 1: Guided Filter Smoothing Depth Map', fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('RGB (Guidance Image)', fontsize=12)
    
    axes[0, 1].imshow(depth_clean, cmap='turbo', vmin=0.5, vmax=3.5)
    axes[0, 1].set_title('Depth Ground Truth', fontsize=12)
    
    axes[0, 2].imshow(depth_noisy, cmap='turbo', vmin=0.5, vmax=3.5)
    axes[0, 2].set_title('Depth + Noise (Input)', fontsize=12)
    
    axes[1, 0].imshow(gaussian, cmap='turbo', vmin=0.5, vmax=3.5)
    axes[1, 0].set_title('Gaussian Blur\n(smooth nhưng MẤT edge ❌)', fontsize=11)
    
    axes[1, 1].imshow(bilateral, cmap='turbo', vmin=0.5, vmax=3.5)
    axes[1, 1].set_title('Bilateral Filter\n(edge từ depth, OK)', fontsize=11)
    
    axes[1, 2].imshow(guided, cmap='turbo', vmin=0.5, vmax=3.5)
    axes[1, 2].set_title('Guided Filter\n(edge từ RGB ✅ tốt nhất)', fontsize=11)
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demo1_depth_smoothing.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Demo 1] Saved: demo1_depth_smoothing.png")
    
    # Error analysis
    err_noisy = np.abs(depth_noisy - depth_clean).mean()
    err_gauss = np.abs(gaussian - depth_clean).mean()
    err_bilat = np.abs(bilateral - depth_clean).mean()
    err_guided = np.abs(guided - depth_clean).mean()
    
    print(f"  Mean Absolute Error:")
    print(f"    Noisy input:      {err_noisy:.4f}m")
    print(f"    Gaussian blur:    {err_gauss:.4f}m (smooth tất cả → blur edge)")
    print(f"    Bilateral filter: {err_bilat:.4f}m (giữ edge depth)")
    print(f"    Guided filter:    {err_guided:.4f}m (giữ edge RGB → tốt nhất)")


# ================================
# Demo 2: Sparse → Dense depth
# ================================

def demo2_sparse_to_dense(rgb, depth_clean, output_dir):
    """
    Demo: Propagate sparse depth sang dense depth bằng Guided Filter + Normalized Convolution.
    
    Đây là ứng dụng quan trọng nhất cho SLAM:
    - Input: sparse depth (tại ORB keypoints, ~2% pixels)
    - Guidance: RGB image
    - Output: dense depth map (depth ở MỌI pixel)
    
    Edge-aware: depth không leak qua biên vật thể.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Tạo sparse depth (2% pixels)
    sparse_depth, mask = create_sparse_depth(depth_clean, sparsity=0.02)
    
    # Guided filter sparse → dense
    dense_depth, confidence = guided_filter_sparse(gray, sparse_depth, mask, radius=16, eps=0.001)
    
    # Simple nearest-neighbor fill (baseline so sánh)
    from scipy.ndimage import distance_transform_edt
    mask_bool = mask > 0
    _, nn_indices = distance_transform_edt(~mask_bool, return_indices=True)
    nn_fill = depth_clean[nn_indices[0], nn_indices[1]]
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Demo 2: Sparse → Dense Depth via Guided Filter', fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('RGB (Guidance)', fontsize=12)
    
    # Show sparse depth as scatter on RGB
    sparse_vis = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).copy()
    ys, xs = np.where(mask > 0)
    axes[0, 1].imshow(sparse_vis)
    scatter = axes[0, 1].scatter(xs, ys, c=sparse_depth[ys, xs], cmap='turbo', 
                                  s=2, vmin=0.5, vmax=3.5)
    axes[0, 1].set_title(f'Sparse Depth ({len(ys)} points, 2%)', fontsize=12)
    
    axes[0, 2].imshow(depth_clean, cmap='turbo', vmin=0.5, vmax=3.5)
    axes[0, 2].set_title('Ground Truth Dense', fontsize=12)
    
    axes[1, 0].imshow(nn_fill, cmap='turbo', vmin=0.5, vmax=3.5)
    axes[1, 0].set_title('Nearest-Neighbor Fill\n(edge bị leak ❌)', fontsize=11)
    
    axes[1, 1].imshow(dense_depth, cmap='turbo', vmin=0.5, vmax=3.5)
    axes[1, 1].set_title('Guided Filter Dense\n(edge preserved ✅)', fontsize=11)
    
    axes[1, 2].imshow(confidence, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 2].set_title('Confidence Map\n(cao = nhiều support)', fontsize=11)
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demo2_sparse_to_dense.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Demo 2] Saved: demo2_sparse_to_dense.png")
    
    # Error analysis
    valid = confidence > 0.01
    err_nn = np.abs(nn_fill[valid] - depth_clean[valid]).mean()
    err_gf = np.abs(dense_depth[valid] - depth_clean[valid]).mean()
    print(f"  Dense Depth Error (where confident):")
    print(f"    Nearest neighbor: {err_nn:.4f}m")
    print(f"    Guided filter:    {err_gf:.4f}m")


# ================================
# Demo 3: Edge-aware property
# ================================

def demo3_edge_aware_property(rgb, depth_noisy, output_dir):
    """
    Demo: Trực quan hóa tại sao guided filter giữ edge.
    
    Ở vùng edge (σ² >> ε): a ≠ 0 → output theo guidance → GIỮA EDGE
    Ở vùng flat (σ² << ε): a ≈ 0 → output = mean(p) → SMOOTH
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    radius = 8
    eps = 0.01
    ksize = 2 * radius + 1
    
    I = gray
    p = depth_noisy
    
    # Tính a_k (hệ số linear)
    mean_I = cv2.boxFilter(I, -1, (ksize, ksize))
    mean_II = cv2.boxFilter(I * I, -1, (ksize, ksize))
    mean_Ip = cv2.boxFilter(I * p, -1, (ksize, ksize))
    mean_p = cv2.boxFilter(p, -1, (ksize, ksize))
    
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Visualize a_k và var_I
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Demo 3: Tại sao Guided Filter giữ Edge?', fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('RGB (Guidance I)', fontsize=12)
    
    axes[0, 1].imshow(var_I, cmap='hot')
    axes[0, 1].set_title('σ²(I) — Variance of Guidance\n(cao tại edge RGB)', fontsize=11)
    
    axes[0, 2].imshow(np.abs(a), cmap='hot')
    axes[0, 2].set_title('|a_k| — Linear Coefficient\n(a≠0 tại edge → GIỮ edge)', fontsize=11)
    
    axes[1, 0].imshow(depth_noisy, cmap='turbo', vmin=0.5, vmax=3.5)
    axes[1, 0].set_title('Input: Noisy Depth', fontsize=12)
    
    # Output
    mean_a = cv2.boxFilter(a, -1, (ksize, ksize))
    mean_b = cv2.boxFilter(b, -1, (ksize, ksize))
    output = mean_a * I + mean_b
    
    axes[1, 1].imshow(output, cmap='turbo', vmin=0.5, vmax=3.5)
    axes[1, 1].set_title('Output: Filtered Depth\n(smooth noise, giữ edge)', fontsize=11)
    
    # Edge map từ RGB (Sobel)
    edge_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    
    axes[1, 2].imshow(edge_mag, cmap='magma')
    axes[1, 2].set_title('Edge Map (Sobel of RGB)\n→ Đây là nơi edge được giữ', fontsize=11)
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demo3_edge_aware.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Demo 3] Saved: demo3_edge_aware.png")
    
    print(f"  Edge region: σ²={var_I[60, 50]:.4f} >> ε={eps} → a={a[60, 50]:.4f} ≠ 0 → GIỮ edge")
    print(f"  Flat region: σ²={var_I[10, 10]:.4f} << ε={eps} → a={a[10, 10]:.4f} ≈ 0 → SMOOTH")


# ================================
# Demo 4: Dùng Replica data thật
# ================================

def demo4_replica_data(output_dir):
    """
    Demo với dữ liệu Replica thật:
    So sánh ORB-only (baseline) vs Edge-Aware Sampling (ours).
    
    Baseline (Photo-SLAM): chỉ có depth tại ORB keypoints (~1500 điểm)
    Ours: ORB + Edge sampling + Uniform grid (~3000-4000 điểm)
    """
    replica_dir = "/media/tam/DATA/data/Replica/office0/results"
    rgb_path = os.path.join(replica_dir, "frame000100.jpg")  # dùng frame 100 cho scene đẹp hơn
    depth_path = os.path.join(replica_dir, "depth000100.png")
    
    # Fallback sang frame 0
    if not os.path.exists(rgb_path):
        rgb_path = os.path.join(replica_dir, "frame000000.jpg")
        depth_path = os.path.join(replica_dir, "depth000000.png")
    
    if not os.path.exists(rgb_path):
        print("[Demo 4] Skipped — Replica data not found at:", replica_dir)
        return
    
    rgb = cv2.imread(rgb_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if depth_raw is None or rgb is None:
        print("[Demo 4] Skipped — cannot read images")
        return
    
    H, W = rgb.shape[:2]
    
    # Replica depth scale: 6553.5
    depth = depth_raw.astype(np.float32) / 6553.5
    
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    
    # ========================================
    #  1. ORB Keypoints (Baseline Photo-SLAM)
    # ========================================
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints = orb.detect(gray, None)
    orb_pts = np.array([[int(kp.pt[0]), int(kp.pt[1])] for kp in keypoints])
    
    # Lọc bỏ points có depth = 0
    valid_orb = []
    for pt in orb_pts:
        x, y = pt
        if 0 <= x < W and 0 <= y < H and depth[y, x] > 0.01:
            valid_orb.append(pt)
    valid_orb = np.array(valid_orb)
    
    # ========================================
    #  2. Edge-Aware Sampling (Ours)
    # ========================================
    # Tính edge strength (Sobel)
    edge_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    edge_norm = edge_mag / (edge_mag.max() + 1e-8)
    
    # Depth gradient (Sobel) — để lọc biên depth không tin cậy
    depth_gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    depth_gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    depth_grad = np.sqrt(depth_gx**2 + depth_gy**2)
    
    # Valid mask: depth hợp lệ + depth gradient nhỏ
    valid_mask = (depth > 0.01) & (depth < 10.0) & (depth_grad < 0.5)
    
    # --- Edge sampling (60% budget = 900 points) ---
    edge_threshold = 0.1
    edge_candidates = np.where(valid_mask & (edge_norm > edge_threshold))
    n_edge = min(900, len(edge_candidates[0]))
    if len(edge_candidates[0]) > n_edge:
        indices = np.random.choice(len(edge_candidates[0]), n_edge, replace=False)
        edge_ys = edge_candidates[0][indices]
        edge_xs = edge_candidates[1][indices]
    else:
        edge_ys = edge_candidates[0]
        edge_xs = edge_candidates[1]
    edge_pts = np.stack([edge_xs, edge_ys], axis=1)
    
    # --- Uniform grid sampling (40% budget = 600 points) ---
    grid_size = 8
    uniform_pts = []
    for gy in range(0, H, grid_size):
        for gx in range(0, W, grid_size):
            # Lấy 1 pixel valid ngẫu nhiên trong cell
            cy, cx = gy + grid_size // 2, gx + grid_size // 2
            if 0 <= cy < H and 0 <= cx < W and valid_mask[cy, cx]:
                uniform_pts.append([cx, cy])
    uniform_pts = np.array(uniform_pts[:600]) if len(uniform_pts) > 600 else np.array(uniform_pts)
    
    # Tổng hợp: ORB + Edge + Uniform
    all_guided_pts = np.vstack([valid_orb, edge_pts, uniform_pts])
    
    # ========================================
    #  3. Visualize
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Demo 4: ORB-only (Baseline) vs Edge-Aware Sampling (Ours) — Replica office0', 
                 fontsize=16, fontweight='bold')
    
    rgb_display = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # --- Row 1: RGB, Edge map, Depth ---
    axes[0, 0].imshow(rgb_display)
    axes[0, 0].set_title('RGB Image', fontsize=12)
    
    axes[0, 1].imshow(edge_norm, cmap='magma')
    axes[0, 1].set_title('Edge Map (Sobel)\nPixels with high edge = sampling priority', fontsize=11)
    
    axes[0, 2].imshow(depth, cmap='turbo', vmin=0, vmax=5)
    axes[0, 2].set_title('Sensor Depth Map', fontsize=12)
    
    # --- Row 2: ORB-only vs Edge-Aware ---
    # ORB only
    orb_vis = rgb_display.copy()
    axes[1, 0].imshow(orb_vis)
    axes[1, 0].scatter(valid_orb[:, 0], valid_orb[:, 1], 
                       c=depth[valid_orb[:, 1], valid_orb[:, 0]], 
                       cmap='turbo', s=3, vmin=0, vmax=5, alpha=0.8)
    axes[1, 0].set_title(f'Baseline: ORB-only\n{len(valid_orb)} points (sparse, gaps)', fontsize=12)
    
    # Edge-Aware (ORB + Edge + Uniform)
    guided_vis = rgb_display.copy()
    axes[1, 1].imshow(guided_vis)
    # ORB points = cyan
    axes[1, 1].scatter(valid_orb[:, 0], valid_orb[:, 1], c='cyan', s=2, alpha=0.5, label='ORB')
    # Edge points = red
    axes[1, 1].scatter(edge_pts[:, 0], edge_pts[:, 1], c='red', s=2, alpha=0.5, label='Edge')
    # Uniform points = yellow
    if len(uniform_pts) > 0:
        axes[1, 1].scatter(uniform_pts[:, 0], uniform_pts[:, 1], c='yellow', s=2, alpha=0.3, label='Uniform')
    axes[1, 1].legend(loc='upper right', fontsize=8, markerscale=3)
    axes[1, 1].set_title(f'Ours: Edge-Aware Sampling\n{len(all_guided_pts)} points (dense, edge-focused)', fontsize=12)
    
    # Coverage comparison
    # Tạo coverage map: binary mask cho từng method
    orb_coverage = np.zeros((H, W), dtype=np.float32)
    for pt in valid_orb:
        cv2.circle(orb_coverage, (pt[0], pt[1]), 5, 1.0, -1)
    
    guided_coverage = np.zeros((H, W), dtype=np.float32)
    for pt in all_guided_pts:
        cv2.circle(guided_coverage, (int(pt[0]), int(pt[1])), 5, 1.0, -1)
    
    # Overlay: green = guided-only coverage, red = no coverage
    coverage_diff = np.zeros((H, W, 3), dtype=np.uint8)
    coverage_diff[guided_coverage > 0] = [0, 200, 0]    # Green: covered by ours
    coverage_diff[orb_coverage > 0] = [0, 100, 255]     # Blue: covered by ORB too
    coverage_diff[(guided_coverage == 0) & (orb_coverage == 0)] = [80, 0, 0]  # Dark red: no coverage
    
    axes[1, 2].imshow(coverage_diff)
    orb_cov_pct = 100.0 * orb_coverage.sum() / valid_mask.sum()
    guided_cov_pct = 100.0 * guided_coverage.sum() / valid_mask.sum()
    axes[1, 2].set_title(f'Coverage Comparison\nORB: {orb_cov_pct:.1f}% vs Ours: {guided_cov_pct:.1f}%\n'
                         f'(Blue=ORB, Green=Ours extra, Red=no coverage)', fontsize=10)
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demo4_replica_orb_vs_guided.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[Demo 4] Saved: demo4_replica_orb_vs_guided.png")
    print(f"  ORB-only:      {len(valid_orb)} points, coverage {orb_cov_pct:.1f}%")
    print(f"  Edge-Aware:    {len(all_guided_pts)} points, coverage {guided_cov_pct:.1f}%")
    print(f"    - ORB:       {len(valid_orb)}")
    print(f"    - Edge:      {len(edge_pts)}")
    print(f"    - Uniform:   {len(uniform_pts)}")


# ================================
# Tổng kết: Guided Filter cho SLAM
# ================================

def print_summary():
    print("""
╔══════════════════════════════════════════════════════════════╗
║               GUIDED FILTER — TÓM TẮT                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  CÔNG THỨC:                                                  ║
║    output(i) = a_k · I(i) + b_k                             ║
║                                                              ║
║    a_k = cov(I, p) / (var(I) + ε)                           ║
║    b_k = mean(p) - a_k · mean(I)                            ║
║                                                              ║
║  TRƯỜNG HỢP:                                                ║
║    Edge (σ² >> ε):  a ≠ 0  → output theo guidance → GIỮ     ║
║    Flat (σ² << ε):  a ≈ 0  → output = mean(p)    → SMOOTH  ║
║                                                              ║
║  ỨNG DỤNG TRONG SLAM:                                       ║
║    • Guidance = RGB image (có edge rõ tại biên vật thể)      ║
║    • Input = depth map (có noise, hoặc sparse)               ║
║    • Output = depth map sạch, edge-preserved                 ║
║                                                              ║
║    → Dùng edge map từ RGB để chọn pixel quan trọng           ║
║      cho Gaussian initialization (Edge-Aware Sampling)       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ================================
# Main
# ================================

if __name__ == "__main__":
    output_dir = "/media/tam/DATA/3D/CG-photo/guided_filter_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  GUIDED FILTER DEMO")
    print("=" * 60)
    
    # Tạo dữ liệu synthetic
    rgb, depth_clean, depth_noisy = create_synthetic_scene()
    
    # Chạy 4 demo
    demo1_depth_smoothing(rgb, depth_clean, depth_noisy, output_dir)
    print()
    demo2_sparse_to_dense(rgb, depth_clean, output_dir)
    print()
    demo3_edge_aware_property(rgb, depth_noisy, output_dir)
    print()
    demo4_replica_data(output_dir)
    print()
    
    print_summary()
    
    print(f"\nTất cả output đã lưu tại: {output_dir}/")
