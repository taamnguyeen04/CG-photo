#!/usr/bin/env python3
"""
So sánh kết quả các thí nghiệm Photo-SLAM / CG-SLAM

Cách sử dụng:
    python compare_experiments.py [dataset] [mode] PA_NAME1 PA_NAME2 [PA_NAME3] [PA_NAME4]

Ví dụ:
    python compare_experiments.py replica rgbd geo07 geo07reg07 geo07var01
    python compare_experiments.py tum mono baseline geo05 geo07
    python compare_experiments.py --legacy geo05 geo07  # Dùng format cũ results_<name>
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
import glob

# Thêm màu cho output terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def get_color(value, best_value, higher_is_better=True):
    """Trả về màu cho giá trị: xanh nếu tốt nhất, vàng nếu gần tốt nhất, đỏ nếu xấu"""
    if pd.isna(value):
        return Colors.RED
    
    if higher_is_better:
        is_best = value >= best_value * 0.999  # tolerance cho float
        is_good = value >= best_value * 0.95
    else:
        is_best = value <= best_value * 1.001
        is_good = value <= best_value * 1.05
    
    if is_best:
        return Colors.GREEN + Colors.BOLD
    elif is_good:
        return Colors.YELLOW
    else:
        return Colors.RED

def load_experiment_data(base_dir, experiment_name, dataset=None, mode=None, legacy=False):
    """Load dữ liệu từ thư mục results
    
    Cấu trúc mới: results/{dataset}/{mode}/{experiment_name}/all_results.csv
    Cấu trúc cũ: results_{experiment_name}/all_results.csv
    """
    if legacy:
        # Legacy format: results_<experiment_name>
        results_dir = os.path.join(base_dir, f"results_{experiment_name}")
    else:
        # New format: results/{dataset}/{mode}/{experiment_name}
        results_dir = os.path.join(base_dir, "results", dataset, mode, experiment_name)
    
    csv_path = os.path.join(results_dir, "all_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"{Colors.RED}Lỗi: Không tìm thấy file {csv_path}{Colors.END}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        df['experiment'] = experiment_name
        return df
    except Exception as e:
        print(f"{Colors.RED}Lỗi đọc file {csv_path}: {e}{Colors.END}")
        return None

def load_from_summary_csv(base_dir, experiment_names, dataset=None, mode=None, legacy=False):
    """Load dữ liệu từ multi_experiments_summary.csv"""
    if legacy:
        summary_path = os.path.join(base_dir, "multi_experiments_summary.csv")
    else:
        summary_path = os.path.join(base_dir, "results", dataset, mode, "multi_experiments_summary.csv")
    
    if not os.path.exists(summary_path):
        print(f"{Colors.YELLOW}Không tìm thấy summary file: {summary_path}{Colors.END}")
        return None
    
    try:
        df = pd.read_csv(summary_path)
        filtered = df[df['experiment'].isin(experiment_names)]
        return filtered
    except Exception as e:
        print(f"{Colors.RED}Lỗi đọc file {summary_path}: {e}{Colors.END}")
        return None

def list_available_experiments(base_dir, dataset=None, mode=None, legacy=False):
    """Liệt kê các experiments có sẵn"""
    if legacy:
        pattern = os.path.join(base_dir, "results_*/all_results.csv")
        files = glob.glob(pattern)
        experiments = []
        for f in files:
            dirname = os.path.dirname(f)
            exp_name = os.path.basename(dirname).replace("results_", "")
            experiments.append(exp_name)
    else:
        pattern = os.path.join(base_dir, "results", dataset, mode, "*/all_results.csv")
        files = glob.glob(pattern)
        experiments = []
        for f in files:
            dirname = os.path.dirname(f)
            exp_name = os.path.basename(dirname)
            experiments.append(exp_name)
    
    return sorted(experiments)

def calculate_scene_averages(df):
    """Tính trung bình các metrics theo scene cho mỗi experiment"""
    # Xác định các cột metrics
    metric_cols = ['avg_psnr', 'avg_ssim', 'avg_lpips', 'avg_accuracy', 
                   'avg_completion', 'avg_comp_ratio']
    
    # Lọc chỉ các cột tồn tại
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    # Group by experiment và scene, tính trung bình
    grouped = df.groupby(['experiment', 'scene'])[available_metrics].mean().reset_index()
    return grouped, available_metrics

def calculate_overall_averages(df, available_metrics):
    """Tính trung bình tổng thể cho mỗi experiment"""
    overall = df.groupby('experiment')[available_metrics].mean()
    return overall

def print_comparison_table(experiments, scene_data, overall_data, available_metrics):
    """In bảng so sánh các thí nghiệm"""
    
    # Định nghĩa metrics và hướng tốt (higher_is_better)
    metric_info = {
        'avg_psnr': ('PSNR ↑', True),        # Cao hơn tốt hơn
        'avg_ssim': ('SSIM ↑', True),         # Cao hơn tốt hơn
        'avg_lpips': ('LPIPS ↓', False),      # Thấp hơn tốt hơn
        'avg_accuracy': ('Accuracy ↓', False), # Thấp hơn tốt hơn (cm)
        'avg_completion': ('Completion ↓', False), # Thấp hơn tốt hơn (cm)
        'avg_comp_ratio': ('Comp.Ratio ↑', True),  # Cao hơn tốt hơn (%)
    }
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}")
    print("             SO SÁNH KẾT QUẢ THÍ NGHIỆM PHOTO-SLAM / CG-SLAM")
    print(f"{'=' * 80}{Colors.END}")
    
    # In header với tên các experiments
    print(f"\n{Colors.BOLD}{'Metric':<20}", end="")
    for exp in experiments:
        print(f"{exp:>15}", end="")
    print(f"{'Δ Best-Worst':>15}{Colors.END}")
    print("-" * (20 + 15 * len(experiments) + 15))
    
    # In từng metric
    for metric in available_metrics:
        if metric not in metric_info:
            continue
            
        display_name, higher_is_better = metric_info[metric]
        
        values = []
        for exp in experiments:
            if exp in overall_data.index:
                val = overall_data.loc[exp, metric]
                values.append(val)
            else:
                values.append(np.nan)
        
        # Tìm best và worst
        valid_values = [v for v in values if not pd.isna(v)]
        if not valid_values:
            continue
            
        if higher_is_better:
            best_val = max(valid_values)
            worst_val = min(valid_values)
        else:
            best_val = min(valid_values)
            worst_val = max(valid_values)
        
        delta = abs(best_val - worst_val)
        
        # In dòng metric
        print(f"{display_name:<20}", end="")
        for val in values:
            color = get_color(val, best_val, higher_is_better)
            if pd.isna(val):
                print(f"{color}{'N/A':>15}{Colors.END}", end="")
            else:
                print(f"{color}{val:>15.4f}{Colors.END}", end="")
        
        # In delta
        delta_pct = (delta / best_val * 100) if best_val != 0 else 0
        print(f"{Colors.CYAN}{delta:>10.4f} ({delta_pct:>4.1f}%){Colors.END}")
    
    print("-" * (20 + 15 * len(experiments) + 15))
    
    # Thêm bảng chi tiết theo scene
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}")
    print("                         CHI TIẾT THEO SCENE")
    print(f"{'=' * 80}{Colors.END}")
    
    scenes = sorted(scene_data['scene'].unique())
    
    for metric in available_metrics:
        if metric not in metric_info:
            continue
        
        display_name, higher_is_better = metric_info[metric]
        print(f"\n{Colors.BOLD}{display_name}{Colors.END}")
        
        # Header
        print(f"{'Scene':<15}", end="")
        for exp in experiments:
            print(f"{exp:>15}", end="")
        print(f"{'Best':>10}")
        print("-" * (15 + 15 * len(experiments) + 10))
        
        for scene in scenes:
            print(f"{scene:<15}", end="")
            scene_vals = []
            
            for exp in experiments:
                mask = (scene_data['experiment'] == exp) & (scene_data['scene'] == scene)
                if mask.any():
                    val = scene_data.loc[mask, metric].values[0]
                    scene_vals.append(val)
                else:
                    scene_vals.append(np.nan)
            
            # Tìm best cho scene này
            valid_vals = [v for v in scene_vals if not pd.isna(v)]
            if valid_vals:
                if higher_is_better:
                    best_scene_val = max(valid_vals)
                else:
                    best_scene_val = min(valid_vals)
                best_exp = experiments[scene_vals.index(best_scene_val)]
            else:
                best_scene_val = np.nan
                best_exp = "N/A"
            
            for val in scene_vals:
                color = get_color(val, best_scene_val, higher_is_better) if not pd.isna(best_scene_val) else Colors.RED
                if pd.isna(val):
                    print(f"{color}{'N/A':>15}{Colors.END}", end="")
                else:
                    print(f"{color}{val:>15.4f}{Colors.END}", end="")
            
            print(f"{Colors.GREEN}{best_exp:>10}{Colors.END}")
        
        # Thêm dòng Average
        print(f"{'AVERAGE':<15}", end="")
        for exp in experiments:
            if exp in overall_data.index:
                val = overall_data.loc[exp, metric]
                best_val = overall_data[metric].max() if higher_is_better else overall_data[metric].min()
                color = get_color(val, best_val, higher_is_better)
                print(f"{color}{val:>15.4f}{Colors.END}", end="")
            else:
                print(f"{Colors.RED}{'N/A':>15}{Colors.END}", end="")
        print()

def print_winner_summary(experiments, overall_data, available_metrics):
    """In tóm tắt experiment thắng nhiều metric nhất"""
    metric_info = {
        'avg_psnr': True,
        'avg_ssim': True,
        'avg_lpips': False,
        'avg_accuracy': False,
        'avg_completion': False,
        'avg_comp_ratio': True,
    }
    
    wins = {exp: 0 for exp in experiments}
    metric_winners = {}
    
    for metric in available_metrics:
        if metric not in metric_info:
            continue
        higher_is_better = metric_info[metric]
        
        values = {}
        for exp in experiments:
            if exp in overall_data.index:
                values[exp] = overall_data.loc[exp, metric]
        
        if not values:
            continue
        
        if higher_is_better:
            winner = max(values, key=lambda x: values[x])
        else:
            winner = min(values, key=lambda x: values[x])
        
        wins[winner] += 1
        metric_winners[metric] = winner
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}")
    print("                              TÓM TẮT")
    print(f"{'=' * 80}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Số lượng metrics thắng:{Colors.END}")
    for exp, count in sorted(wins.items(), key=lambda x: -x[1]):
        bar = "█" * count + "░" * (len(available_metrics) - count)
        color = Colors.GREEN if count == max(wins.values()) else Colors.YELLOW
        print(f"  {color}{exp:<25} {bar} {count}/{len(available_metrics)}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Experiment thắng theo từng metric:{Colors.END}")
    metric_names = {
        'avg_psnr': 'PSNR',
        'avg_ssim': 'SSIM',
        'avg_lpips': 'LPIPS',
        'avg_accuracy': 'Accuracy',
        'avg_completion': 'Completion',
        'avg_comp_ratio': 'Comp.Ratio',
    }
    for metric, winner in metric_winners.items():
        display = metric_names.get(metric, metric)
        print(f"  {display:<15}: {Colors.GREEN}{winner}{Colors.END}")

def export_comparison_csv(experiments, scene_data, overall_data, available_metrics, output_path):
    """Xuất kết quả so sánh ra file CSV"""
    # Tạo bảng so sánh tổng thể
    comparison = overall_data.loc[experiments].copy()
    comparison = comparison.round(4)
    comparison.to_csv(output_path)
    print(f"\n{Colors.CYAN}Đã lưu kết quả so sánh vào: {output_path}{Colors.END}")

def main():
    parser = argparse.ArgumentParser(
        description='So sánh kết quả các thí nghiệm Photo-SLAM / CG-SLAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
    %(prog)s replica rgbd geo07 geo07reg07 geo07var01
    %(prog)s tum mono baseline geo05 geo07
    %(prog)s --legacy geo05 geo07  # Dùng format cũ results_<name>
    %(prog)s --list replica rgbd   # Liệt kê các experiments có sẵn
        """
    )
    
    parser.add_argument('dataset', nargs='?', default='replica',
                        help='Dataset: replica hoặc tum (mặc định: replica)')
    parser.add_argument('mode', nargs='?', default='rgbd',
                        help='Mode: rgbd, mono, stereo (mặc định: rgbd)')
    parser.add_argument('experiments', nargs='*', 
                        help='Tên các thí nghiệm để so sánh (2-4 tên)')
    parser.add_argument('-b', '--base-dir', 
                        default='/media/tam/DATA/3D/CG-photo',
                        help='Thư mục gốc chứa results (mặc định: /media/tam/DATA/3D/CG-photo)')
    parser.add_argument('-s', '--use-summary', action='store_true',
                        help='Sử dụng file multi_experiments_summary.csv thay vì đọc từng thư mục')
    parser.add_argument('-o', '--output', 
                        help='Xuất kết quả so sánh ra file CSV')
    parser.add_argument('--no-color', action='store_true',
                        help='Tắt màu trong output')
    parser.add_argument('--legacy', action='store_true',
                        help='Sử dụng format thư mục cũ results_<name>')
    parser.add_argument('--list', action='store_true',
                        help='Liệt kê các experiments có sẵn')
    
    args = parser.parse_args()
    
    # Tắt màu nếu cần
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    # Liệt kê experiments nếu --list
    if args.list:
        experiments = list_available_experiments(
            args.base_dir, args.dataset, args.mode, args.legacy
        )
        if experiments:
            print(f"{Colors.HEADER}Experiments có sẵn ({args.dataset}/{args.mode}):{Colors.END}")
            for exp in experiments:
                print(f"  - {exp}")
        else:
            print(f"{Colors.RED}Không tìm thấy experiments nào{Colors.END}")
        sys.exit(0)
    
    # Validate số lượng experiments
    if len(args.experiments) < 2:
        print(f"{Colors.RED}Lỗi: Cần ít nhất 2 experiments để so sánh{Colors.END}")
        print(f"\nSử dụng: python compare_experiments.py [dataset] [mode] exp1 exp2 [exp3] [exp4]")
        print(f"\nExperiments có sẵn ({args.dataset}/{args.mode}):")
        experiments = list_available_experiments(
            args.base_dir, args.dataset, args.mode, args.legacy
        )
        for exp in experiments:
            print(f"  - {exp}")
        sys.exit(1)
    if len(args.experiments) > 4:
        print(f"{Colors.YELLOW}Cảnh báo: Chỉ hiển thị 4 experiments đầu tiên{Colors.END}")
        args.experiments = args.experiments[:4]
    
    print(f"{Colors.HEADER}Đang so sánh: {', '.join(args.experiments)}")
    print(f"Dataset: {args.dataset}, Mode: {args.mode}{Colors.END}")
    
    # Load dữ liệu
    all_data = []
    
    if args.use_summary:
        data = load_from_summary_csv(
            args.base_dir, args.experiments, 
            args.dataset, args.mode, args.legacy
        )
        if data is not None and not data.empty:
            all_data.append(data)
    else:
        for exp in args.experiments:
            data = load_experiment_data(
                args.base_dir, exp, 
                args.dataset, args.mode, args.legacy
            )
            if data is not None:
                all_data.append(data)
    
    if not all_data:
        print(f"{Colors.RED}Lỗi: Không tìm thấy dữ liệu cho bất kỳ experiment nào{Colors.END}")
        print(f"\nKiểm tra các thư mục:")
        for exp in args.experiments:
            if args.legacy:
                path = os.path.join(args.base_dir, f"results_{exp}")
            else:
                path = os.path.join(args.base_dir, "results", args.dataset, args.mode, exp)
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {path}")
        sys.exit(1)
    
    # Gộp dữ liệu
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Kiểm tra experiments được load
    loaded_experiments = combined_df['experiment'].unique().tolist()
    missing = [e for e in args.experiments if e not in loaded_experiments]
    if missing:
        print(f"{Colors.YELLOW}Cảnh báo: Không tìm thấy dữ liệu cho: {', '.join(missing)}{Colors.END}")
    
    # Chỉ giữ experiments được load thành công
    experiments = [e for e in args.experiments if e in loaded_experiments]
    
    if len(experiments) < 2:
        print(f"{Colors.RED}Lỗi: Cần ít nhất 2 experiments có dữ liệu để so sánh{Colors.END}")
        sys.exit(1)
    
    # Tính toán
    scene_data, available_metrics = calculate_scene_averages(combined_df)
    overall_data = calculate_overall_averages(combined_df, available_metrics)
    
    # In bảng so sánh
    print_comparison_table(experiments, scene_data, overall_data, available_metrics)
    
    # In tóm tắt
    print_winner_summary(experiments, overall_data, available_metrics)
    
    # Xuất CSV nếu cần
    if args.output:
        export_comparison_csv(experiments, scene_data, overall_data, 
                              available_metrics, args.output)
    
    print(f"\n{Colors.HEADER}Hoàn thành so sánh!{Colors.END}")

if __name__ == '__main__':
    main()
