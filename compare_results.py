import argparse
from pathlib import Path
import sys
import pandas as pd


# Định nghĩa metric và chiều tốt/xấu
# True  = lớn hơn là tốt hơn
# False = nhỏ hơn là tốt hơn
METRIC_DIRECTIONS = {
    "psnr": True,
    "ssim": True,
    "lpips": False,
    "ate_rmse": False,
    "accuracy": False,
    "completion": False,
    "comp_ratio": True,
    "chamfer": False,
}


def load_and_clean_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    if p.stat().st_size == 0:
        raise ValueError(f"File rỗng: {path}")

    # Bỏ qua dòng lỗi format
    df = pd.read_csv(path, on_bad_lines="skip")

    if "scene" not in df.columns:
        raise ValueError(f"File {path} không có cột 'scene'.")

    # Chuẩn hóa cột scene
    df["scene"] = df["scene"].astype(str).str.strip()
    df = df[df["scene"] != ""]

    # Chỉ giữ các metric có trong file
    available_metrics = [m for m in METRIC_DIRECTIONS.keys() if m in df.columns]
    if not available_metrics:
        raise ValueError(f"File {path} không có metric hợp lệ nào trong {list(METRIC_DIRECTIONS.keys())}")

    # Convert numeric; lỗi thì thành NaN (vd N/A)
    for m in available_metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # Bỏ dòng không có bất kỳ metric nào
    df = df.dropna(subset=available_metrics, how="all")
    if df.empty:
        raise ValueError(f"File {path} không còn dữ liệu hợp lệ sau khi làm sạch.")

    # Giữ cột cần thiết
    keep_cols = ["scene"] + available_metrics
    if "run" in df.columns:
        keep_cols.insert(1, "run")
    return df[keep_cols]


def aggregate_by_scene(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in df.columns if c in METRIC_DIRECTIONS]
    # mean bỏ qua NaN mặc định
    agg = df.groupby("scene", dropna=False)[metrics].mean().reset_index()
    # thêm số run hợp lệ trên mỗi scene
    count_df = df.groupby("scene", dropna=False).size().rename("num_rows").reset_index()
    agg = agg.merge(count_df, on="scene", how="left")
    return agg


def compare_scene_metrics(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str):
    merged = a.merge(b, on="scene", how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        return pd.DataFrame(), {"wins_a": 0, "wins_b": 0, "ties": 0, "total": 0}

    rows = []
    wins_a = wins_b = ties = 0

    metrics_common = [
        m for m in METRIC_DIRECTIONS
        if f"{m}_a" in merged.columns and f"{m}_b" in merged.columns
    ]

    for _, r in merged.iterrows():
        scene = r["scene"]
        for m in metrics_common:
            va = r[f"{m}_a"]
            vb = r[f"{m}_b"]

            if pd.isna(va) or pd.isna(vb):
                winner = "NA"
            else:
                higher_better = METRIC_DIRECTIONS[m]
                if va == vb:
                    winner = "tie"
                    ties += 1
                else:
                    a_better = (va > vb) if higher_better else (va < vb)
                    if a_better:
                        winner = name_a
                        wins_a += 1
                    else:
                        winner = name_b
                        wins_b += 1

            rows.append({
                "scene": scene,
                "metric": m,
                f"{name_a}_mean": va,
                f"{name_b}_mean": vb,
                "winner": winner
            })

    out = pd.DataFrame(rows)
    summary = {"wins_a": wins_a, "wins_b": wins_b, "ties": ties, "total": wins_a + wins_b + ties}
    return out, summary


def main():
    parser = argparse.ArgumentParser(description="So sánh 2 file kết quả theo mean từng scene.")
    parser.add_argument("--a", required=True, help="CSV A")
    parser.add_argument("--b", required=True, help="CSV B")
    parser.add_argument("--name-a", default="A", help="Tên hiển thị cho file A")
    parser.add_argument("--name-b", default="B", help="Tên hiển thị cho file B")
    parser.add_argument("--out-dir", default=".", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    try:
        df_a = load_and_clean_csv(args.a)
        df_b = load_and_clean_csv(args.b)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    agg_a = aggregate_by_scene(df_a)
    agg_b = aggregate_by_scene(df_b)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lưu mean theo scene
    agg_a.to_csv(out_dir / f"mean_by_scene_{args.name_a}.csv", index=False)
    agg_b.to_csv(out_dir / f"mean_by_scene_{args.name_b}.csv", index=False)

    # Compare
    cmp_df, summary = compare_scene_metrics(agg_a, agg_b, args.name_a, args.name_b)
    if cmp_df.empty:
        print("[WARN] Không có scene chung để so sánh.")
    else:
        cmp_path = out_dir / f"comparison_{args.name_a}_vs_{args.name_b}.csv"
        cmp_df.to_csv(cmp_path, index=False)
        print(f"[OK] Đã lưu: {cmp_path}")

    # Báo cáo scene chỉ có ở 1 file
    scenes_a = set(agg_a["scene"].tolist())
    scenes_b = set(agg_b["scene"].tolist())
    only_a = sorted(scenes_a - scenes_b)
    only_b = sorted(scenes_b - scenes_a)

    print("\n=== SUMMARY ===")
    print(f"{args.name_a} wins: {summary['wins_a']}")
    print(f"{args.name_b} wins: {summary['wins_b']}")
    print(f"Ties: {summary['ties']}")
    print(f"Total compared cells: {summary['total']}")

    if only_a:
        print(f"\nScene chỉ có trong {args.name_a}: {only_a}")
    if only_b:
        print(f"Scene chỉ có trong {args.name_b}: {only_b}")

    # Overall winner
    if summary["wins_a"] > summary["wins_b"]:
        print(f"\n=> Overall: {args.name_a} tốt hơn")
    elif summary["wins_b"] > summary["wins_a"]:
        print(f"\n=> Overall: {args.name_b} tốt hơn")
    else:
        print("\n=> Overall: Hòa")


if __name__ == "__main__":
    main()