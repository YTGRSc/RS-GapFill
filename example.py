"""
遥感影像缺失修复示例

演示:
  1. 模拟缺失（随机、条带、块状）
  2. 多种修复方法对比
  3. 质量评估
"""

import numpy as np
from gapfill import spatial_fill, temporal_fill, spatiotemporal_fill
from gapfill.utils import simulate_missing, compute_metrics


def main():
    # ==========================================================
    # 创建模拟数据: 一个 3 波段 × 100 行 × 100 列 的简单网格
    # ==========================================================
    print("=" * 60)
    print("遥感影像缺失修复 — 方法对比演示")
    print("=" * 60)

    np.random.seed(42)
    rows, cols, bands = 100, 100, 3

    # 模拟多光谱影像
    x = np.linspace(0, 4 * np.pi, rows)
    y = np.linspace(0, 4 * np.pi, cols)
    xx, yy = np.meshgrid(x, y)
    base = np.sin(xx) * np.cos(yy)  # (rows, cols)
    image = np.stack([base * (0.5 + 0.3 * b) for b in range(bands)], axis=0)
    # → (bands, rows, cols)

    print(f"\n原始影像: {image.shape}")

    # ==========================================================
    # 模拟条带缺失 (Landsat 7 SLC-off 风格)
    # ==========================================================
    print("\n--- 模拟条带缺失 ---")
    corrupted, mask = simulate_missing(image, ratio=0.22, pattern='stripe', gap=7)
    print(f"缺失率: {mask.sum() / mask.size:.2%}")

    # ==========================================================
    # 方法 1: 空间 IDW 修复
    # ==========================================================
    print("\n[1] 空间 IDW 修复...")
    filled_idw = spatial_fill(corrupted, mask[0], method='idw', radius=5)
    m_idw = compute_metrics(image, filled_idw, mask)
    print(f"    PSNR={m_idw['psnr']:.2f} dB, RMSE={m_idw['rmse']:.4f}")

    # ==========================================================
    # 方法 2: 空间中值修复
    # ==========================================================
    print("[2] 空间中值修复...")
    filled_median = spatial_fill(corrupted, mask[0], method='median', size=5)
    m_med = compute_metrics(image, filled_median, mask)
    print(f"    PSNR={m_med['psnr']:.2f} dB, RMSE={m_med['rmse']:.4f}")

    # ==========================================================
    # 方法 3: 时间维度修复（模拟 10 时相）
    # ==========================================================
    print("\n[3] 时间线性插值（模拟时序)...")
    n_time = 10
    time_series = np.stack([image + 0.1 * i * np.random.randn(*image.shape)
                            for i in range(n_time)], axis=0)
    # (time, bands, rows, cols) → 重新组织为 (bands, time, rows, cols)
    # 为简单起见，在第一个波段上演示

    ts_band0 = time_series[:, 0, :, :]  # (time, rows, cols)
    # 随机缺失 10% 时间步
    corrupted_ts = ts_band0.copy()
    for t in range(n_time):
        rand_mask = np.random.random((rows, cols)) < 0.1
        corrupted_ts[t][rand_mask] = np.nan

    filled_temporal = temporal_fill(corrupted_ts, method='temporal_linear')
    m_temp = compute_metrics(ts_band0, filled_temporal)
    print(f"    PSNR={m_temp['psnr']:.2f} dB, RMSE={m_temp['rmse']:.4f}")

    # ==========================================================
    # 方法对比汇总
    # ==========================================================
    print("\n" + "=" * 60)
    print("方法对比汇总")
    print("=" * 60)
    print(f"{'方法':<20s} {'PSNR(dB)':<12s} {'RMSE':<10s} {'MAE':<10s}")
    print("-" * 52)
    for name, m in [("Spatial-IDW", m_idw),
                     ("Spatial-Median", m_med),
                     ("Temporal-Linear", m_temp)]:
        print(f"{name:<20s} {m['psnr']:<12.2f} {m['rmse']:<10.4f} {m['mae']:<10.4f}")

    print("\n✅ 演示完成！")


if __name__ == '__main__':
    main()
