"""
工具函数 & 评价指标模块。
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def missing_ratio(data):
    """计算缺失率。"""
    return np.isnan(data).sum() / data.size


def simulate_missing(data, ratio=0.2, pattern='random', **kwargs):
    """
    生成模拟缺失数据用于方法评估。

    Parameters
    ----------
    data : np.ndarray
        原始完整数据（2D 或 3D）。
    ratio : float
        缺失比例。
    pattern : {'random', 'stripe', 'block'}
        - 'random': 随机缺失
        - 'stripe': 条带缺失（模拟 Landsat 7 SLC-off）
        - 'block': 块状缺失（模拟云覆盖）

    Returns
    -------
    corrupted : np.ndarray
    mask : np.ndarray, bool
    """
    corrupted = data.copy().astype(np.float32)
    shape = data.shape

    if pattern == 'random':
        mask = np.random.random(shape) < ratio

    elif pattern == 'stripe':
        # 模拟条带: 沿列方向周期性缺失
        mask = np.zeros(shape, dtype=bool)
        if data.ndim == 3:
            for b in range(shape[1]):
                if np.random.random() < ratio:
                    mask[:, :, b::kwargs.get('gap', 8)] = True
        else:
            for b in range(shape[1]):
                if np.random.random() < ratio:
                    mask[:, b::kwargs.get('gap', 8)] = True

    elif pattern == 'block':
        # 模拟块状: 随机矩形块
        mask = np.zeros(shape, dtype=bool)
        block_size = kwargs.get('block_size', 16)
        n_blocks = int(ratio * shape[-2] * shape[-1] / (block_size ** 2))

        if data.ndim == 3:
            t, r, c = shape
            for _ in range(n_blocks):
                ti = np.random.randint(0, t)
                ri = np.random.randint(0, max(1, r - block_size))
                ci = np.random.randint(0, max(1, c - block_size))
                mask[ti, ri:ri + block_size, ci:ci + block_size] = True
        else:
            r, c = shape
            for _ in range(n_blocks):
                ri = np.random.randint(0, max(1, r - block_size))
                ci = np.random.randint(0, max(1, c - block_size))
                mask[ri:ri + block_size, ci:ci + block_size] = True

    else:
        raise ValueError(f"未知 pattern: {pattern}")

    corrupted[mask] = np.nan
    return corrupted, mask


def compute_metrics(original, filled, mask=None):
    """
    计算修复质量评价指标。

    Parameters
    ----------
    original : np.ndarray
        原始真值。
    filled : np.ndarray
        修复结果。
    mask : np.ndarray, bool, optional
        仅评估 mask 区域。

    Returns
    -------
    metrics : dict
        {'psnr': float, 'ssim': float, 'rmse': float, 'mae': float}
    """
    if mask is not None:
        orig = original[mask]
        fill = filled[mask]
    else:
        orig = original.ravel()
        fill = filled.ravel()

    # 排除 NaN
    valid = ~(np.isnan(orig) | np.isnan(fill))
    orig = orig[valid]
    fill = fill[valid]

    if len(orig) == 0:
        return {'psnr': float('nan'), 'ssim': float('nan'),
                'rmse': float('nan'), 'mae': float('nan')}

    mse = np.mean((orig - fill) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(orig - fill))

    # PSNR
    data_range = orig.max() - orig.min()
    if data_range > 0:
        psnr_val = 10 * np.log10(data_range ** 2 / mse) if mse > 0 else float('inf')
    else:
        psnr_val = float('nan')

    # SSIM (需要 2D 输入)
    if mask is not None and mask.ndim == 2:
        ssim_val = ssim(orig.reshape(-1, 1), fill.reshape(-1, 1),
                        data_range=1)  # 简化处理
    else:
        ssim_val = float('nan')

    return {'psnr': psnr_val, 'ssim': ssim_val, 'rmse': rmse, 'mae': mae}
