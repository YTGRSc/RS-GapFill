"""
传统空间插值修复方法。

支持:
  - IDW (反距离加权)
  - 双线性插值
  - 中值滤波修复
  - 邻域均值填充
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import median_filter


def _validate_input(data):
    """验证输入为 2D 或 3D 数组。"""
    if data.ndim not in (2, 3):
        raise ValueError(f"输入应为 2D (rows, cols) 或 3D (time, rows, cols)，得到 {data.ndim}D")


def idw_fill(data, mask, radius=5, power=2):
    """
    反距离加权 (Inverse Distance Weighting) 填充。

    Parameters
    ----------
    data : np.ndarray, shape (rows, cols)
        含缺失值的图像。
    mask : np.ndarray, bool, shape (rows, cols)
        True 表示缺失像素。
    radius : int
        搜索半径（像素）。
    power : int
        距离权重指数。

    Returns
    -------
    filled : np.ndarray
    """
    rows, cols = data.shape
    filled = data.copy()
    missing = np.argwhere(mask)

    # 有效像素索引和值
    valid = np.argwhere(~mask)
    valid_vals = data[~mask]

    if len(valid) == 0:
        return filled  # 全部缺失，无法修复

    for r, c in missing:
        r_min = max(0, r - radius)
        r_max = min(rows, r + radius + 1)
        c_min = max(0, c - radius)
        c_max = min(cols, c + radius + 1)

        local_mask = mask[r_min:r_max, c_min:c_max]
        local_data = data[r_min:r_max, c_min:c_max]
        local_valid = ~local_mask

        if local_valid.sum() == 0:
            continue  # 邻域全缺失

        # 计算距离
        lr, lc = np.where(local_valid)
        distances = np.sqrt((lr - (r - r_min))**2 + (lc - (c - c_min))**2)
        distances = np.clip(distances, 1e-10, None)  # 避免除零
        weights = 1.0 / (distances ** power)
        weights /= weights.sum()

        filled[r, c] = np.sum(local_data[local_valid] * weights)

    return filled


def bilinear_fill(data, mask):
    """
    双线性插值填充（使用 scipy.griddata）。

    Parameters
    ----------
    data : np.ndarray, shape (rows, cols)
    mask : np.ndarray, bool

    Returns
    -------
    filled : np.ndarray
    """
    rows, cols = data.shape
    filled = data.copy()

    valid = np.argwhere(~mask)
    missing = np.argwhere(mask)

    if len(valid) < 4 or len(missing) == 0:
        return filled

    valid_vals = data[~mask]

    filled_vals = griddata(
        points=valid, values=valid_vals,
        xi=missing, method='linear', fill_value=np.nan
    )

    for (r, c), val in zip(missing, filled_vals):
        filled[r, c] = val

    return filled


def median_fill(data, mask, size=5):
    """
    中值滤波修复：用邻域有效值的中位数填充。

    Parameters
    ----------
    data : np.ndarray, shape (rows, cols)
    mask : np.ndarray, bool
    size : int
        滤波窗口大小。

    Returns
    -------
    filled : np.ndarray
    """
    # 先将缺失值设为 NaN，再中值滤波
    tmp = data.copy().astype(np.float32)
    tmp[mask] = np.nan

    # 对于每个缺失像素，取邻域非 NaN 中值
    filled = data.copy()
    rows, cols = data.shape
    h = size // 2

    for r in range(rows):
        for c in range(cols):
            if mask[r, c]:
                r_min, r_max = max(0, r - h), min(rows, r + h + 1)
                c_min, c_max = max(0, c - h), min(cols, c + h + 1)
                window = tmp[r_min:r_max, c_min:c_max]
                valid_vals = window[~np.isnan(window)]
                if len(valid_vals) > 0:
                    filled[r, c] = np.median(valid_vals)

    return filled


def spatial_fill(data, mask=None, method='idw', **kwargs):
    """
    空间插值修复的统一接口。

    Parameters
    ----------
    data : np.ndarray, shape (rows, cols) or (time, rows, cols)
    mask : np.ndarray, bool, optional
        缺失掩膜。如不提供，自动根据 NaN 生成。
    method : {'idw', 'bilinear', 'median'}
    **kwargs : 传递给具体方法。

    Returns
    -------
    filled : np.ndarray
    """
    _validate_input(data)

    if mask is None:
        mask = np.isnan(data) if data.ndim == 2 else np.isnan(data[0])

    if data.ndim == 2:
        if method == 'idw':
            return idw_fill(data, mask, **kwargs)
        elif method == 'bilinear':
            return bilinear_fill(data, mask)
        elif method == 'median':
            return median_fill(data, mask, **kwargs)
        else:
            raise ValueError(f"未知方法: {method}")
    else:
        # 3D: 逐时间片处理
        filled = np.empty_like(data)
        for t in range(data.shape[0]):
            filled[t] = spatial_fill(data[t], mask, method, **kwargs)
        return filled
