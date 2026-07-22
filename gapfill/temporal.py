"""
时间维度缺失修复方法。

支持:
  - 前后时间平均
  - 最近邻时间替换
  - 线性时间插值
  - 时空联合修复
"""

import numpy as np


def temporal_mean_fill(data, window=2):
    """
    用前后 window 帧的均值填充缺失值。

    Parameters
    ----------
    data : np.ndarray, shape (time, rows, cols)
    window : int
        向前/向后各取 window 帧。

    Returns
    -------
    filled : np.ndarray
    """
    n_time, rows, cols = data.shape
    filled = data.copy()

    for t in range(n_time):
        mask = np.isnan(filled[t])
        if not mask.any():
            continue

        # 收集前后帧
        t_start = max(0, t - window)
        t_end = min(n_time, t + window + 1)
        neighbors = []
        for ti in range(t_start, t_end):
            if ti != t:
                neighbors.append(filled[ti])

        if neighbors:
            stack = np.stack(neighbors, axis=0)  # (n_neighbors, rows, cols)
            filled[t][mask] = np.nanmean(stack[:, mask], axis=0)

    return filled


def temporal_nearest_fill(data):
    """
    用时间上最近的未缺失值填充。

    Parameters
    ----------
    data : np.ndarray, shape (time, rows, cols)

    Returns
    -------
    filled : np.ndarray
    """
    n_time, rows, cols = data.shape
    filled = data.copy()

    for r in range(rows):
        for c in range(cols):
            series = filled[:, r, c].copy()
            nan_mask = np.isnan(series)

            if not nan_mask.any():
                continue
            if nan_mask.all():
                continue  # 全部缺失，跳过

            valid_idx = np.where(~nan_mask)[0]
            missing_idx = np.where(nan_mask)[0]

            for mi in missing_idx:
                # 找最近的前后有效值
                if mi < valid_idx[0]:
                    series[mi] = series[valid_idx[0]]
                elif mi > valid_idx[-1]:
                    series[mi] = series[valid_idx[-1]]
                else:
                    # 二分搜索前后
                    prev = valid_idx[valid_idx < mi]
                    nxt = valid_idx[valid_idx > mi]
                    prev = prev[-1]
                    nxt = nxt[0]
                    # 取近的那个
                    if mi - prev <= nxt - mi:
                        series[mi] = series[prev]
                    else:
                        series[mi] = series[nxt]

            filled[:, r, c] = series

    return filled


def temporal_linear_fill(data):
    """
    逐像素时间序列线性插值。

    Parameters
    ----------
    data : np.ndarray, shape (time, rows, cols)

    Returns
    -------
    filled : np.ndarray
    """
    import pandas as pd

    n_time, rows, cols = data.shape
    filled = data.copy()

    for r in range(rows):
        for c in range(cols):
            series = filled[:, r, c]
            s = pd.Series(series)
            if s.isna().any() and not s.isna().all():
                s = s.interpolate(method='linear', limit_direction='both')
                filled[:, r, c] = s.values

    return filled


def spatiotemporal_fill(data, method='temporal_nearest', **kwargs):
    """
    时空联合修复：先时间，再空间。

    Parameters
    ----------
    data : np.ndarray, shape (time, rows, cols)
    method : str
        'temporal_mean' | 'temporal_nearest' | 'temporal_linear'

    Returns
    -------
    filled : np.ndarray
    """
    # Step 1: 时间修复
    if method == 'temporal_mean':
        filled = temporal_mean_fill(data, **kwargs)
    elif method == 'temporal_nearest':
        filled = temporal_nearest_fill(data)
    elif method == 'temporal_linear':
        filled = temporal_linear_fill(data)
    else:
        raise ValueError(f"未知方法: {method}")

    # Step 2: 空间残差修复（对时间无法修复的像素）
    from .traditional import spatial_fill
    for t in range(filled.shape[0]):
        residual_mask = np.isnan(filled[t])
        if residual_mask.any():
            filled[t] = spatial_fill(filled[t], residual_mask, method='median')

    return filled


def temporal_fill(data, method='temporal_nearest', **kwargs):
    """
    时间修复统一接口。

    Parameters
    ----------
    data : np.ndarray, shape (time, rows, cols)
    method : 'temporal_mean' | 'temporal_nearest' | 'temporal_linear'

    Returns
    -------
    filled : np.ndarray
    """
    if method == 'temporal_mean':
        return temporal_mean_fill(data, **kwargs)
    elif method == 'temporal_nearest':
        return temporal_nearest_fill(data)
    elif method == 'temporal_linear':
        return temporal_linear_fill(data)
    else:
        raise ValueError(f"未知方法: {method}")
