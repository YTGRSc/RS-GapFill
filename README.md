# 遥感影像缺失数据重建 / Remote Sensing Image Gap Filling

> 综述 · 方法 · 代码框架 — 从传统插值到扩散模型

---

## 1. 问题背景

遥感影像常因以下原因产生数据缺失：

| 缺失来源 | 说明 | 典型场景 |
|---|---|---|
| ☁️ 云/云影遮挡 | 光学遥感最普遍的干扰 | Landsat, Sentinel-2, MODIS |
| 🛰️ 传感器故障 | 硬件导致系统性缺失 | Landsat 7 ETM+ SLC-off (2003年起, 22%条带缺失) |
| 🌫️ 大气条件 | 气溶胶、雾霾、沙尘 | 城市区域、干旱季节 |
| ⚫ 扫描间隙 | 宽幅传感器拼接空隙 | MODIS Aqua Band 6 (2005年失效) |
| 🔭 观测几何 | BRDF 效应导致的异常值 | 高纬度地区、山区 |

**目标**: 从已有的时空邻域信息中，推断并重建合理的像素值。

---

## 2. 方法全景

```
                       遥感影像缺失修复方法
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        传统方法          机器学习方法      深度学习方法
              │               │               │
     ┌────────┼────────┐  ┌───┴───┐  ┌───────┼───────┐
     │        │        │  │       │  │       │       │
   空间    时间    时空    RF    CNN    GAN  Transformer
   插值    替换    融合   SVM    LSTM   扩散模型  MAE
```

### 2.1 传统方法

#### 空间插值 (Spatial Interpolation)
利用同一时相邻近像素填充缺失区域。

| 方法 | 原理 | 适用场景 |
|---|---|---|
| 反距离加权 (IDW) | 距离越近权重越大 | 小范围缺失 |
| 克里金 (Kriging) | 基于半变异函数的空间最优估计 | 连续地物 |
| 样条插值 (Spline) | 通过已知点拟合光滑曲面 | 地形数据 |

#### 时间替换 (Temporal Substitution)
用相邻日期同一位置的影像填补。

$$
\hat{X}(t) = X(t-1) \quad \text{或} \quad \hat{X}(t) = \frac{X(t-1) + X(t+1)}{2}
$$

简单有效，但对地表突变（如收割、火灾）不鲁棒。

#### 时空融合方法 (Spatiotemporal Fusion)

| 方法 | 论文 | 核心思路 |
|---|---|---|
| **STARFM** | Gao et al., 2006, RSE | 基于光谱相似性加权的 MODIS-Landsat 融合 |
| **ESTARFM** | Zhu et al., 2010, RSE | 引入混合像元分解，增强异质性区域表现 |
| **NSPI** | Chen et al., 2011, RSE | 邻域相似像元插值 — Landsat 7 SLC-off 填充 |
| **GNSPI** | Zhu et al., 2012, RSE | 地理统计+非局部滤波改进版 NSPI |
| **FSDAF** | Zhu et al., 2016, RSE | 灵活的亚像元级时空融合 |

---

### 2.2 深度学习方法

#### CNN 系列

| 方法 | 论文 | 核心思路 |
|---|---|---|
| **Context Encoder** | Pathak et al., CVPR 2016 | 编码器-解码器 + 对抗损失，开创图像修复范式 |
| **PConv (Partial Conv)** | Liu et al., ECCV 2018 | 只在有效像素上卷积，mask 随层自动更新 |
| **GatedConv** | Yu et al., ICCV 2019 | 门控卷积替代部分卷积，适用于不规则 mask |
| **DeepFill v2** | Yu et al., CVPR 2019 | 上下文注意力 + 门控卷积 |

> 🛠️ **PConv 在遥感中的应用**: 适用于不规则云斑块修复，mask 自动收缩特性对厚云尤其有效。

#### GAN 系列

| 方法 | 论文 | 核心思路 |
|---|---|---|
| **Pix2Pix** | Isola et al., CVPR 2017 | 条件 GAN，L1 + 对抗损失 |
| **CycleGAN** | Zhu et al., ICCV 2017 | 无配对训练，适合 SAR→光学翻译替代云区 |
| **SpA-GAN** | Pan et al., 2021 | 空间注意力 GAN 专门针对遥感去云 |
| **CTGAN** | Wen et al., 2021 | 基于 Transformer 的遥感影像薄云去除 |

#### Transformer & 自注意力

| 方法 | 论文 | 核心思路 |
|---|---|---|
| **SwinIR** | Liang et al., ICCVW 2021 | Swin Transformer + 残差块用于图像复原 |
| **Restormer** | Zamir et al., CVPR 2022 | Transformer 架构的多尺度图像复原 |
| **Uformer** | Wang et al., CVPR 2022 | U-Net 结构 Transformer，通用图像修复 |
| **Stripformer** | Tsai et al., ECCV 2022 | 水平和垂直条带注意力去模糊/修复 |

#### MAE 自监督范式 (⭐ 当前热门)

| 方法 | 论文 | 核心思路 |
|---|---|---|
| **MAE** | He et al., CVPR 2022 | 随机遮蔽 → 编码可见块 → 解码重建，简单而强大 |
| **SatMAE** | Cong et al., NeurIPS 2022 | MAE + 时空位置编码，适配遥感多光谱特性 |
| **SpectralMAE** | Sun et al., 2023 | 光谱维度 MAE 预训练 |
| **GFMM** | Wang et al., 2024 | Generative Fusion of Multi-source and Multi-temporal |

> 💡 **核心洞察**: MAE 让网络学会从部分观察推测完整场景——本质上就是一个天然的"缺失修复"预训练任务。

#### 扩散模型 (Diffusion Models) 🔥

| 方法 | 论文 | 核心思路 |
|---|---|---|
| **RePaint** | Lugmayr et al., CVPR 2022 | 利用预训练扩散模型，仅对缺失区域去噪 |
| **DDPM-CR** | Jing et al., 2023 | 去噪扩散概率模型遥感去云 |
| **DiffusionRSS** | Liu et al., 2024 | 扩散模型遥感影像修复基准 |

> 🎯 **为什么扩散模型适合缺失修复？**
> 扩散过程的逆向本质上就是"从噪声/部分观测 → 完整图像"的重建过程，天然匹配修复任务。

---

### 2.3 多源数据融合方法

| 方法 | 思路 |
|---|---|
| **SAR-光学融合** | SAR 不受云雨影响，通过跨模态翻译填补光学缺失 |
| **多传感器协同** | MODIS (高时间) + Landsat (高空间) 互补 |
| **DEM/地形辅助** | 利用高程信息约束山地阴影区域的修复 |

---

## 3. 常用数据集

| 数据集 | 描述 | 规模 |
|---|---|---|
| **SEN12MS-CR** | Sentinel-2 全球分布 + 人工云覆盖 | 169 对 |
| **RICE** | 遥感影像云去除基准 | 500 对 |
| **WHU Cloud** | 武汉大学标注云检测数据集 | 859 景 |
| **T-Cloud** | 时间序列去云数据集 | 多时相 |
| **Landsat 7 SLC-off** | 自然条带缺失场景 | 全球覆盖 |

---

## 4. 评价指标

| 指标 | 公式/说明 | 关注点 |
|---|---|---|
| **PSNR** | $10\log_{10}(MAX^2/MSE)$ | 像素级保真度 |
| **SSIM** | 亮度 × 对比度 × 结构 | 结构相似性 |
| **SAM** | 光谱角制图 | 光谱保真度 |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(\hat{x}-x)^2}$ | 绝对误差 |
| **LPIPS** | 深度特征感知距离 | 感知质量（人眼） |

---

## 5. 开源工具速览

| 工具 | 生态 | 能力 |
|---|---|---|
| [scikit-image](https://scikit-image.org) | Python | 基础图像修复（inpaint_biharmonic, NL-means） |
| [OpenCV inpainting](https://docs.opencv.org) | C++/Python | Navier-Stokes, Telea 快速修补 |
| [LAMA](https://github.com/advimman/lama) | PyTorch | 傅里叶卷积 + 大感受野修复（通用图像） |
| [MAT](https://github.com/fenglinglwb/MAT) | PyTorch | Mask-Aware Transformer 修复 |
| [I2INet](https://github.com/HIT-LEI/I2INet) | PyTorch | 遥感云修复专用 |
| **本仓库** | Python | 栅格时序缺失修复框架 ↓ |

---

## 6. 本仓库代码框架

```
RS-GapFill/
├── README.md
├── requirements.txt
├── gapfill/
│   ├── __init__.py
│   ├── traditional.py    # 传统插值修复
│   ├── temporal.py       # 时间维度修复
│   └── utils.py          # 工具函数 & 评价指标
└── example.py
```

### 已实现的功能

- **空间修复**: IDW、双线性、中值滤波修复
- **时间修复**: 前后时间平均、最近邻时间替换
- **时空修复**: 时空平滑窗口填补
- **STL 预处理**: 去季节/趋势后修复（避免季节断裂）
- **评价**: PSNR, SSIM, RMSE, 缺失率统计

### 适用场景

- 遥感时序数据 (NDVI, EVI, LST, 反射率等) 的批量缺失修复
- Landsat 7 SLC-off 条带填补
- 云覆盖区域重建
- Sentinel-2 / MODIS 时序质量控制

---

## 7. 参考文献 (精选)

1. **Gao, F.**, et al. (2006). On the blending of the Landsat and MODIS surface reflectance. *Remote Sensing of Environment*.
2. **Chen, J.**, et al. (2011). A simple and effective method for filling gaps in Landsat ETM+ SLC-off images. *Remote Sensing of Environment*.
3. **Zhu, X.**, et al. (2010). An enhanced spatial and temporal adaptive reflectance fusion model. *Remote Sensing of Environment*.
4. **Zhu, X.**, et al. (2016). A flexible spatiotemporal method for fusing satellite images. *Remote Sensing of Environment*.
5. **Liu, G.**, et al. (2018). Image inpainting for irregular holes using partial convolutions. *ECCV*.
6. **He, K.**, et al. (2022). Masked autoencoders are scalable vision learners. *CVPR*.
7. **Cong, Y.**, et al. (2022). SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. *NeurIPS*.
8. **Lugmayr, A.**, et al. (2022). RePaint: Inpainting using denoising diffusion probabilistic models. *CVPR*.

---

## 8. 趋势与展望

| 方向 | 描述 |
|---|---|
| 🔮 **基础模型 (FM4RS)** | 遥感大模型 (Prithvi, SatCLIP, RemoteCLIP) 的零样本缺失修复能力 |
| 🔮 **扩散+生成** | 从"修复"到"生成"——不只是填补，而是最合理的"想象" |
| 🔮 **多模态融合** | SAR + 光学 + DEM + 气象 = 全维度约束修复 |
| 🔮 **自监督预训练** | MAE 范式的持续演进，无需标注的修复能力 |
| 🔮 **3D 时空修复** | 将时间维度作为第三维度统一建模，而非逐帧处理 |

---

> 📌 **License**: MIT — 欢迎 Star ⭐ & PR
>
> 📬 **作者**: YTGRSc | 研究方向: GIS · Remote Sensing · Deep Learning
