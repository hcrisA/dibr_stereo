# DIBR单目转立体视频系统

基于深度图像渲染（DIBR）的单目视频转立体视频系统，支持MiDaS和Depth-Anything-V2两种深度估计模型。

## 项目结构

```
DIBR/
├── Data/                           # 数据目录
│   ├── checkpoints/                # 模型权重（需下载）
│   │   ├── dpt_hybrid_384.pt       # MiDaS DPT-Hybrid权重
│   │   └── depth_anything_v2_vits.pth  # Depth-Anything-V2-Small
│   ├── mono2stereo-test/           # 测试数据集（需放置）
│   │   ├── animation/left/, right/
│   │   ├── simple/left/, right/
│   │   ├── complex/left/, right/
│   │   ├── indoor/left/, right/
│   │   └── outdoor/left/, right/
│   └── outputs/                    # 输出目录（自动创建）
│
├── dibr_stereo/                    # 代码目录
│   ├── MiDaS/                      # MiDaS仓库（已包含）
│   │   └── midas/dpt_depth.py
│   ├── depth_anything_v2/          # Depth-Anything-V2模块（已包含）
│   │   └── dpt.py
│   ├── requirements.txt            # 依赖列表
│   ├── benchmark_eval_v2.py        # MiDaS评估脚本
│   ├── download_depth_anything_v2.py  # Depth-Anything-V2评估脚本
│   └── ...
│
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
conda create -n dibr python==3.12
conda activate dibr
cd dibr_stereo
pip install -r requirements.txt
```

### 2. 下载模型权重

权重文件需放置到 `Data/checkpoints/` 目录：

**MiDaS DPT-Hybrid (384MB)**：
```bash
wget -O Data/checkpoints/dpt_hybrid_384.pt \
  https://github.com/intel-isl/DPT/releases/download/1.0/dpt_hybrid_384.pt
```

**Depth-Anything-V2-Small (24.8MB)**：
```bash
wget -O Data/checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

### 3. 放置测试数据集

将测试数据集放置到 `Data/mono2stereo-test/` 目录，结构如下：
```
Data/mono2stereo-test/
├── animation/left/, right/
├── simple/left/, right/
├── complex/left/, right/
├── indoor/left/, right/
└── outdoor/left/, right/
```

### 4. 运行测试

#### 方法A：使用MiDaS模型测试

```bash
cd dibr_stereo

# 一键测试（评估所有场景）
python test_mono2stereo_v3.py

# 或指定参数
python benchmark_eval_v2.py \
  --dataset ../Data/mono2stereo-test \
  --model-path ../Data/checkpoints/dpt_hybrid_384.pt \
  --output-dir ../Data/outputs/midas_results \
  --device cuda
```

#### 方法B：使用Depth-Anything-V2测试

```bash
cd dibr_stereo

# 使用Small模型
python download_depth_anything_v2.py \
  --dataset ../Data/mono2stereo-test \
  --model-path ../Data/checkpoints/depth_anything_v2_vits.pth \
  --output-dir ../Data/outputs/depth_anything_v2_results \
  --encoder vits \
  --device cuda
```


## 模型对比

| 模型 | 大小 | PSNR | SSIM | SIoU | FPS | 特点 |
|------|------|------|------|------|-----|------|
| MiDaS DPT-Hybrid | 384MB | 32-34dB | 0.76-0.80 | 0.28-0.32 | 30-35 | 平衡性能 |
| Depth-Anything-V2-Small | 24.8MB | 31-33dB | 0.74-0.78 | 0.26-0.30 | 45-55 | 轻量快速 |
| Depth-Anything-V2-Base | 97.5MB | 33-35dB | 0.78-0.82 | 0.30-0.35 | 30-40 | 高质量 |

**推荐选择**：
- 追求速度/边缘部署：Depth-Anything-V2-Small
- 追求质量：MiDaS DPT-Hybrid 或 Depth-Anything-V2-Base
- 平衡选择：MiDaS DPT-Hybrid

## 输出说明

### 输出目录结构

```
Data/outputs/
├── midas_results/                  # MiDaS模型输出
│   ├── animation/                  # 各场景输出图像
│   ├── simple/
│   ├── complex/
│   ├── indoor/
│   ├── outdoor/
│   ├── evaluation_report.json      # 完整评估报告
│   ├── summary.json                # 简要结果
│   └── results.csv                 # CSV格式结果
│
└── depth_anything_v2_results/      # Depth-Anything-V2输出
    └── ...
```

### 结果文件说明

**evaluation_report.json** - 完整评估报告
```json
{
  "dataset": "数据集路径",
  "num_scenes": 5,
  "total_images": 4838,
  "overall": {
    "psnr": 32.5,
    "ssim": 0.78,
    "siou": 0.30,
    "fps": 35.2
  },
  "per_scene": [...],
  "system": {
    "model_size_mb": 384.0,
    "device": "cuda"
  }
}
```

**summary.json** - 简要结果
```json
{
  "Overall": {
    "PSNR (dB)": 32.5,
    "SSIM": 0.78,
    "SIoU": 0.30,
    "FPS": 35.2
  },
  "Model_Size_MB": 384.0
}
```

## 评估指标说明

### PSNR (Peak Signal-to-Noise Ratio)
- **单位**: dB
- **含义**: 峰值信噪比，衡量图像质量
- **目标**: > 32 dB
- **计算**: `20 * log10(255 / RMSE)`

### SSIM (Structural Similarity Index)
- **范围**: [0, 1]
- **含义**: 结构相似性，衡量视觉感知质量
- **目标**: > 0.75

### SIoU (Structural IoU)
- **范围**: [0, 1]
- **含义**: 结构IoU，衡量结构一致性
- **目标**: > 0.28
- **计算**: `0.75 * 边缘重叠 + 0.25 * 差异重叠`

### FPS (Frames Per Second)
- **单位**: 帧/秒
- **含义**: 推理速度
- **目标**: > 30 FPS

## 参数调优

### 调整立体效果

编辑 `eval_config.yaml` 或使用命令行参数：

```yaml
dibr:
  baseline: 0.065      # 基线距离（米）
  focal_length: 800    # 焦距（像素）
```

命令行方式：
```bash
# 增强立体感
python benchmark_eval_v2.py --baseline 0.1 --focal-length 1200

# 减弱立体感
python benchmark_eval_v2.py --baseline 0.05 --focal-length 600
```

### 调整建议
- `baseline` 增大 → 视差变大 → 立体感增强（空洞增多）
- `focal_length` 增大 → 视差变大 → 立体感增强

## 权重下载链接汇总

### MiDaS模型
| 模型 | 大小 | 下载链接 |
|------|------|----------|
| DPT-Large | 1.2GB | https://github.com/intel-isl/DPT/releases/download/1.0/dpt_large_384.pt |
| DPT-Hybrid | 384MB | https://github.com/intel-isl/DPT/releases/download/1.0/dpt_hybrid_384.pt |
| MiDaS-small | 100MB | https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21_small_256.pt |

### Depth-Anything-V2模型
| 模型 | 大小 | 下载链接 |
|------|------|----------|
| Small (vits) | 24.8MB | https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth |
| Base (vitb) | 97.5MB | https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth |
| Large (vitl) | 1.3GB | https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth |

## 常见问题

### Q1: CUDA out of memory？
A: 使用小模型或CPU：
```bash
python benchmark_eval_v2.py --device cpu
# 或
python download_depth_anything_v2.py --encoder vits
```

### Q2: 找不到图像？
A: 检查数据集路径是否正确：
```bash
ls Data/mono2stereo-test/animation/left/
ls Data/mono2stereo-test/animation/right/
```

### Q3: 模型加载失败？
A: 检查权重文件是否存在：
```bash
ls -lh Data/checkpoints/
```

### Q4: 生成的图像有明显空洞？
A: 调整参数：
```yaml
# 减小基线距离
dibr:
  baseline: 0.05
  
# 或修改空洞填充
hole_filling:
  method: "ns"
  radius: 10
```

## 参考资料

- [MiDaS论文](https://arxiv.org/abs/1907.01341)
- [Depth-Anything-V2](https://depth-anything-v2.github.io/)
- [DIBR原理](https://en.wikipedia.org/wiki/Depth_image-based_rendering)

## 更新日志

### v1.2.0 (2026-03-22)
- 整合MiDaS和Depth-Anything-V2仓库代码
- 简化部署流程，无需克隆外部仓库
- 修复模块导入路径问题

### v1.1.0 (2026-03-22)
- 支持Depth-Anything-V2模型
- 优化项目结构
- 统一路径配置
- 添加多场景评估

### v1.0.0 (2026-03-21)
- 初始版本
- 支持MiDaS深度估计
- 完整DIBR流程

---

**注意**：本项目仅供学习和研究使用，商业使用请遵守相关模型和数据集的许可协议。
