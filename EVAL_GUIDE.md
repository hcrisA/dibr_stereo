# Mono2Stereo数据集评估指南

## 快速开始

### 方法1：一键测试（推荐）

```bash
cd /root/autodl-tmp/dibr_stereo
python test_mono2stereo.py
```

### 方法2：自定义参数

```bash
python benchmark_eval.py \
  --left-dir /root/autodl-tmp/mono2stereo-test/left \
  --right-dir /root/autodl-tmp/mono2stereo-test/right \
  --output-dir ./eval_output \
  --model-path /root/autodl-tmp/torch/hub/checkpoints/dpt_hybrid_384.pt \
  --device cuda \
  --baseline 0.065 \
  --focal-length 800
```

## 输出结果

评估完成后，会在输出目录生成以下文件：

```
mono2stereo_results/
├── *.png                     # 生成的右视角图像
├── evaluation_report.json    # 完整评估报告
├── summary.json              # 简要结果
└── results.csv               # CSV格式结果
```

### 查看结果

```bash
# 查看简要结果
cat mono2stereo_results/summary.json

# 查看CSV结果
cat mono2stereo_results/results.csv

# 查看完整报告
cat mono2stereo_results/evaluation_report.json
```

## 评估指标说明

### 1. PSNR (Peak Signal-to-Noise Ratio)
- **单位**: dB
- **含义**: 峰值信噪比，衡量图像质量
- **目标**: > 32 dB
- **计算**: `10 * log10(MAX^2 / MSE)`

### 2. SSIM (Structural Similarity Index)
- **范围**: [0, 1]
- **含义**: 结构相似性，衡量视觉感知质量
- **目标**: > 0.75
- **计算**: 基于亮度、对比度、结构信息

### 3. SIoU (Structural IoU)
- **范围**: [0, 1]
- **含义**: 结构IoU，衡量结构一致性
- **目标**: > 0.28
- **计算**: 基于梯度相似性

### 4. FPS (Frames Per Second)
- **单位**: 帧/秒
- **含义**: 推理速度
- **目标**: > 60 FPS

### 5. Model Size
- **单位**: MB
- **含义**: 模型参数大小
- **目标**: < 500 MB

## 参数调优

### 调整立体效果

编辑 `eval_config.yaml` 或使用命令行参数：

```bash
# 增强立体感（视差变大）
python benchmark_eval.py --baseline 0.1 --focal-length 1200

# 减弱立体感（视差变小）
python benchmark_eval.py --baseline 0.05 --focal-length 600
```

### 修改配置文件

```yaml
dibr:
  baseline: 0.065      # 基线距离
  focal_length: 800    # 焦距
```

**调整建议**：
- `baseline` 增大 → 视差变大 → 立体感增强（但空洞增多）
- `focal_length` 增大 → 视差变大 → 立体感增强

## 性能对比

### 不同模型对比

```bash
# DPT_Large（最高质量）
python benchmark_eval.py --model-path /path/to/dpt_large_384.pt

# DPT_Hybrid（平衡）
python benchmark_eval.py --model-path /path/to/dpt_hybrid_384.pt

# MiDaS_small（最快）
python benchmark_eval.py --model-path /path/to/midas_v21_small_256.pt
```

### 预期性能（RTX 3090）

| 模型 | PSNR | SSIM | SIoU | FPS | Size |
|------|------|------|------|-----|------|
| DPT_Large | 33-35 | 0.78-0.82 | 0.30-0.35 | 18 | 1.2GB |
| DPT_Hybrid | 32-34 | 0.76-0.80 | 0.28-0.32 | 33 | 384MB |
| MiDaS_small | 30-32 | 0.72-0.76 | 0.25-0.28 | 66 | 100MB |

## 故障排查

### 问题1：CUDA out of memory

**解决方案**：
```bash
# 使用CPU
python benchmark_eval.py --device cpu

# 或使用小模型
python benchmark_eval.py --model-path /path/to/midas_v21_small_256.pt
```

### 问题2：找不到图像

**检查路径**：
```bash
ls /root/autodl-tmp/mono2stereo-test/left/
ls /root/autodl-tmp/mono2stereo-test/right/
```

### 问题3：模型加载失败

**检查模型文件**：
```bash
ls -lh /root/autodl-tmp/torch/hub/checkpoints/
```

## 提交结果

评估完成后，需要提交：

1. **summary.json** - 包含所有指标
2. **evaluation_report.json** - 完整报告
3. **样张对比** - 选几张典型结果对比

## 进阶功能

### 批量测试不同参数

```bash
# 创建测试脚本
cat > test_all_params.sh << 'EOF'
#!/bin/bash

baselines=(0.05 0.065 0.1)
focals=(600 800 1000)

for b in "${baselines[@]}"; do
  for f in "${focals[@]}"; do
    echo "Testing: baseline=$b, focal=$f"
    python benchmark_eval.py \
      --baseline $b \
      --focal-length $f \
      --output-dir ./results_b${b}_f${f}
  done
done
EOF

chmod +x test_all_params.sh
./test_all_params.sh
```

### 可视化对比

```python
import cv2
import numpy as np

# 读取图像
left = cv2.imread('left/001.png')
pred = cv2.imread('mono2stereo_results/001.png')
gt = cv2.imread('right/001.png')

# 拼接对比
comparison = np.hstack([left, pred, gt])
cv2.imwrite('comparison.png', comparison)
```

---

**祝测试顺利！** 🎉
