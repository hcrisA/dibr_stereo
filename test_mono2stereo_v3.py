#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mono2Stereo数据集快速测试脚本（支持多场景 + 详细输出）
一键运行评估
"""

import os
import sys
import torch
from pathlib import Path
import time

# 项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def quick_test():
    """快速测试Mono2Stereo数据集"""
    
    print("="*70)
    print("DIBR系统 - Mono2Stereo数据集评估")
    print("="*70)
    print()
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    # 配置路径（使用相对路径）
    config = {
        'dataset': str(project_root / "Data" / "mono2stereo-test"),
        'model_path': str(project_root / "Data" / "checkpoints" / "dpt_hybrid_384.pt"),
        'output_dir': str(project_root / "Data" / "outputs" / "midas_results"),
        'device': 'cuda',
        'baseline': 0.065,
        'focal_length': 800
    }
    
    # 检查CUDA
    print("[设备检查]")
    if torch.cuda.is_available():
        print(f"✓ CUDA可用")
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  当前GPU: {torch.cuda.current_device()}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        config['device'] = 'cuda'
    else:
        print("✗ CUDA不可用，将使用CPU")
        config['device'] = 'cpu'
    
    print()
    
    # 检查路径
    print("[1/3] 检查路径...")
    
    if not os.path.exists(config['dataset']):
        print(f"✗ 数据集目录不存在: {config['dataset']}")
        return
    
    if not os.path.exists(config['model_path']):
        print(f"✗ 模型文件不存在: {config['model_path']}")
        return
    
    # 检查模型大小
    model_size_mb = os.path.getsize(config['model_path']) / 1024 / 1024
    print(f"✓ 模型文件: {config['model_path']}")
    print(f"  文件大小: {model_size_mb:.2f} MB")
    
    # 检查场景
    dataset_root = Path(config['dataset'])
    scenes = []
    for d in dataset_root.iterdir():
        if d.is_dir() and (d / "left").exists() and (d / "right").exists():
            scenes.append(d.name)
    scenes = sorted(scenes)
    
    if len(scenes) == 0:
        print(f"✗ 未找到有效场景（需要 left/right 子目录）")
        return
    
    print(f"✓ 数据集目录: {config['dataset']}")
    print(f"  找到 {len(scenes)} 个场景: {', '.join(scenes)}")
    
    # 统计各场景图像数量
    print(f"\n  场景图像数量:")
    for scene in scenes:
        left_dir = dataset_root / scene / "left"
        num_images = len(list(left_dir.glob("*.png")) + list(left_dir.glob("*.jpg")))
        print(f"    - {scene}: {num_images} 张")
    
    print()
    
    # 运行评估
    print("[2/3] 开始评估...")
    print()
    
    # 加载模型
    print("步骤1: 加载MiDaS模型...")
    print(f"  设备: {config['device']}")
    print(f"  模型路径: {config['model_path']}")
    
    t_load_start = time.time()
    
    from benchmark_eval_v2 import DIBREvaluator
    
    evaluator = DIBREvaluator(
        model_path=config['model_path'],
        device=config['device'],
        baseline=config['baseline'],
        focal_length=config['focal_length']
    )
    
    t_load_time = time.time() - t_load_start
    print(f"✓ 模型加载完成，耗时: {t_load_time:.2f} 秒")
    print()
    
    # 评估
    print("步骤2: 评估数据集...")
    print(f"  输出目录: {config['output_dir']}")
    print(f"  DIBR参数: baseline={config['baseline']}m, focal_length={config['focal_length']}px")
    print()
    
    results = evaluator.evaluate_dataset(
        dataset_root=config['dataset'],
        output_dir=config['output_dir'],
        save_results=True
    )
    
    # 打印结果
    print("\n[3/3] 评估完成！\n")
    evaluator.print_summary(results)
    
    # 保存报告
    import json
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # 完整报告
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n完整报告: {report_path}")
    
    # 简要结果
    summary = {
        'Overall': {
            'PSNR (dB)': round(results['overall']['psnr'], 2),
            'SSIM': round(results['overall']['ssim'], 4),
            'SIoU': round(results['overall']['siou'], 4),
            'FPS': round(results['overall']['fps'], 2),
            'Inference_Time_ms': round(results['overall']['avg_time_ms'], 2)
        },
        'Per_Scene': results['per_scene'],
        'Model_Size_MB': round(results['system']['model_size_mb'], 2)
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"简要结果: {summary_path}")
    
    # CSV结果
    csv_path = output_dir / "results.csv"
    with open(csv_path, 'w') as f:
        f.write("Scene,PSNR(dB),SSIM,SIoU,FPS\n")
        for scene_data in results['per_scene']:
            f.write(f"{scene_data['scene']},"
                   f"{scene_data['psnr']:.2f},"
                   f"{scene_data['ssim']:.4f},"
                   f"{scene_data['siou']:.4f},"
                   f"{scene_data['fps']:.2f}\n")
        f.write(f"\nOverall,"
               f"{results['overall']['psnr']:.2f},"
               f"{results['overall']['ssim']:.4f},"
               f"{results['overall']['siou']:.4f},"
               f"{results['overall']['fps']:.2f}\n")
    print(f"CSV结果: {csv_path}")
    
    # 与目标对比
    print("\n" + "="*70)
    print("与华为赛题指标对比")
    print("="*70)
    
    targets = {
        'PSNR': 32.0,
        'SSIM': 0.75,
        'SIoU': 0.28
    }
    
    overall = results['overall']
    
    print(f"PSNR: {overall['psnr']:.2f} dB ", end='')
    if overall['psnr'] >= targets['PSNR']:
        print(f"✓ (目标 > {targets['PSNR']} dB)")
    else:
        print(f"✗ (目标 > {targets['PSNR']} dB)")
    
    print(f"SSIM: {overall['ssim']:.4f} ", end='')
    if overall['ssim'] >= targets['SSIM']:
        print(f"✓ (目标 > {targets['SSIM']})")
    else:
        print(f"✗ (目标 > {targets['SSIM']})")
    
    print(f"SIoU: {overall['siou']:.4f} ", end='')
    if overall['siou'] >= targets['SIoU']:
        print(f"✓ (目标 > {targets['SIoU']})")
    else:
        print(f"✗ (目标 > {targets['SIoU']})")
    
    print(f"FPS: {overall['fps']:.2f}")
    print(f"模型大小: {results['system']['model_size_mb']:.2f} MB")
    
    if torch.cuda.is_available():
        gpu_mem = results['system']['gpu_memory_mb']
        print(f"GPU显存占用: {gpu_mem:.2f} MB")
    
    print("\n" + "="*70)
    print("评估完成！结果已保存到:", output_dir)
    print("="*70)


if __name__ == "__main__":
    quick_test()
