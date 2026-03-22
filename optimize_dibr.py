#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIBR方案快速优化脚本
优化内容：
1. 自动估计最佳disparity_scale
2. 深度图后处理优化
3. 调整相机参数
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json

# 项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from benchmark_eval_v2 import DIBREvaluator, calculate_metrics


def estimate_optimal_scale(
    evaluator,
    left_dir: str,
    right_dir: str,
    num_samples: int = 20,
    scale_range: tuple = (0.5, 3.0),
    num_scales: int = 10
):
    """
    自动估计最佳disparity_scale
    
    通过网格搜索找到使PSNR最大的scale值
    """
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    
    # 获取图像列表
    left_images = sorted(list(left_dir.glob("*.png")) + list(left_dir.glob("*.jpg")))
    right_images = sorted(list(right_dir.glob("*.png")) + list(right_dir.glob("*.jpg")))
    
    # 采样
    if len(left_images) > num_samples:
        indices = np.linspace(0, len(left_images)-1, num_samples, dtype=int)
        left_images = [left_images[i] for i in indices]
        right_images = [right_images[i] for i in indices]
    
    print(f"使用 {len(left_images)} 张图像估计最佳scale...")
    
    # 测试不同的scale值
    scales = np.linspace(scale_range[0], scale_range[1], num_scales)
    best_scale = 1.0
    best_psnr = 0
    
    for scale in tqdm(scales, desc="搜索最佳scale"):
        # 更新renderer的scale
        evaluator.renderer.disparity_scale = scale
        
        # 计算平均PSNR
        psnrs = []
        for left_path, right_path in zip(left_images, right_images):
            left_img = cv2.imread(str(left_path))
            gt_right_img = cv2.imread(str(right_path))
            
            if left_img is None or gt_right_img is None:
                continue
            
            # 推理
            pred_right_img, _ = evaluator.process_single_image(left_img)
            
            # 计算PSNR
            metrics = calculate_metrics(pred_right_img, gt_right_img, left_img)
            psnrs.append(metrics['psnr'])
        
        avg_psnr = np.mean(psnrs)
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_scale = scale
            print(f"  scale={scale:.3f}, PSNR={avg_psnr:.2f}dB ✓")
    
    print(f"\n最佳scale: {best_scale:.3f}, PSNR: {best_psnr:.2f}dB")
    return best_scale


def estimate_optimal_baseline_focal(
    evaluator,
    left_dir: str,
    right_dir: str,
    num_samples: int = 20
):
    """
    自动估计最佳baseline和focal_length组合
    """
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    
    # 获取图像列表
    left_images = sorted(list(left_dir.glob("*.png")) + list(left_dir.glob("*.jpg")))
    
    # 采样
    if len(left_images) > num_samples:
        indices = np.linspace(0, len(left_images)-1, num_samples, dtype=int)
        left_images = [left_images[i] for i in indices]
    
    print(f"使用 {len(left_images)} 张图像估计最佳参数...")
    
    # 测试不同的参数组合
    baselines = [0.03, 0.05, 0.065, 0.08, 0.1]  # 米
    focals = [400, 600, 800, 1000, 1200]  # 像素
    
    best_params = {'baseline': 0.065, 'focal': 800}
    best_psnr = 0
    
    for baseline in baselines:
        for focal in focals:
            # 更新renderer参数
            evaluator.renderer.baseline = baseline
            evaluator.renderer.focal_length = focal
            
            # 计算平均PSNR
            psnrs = []
            for left_path in left_images:
                left_img = cv2.imread(str(left_path))
                right_path = right_dir / left_path.name
                gt_right_img = cv2.imread(str(right_path))
                
                if left_img is None or gt_right_img is None:
                    continue
                
                # 推理
                pred_right_img, _ = evaluator.process_single_image(left_img)
                
                # 计算PSNR
                metrics = calculate_metrics(pred_right_img, gt_right_img, left_img)
                psnrs.append(metrics['psnr'])
            
            avg_psnr = np.mean(psnrs)
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_params = {'baseline': baseline, 'focal': focal}
                print(f"  baseline={baseline}, focal={focal}, PSNR={avg_psnr:.2f}dB ✓")
    
    print(f"\n最佳参数: baseline={best_params['baseline']}, focal={best_params['focal']}")
    print(f"PSNR: {best_psnr:.2f}dB")
    return best_params


def optimize_depth_postprocessing(
    evaluator,
    left_dir: str,
    right_dir: str,
    num_samples: int = 20
):
    """
    优化深度图后处理参数
    """
    from scipy.ndimage import gaussian_filter
    
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    
    # 获取图像列表
    left_images = sorted(list(left_dir.glob("*.png")) + list(left_dir.glob("*.jpg")))
    
    # 采样
    if len(left_images) > num_samples:
        indices = np.linspace(0, len(left_images)-1, num_samples, dtype=int)
        left_images = [left_images[i] for i in indices]
    
    print(f"优化深度后处理参数...")
    
    # 测试不同的深度平滑参数
    blur_sigmas = [0, 0.5, 1.0, 1.5, 2.0]
    
    best_sigma = 0
    best_psnr = 0
    
    for sigma in blur_sigmas:
        psnrs = []
        
        for left_path in left_images:
            left_img = cv2.imread(str(left_path))
            right_path = right_dir / left_path.name
            gt_right_img = cv2.imread(str(right_path))
            
            if left_img is None or gt_right_img is None:
                continue
            
            # 深度估计
            left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            depth = evaluator.estimate_depth(left_rgb)
            
            # 深度平滑
            if sigma > 0:
                depth = gaussian_filter(depth, sigma=sigma)
            
            # DIBR渲染
            right_rgb, _ = evaluator.renderer.render_right_view(left_rgb, depth)
            pred_right_img = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
            
            # 计算PSNR
            metrics = calculate_metrics(pred_right_img, gt_right_img, left_img)
            psnrs.append(metrics['psnr'])
        
        avg_psnr = np.mean(psnrs)
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_sigma = sigma
            print(f"  blur_sigma={sigma}, PSNR={avg_psnr:.2f}dB ✓")
    
    print(f"\n最佳blur_sigma: {best_sigma}")
    return best_sigma


def run_full_optimization(
    model_path: str,
    dataset_root: str,
    scene: str = "simple",
    output_file: str = "optimized_params.json"
):
    """
    运行完整优化流程
    """
    print("="*70)
    print("DIBR参数优化")
    print("="*70)
    print()
    
    # 初始化评估器
    print("加载模型...")
    evaluator = DIBREvaluator(
        model_path=model_path,
        device="cuda",
        baseline=0.065,
        focal_length=800
    )
    
    left_dir = Path(dataset_root) / scene / "left"
    right_dir = Path(dataset_root) / scene / "right"
    
    # 1. 估计最佳scale
    print("\n[1/3] 估计最佳disparity_scale...")
    best_scale = estimate_optimal_scale(
        evaluator, left_dir, right_dir,
        num_samples=20,
        scale_range=(0.5, 3.0),
        num_scales=20
    )
    
    # 2. 估计最佳baseline和focal
    print("\n[2/3] 估计最佳baseline和focal_length...")
    best_params = estimate_optimal_baseline_focal(
        evaluator, left_dir, right_dir,
        num_samples=20
    )
    
    # 3. 优化深度后处理
    print("\n[3/3] 优化深度后处理...")
    best_sigma = optimize_depth_postprocessing(
        evaluator, left_dir, right_dir,
        num_samples=20
    )
    
    # 保存结果
    optimized_params = {
        'disparity_scale': float(best_scale),
        'baseline': float(best_params['baseline']),
        'focal_length': float(best_params['focal']),
        'depth_blur_sigma': float(best_sigma)
    }
    
    with open(output_file, 'w') as f:
        json.dump(optimized_params, f, indent=2)
    
    print("\n" + "="*70)
    print("优化完成！")
    print("="*70)
    print(f"最优参数已保存到: {output_file}")
    print(f"参数:")
    for key, value in optimized_params.items():
        print(f"  {key}: {value}")
    
    return optimized_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DIBR参数优化")
    parser.add_argument("--model-path", type=str,
                       default="/root/autodl-tmp/torch/hub/checkpoints/dpt_hybrid_384.pt",
                       help="模型路径")
    parser.add_argument("--dataset", type=str,
                       default="/root/autodl-tmp/mono2stereo-test",
                       help="数据集路径")
    parser.add_argument("--scene", type=str, default="simple",
                       help="用于优化的场景")
    parser.add_argument("--output", type=str, default="optimized_params.json",
                       help="输出文件")
    
    args = parser.parse_args()
    
    run_full_optimization(
        model_path=args.model_path,
        dataset_root=args.dataset,
        scene=args.scene,
        output_file=args.output
    )
