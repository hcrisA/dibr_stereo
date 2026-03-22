#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIBR系统快速启动脚本
支持选择不同深度模型进行测试
"""

import os
import sys
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))


def print_header():
    """打印标题"""
    print("\n" + "="*70)
    print("DIBR单目转立体视频系统 - 快速启动")
    print("="*70)


def check_weights():
    """检查模型权重"""
    print("\n[检查模型权重]")
    
    weights = {
        "MiDaS DPT-Hybrid": project_root / "Data" / "checkpoints" / "dpt_hybrid_384.pt",
        "Depth-Anything-V2-Small": project_root / "Data" / "checkpoints" / "depth_anything_v2_vits.pth",
        "Depth-Anything-V2-Base": project_root / "Data" / "checkpoints" / "depth_anything_v2_vitb.pth",
    }
    
    available = {}
    for name, path in weights.items():
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  ✓ {name}: {size_mb:.1f}MB")
            available[name] = path
        else:
            print(f"  ✗ {name}: 未找到")
    
    return available


def check_dataset():
    """检查数据集"""
    print("\n[检查数据集]")
    
    dataset_path = project_root / "Data" / "mono2stereo-test"
    
    if not dataset_path.exists():
        print(f"  ✗ 数据集不存在: {dataset_path}")
        return None
    
    scenes = []
    for d in dataset_path.iterdir():
        if d.is_dir() and (d / "left").exists() and (d / "right").exists():
            num_images = len(list((d / "left").glob("*.jpg")) + list((d / "left").glob("*.png")))
            scenes.append((d.name, num_images))
    
    scenes.sort()
    
    print(f"  ✓ 数据集路径: {dataset_path}")
    print(f"  ✓ 场景数量: {len(scenes)}")
    for scene, count in scenes:
        print(f"    - {scene}: {count}张图像")
    
    return dataset_path


def run_midas_evaluation(dataset_path, output_dir):
    """运行MiDaS评估"""
    print("\n" + "="*70)
    print("使用MiDaS DPT-Hybrid模型进行评估")
    print("="*70)
    
    from benchmark_eval_v2 import DIBREvaluator
    
    model_path = project_root / "Data" / "checkpoints" / "dpt_hybrid_384.pt"
    
    evaluator = DIBREvaluator(
        model_path=str(model_path),
        device="cuda",
        baseline=0.065,
        focal_length=800
    )
    
    results = evaluator.evaluate_dataset(
        dataset_root=str(dataset_path),
        output_dir=str(output_dir),
        save_results=True
    )
    
    evaluator.print_summary(results)
    
    # 保存报告
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n评估报告已保存: {report_path}")
    return results


def run_depth_anything_v2_evaluation(dataset_path, output_dir, encoder="vits"):
    """运行Depth-Anything-V2评估"""
    print("\n" + "="*70)
    print(f"使用Depth-Anything-V2-{encoder.upper()}模型进行评估")
    print("="*70)
    
    from download_depth_anything_v2 import LightweightDIBREvaluator
    
    model_name = f"depth_anything_v2_{encoder}.pth"
    model_path = project_root / "Data" / "checkpoints" / model_name
    
    evaluator = LightweightDIBREvaluator(
        model_path=str(model_path),
        device="cuda",
        baseline=0.065,
        focal_length=800,
        encoder=encoder
    )
    
    results = evaluator.evaluate_dataset(
        dataset_root=str(dataset_path),
        output_dir=str(output_dir),
        save_results=True
    )
    
    evaluator.print_summary(results)
    
    # 保存报告
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n评估报告已保存: {report_path}")
    return results


def main():
    """主函数"""
    print_header()
    
    # 检查权重
    available_weights = check_weights()
    
    # 检查数据集
    dataset_path = check_dataset()
    
    if not dataset_path:
        print("\n错误: 数据集不存在，请先下载数据集")
        return
    
    if not available_weights:
        print("\n错误: 未找到任何模型权重，请先下载模型")
        print("\n下载链接:")
        print("  MiDaS: https://github.com/intel-isl/DPT/releases")
        print("  Depth-Anything-V2: https://huggingface.co/depth-anything")
        return
    
    # 选择模型
    print("\n" + "-"*70)
    print("选择深度估计模型:")
    print("-"*70)
    
    options = list(available_weights.keys())
    for i, name in enumerate(options, 1):
        print(f"  {i}. {name}")
    
    print(f"  {len(options)+1}. 运行所有可用模型进行对比")
    print(f"  0. 退出")
    
    choice = input("\n请输入选项 [0-{}]: ".format(len(options)+1)).strip()
    
    if choice == "0":
        print("退出")
        return
    
    try:
        choice_idx = int(choice) - 1
        
        if choice_idx == len(options):
            # 运行所有模型
            print("\n将依次运行所有可用模型...")
            for name, path in available_weights.items():
                if "MiDaS" in name:
                    output_dir = project_root / "Data" / "outputs" / "midas_results"
                    run_midas_evaluation(dataset_path, output_dir)
                elif "Depth-Anything-V2-Small" in name:
                    output_dir = project_root / "Data" / "outputs" / "depth_anything_v2_small_results"
                    run_depth_anything_v2_evaluation(dataset_path, output_dir, "vits")
                elif "Depth-Anything-V2-Base" in name:
                    output_dir = project_root / "Data" / "outputs" / "depth_anything_v2_base_results"
                    run_depth_anything_v2_evaluation(dataset_path, output_dir, "vitb")
        
        elif 0 <= choice_idx < len(options):
            selected = options[choice_idx]
            print(f"\n已选择: {selected}")
            
            if "MiDaS" in selected:
                output_dir = project_root / "Data" / "outputs" / "midas_results"
                run_midas_evaluation(dataset_path, output_dir)
            elif "Depth-Anything-V2-Small" in selected:
                output_dir = project_root / "Data" / "outputs" / "depth_anything_v2_small_results"
                run_depth_anything_v2_evaluation(dataset_path, output_dir, "vits")
            elif "Depth-Anything-V2-Base" in selected:
                output_dir = project_root / "Data" / "outputs" / "depth_anything_v2_base_results"
                run_depth_anything_v2_evaluation(dataset_path, output_dir, "vitb")
        else:
            print("无效选项")
    
    except (ValueError, IndexError):
        print("无效输入")
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)


if __name__ == "__main__":
    main()
