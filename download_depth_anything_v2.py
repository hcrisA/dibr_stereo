#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用Depth-Anything-V2-Small的轻量级DIBR评估
模型大小: 24.8MB (vs MiDaS 470MB)
预期FPS: 40+ (vs MiDaS 13)
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
import json
from typing import Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def detect_edges(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """边缘检测"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return edges


def edge_overlap(edge1: np.ndarray, edge2: np.ndarray) -> float:
    """计算边缘IoU"""
    intersection = np.logical_and(edge1, edge2).sum()
    union = np.logical_or(edge1, edge2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def calculate_siou(pred: np.ndarray, gt: np.ndarray, left: np.ndarray) -> float:
    """计算SIoU - 官方实现"""
    left_edges = detect_edges(left, 100, 200)
    pred_edges = detect_edges(pred, 100, 200)
    right_edges = detect_edges(gt, 100, 200)
    
    diff_gl = np.abs(pred.astype(np.float32) - left.astype(np.float32))
    diff_rl = np.abs(gt.astype(np.float32) - left.astype(np.float32))
    
    diff_gl_gray = cv2.cvtColor(diff_gl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    diff_rl_gray = cv2.cvtColor(diff_rl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    diff_gl_bin = np.zeros(diff_gl_gray.shape, dtype=np.uint8)
    diff_rl_bin = np.zeros(diff_rl_gray.shape, dtype=np.uint8)
    diff_gl_bin[diff_gl_gray > 5] = 1
    diff_rl_bin[diff_rl_gray > 5] = 1
    
    edge_overlap_gr = edge_overlap(pred_edges, right_edges)
    diff_overlap_grl = edge_overlap(diff_gl_bin, diff_rl_bin)
    
    return 0.75 * edge_overlap_gr + 0.25 * diff_overlap_grl


def calculate_metrics(pred: np.ndarray, gt: np.ndarray, left: np.ndarray = None) -> Dict[str, float]:
    """计算评估指标"""
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
    if left is not None and pred.shape != left.shape:
        left = cv2.resize(left, (pred.shape[1], pred.shape[0]))
    
    # PSNR
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0:
        psnr_value = 100.0
    else:
        rmse = np.sqrt(mse)
        psnr_value = 20 * np.log10(255.0 / rmse)
    
    # SSIM
    ssim_value = ssim(gt, pred, multichannel=True, channel_axis=2, win_size=7, data_range=255)
    
    # SIoU
    if left is not None:
        siou_value = calculate_siou(pred, gt, left)
    else:
        siou_value = 0.0
    
    return {
        'psnr': float(psnr_value),
        'ssim': float(ssim_value),
        'siou': float(siou_value)
    }


class DepthAnythingV2Wrapper:
    """Depth-Anything-V2模型包装器"""
    
    def __init__(self, model_path: str, device: str = 'cuda', encoder: str = 'vits'):
        """
        初始化Depth-Anything-V2
        
        Args:
            model_path: 模型权重路径
            device: 运行设备
            encoder: 模型类型 'vits', 'vitb', 'vitl'
        """
        self.device = device
        self.encoder = encoder
        
        # 模型配置
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        print(f"加载Depth-Anything-V2-{encoder.upper()}...")
        print(f"  模型路径: {model_path}")
        
        # 检查是否需要下载模型
        if not os.path.exists(model_path):
            print(f"  ⚠️ 模型不存在，尝试下载...")
            self._download_model(model_path, encoder)
        
        # 加载模型
        try:
            # 尝试从Depth-Anything-V2仓库导入
            from depth_anything_v2.dpt import DepthAnythingV2
            
            self.model = DepthAnythingV2(**model_configs[encoder])
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model = self.model.to(device).eval()
            self.use_official = True
            
        except ImportError:
            print(f"  ⚠️ 未找到depth_anything_v2模块，尝试使用Hugging Face...")
            self._load_from_huggingface(encoder)
            self.use_official = False
        
        # 计算模型大小
        self.model_size_mb = self._get_model_size()
        print(f"  ✓ 模型加载完成 (大小: {self.model_size_mb:.2f} MB)")
    
    def _download_model(self, model_path: str, encoder: str):
        """下载模型权重"""
        import urllib.request
        
        urls = {
            'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
            'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
            'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
        }
        
        url = urls[encoder]
        print(f"  正在下载: {url}")
        
        # 创建目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 下载
        urllib.request.urlretrieve(url, model_path)
        print(f"  ✓ 下载完成: {model_path}")
    
    def _load_from_huggingface(self, encoder: str):
        """从Hugging Face加载模型"""
        from transformers import pipeline
        
        model_names = {
            'vits': 'depth-anything/Depth-Anything-V2-Small-hf',
            'vitb': 'depth-anything/Depth-Anything-V2-Base-hf',
            'vitl': 'depth-anything/Depth-Anything-V2-Large-hf',
        }
        
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_names[encoder],
            device=0 if self.device == 'cuda' else -1
        )
    
    def _get_model_size(self) -> float:
        """获取模型大小"""
        if self.use_official:
            param_size = 0
            for param in self.model.parameters():
                param_size += param.nelement() * param.element_size()
            
            buffer_size = 0
            for buffer in self.model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            return (param_size + buffer_size) / 1024 / 1024
        else:
            return 24.8  # Small模型的参数量
    
    def infer_image(self, image: np.ndarray) -> np.ndarray:
        """
        推理单张图像
        
        Args:
            image: RGB图像 [H, W, 3] 或 BGR图像
            
        Returns:
            depth: 深度图 [H, W], 值域[0, 1]
        """
        if self.use_official:
            # 使用官方实现
            depth = self.model.infer_image(image, input_size=518)
            
            # 归一化到[0, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
        else:
            # 使用Hugging Face
            from PIL import Image
            
            # BGR转RGB
            if image.shape[2] == 3 and image.dtype == np.uint8:
                # 假设是BGR
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            result = self.pipe(pil_image)
            depth = np.array(result["depth"])
            
            # 归一化到[0, 1]
            depth = depth.astype(np.float32) / 255.0
        
        return depth


class LightweightDIBREvaluator:
    """轻量级DIBR评估器（使用Depth-Anything-V2）"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        baseline: float = 0.065,
        focal_length: float = 800,
        disparity_scale: float = 1.0,
        encoder: str = 'vits'
    ):
        self.device = device
        self.baseline = baseline
        self.focal_length = focal_length
        self.disparity_scale = disparity_scale
        
        # 加载深度模型
        self.depth_model = DepthAnythingV2Wrapper(
            model_path=model_path,
            device=device,
            encoder=encoder
        )
        
        # 加载渲染器
        from dibr_renderer import DIBRRenderer
        self.renderer = DIBRRenderer(
            baseline=baseline,
            focal_length=focal_length,
            disparity_scale=disparity_scale
        )
        
        self.model_size_mb = self.depth_model.model_size_mb
        
        print(f"✓ 轻量级DIBR系统初始化完成")
        print(f"  模型大小: {self.model_size_mb:.2f} MB")
        print(f"  参数: baseline={baseline}, focal={focal_length}, scale={disparity_scale}")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """估计深度图"""
        return self.depth_model.infer_image(image)
    
    @torch.no_grad()
    def process_single_image(
        self, 
        left_image: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """处理单张图像"""
        # 计时
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_start = time.time()
        
        # BGR转RGB
        left_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        
        # 深度估计
        depth = self.estimate_depth(left_rgb)
        
        # DIBR渲染
        right_rgb, _ = self.renderer.render_right_view(left_rgb, depth)
        
        # RGB转BGR
        right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
        
        # 计时结束
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = (time.time() - t_start) * 1000
        
        return right_bgr, inference_time
    
    def evaluate_dataset(
        self,
        dataset_root: str,
        output_dir: Optional[str] = None,
        save_results: bool = True,
        scenes: Optional[list] = None
    ) -> Dict:
        """评估整个数据集"""
        dataset_root = Path(dataset_root)
        
        # 自动检测场景
        if scenes is None:
            scenes = []
            for d in dataset_root.iterdir():
                if d.is_dir() and (d / "left").exists() and (d / "right").exists():
                    scenes.append(d.name)
            scenes = sorted(scenes)
        
        print(f"\n找到 {len(scenes)} 个场景: {', '.join(scenes)}\n")
        
        # 创建输出目录
        if output_dir and save_results:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 评估每个场景
        all_scene_results = []
        all_metrics_global = []
        
        print("开始评估各场景...")
        for scene in scenes:
            print(f"\n[{scene}]")
            
            left_dir = dataset_root / scene / "left"
            right_dir = dataset_root / scene / "right"
            
            scene_result = self._evaluate_single_scene(
                left_dir=str(left_dir),
                right_dir=str(right_dir),
                scene_name=scene,
                output_dir=output_dir,
                save_results=save_results
            )
            
            if scene_result:
                all_scene_results.append(scene_result)
                all_metrics_global.extend(scene_result['detailed_results'])
        
        # 计算全局指标
        global_psnr = np.mean([m['psnr'] for m in all_metrics_global])
        global_ssim = np.mean([m['ssim'] for m in all_metrics_global])
        global_siou = np.mean([m['siou'] for m in all_metrics_global])
        global_time = np.mean([m['inference_time_ms'] for m in all_metrics_global])
        global_fps = 1000.0 / global_time
        
        results = {
            'dataset': str(dataset_root),
            'num_scenes': len(all_scene_results),
            'total_images': len(all_metrics_global),
            'overall': {
                'psnr': float(global_psnr),
                'ssim': float(global_ssim),
                'siou': float(global_siou),
                'avg_time_ms': float(global_time),
                'fps': float(global_fps)
            },
            'per_scene': [
                {
                    'scene': sr['scene'],
                    'psnr': sr['psnr_mean'],
                    'ssim': sr['ssim_mean'],
                    'siou': sr['siou_mean'],
                    'fps': sr['fps']
                }
                for sr in all_scene_results
            ],
            'system': {
                'model_size_mb': self.model_size_mb,
                'device': self.device,
                'depth_model': 'Depth-Anything-V2-Small',
                'params': {
                    'baseline': self.baseline,
                    'focal_length': self.focal_length,
                    'disparity_scale': self.disparity_scale
                }
            }
        }
        
        return results
    
    def _evaluate_single_scene(
        self,
        left_dir: str,
        right_dir: str,
        scene_name: str,
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict:
        """评估单个场景"""
        left_dir = Path(left_dir)
        right_dir = Path(right_dir)
        
        # 获取图像列表
        left_images = sorted(list(left_dir.glob("*.png")) + 
                            list(left_dir.glob("*.jpg")) + 
                            list(left_dir.glob("*.jpeg")))
        
        if len(left_images) == 0:
            print(f"警告: 场景 {scene_name} 未找到图像")
            return None
        
        # 创建输出目录
        if output_dir and save_results:
            scene_output = Path(output_dir) / scene_name
            scene_output.mkdir(parents=True, exist_ok=True)
        
        # 评估统计
        all_metrics = []
        all_times = []
        
        print(f"  开始处理 {len(left_images)} 张图像...")
        
        # 逐张处理
        for idx, left_path in enumerate(left_images, 1):
            left_name = left_path.name
            right_path = right_dir / left_name
            
            if not right_path.exists():
                continue
            
            # 读取图像
            left_img = cv2.imread(str(left_path))
            gt_right_img = cv2.imread(str(right_path))
            
            if left_img is None or gt_right_img is None:
                continue
            
            # 处理
            pred_right_img, infer_time = self.process_single_image(left_img)
            
            # 计算指标
            metrics = calculate_metrics(pred_right_img, gt_right_img, left_img)
            metrics['inference_time_ms'] = infer_time
            metrics['image_name'] = left_name
            
            all_metrics.append(metrics)
            all_times.append(infer_time)
            
            # 保存结果
            if save_results and output_dir:
                cv2.imwrite(str(scene_output / left_name), pred_right_img)
            
            # 显示进度
            if idx % 20 == 0 or idx == len(left_images):
                avg_time = np.mean(all_times[-20:]) if len(all_times) >= 20 else np.mean(all_times)
                current_fps = 1000.0 / avg_time
                avg_psnr = np.mean([m['psnr'] for m in all_metrics[-20:]]) if len(all_metrics) >= 20 else np.mean([m['psnr'] for m in all_metrics])
                print(f"    [{idx}/{len(left_images)}] FPS: {current_fps:.1f}, PSNR: {avg_psnr:.2f}dB")
        
        print(f"  ✓ 场景 {scene_name} 完成")
        print(f"    平均FPS: {1000.0/np.mean(all_times):.1f}")
        print(f"    平均PSNR: {np.mean([m['psnr'] for m in all_metrics]):.2f}dB")
        print(f"    平均SSIM: {np.mean([m['ssim'] for m in all_metrics]):.4f}")
        print(f"    平均SIoU: {np.mean([m['siou'] for m in all_metrics]):.4f}")
        
        if len(all_metrics) == 0:
            return None
        
        results = {
            'scene': scene_name,
            'num_images': len(all_metrics),
            'psnr_mean': float(np.mean([m['psnr'] for m in all_metrics])),
            'ssim_mean': float(np.mean([m['ssim'] for m in all_metrics])),
            'siou_mean': float(np.mean([m['siou'] for m in all_metrics])),
            'fps': float(1000.0 / np.mean(all_times)),
            'detailed_results': all_metrics
        }
        
        return results
    
    def print_summary(self, results: Dict):
        """打印评估摘要"""
        print("\n" + "="*70)
        print("轻量级DIBR系统评估报告")
        print("="*70)
        
        print(f"\n深度模型: {results['system']['depth_model']}")
        print(f"模型大小: {results['system']['model_size_mb']:.2f} MB")
        
        print(f"\n数据集: {results['dataset']}")
        print(f"场景数量: {results['num_scenes']}")
        print(f"图像总数: {results['total_images']}")
        
        # 各场景结果
        print("\n" + "-"*70)
        print("各场景性能")
        print("-"*70)
        print(f"{'场景':<12} {'PSNR(dB)':<10} {'SSIM':<10} {'SIoU':<10} {'FPS':<10}")
        print("-"*70)
        
        for scene_data in results['per_scene']:
            print(f"{scene_data['scene']:<12} "
                  f"{scene_data['psnr']:<10.2f} "
                  f"{scene_data['ssim']:<10.4f} "
                  f"{scene_data['siou']:<10.4f} "
                  f"{scene_data['fps']:<10.2f}")
        
        # 总体结果
        print("\n" + "-"*70)
        print("总体性能")
        print("-"*70)
        
        overall = results['overall']
        print(f"PSNR: {overall['psnr']:.2f} dB")
        print(f"SSIM: {overall['ssim']:.4f}")
        print(f"SIoU: {overall['siou']:.4f}")
        print(f"平均FPS: {overall['fps']:.2f}")
        
        # 与目标对比
        print("\n" + "-"*70)
        print("与华为赛题指标对比")
        print("-"*70)
        
        targets = {'PSNR': 32.0, 'SSIM': 0.75, 'SIoU': 0.28}
        
        print(f"PSNR: {overall['psnr']:.2f} dB ", end='')
        print(f"✓" if overall['psnr'] >= targets['PSNR'] else f"✗ (目标 > {targets['PSNR']} dB)")
        
        print(f"SSIM: {overall['ssim']:.4f} ", end='')
        print(f"✓" if overall['ssim'] >= targets['SSIM'] else f"✗ (目标 > {targets['SSIM']})")
        
        print(f"SIoU: {overall['siou']:.4f} ", end='')
        print(f"✓" if overall['siou'] >= targets['SIoU'] else f"✗ (目标 > {targets['SIoU']})")
        
        print(f"FPS: {overall['fps']:.2f} (目标 > 30)")
        print(f"模型大小: {results['system']['model_size_mb']:.2f} MB (目标 < 100 MB)")
        
        print("\n" + "="*70)


def main():
    import argparse
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser(description="轻量级DIBR评估 (Depth-Anything-V2)")
    parser.add_argument("--dataset", "-d", type=str, 
                       default=str(project_root / "Data" / "mono2stereo-test"),
                       help="数据集根目录")
    parser.add_argument("--output-dir", "-o", type=str,
                       default=str(project_root / "Data" / "outputs" / "depth_anything_v2_results"),
                       help="输出目录")
    parser.add_argument("--device", type=str, default="cuda")
    
    # 模型参数
    parser.add_argument("--encoder", type=str, default="vits",
                       choices=['vits', 'vitb', 'vitl'],
                       help="模型大小: vits=Small, vitb=Base, vitl=Large")
    parser.add_argument("--model-path", type=str,
                       default=str(project_root / "Data" / "checkpoints" / "depth_anything_v2_vits.pth"),
                       help="模型权重路径")
    
    # DIBR参数
    parser.add_argument("--baseline", type=float, default=0.065)
    parser.add_argument("--focal-length", type=float, default=800)
    parser.add_argument("--disparity-scale", type=float, default=1.0)
    
    args = parser.parse_args()
    
    print("="*70)
    print("轻量级DIBR系统 - Depth-Anything-V2")
    print("="*70)
    print()
    
    # 初始化评估器
    evaluator = LightweightDIBREvaluator(
        model_path=args.model_path,
        device=args.device,
        baseline=args.baseline,
        focal_length=args.focal_length,
        disparity_scale=args.disparity_scale,
        encoder=args.encoder
    )
    
    # 评估数据集
    results = evaluator.evaluate_dataset(
        dataset_root=args.dataset,
        output_dir=args.output_dir
    )
    
    # 打印摘要
    evaluator.print_summary(results)
    
    # 保存报告
    output_dir = Path(args.output_dir)
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n评估报告已保存: {report_path}")


if __name__ == "__main__":
    main()
