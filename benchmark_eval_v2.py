"""
DIBR系统性能评估脚本 - 支持Mono2Stereo数据集结构
数据集结构：
  mono2stereo-test/
    ├── animation/left/, right/
    ├── simple/left/, right/
    ├── complex/left/, right/
    ├── indoor/left/, right/
    └── outdoor/left/, right/
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 图像质量评估
from skimage.metrics import structural_similarity as ssim

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class EvaluationMetrics:
    """评估指标"""
    psnr: float
    ssim: float
    siou: float
    inference_time: float  # 毫秒
    fps: float


@dataclass 
class SystemInfo:
    """系统信息"""
    model_size_mb: float
    gpu_memory_mb: float
    device: str


def detect_edges(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """
    边缘检测
    
    Args:
        image: BGR图像 [H, W, 3]
        low: Canny低阈值
        high: Canny高阈值
        
    Returns:
        edges: 边缘图像 [H, W], 二值图
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return edges


def edge_overlap(edge1: np.ndarray, edge2: np.ndarray) -> float:
    """
    计算两个边缘图的IoU
    
    Args:
        edge1: 边缘图1 [H, W]
        edge2: 边缘图2 [H, W]
        
    Returns:
        iou: 边缘IoU
    """
    intersection = np.logical_and(edge1, edge2).sum()
    union = np.logical_or(edge1, edge2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def calculate_siou(pred: np.ndarray, gt: np.ndarray, left: np.ndarray) -> float:
    """
    计算结构相似度指标 (Structural IoU) - 官方实现版本
    
    SIoU = 0.75 * 边缘重叠 + 0.25 * 差异重叠
    
    Args:
        pred: 预测的右视角图像 [H, W, 3], BGR格式
        gt: 真实右视角图像 [H, W, 3], BGR格式
        left: 左视角图像 [H, W, 3], BGR格式 (参考图像)
        
    Returns:
        siou: 结构IoU分数 [0, 1]
    """
    # 边缘检测
    left_edges = detect_edges(left, 100, 200)
    pred_edges = detect_edges(pred, 100, 200)
    right_edges = detect_edges(gt, 100, 200)
    
    # 计算差异图
    diff_gl = np.abs(pred.astype(np.float32) - left.astype(np.float32))
    diff_rl = np.abs(gt.astype(np.float32) - left.astype(np.float32))
    
    # 转换为灰度
    diff_gl_gray = cv2.cvtColor(diff_gl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    diff_rl_gray = cv2.cvtColor(diff_rl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    # 差异二值化
    diff_gl_bin = np.zeros(diff_gl_gray.shape, dtype=np.uint8)
    diff_rl_bin = np.zeros(diff_rl_gray.shape, dtype=np.uint8)
    diff_gl_bin[diff_gl_gray > 5] = 1
    diff_rl_bin[diff_rl_gray > 5] = 1
    
    # 计算边缘重叠
    edge_overlap_gr = edge_overlap(pred_edges, right_edges)
    
    # 计算差异重叠
    diff_overlap_grl = edge_overlap(diff_gl_bin, diff_rl_bin)
    
    # 最终SIoU
    siou = 0.75 * edge_overlap_gr + 0.25 * diff_overlap_grl
    
    return siou


def calculate_metrics(pred: np.ndarray, gt: np.ndarray, left: np.ndarray = None) -> Dict[str, float]:
    """
    计算所有评估指标 - 与官方metrics.py一致
    
    Args:
        pred: 预测图像 [H, W, 3], BGR格式, uint8
        gt: 真实图像 [H, W, 3], BGR格式, uint8
        left: 左视角图像 [H, W, 3], BGR格式, uint8 (计算SIoU需要)
        
    Returns:
        metrics: 指标字典
    """
    # 确保图像尺寸一致
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
    if left is not None and pred.shape != left.shape:
        left = cv2.resize(left, (pred.shape[1], pred.shape[0]))
    
    # PSNR - 与官方一致: 20 * log10(255 / RMSE)
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0:
        psnr_value = 100.0  # 最大合理值
    else:
        rmse = np.sqrt(mse)
        psnr_value = 20 * np.log10(255.0 / rmse)
    
    # SSIM - 与官方一致: win_size=7
    ssim_value = ssim(gt, pred, multichannel=True, channel_axis=2, win_size=7, data_range=255)
    
    # SIoU - 需要left图像
    if left is not None:
        siou_value = calculate_siou(pred, gt, left)
    else:
        siou_value = 0.0  # 如果没有left图像，返回0
    
    return {
        'psnr': float(psnr_value),
        'ssim': float(ssim_value),
        'siou': float(siou_value)
    }


class DIBREvaluator:
    """DIBR系统评估器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        baseline: float = 0.065,
        focal_length: float = 800
    ):
        """
        初始化评估器
        
        Args:
            model_path: MiDaS模型路径
            device: 运行设备
            baseline: 双目基线距离
            focal_length: 焦距
        """
        self.device = device
        self.model_path = model_path
        
        # 加载模型
        print("加载MiDaS模型...")
        self.model = self._load_model(model_path)
        
        # 加载渲染器
        from dibr_renderer import DIBRRenderer
        self.renderer = DIBRRenderer(
            baseline=baseline,
            focal_length=focal_length
        )
        
        # 设置transform
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # 记录模型大小
        self.model_size_mb = self._get_model_size()
        
        print(f"✓ 模型加载完成 (大小: {self.model_size_mb:.2f} MB)")
    
    def _load_model(self, model_path: str):
        """加载MiDaS模型"""
        try:
            print(f"  正在加载模型权重...")
            
            # 尝试直接加载checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            print(f"  模型权重已加载到 {self.device}")
            
            # 检查checkpoint类型
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    print(f"  检测到完整模型对象")
                    model = checkpoint['model']
                    model.to(self.device)
                    print(f"  ✓ 模型已移动到 {self.device}")
                    return model
                else:
                    # 需要构建模型架构
                    print(f"  检测到state_dict格式，需要构建模型架构...")
                    
                    import sys
                    from pathlib import Path
                    
                    # 优先从项目目录查找MiDaS仓库
                    # benchmark_eval_v2.py 在 dibr_stereo 目录下，所以同级查找 MiDaS 文件夹
                    project_root = Path(__file__).parent
                    local_midas_repo = project_root / "MiDaS"
                    
                    # 如果项目目录没有，再查找torch hub缓存
                    hub_dir = Path.home() / ".cache" / "torch" / "hub"
                    cache_midas_repo = hub_dir / "intel-isl_MiDaS_master"
                    
                    midas_repo = None
                    if local_midas_repo.exists():
                        midas_repo = local_midas_repo
                        print(f"  找到项目MiDaS仓库: {midas_repo}")
                    elif cache_midas_repo.exists():
                        midas_repo = cache_midas_repo
                        print(f"  找到缓存MiDaS仓库: {midas_repo}")
                    
                    if midas_repo and midas_repo.exists():
                        sys.path.insert(0, str(midas_repo))
                        
                        try:
                            from midas.dpt_depth import DPTDepthModel
                            
                            print(f"  正在构建DPT模型...")
                            model = DPTDepthModel(
                                path=None,
                                backbone="vitb_rn50_384",
                                non_negative=True,
                            )
                            
                            # 加载权重
                            state_dict = checkpoint.get('state_dict', checkpoint)
                            model.load_state_dict(state_dict, strict=False)
                            model.to(self.device)
                            model.eval()
                            
                            print(f"  ✓ 模型构建成功并已加载到 {self.device}")
                            return model
                            
                        except Exception as e:
                            print(f"  从本地仓库构建失败: {e}，尝试使用torch.hub...")
                    
                    # 方法2：使用torch.hub加载模型架构并加载本地权重
                    print(f"  使用torch.hub下载MiDaS模型架构...")
                    try:
                        # 先通过torch.hub加载模型（会自动下载仓库）
                        hub_model = torch.hub.load(
                            "intel-isl/MiDaS",
                            "DPT_Hybrid",
                            pretrained=False,
                            trust_repo=True
                        )
                        
                        # 加载本地权重
                        state_dict = checkpoint.get('state_dict', checkpoint)
                        hub_model.load_state_dict(state_dict, strict=False)
                        hub_model.to(self.device)
                        hub_model.eval()
                        
                        print(f"  ✓ 通过torch.hub加载成功")
                        return hub_model
                        
                    except Exception as e:
                        print(f"  torch.hub加载失败: {e}")
                        raise RuntimeError(
                            "无法加载MiDaS模型。\n"
                            "请确保网络连接正常，或手动运行：\n"
                            "python -c \"import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', trust_repo=True)\""
                        )
            else:
                print(f"  检测到模型对象")
                checkpoint.to(self.device)
                print(f"  ✓ 模型已移动到 {self.device}")
                return checkpoint
                
        except Exception as e:
            print(f"  ✗ 模型加载失败: {e}")
            raise
    
    def _get_model_size(self) -> float:
        """获取模型大小（MB）"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        估计深度图
        
        Args:
            image: RGB图像 [H, W, 3]
            
        Returns:
            depth: 深度图 [H, W], 值域[0, 1]
        """
        from PIL import Image
        
        # 保存原始尺寸
        original_h, original_w = image.shape[:2]
        
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            original_w, original_h = image_pil.size
        
        # DPT模型需要384x384的输入，使用MiDaS官方的transform
        # 使用MiDaS官方的预处理方式
        import torch.nn.functional as F
        
        # 转换为tensor并归一化
        img_np = np.array(image_pil)
        img_tensor = torch.from_numpy(img_np).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # 归一化 (ImageNet标准)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Resize到384x384 (DPT模型要求)
        img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
        img_tensor = F.interpolate(img_tensor, size=(384, 384), mode='bilinear', align_corners=False)
        
        # 移动到设备
        input_tensor = img_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # 后处理
        prediction = prediction.squeeze().cpu().numpy()
        
        # Resize回原始尺寸
        prediction = cv2.resize(prediction, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        
        # 归一化到[0, 1]
        min_val = np.percentile(prediction, 0)
        max_val = np.percentile(prediction, 100)
        depth = (prediction - min_val) / (max_val - min_val + 1e-8)
        depth = 1.0 - depth  # 反转（MiDaS输出逆深度）
        
        return depth.astype(np.float32)
    
    @torch.no_grad()
    def process_single_image(
        self, 
        left_image: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        处理单张图像
        
        Args:
            left_image: 左视角图像（BGR格式）
            
        Returns:
            right_image: 生成的右视角图像（BGR格式）
            inference_time: 推理时间（毫秒）
        """
        # 计时开始
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
        inference_time = (time.time() - t_start) * 1000  # 毫秒
        
        return right_bgr, inference_time
    
    def evaluate_single_scene(
        self,
        left_dir: str,
        right_dir: str,
        scene_name: str,
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict:
        """
        评估单个场景
        
        Args:
            left_dir: 左视角图像目录
            right_dir: 右视角图像目录（GT）
            scene_name: 场景名称
            output_dir: 输出目录
            save_results: 是否保存结果
            
        Returns:
            results: 场景评估结果
        """
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
            # 查找对应的右视角图像
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
            
            # 计算指标 - 传入left图像用于SIoU计算
            metrics = calculate_metrics(pred_right_img, gt_right_img, left_img)
            metrics['inference_time_ms'] = infer_time
            metrics['image_name'] = left_name
            
            all_metrics.append(metrics)
            all_times.append(infer_time)
            
            # 保存结果
            if save_results and output_dir:
                cv2.imwrite(str(scene_output / left_name), pred_right_img)
            
            # 显示进度
            if idx % 10 == 0 or idx == len(left_images):
                avg_time = np.mean(all_times[-10:]) if len(all_times) >= 10 else np.mean(all_times)
                current_fps = 1000.0 / avg_time
                avg_psnr = np.mean([m['psnr'] for m in all_metrics[-10:]]) if len(all_metrics) >= 10 else np.mean([m['psnr'] for m in all_metrics])
                print(f"    [{idx}/{len(left_images)}] FPS: {current_fps:.1f}, PSNR: {avg_psnr:.2f}dB")
        
        print(f"  ✓ 场景 {scene_name} 完成")
        print(f"    平均FPS: {1000.0/np.mean(all_times):.1f}")
        print(f"    平均PSNR: {np.mean([m['psnr'] for m in all_metrics]):.2f}dB")
        print(f"    平均SSIM: {np.mean([m['ssim'] for m in all_metrics]):.4f}")
        print(f"    平均SIoU: {np.mean([m['siou'] for m in all_metrics]):.4f}")
        print()
        
        # 计算场景平均指标
        if len(all_metrics) == 0:
            return None
        
        results = {
            'scene': scene_name,
            'num_images': len(all_metrics),
            'psnr_mean': float(np.mean([m['psnr'] for m in all_metrics])),
            'psnr_std': float(np.std([m['psnr'] for m in all_metrics])),
            'ssim_mean': float(np.mean([m['ssim'] for m in all_metrics])),
            'ssim_std': float(np.std([m['ssim'] for m in all_metrics])),
            'siou_mean': float(np.mean([m['siou'] for m in all_metrics])),
            'siou_std': float(np.std([m['siou'] for m in all_metrics])),
            'avg_time_ms': float(np.mean(all_times)),
            'fps': float(1000.0 / np.mean(all_times)),
            'detailed_results': all_metrics
        }
        
        return results
    
    def evaluate_dataset(
        self,
        dataset_root: str,
        output_dir: Optional[str] = None,
        save_results: bool = True,
        scenes: Optional[List[str]] = None
    ) -> Dict:
        """
        评估整个数据集
        
        Args:
            dataset_root: 数据集根目录
            output_dir: 输出目录
            save_results: 是否保存结果
            scenes: 要评估的场景列表，None表示评估所有
            
        Returns:
            results: 完整评估结果
        """
        dataset_root = Path(dataset_root)
        
        # 自动检测场景
        if scenes is None:
            scenes = []
            for d in dataset_root.iterdir():
                if d.is_dir() and (d / "left").exists() and (d / "right").exists():
                    scenes.append(d.name)
            scenes = sorted(scenes)
        
        print(f"\n找到 {len(scenes)} 个场景: {', '.join(scenes)}\n")
        
        if len(scenes) == 0:
            raise ValueError(f"未找到有效场景目录: {dataset_root}")
        
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
            
            scene_result = self.evaluate_single_scene(
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
        
        # 汇总结果
        results = {
            'dataset': str(dataset_root),
            'num_scenes': len(all_scene_results),
            'total_images': len(all_metrics_global),
            'scenes': {sr['scene']: sr for sr in all_scene_results},
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
                'gpu_memory_mb': self._get_gpu_memory() if torch.cuda.is_available() else 0
            }
        }
        
        return results
    
    def _get_gpu_memory(self) -> float:
        """获取GPU显存使用量（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0
    
    def save_report(self, results: Dict, output_path: str):
        """保存评估报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 评估报告已保存: {output_path}")
    
    def print_summary(self, results: Dict):
        """打印评估摘要"""
        print("\n" + "="*70)
        print("DIBR系统评估报告")
        print("="*70)
        
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
        print(f"平均推理时间: {overall['avg_time_ms']:.2f} ms")
        print(f"平均FPS: {overall['fps']:.2f}")
        
        # 系统信息
        print("\n" + "-"*70)
        print("系统信息")
        print("-"*70)
        
        sys_info = results['system']
        print(f"模型大小: {sys_info['model_size_mb']:.2f} MB")
        print(f"运行设备: {sys_info['device']}")
        if sys_info['gpu_memory_mb'] > 0:
            print(f"GPU显存占用: {sys_info['gpu_memory_mb']:.2f} MB")
        
        print("\n" + "="*70)


def main():
    """主函数"""
    import argparse
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser(description="DIBR系统性能评估")
    parser.add_argument("--dataset", "-d", type=str, 
                       default=str(project_root / "Data" / "mono2stereo-test"),
                       help="数据集根目录")
    parser.add_argument("--output-dir", "-o", type=str,
                       default=str(project_root / "Data" / "outputs" / "midas_results"),
                       help="输出目录")
    parser.add_argument("--model-path", "-m", type=str,
                       default=str(project_root / "Data" / "checkpoints" / "dpt_hybrid_384.pt"),
                       help="MiDaS模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="运行设备")
    parser.add_argument("--baseline", type=float, default=0.065,
                       help="双目基线距离（米）")
    parser.add_argument("--focal-length", type=float, default=800,
                       help="焦距（像素）")
    parser.add_argument("--scenes", nargs='+', default=None,
                       help="要评估的场景列表（默认全部）")
    parser.add_argument("--no-save", action="store_true",
                       help="不保存生成的图像")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = DIBREvaluator(
        model_path=args.model_path,
        device=args.device,
        baseline=args.baseline,
        focal_length=args.focal_length
    )
    
    # 评估数据集
    results = evaluator.evaluate_dataset(
        dataset_root=args.dataset,
        output_dir=args.output_dir if not args.no_save else None,
        save_results=not args.no_save,
        scenes=args.scenes
    )
    
    # 打印摘要
    evaluator.print_summary(results)
    
    # 保存报告
    output_dir = Path(args.output_dir)
    report_path = output_dir / "evaluation_report.json"
    evaluator.save_report(results, report_path)
    
    # 保存简要结果
    summary = {
        'Overall': {
            'PSNR (dB)': round(results['overall']['psnr'], 2),
            'SSIM': round(results['overall']['ssim'], 4),
            'SIoU': round(results['overall']['siou'], 4),
            'FPS': round(results['overall']['fps'], 2),
            'Inference Time (ms)': round(results['overall']['avg_time_ms'], 2)
        },
        'Per_Scene': results['per_scene'],
        'Model_Size_MB': round(results['system']['model_size_mb'], 2)
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 保存CSV
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
    
    print(f"\n简要结果: {summary_path}")
    print(f"CSV结果: {csv_path}")
    print("\n评估完成！")


if __name__ == "__main__":
    main()
