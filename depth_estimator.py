"""
单目深度估计模块
使用预训练的MiDaS模型进行深度估计，无需训练
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import cv2


class MiDaSDepthEstimator:
    """
    基于MiDaS的单目深度估计器
    
    支持3种预训练模型：
    - DPT_Large: 最高质量，速度最慢
    - DPT_Hybrid: 质量与速度平衡
    - MiDaS_small: 最快速度
    """
    
    # MiDaS模型变体
    MODEL_TYPES = {
        "DPT_Large": "intel-isl/MiDaS/DPT_Large",
        "DPT_Hybrid": "intel-isl/MiDaS/DPT_Hybrid",
        "MiDaS_small": "intel-isl/MiDaS/MiDaS_small"
    }
    
    def __init__(
        self,
        model_type: str = "DPT_Large",
        device: str = "cuda",
        fp16: bool = False
    ):
        """
        初始化深度估计器
        
        Args:
            model_type: 模型类型，可选 "DPT_Large", "DPT_Hybrid", "MiDaS_small"
            device: 运行设备，"cuda" 或 "cpu"
            fp16: 是否使用半精度推理
        """
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device.type == "cuda"
        
        print(f"正在加载MiDaS模型: {model_type}")
        print(f"设备: {self.device}, 半精度: {self.fp16}")
        
        # 加载模型
        self._load_model()
        
        # 存储输入尺寸（MiDaS需要特定输入尺寸）
        self.input_size = None
        
    def _load_model(self):
        """加载MiDaS预训练模型"""
        try:
            # 方法1：使用torch.hub加载
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                self.model_type,
                pretrained=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            # 加载对应的transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if "DPT" in self.model_type:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
                
            print("✓ 模型加载成功")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("尝试使用备用加载方式...")
            self._load_model_alternative()
    
    def _load_model_alternative(self):
        """备用模型加载方式"""
        from torchvision import transforms
        
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            self.model_type,
            source="github",
            pretrained=True,
            force_reload=False
        )
        self.model.to(self.device)
        self.model.eval()
        
        # 标准变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        
        print("✓ 备用加载成功")
    
    @torch.no_grad()
    def estimate_depth(
        self,
        image: Union[np.ndarray, Image.Image],
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        估计单张图像的深度图
        
        Args:
            image: 输入图像，RGB格式，可以是numpy数组或PIL Image
            output_size: 输出深度图尺寸 (width, height)，None表示与输入相同
            
        Returns:
            depth: 深度图，numpy数组，值域[0, 1]，越大表示越近
        """
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            if image.shape[2] == 4:
                image = image[:, :, :3]
            image_pil = Image.fromarray(image)
            original_size = image.shape[:2][::-1]  # (width, height)
        else:
            original_size = image.size
            image_pil = image
        
        # 预处理
        input_tensor = self.transform(image_pil).to(self.device)
        
        # 添加batch维度
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # 半精度推理
        if self.fp16:
            input_tensor = input_tensor.half()
        
        # 推理
        with torch.cuda.amp.autocast(enabled=self.fp16):
            prediction = self.model(input_tensor)
        
        # 后处理
        prediction = prediction.squeeze().cpu().numpy()
        
        # 插值回原始尺寸
        prediction = cv2.resize(
            prediction,
            original_size,
            interpolation=cv2.INTER_CUBIC
        )
        
        # 如果指定了输出尺寸，再次调整
        if output_size is not None:
            prediction = cv2.resize(
                prediction,
                output_size,
                interpolation=cv2.INTER_CUBIC
            )
        
        # 归一化到[0, 1]
        depth = self._normalize_depth(prediction)
        
        return depth
    
    @torch.no_grad()
    def estimate_depth_batch(
        self,
        images: list,
        output_size: Optional[Tuple[int, int]] = None
    ) -> list:
        """
        批量估计深度图
        
        Args:
            images: 图像列表
            output_size: 输出尺寸
            
        Returns:
            depths: 深度图列表
        """
        depths = []
        for img in images:
            depth = self.estimate_depth(img, output_size)
            depths.append(depth)
        return depths
    
    def _normalize_depth(
        self,
        depth: np.ndarray,
        min_percentile: float = 0.0,
        max_percentile: float = 100.0
    ) -> np.ndarray:
        """
        归一化深度图到[0, 1]
        
        MiDaS输出的是逆深度，需要反转：
        - 逆深度：值越大，距离越近
        - 深度：值越大，距离越远
        """
        # 百分位裁剪
        min_val = np.percentile(depth, min_percentile)
        max_val = np.percentile(depth, max_percentile)
        
        depth_clipped = np.clip(depth, min_val, max_val)
        
        # 归一化到[0, 1]
        depth_norm = (depth_clipped - min_val) / (max_val - min_val + 1e-8)
        
        # MiDaS输出逆深度，反转得到真实深度
        # 逆深度：1=近，0=远
        # 真实深度：0=近，1=远
        depth_real = 1.0 - depth_norm
        
        return depth_real.astype(np.float32)
    
    def depth_to_disparity(
        self,
        depth: np.ndarray,
        baseline: float,
        focal_length: float,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        将深度图转换为视差图
        
        视差公式：disparity = (baseline * focal_length) / depth
        
        Args:
            depth: 深度图，单位：米
            baseline: 双目基线距离，单位：米
            focal_length: 焦距，单位：像素
            scale: 缩放因子
            
        Returns:
            disparity: 视差图，单位：像素
        """
        # 避免除零
        depth_safe = np.maximum(depth, 1e-3)
        
        # 计算视差
        disparity = (baseline * focal_length * scale) / depth_safe
        
        return disparity.astype(np.float32)
    
    def visualize_depth(
        self,
        depth: np.ndarray,
        colormap: int = cv2.COLORMAP_MAGMA
    ) -> np.ndarray:
        """
        可视化深度图
        
        Args:
            depth: 深度图 [0, 1]
            colormap: OpenCV色彩映射
            
        Returns:
            colored_depth: 彩色深度图
        """
        # 归一化到[0, 255]
        depth_uint8 = (depth * 255).astype(np.uint8)
        
        # 应用色彩映射
        colored_depth = cv2.applyColorMap(depth_uint8, colormap)
        
        return colored_depth
    
    def visualize_disparity(
        self,
        disparity: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        可视化视差图
        """
        # 归一化到[0, 255]
        disp_norm = cv2.normalize(
            disparity, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        
        # 应用色彩映射
        colored_disp = cv2.applyColorMap(disp_norm, colormap)
        
        return colored_disp


def test_depth_estimator():
    """测试深度估计器"""
    import matplotlib.pyplot as plt
    
    # 创建估计器
    estimator = MiDaSDepthEstimator(model_type="DPT_Hybrid")
    
    # 创建测试图像（渐变色）
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(640):
        test_img[:, i, :] = int(255 * i / 640)
    
    # 估计深度
    depth = estimator.estimate_depth(test_img)
    
    # 可视化
    colored_depth = estimator.visualize_depth(depth)
    
    print(f"深度图形状: {depth.shape}")
    print(f"深度值范围: [{depth.min():.3f}, {depth.max():.3f}]")
    
    # 显示
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(test_img)
    axes[0].set_title("Input Image")
    axes[1].imshow(depth, cmap='magma')
    axes[1].set_title("Estimated Depth")
    plt.tight_layout()
    plt.savefig("test_depth_result.png")
    print("结果已保存到 test_depth_result.png")


if __name__ == "__main__":
    test_depth_estimator()
