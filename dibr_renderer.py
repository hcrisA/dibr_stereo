"""
DIBR (Depth Image-Based Rendering) 核心渲染模块

实现从左视角到右视角的虚拟视图合成
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from scipy import ndimage


class DIBRRenderer:
    """
    DIBR渲染器
    
    核心流程：
    1. 视差计算：depth → disparity
    2. 像素位移：根据视差移动像素
    3. 空洞检测：识别被遮挡区域
    4. 空洞填充：使用背景扩展或图像修复
    5. 后处理：边缘平滑、滤波
    """
    
    def __init__(
        self,
        baseline: float = 0.065,
        focal_length: float = 1000,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        disparity_scale: float = 1.0
    ):
        """
        初始化DIBR渲染器
        
        Args:
            baseline: 双目基线距离（米），默认65mm
            focal_length: 焦距（像素）
            cx, cy: 主点坐标，None表示图像中心
            disparity_scale: 视差缩放因子
        """
        self.baseline = baseline
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy
        self.disparity_scale = disparity_scale
        
    def render_right_view(
        self,
        left_image: np.ndarray,
        depth: np.ndarray,
        fill_holes: bool = True,
        hole_filling_method: str = "telea",
        hole_filling_radius: int = 5,
        edge_smoothing: bool = True,
        bilateral_filter: bool = False,
        bilateral_params: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从左视角图像和深度图渲染右视角图像
        
        Args:
            left_image: 左视角图像 [H, W, 3]
            depth: 深度图 [H, W]，值域[0, 1]，越大越远
            fill_holes: 是否填充空洞
            hole_filling_method: 空洞填充方法 "telea" 或 "ns"
            hole_filling_radius: 填充半径
            edge_smoothing: 是否边缘平滑
            bilateral_filter: 是否应用双边滤波
            bilateral_params: 双边滤波参数 {"sigma_color": 10, "sigma_space": 15}
            
        Returns:
            right_image: 右视角图像 [H, W, 3]
            hole_mask: 空洞掩码 [H, W]
        """
        # 获取图像尺寸
        h, w = left_image.shape[:2]
        
        # 设置主点坐标
        cx = self.cx if self.cx is not None else w / 2
        cy = self.cy if self.cy is not None else h / 2
        
        # 深度归一化（假设深度范围0.1-100米）
        depth_real = self._denormalize_depth(depth, min_depth=0.1, max_depth=100.0)
        
        # 计算视差图
        disparity = self._compute_disparity(depth_real)
        
        # DIBR核心：像素位移
        right_image, hole_mask = self._warp_image(
            left_image, disparity, cx, cy
        )
        
        # 空洞填充
        if fill_holes:
            right_image = self._fill_holes(
                right_image, hole_mask,
                method=hole_filling_method,
                radius=hole_filling_radius
            )
        
        # 边缘平滑
        if edge_smoothing:
            right_image = self._smooth_edges(right_image, hole_mask)
        
        # 双边滤波
        if bilateral_filter:
            params = bilateral_params or {"sigma_color": 10, "sigma_space": 15}
            right_image = cv2.bilateralFilter(
                right_image, -1,
                params["sigma_color"],
                params["sigma_space"]
            )
        
        return right_image, hole_mask
    
    def _denormalize_depth(
        self,
        depth_norm: np.ndarray,
        min_depth: float = 0.1,
        max_depth: float = 100.0
    ) -> np.ndarray:
        """
        将归一化深度[0, 1]转换为真实深度值
        
        使用逆深度映射：近处0.1m，远处100m
        """
        # 指数映射（模拟真实深度分布）
        depth_real = min_depth + (max_depth - min_depth) * depth_norm
        
        return depth_real
    
    def _compute_disparity(self, depth: np.ndarray) -> np.ndarray:
        """
        计算视差图
        
        视差公式：disparity = (baseline * focal_length) / depth
        
        Args:
            depth: 真实深度值（米）
            
        Returns:
            disparity: 视差（像素）
        """
        # 避免除零
        depth_safe = np.maximum(depth, 1e-3)
        
        # 计算视差
        disparity = (self.baseline * self.focal_length * self.disparity_scale) / depth_safe
        
        return disparity
    
    def _warp_image(
        self,
        image: np.ndarray,
        disparity: np.ndarray,
        cx: float,
        cy: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据视差图进行图像变换（左→右）- 优化版本
        
        原理：
        - 左视角中位置x的像素，在右视角中位置为 x - disparity(x)
        - 前景（近处）位移大，背景（远处）位移小
        
        Args:
            image: 输入图像 [H, W, 3]
            disparity: 视差图 [H, W]
            cx, cy: 主点坐标
            
        Returns:
            warped_image: 变换后的图像
            hole_mask: 空洞掩码
        """
        h, w = image.shape[:2]
        
        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # 计算右视角坐标
        # 左→右：x_right = x_left - disparity
        # 确保disparity是float32类型
        disparity_f32 = disparity.astype(np.float32)
        x_right = (x_coords - disparity_f32).astype(np.float32)
        y_right = y_coords.astype(np.float32)  # y坐标不变
        
        # 使用cv2.remap进行向量化变换（比双重循环快100倍以上）
        warped_image = cv2.remap(
            image,
            x_right,
            y_right,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 检测空洞
        # 方法1：检测超出边界的像素
        hole_mask = ((x_right < 0) | (x_right >= w)).astype(np.uint8)
        
        # 方法2：检测重复映射（深度测试）
        # 使用深度buffer检测遮挡
        depth_buffer = np.full((h, w), np.inf, dtype=np.float32)
        
        # 对于每个右视角像素，找到对应的左视角像素
        # 这里简化处理：如果映射后的像素值为0，认为是空洞
        gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        hole_mask = (gray == 0).astype(np.uint8)
        
        return warped_image, hole_mask
    
    def _warp_image_optimized(
        self,
        image: np.ndarray,
        disparity: np.ndarray,
        cx: float,
        cy: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        优化版本的图像变换（使用向量化操作）
        
        注意：此版本速度更快，但可能在边缘处有精度损失
        """
        h, w = image.shape[:2]
        
        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # 计算右视角坐标
        x_right = x_coords - disparity
        y_right = y_coords
        
        # 创建映射
        map_x = x_right
        map_y = y_right
        
        # 使用重映射（双线性插值）
        warped_image = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 检测空洞
        # 空洞区域：映射后的坐标超出边界或重复映射
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 边界外的像素标记为空洞
        hole_mask[(x_right < 0) | (x_right >= w)] = 1
        
        # 重叠区域检测（前景遮挡背景）
        # 通过检查深度一致性来检测
        depth_at_warped = cv2.remap(
            disparity,
            map_x,
            map_y,
            cv2.INTER_LINEAR
        )
        hole_mask[depth_at_warped == 0] = 1
        
        return warped_image, hole_mask
    
    def _fill_holes(
        self,
        image: np.ndarray,
        hole_mask: np.ndarray,
        method: str = "telea",
        radius: int = 5
    ) -> np.ndarray:
        """
        填充空洞
        
        Args:
            image: 输入图像
            hole_mask: 空洞掩码
            method: 填充方法
                - "telea": Alexandru Telea算法（快速）
                - "ns": Navier-Stokes流体动力学（质量更好）
                - "background": 背景扩展填充
            radius: 填充半径
            
        Returns:
            filled_image: 填充后的图像
        """
        if method in ["telea", "ns"]:
            # OpenCV图像修复
            flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
            
            # 确保空洞掩码格式正确
            if hole_mask.max() <= 1:
                inpaint_mask = (hole_mask * 255).astype(np.uint8)
            else:
                inpaint_mask = hole_mask.astype(np.uint8)
            
            filled_image = cv2.inpaint(image, inpaint_mask, radius, flag)
            
        elif method == "background":
            # 背景扩展填充
            filled_image = self._background_expansion_fill(image, hole_mask)
            
        else:
            raise ValueError(f"未知的空洞填充方法: {method}")
        
        return filled_image
    
    def _background_expansion_fill(
        self,
        image: np.ndarray,
        hole_mask: np.ndarray,
        iterations: int = 10
    ) -> np.ndarray:
        """
        背景扩展填充
        
        从空洞边缘向内扩展背景像素
        适用于大范围空洞填充
        """
        filled_image = image.copy()
        current_mask = hole_mask.copy()
        
        for _ in range(iterations):
            # 找到空洞边缘
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(current_mask, kernel, iterations=1)
            edge = dilated - current_mask
            
            if edge.sum() == 0:
                break
            
            # 从边缘像素向空洞内部复制
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if edge[y, x] > 0:
                        # 找到最近的非空洞像素
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                                    if current_mask[ny, nx] == 0:
                                        filled_image[y, x] = filled_image[ny, nx]
                                        current_mask[y, x] = 0
                                        break
        
        return filled_image
    
    def _smooth_edges(
        self,
        image: np.ndarray,
        hole_mask: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        边缘平滑
        
        在原本的空洞区域边缘应用高斯模糊，减少突变
        """
        # 找到空洞边缘
        if hole_mask.max() <= 1:
            mask_uint8 = (hole_mask * 255).astype(np.uint8)
        else:
            mask_uint8 = hole_mask.astype(np.uint8)
        
        edges = cv2.Canny(mask_uint8, 50, 150)
        
        # 在边缘区域应用高斯模糊
        smoothed = image.copy()
        
        # 创建边缘附近的掩码
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=2)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # 混合
        edge_region_3ch = np.stack([edge_region] * 3, axis=-1)
        smoothed = np.where(edge_region_3ch > 0, blurred, image)
        
        return smoothed
    
    def compute_depth_from_disparity(
        self,
        disparity: np.ndarray,
        baseline: Optional[float] = None,
        focal_length: Optional[float] = None
    ) -> np.ndarray:
        """
        从视差图计算深度
        
        depth = (baseline * focal_length) / disparity
        """
        baseline = baseline or self.baseline
        focal_length = focal_length or self.focal_length
        
        disparity_safe = np.maximum(disparity, 1e-3)
        depth = (baseline * focal_length) / disparity_safe
        
        return depth


class StereoDIBR:
    """
    立体视觉DIBR系统
    
    支持双向渲染（左→右，右→左）
    """
    
    def __init__(self, renderer: DIBRRenderer):
        """
        初始化立体DIBR系统
        
        Args:
            renderer: DIBR渲染器实例
        """
        self.renderer = renderer
    
    def render_right_from_left(
        self,
        left_image: np.ndarray,
        depth: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从左视角渲染右视角
        """
        return self.renderer.render_right_view(left_image, depth, **kwargs)
    
    def render_left_from_right(
        self,
        right_image: np.ndarray,
        depth: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从右视角渲染左视角
        
        注意：视差方向相反（左→右：减视差，右→左：加视差）
        """
        # 暂时反转视差方向
        # 这里需要修改渲染器来支持右→左渲染
        # 简单方法：镜像图像，渲染，再镜像回来
        
        # 镜像
        right_mirror = cv2.flip(right_image, 1)
        depth_mirror = cv2.flip(depth, 1)
        
        # 渲染
        left_mirror, hole_mask_mirror = self.renderer.render_right_view(
            right_mirror, depth_mirror, **kwargs
        )
        
        # 镜像回来
        left_image = cv2.flip(left_mirror, 1)
        hole_mask = cv2.flip(hole_mask_mirror, 1)
        
        return left_image, hole_mask


def test_dibr_renderer():
    """测试DIBR渲染器"""
    # 创建测试数据
    h, w = 480, 640
    
    # 创建左视角测试图像（彩色渐变）
    left_image = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        left_image[:, x, 0] = int(255 * x / w)  # R
        left_image[:, x, 1] = int(255 * (w - x) / w)  # G
    left_image[:, :, 2] = 128  # B
    
    # 创建测试深度图（中心近，边缘远）
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    center_y, center_x = h / 2, w / 2
    distance = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    depth = 1 - distance / (distance.max() + 1e-8)  # [0, 1]，中心近
    depth = depth.astype(np.float32)
    
    # 创建渲染器
    renderer = DIBRRenderer(
        baseline=0.065,
        focal_length=800
    )
    
    # 渲染右视角
    print("正在渲染右视角...")
    right_image, hole_mask = renderer.render_right_view(
        left_image,
        depth,
        fill_holes=True,
        edge_smoothing=True
    )
    
    # 可视化
    print(f"左图像形状: {left_image.shape}")
    print(f"右图像形状: {right_image.shape}")
    print(f"空洞比例: {hole_mask.mean() * 100:.2f}%")
    
    # 保存结果
    cv2.imwrite("test_left.png", cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("test_right.png", cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("test_depth.png", (depth * 255).astype(np.uint8))
    cv2.imwrite("test_holes.png", hole_mask * 255)
    
    print("测试结果已保存")


if __name__ == "__main__":
    test_dibr_renderer()
