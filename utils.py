"""
工具函数模块
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from pathlib import Path


def create_test_video(
    output_path: str,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    duration: float = 5.0,
    pattern: str = "moving_circle"
) -> str:
    """
    创建测试视频
    
    Args:
        output_path: 输出路径
        width: 视频宽度
        height: 视频高度
        fps: 帧率
        duration: 时长（秒）
        pattern: 测试模式
            - "moving_circle": 移动的圆形
            - "depth_layers": 深度层
            - "checkerboard": 棋盘格
            
    Returns:
        output_path: 输出路径
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(fps * duration)
    
    for i in range(total_frames):
        if pattern == "moving_circle":
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 前景圆形（近处）
            cx = int(width * (0.3 + 0.4 * np.sin(2 * np.pi * i / total_frames)))
            cy = height // 2
            cv2.circle(frame, (cx, cy), 50, (0, 0, 255), -1)
            
            # 背景矩形（远处）
            cv2.rectangle(frame, (100, 100), (width-100, height-100), (255, 0, 0), -1)
            
            # 中景三角形
            pts = np.array([
                [width//2 - 40, height//2 + 60],
                [width//2 + 40, height//2 + 60],
                [width//2, height//2 - 40]
            ], np.int32)
            cv2.fillPoly(frame, [pts], (0, 255, 0))
            
        elif pattern == "depth_layers":
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 背景（远）
            cv2.rectangle(frame, (0, 0), (width, height), (50, 50, 50), -1)
            
            # 中景层
            cv2.rectangle(frame, (50, 50), (width-50, height-50), (100, 100, 100), -1)
            
            # 前景层
            cv2.rectangle(frame, (100, 100), (width-100, height-100), (200, 200, 200), -1)
            
            # 最近层
            cv2.rectangle(frame, (150, 150), (width-150, height-150), (255, 255, 255), -1)
            
        elif pattern == "checkerboard":
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cell_size = 40
            
            for y in range(0, height, cell_size):
                for x in range(0, width, cell_size):
                    if (x // cell_size + y // cell_size) % 2 == 0:
                        cv2.rectangle(frame, (x, y), (x+cell_size, y+cell_size), (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    
    return output_path


def visualize_stereo_pair(
    left: np.ndarray,
    right: np.ndarray,
    title: str = "Stereo Pair",
    save_path: Optional[str] = None
):
    """
    可视化立体图像对
    
    Args:
        left: 左图像
        right: 右图像
        title: 标题
        save_path: 保存路径（可选）
    """
    # 并排显示
    stereo = np.hstack([left, right])
    
    cv2.imshow(title, stereo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if save_path:
        cv2.imwrite(save_path, stereo)


def calculate_metrics(
    left: np.ndarray,
    right: np.ndarray,
    right_gt: Optional[np.ndarray] = None
) -> dict:
    """
    计算评估指标
    
    Args:
        left: 左图像
        right: 生成的右图像
        right_gt: 真实右图像（可选）
        
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 视差统计
    # 通过左右图像的差异估计视差范围
    if right_gt is not None:
        # PSNR
        mse = np.mean((right.astype(float) - right_gt.astype(float)) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
        metrics["PSNR"] = psnr
        
        # SSIM
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(
            right, right_gt,
            multichannel=True,
            channel_axis=2
        )
        metrics["SSIM"] = ssim
    
    # 视差范围估计
    # 通过左右图像的水平位移估计
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    
    # 使用模板匹配估计最大视差
    template = gray_left[:, left.shape[1]//3:2*left.shape[1]//3]
    result = cv2.matchTemplate(
        gray_right, template, cv2.TM_CCOEFF_NORMED
    )
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    estimated_disparity = left.shape[1]//3 - max_loc[0]
    metrics["estimated_max_disparity"] = estimated_disparity
    
    return metrics


def resize_video(
    input_path: str,
    output_path: str,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    scale: Optional[float] = None
):
    """
    调整视频分辨率
    
    Args:
        input_path: 输入路径
        output_path: 输出路径
        target_width: 目标宽度
        target_height: 目标高度
        scale: 缩放因子（如果提供，则忽略target_width和target_height）
    """
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if scale:
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width = target_width or width
        new_height = target_height or height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized = cv2.resize(frame, (new_width, new_height))
        out.write(resized)
    
    cap.release()
    out.release()


def extract_frames(video_path: str, output_dir: str, pattern: str = "frame_{:06d}.png"):
    """
    提取视频帧
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        pattern: 文件命名模式
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        output_path = str(Path(output_dir) / pattern.format(frame_idx))
        cv2.imwrite(output_path, frame)
        frame_idx += 1
    
    cap.release()
    print(f"已提取 {frame_idx} 帧到 {output_dir}")


def frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: float = 30.0,
    pattern: str = "*.png"
):
    """
    将帧序列合成视频
    
    Args:
        frames_dir: 帧目录
        output_path: 输出路径
        fps: 帧率
        pattern: 文件匹配模式
    """
    from glob import glob
    
    frame_files = sorted(glob(str(Path(frames_dir) / pattern)))
    
    if not frame_files:
        raise ValueError(f"未找到匹配的帧文件: {Path(frames_dir) / pattern}")
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(frame_files[0])
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)
    
    out.release()
    print(f"已合成视频: {output_path}")


def create_anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    创建红蓝立体图
    
    Args:
        left: 左图像
        right: 右图像
        
    Returns:
        anaglyph: 红蓝立体图
    """
    # 左眼红色通道
    # 右眼绿色和蓝色通道
    anaglyph = np.zeros_like(left)
    anaglyph[:, :, 2] = left[:, :, 2]  # R from left
    anaglyph[:, :, 1] = right[:, :, 1]  # G from right
    anaglyph[:, :, 0] = right[:, :, 0]  # B from right
    
    return anaglyph


if __name__ == "__main__":
    # 测试：创建测试视频
    print("创建测试视频...")
    test_video = create_test_video(
        "test_input.mp4",
        pattern="moving_circle",
        duration=3.0
    )
    print(f"测试视频已创建: {test_video}")
