"""
视频处理流水线

整合深度估计和DIBR渲染，实现从单目左视频到右视频的完整处理流程
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
from tqdm import tqdm
import yaml
import time
from dataclasses import dataclass

# 导入本地模块
from depth_estimator import MiDaSDepthEstimator
from dibr_renderer import DIBRRenderer


@dataclass
class VideoInfo:
    """视频信息"""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str


class VideoProcessor:
    """
    视频处理器
    
    负责：
    1. 视频读取
    2. 帧级处理（深度估计 + DIBR渲染）
    3. 视频写入
    4. 进度显示
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化视频处理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化模型和渲染器
        self.depth_estimator = None
        self.renderer = None
        
        # 性能统计
        self.stats = {
            "depth_time": [],
            "render_time": [],
            "total_time": []
        }
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _init_models(self, device: Optional[str] = None):
        """初始化模型"""
        if self.depth_estimator is None:
            print("初始化深度估计模型...")
            self.depth_estimator = MiDaSDepthEstimator(
                model_type=self.config["depth"]["model"],
                device=device or self.config["depth"]["device"],
                fp16=self.config["performance"]["fp16"]
            )
        
        if self.renderer is None:
            print("初始化DIBR渲染器...")
            camera_config = self.config["camera"]
            self.renderer = DIBRRenderer(
                baseline=camera_config["baseline"],
                focal_length=camera_config["focal_length"],
                cx=camera_config["cx"],
                cy=camera_config["cy"],
                disparity_scale=self.config["dibr"]["disparity_scale"]
            )
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """获取视频信息"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        codec_code = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((codec_code >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec
        )
    
    def process_video(
        self,
        input_video: str,
        output_video: str,
        device: Optional[str] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        save_depth: bool = True,
        save_intermediate: bool = False,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        处理视频
        
        Args:
            input_video: 输入视频路径（左视频）
            output_video: 输出视频路径（右视频）
            device: 运行设备
            start_frame: 起始帧
            end_frame: 结束帧
            save_depth: 是否保存深度图视频
            save_intermediate: 是否保存中间结果
            callback: 进度回调函数 callback(frame_idx, total_frames)
            
        Returns:
            stats: 处理统计信息
        """
        # 初始化模型
        self._init_models(device)
        
        # 获取视频信息
        video_info = self.get_video_info(input_video)
        print(f"\n视频信息:")
        print(f"  分辨率: {video_info.width}x{video_info.height}")
        print(f"  帧率: {video_info.fps:.2f} FPS")
        print(f"  总帧数: {video_info.frame_count}")
        print(f"  时长: {video_info.duration:.2f}秒")
        
        # 设置处理范围
        end_frame = end_frame or video_info.frame_count
        total_frames = end_frame - start_frame
        
        # 打开输入视频
        cap = cv2.VideoCapture(input_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 准备输出
        output_dir = Path(output_video).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 输出视频设置
        output_fps = self.config["output"]["video_fps"] or video_info.fps
        output_codec = cv2.VideoWriter_fourcc(*self.config["output"]["video_codec"])
        
        out_right = cv2.VideoWriter(
            output_video,
            output_codec,
            output_fps,
            (video_info.width, video_info.height)
        )
        
        out_depth = None
        if save_depth:
            depth_video_path = str(output_video).replace(".mp4", "_depth.mp4")
            out_depth = cv2.VideoWriter(
                depth_video_path,
                output_codec,
                output_fps,
                (video_info.width, video_info.height),
                isColor=True
            )
        
        # 中间结果目录
        if save_intermediate:
            intermediate_dir = output_dir / "intermediate"
            intermediate_dir.mkdir(exist_ok=True)
        
        # 重置统计
        self.stats = {"depth_time": [], "render_time": [], "total_time": []}
        
        print(f"\n开始处理 {total_frames} 帧...")
        
        try:
            frame_idx = 0
            pbar = tqdm(total=total_frames, desc="处理进度")
            
            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= end_frame:
                    break
                
                # 计时开始
                t_start = time.time()
                
                # BGR转RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 深度估计
                t_depth_start = time.time()
                depth = self.depth_estimator.estimate_depth(frame_rgb)
                t_depth = time.time() - t_depth_start
                self.stats["depth_time"].append(t_depth)
                
                # DIBR渲染
                t_render_start = time.time()
                right_image, hole_mask = self.renderer.render_right_view(
                    frame_rgb,
                    depth,
                    fill_holes=self.config["dibr"]["hole_filling"]["method"] != "none",
                    hole_filling_method=self.config["dibr"]["hole_filling"]["method"],
                    hole_filling_radius=self.config["dibr"]["hole_filling"]["radius"],
                    edge_smoothing=self.config["dibr"]["postprocess"]["edge_smoothing"],
                    bilateral_filter=self.config["dibr"]["postprocess"]["bilateral_filter"],
                    bilateral_params={
                        "sigma_color": self.config["dibr"]["postprocess"]["sigma_color"],
                        "sigma_space": self.config["dibr"]["postprocess"]["sigma_space"]
                    } if self.config["dibr"]["postprocess"]["bilateral_filter"] else None
                )
                t_render = time.time() - t_render_start
                self.stats["render_time"].append(t_render)
                
                # RGB转BGR并写入
                right_image_bgr = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)
                out_right.write(right_image_bgr)
                
                # 保存深度图视频
                if out_depth is not None:
                    depth_colored = self.depth_estimator.visualize_depth(depth)
                    out_depth.write(depth_colored)
                
                # 保存中间结果
                if save_intermediate and frame_idx % 10 == 0:
                    cv2.imwrite(
                        str(intermediate_dir / f"frame_{frame_idx:06d}_left.png"),
                        frame
                    )
                    cv2.imwrite(
                        str(intermediate_dir / f"frame_{frame_idx:06d}_right.png"),
                        right_image_bgr
                    )
                    cv2.imwrite(
                        str(intermediate_dir / f"frame_{frame_idx:06d}_depth.png"),
                        (depth * 255).astype(np.uint8)
                    )
                
                # 总时间
                t_total = time.time() - t_start
                self.stats["total_time"].append(t_total)
                
                # 回调
                if callback:
                    callback(frame_idx, total_frames)
                
                frame_idx += 1
                pbar.update(1)
                pbar.set_postfix({
                    "FPS": f"{1.0/t_total:.1f}",
                    "Depth": f"{t_depth:.3f}s",
                    "Render": f"{t_render:.3f}s"
                })
            
            pbar.close()
            
        finally:
            cap.release()
            out_right.release()
            if out_depth:
                out_depth.release()
        
        # 计算统计信息
        result_stats = {
            "total_frames": frame_idx - start_frame,
            "avg_depth_time": np.mean(self.stats["depth_time"]),
            "avg_render_time": np.mean(self.stats["render_time"]),
            "avg_total_time": np.mean(self.stats["total_time"]),
            "avg_fps": 1.0 / np.mean(self.stats["total_time"]),
            "total_time": np.sum(self.stats["total_time"])
        }
        
        return result_stats
    
    def process_single_frame(
        self,
        frame: np.ndarray,
        device: Optional[str] = None,
        return_depth: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        处理单帧图像
        
        Args:
            frame: 输入帧（BGR格式）
            device: 运行设备
            return_depth: 是否返回深度图
            
        Returns:
            right_image: 右视角图像（BGR）
            depth: 深度图（如果return_depth=True）
        """
        # 初始化模型
        self._init_models(device)
        
        # BGR转RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 深度估计
        depth = self.depth_estimator.estimate_depth(frame_rgb)
        
        # DIBR渲染
        right_image, _ = self.renderer.render_right_view(
            frame_rgb,
            depth,
            fill_holes=True,
            edge_smoothing=True
        )
        
        # RGB转BGR
        right_image_bgr = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)
        
        if return_depth:
            return right_image_bgr, depth
        else:
            return right_image_bgr, None


class RealTimeProcessor:
    """
    实时处理器
    
    用于摄像头实时处理
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.processor = VideoProcessor(config_path)
        self.processor._init_models()
    
    def run(
        self,
        camera_id: int = 0,
        display: bool = True,
        save_output: bool = False,
        output_path: str = "realtime_output.mp4"
    ):
        """
        实时处理摄像头视频
        
        Args:
            camera_id: 摄像头ID
            display: 是否显示结果
            save_output: 是否保存输出
            output_path: 输出路径
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头 {camera_id}")
        
        # 获取摄像头参数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # 输出视频
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        print(f"实时处理中... 按 'q' 退出")
        
        cv2.namedWindow("DIBR Stereo", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理
                right_frame, depth = self.processor.process_single_frame(frame)
                
                # 拼接左右图像
                stereo = np.hstack([frame, right_frame])
                
                # 显示
                if display:
                    cv2.imshow("DIBR Stereo", stereo)
                
                # 保存
                if out:
                    out.write(stereo)
                
                # 按键检测
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        
        print("实时处理结束")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DIBR单目转立体视频系统")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入左视频路径")
    parser.add_argument("--output", "-o", type=str, default="output_right.mp4", help="输出右视频路径")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="运行设备")
    parser.add_argument("--start", "-s", type=int, default=0, help="起始帧")
    parser.add_argument("--end", "-e", type=int, default=None, help="结束帧")
    parser.add_argument("--save-depth", action="store_true", help="保存深度图视频")
    parser.add_argument("--realtime", action="store_true", help="实时处理模式")
    
    args = parser.parse_args()
    
    if args.realtime:
        # 实时处理模式
        processor = RealTimeProcessor(args.config)
        processor.run()
    else:
        # 视频文件处理模式
        processor = VideoProcessor(args.config)
        
        stats = processor.process_video(
            input_video=args.input,
            output_video=args.output,
            device=args.device,
            start_frame=args.start,
            end_frame=args.end,
            save_depth=args.save_depth
        )
        
        print("\n" + "="*50)
        print("处理完成！")
        print("="*50)
        print(f"总帧数: {stats['total_frames']}")
        print(f"平均深度估计时间: {stats['avg_depth_time']*1000:.1f}ms")
        print(f"平均渲染时间: {stats['avg_render_time']*1000:.1f}ms")
        print(f"平均处理时间: {stats['avg_total_time']*1000:.1f}ms")
        print(f"平均FPS: {stats['avg_fps']:.2f}")
        print(f"总处理时间: {stats['total_time']:.2f}秒")
        print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
