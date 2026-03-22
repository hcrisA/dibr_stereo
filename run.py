#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIBR单目转立体视频系统 - 一键运行脚本
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from video_pipeline import VideoProcessor, RealTimeProcessor


def check_dependencies():
    """检查依赖"""
    print("检查依赖...")
    
    missing = []
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"    CUDA: {torch.version.cuda}")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        missing.append("torch")
    
    try:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError:
        missing.append("numpy")
    
    try:
        import yaml
        print(f"  ✓ PyYAML")
    except ImportError:
        missing.append("pyyaml")
    
    try:
        from PIL import Image
        print(f"  ✓ Pillow")
    except ImportError:
        missing.append("Pillow")
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        return False
    
    print("\n✓ 所有依赖已安装\n")
    return True


def process_video(args):
    """处理视频文件"""
    processor = VideoProcessor(args.config)
    
    # 获取视频信息
    video_info = processor.get_video_info(args.input)
    print(f"输入视频: {args.input}")
    print(f"分辨率: {video_info.width}x{video_info.height}")
    print(f"帧率: {video_info.fps:.2f} FPS")
    print(f"总帧数: {video_info.frame_count}")
    print(f"时长: {video_info.duration:.2f}秒\n")
    
    # 处理
    stats = processor.process_video(
        input_video=args.input,
        output_video=args.output,
        device=args.device,
        start_frame=args.start,
        end_frame=args.end,
        save_depth=args.save_depth
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"总帧数: {stats['total_frames']}")
    print(f"平均深度估计时间: {stats['avg_depth_time']*1000:.1f}ms")
    print(f"平均渲染时间: {stats['avg_render_time']*1000:.1f}ms")
    print(f"平均处理时间: {stats['avg_total_time']*1000:.1f}ms")
    print(f"平均FPS: {stats['avg_fps']:.2f}")
    print(f"总处理时间: {stats['total_time']:.2f}秒")
    print(f"\n输出文件: {args.output}")
    if args.save_depth:
        print(f"深度图视频: {args.output.replace('.mp4', '_depth.mp4')}")


def process_realtime(args):
    """实时处理"""
    processor = RealTimeProcessor(args.config)
    processor.run(
        camera_id=args.camera,
        display=not args.no_display,
        save_output=args.save,
        output_path=args.output
    )


def main():
    parser = argparse.ArgumentParser(
        description="DIBR单目转立体视频系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理视频文件
  python run.py -i input_left.mp4 -o output_right.mp4
  
  # 实时处理摄像头
  python run.py --realtime --camera 0
  
  # 使用GPU处理视频
  python run.py -i input.mp4 -o output.mp4 --device cuda
  
  # 保存深度图视频
  python run.py -i input.mp4 -o output.mp4 --save-depth

配置文件说明 (config.yaml):
  camera.baseline: 双目基线距离（米）
  camera.focal_length: 焦距（像素）
  depth.model: MiDaS模型类型（DPT_Large/DPT_Hybrid/MiDaS_small）
  dibr.hole_filling.method: 空洞填充方法（telea/ns）
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="处理模式")
    
    # 视频文件处理模式
    video_parser = subparsers.add_parser("video", help="处理视频文件")
    video_parser.add_argument("-i", "--input", required=True, help="输入左视频路径")
    video_parser.add_argument("-o", "--output", default="output_right.mp4", help="输出右视频路径")
    video_parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    video_parser.add_argument("-d", "--device", default="cuda", help="运行设备")
    video_parser.add_argument("-s", "--start", type=int, default=0, help="起始帧")
    video_parser.add_argument("-e", "--end", type=int, default=None, help="结束帧")
    video_parser.add_argument("--save-depth", action="store_true", help="保存深度图视频")
    
    # 实时处理模式
    realtime_parser = subparsers.add_parser("realtime", help="实时处理摄像头")
    realtime_parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    realtime_parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    realtime_parser.add_argument("--no-display", action="store_true", help="不显示结果")
    realtime_parser.add_argument("--save", action="store_true", help="保存输出视频")
    realtime_parser.add_argument("-o", "--output", default="realtime_output.mp4", help="输出路径")
    
    # 默认模式（兼容旧版）
    parser.add_argument("-i", "--input", help="输入左视频路径")
    parser.add_argument("-o", "--output", default="output_right.mp4", help="输出右视频路径")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("-d", "--device", default="cuda", help="运行设备")
    parser.add_argument("-s", "--start", type=int, default=0, help="起始帧")
    parser.add_argument("-e", "--end", type=int, default=None, help="结束帧")
    parser.add_argument("--save-depth", action="store_true", help="保存深度图视频")
    parser.add_argument("--realtime", action="store_true", help="实时处理模式")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    parser.add_argument("--no-display", action="store_true", help="不显示结果")
    parser.add_argument("--check-deps", action="store_true", help="检查依赖")
    
    args = parser.parse_args()
    
    # 检查依赖
    if hasattr(args, 'check_deps') and args.check_deps:
        check_dependencies()
        return
    
    if not check_dependencies():
        sys.exit(1)
    
    # 根据模式处理
    if args.realtime or getattr(args, 'mode', None) == "realtime":
        process_realtime(args)
    else:
        if not hasattr(args, 'input') or args.input is None:
            parser.print_help()
            print("\n错误: 请提供输入视频路径 (-i/--input)")
            sys.exit(1)
        process_video(args)


if __name__ == "__main__":
    main()
