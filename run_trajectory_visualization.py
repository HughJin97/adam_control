#!/usr/bin/env python3
"""
启动MuJoCo轨迹可视化演示的便捷脚本
"""

import os
import sys
import subprocess

def main():
    """主函数"""
    print("🚀 启动AzureLoong机器人轨迹可视化演示")
    print("=" * 50)
    
    # 检查当前目录
    if not os.path.exists("models/scene.xml"):
        print("❌ 错误: 请在项目根目录运行此脚本")
        print("   当前目录:", os.getcwd())
        print("   需要的文件: models/scene.xml")
        return 1
    
    # 检查Python环境
    try:
        import mujoco
        print("✓ MuJoCo已安装")
    except ImportError:
        print("❌ 错误: MuJoCo未安装")
        print("   请运行: pip install mujoco")
        return 1
    
    try:
        import numpy as np
        print("✓ NumPy已安装")
    except ImportError:
        print("❌ 错误: NumPy未安装")
        print("   请运行: pip install numpy")
        return 1
    
    # 运行演示
    print("\n🎬 启动轨迹可视化演示...")
    try:
        # 直接运行演示脚本
        from examples.mujoco_trajectory_visualization import main as demo_main
        demo_main()
        return 0
    except Exception as e:
        print(f"❌ 运行演示时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 