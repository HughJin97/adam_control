#!/bin/bash
# macOS MuJoCo轨迹可视化启动脚本

echo "🍎 启动macOS MuJoCo轨迹可视化演示"
echo "=================================="

# 检查MjPython是否可用
if command -v mjpython &> /dev/null; then
    echo "✅ 找到MjPython"
    mjpython /Users/hugh/Documents/adam_control/examples/mujoco_macos_trajectory_demo.py
else
    echo "❌ 未找到MjPython"
    echo "请安装MuJoCo并确保mjpython在PATH中"
    echo ""
    echo "安装方法:"
    echo "1. 下载MuJoCo: https://mujoco.org/"
    echo "2. 安装Python绑定: pip install mujoco"
    echo "3. 或使用conda: conda install -c conda-forge mujoco"
fi
