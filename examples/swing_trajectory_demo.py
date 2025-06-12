#!/usr/bin/env python3
"""
摆动足轨迹规划演示脚本

演示和比较不同的轨迹规划方法：
1. 三次多项式插值
2. 贝塞尔曲线
3. 正弦函数
4. 摆线轨迹

并展示关键参数的影响
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from gait_core.swing_trajectory import (
    create_swing_trajectory_planner, 
    TrajectoryParameterOptimizer,
    SwingTrajectoryConfig,
    TrajectoryType
)


def demo_trajectory_comparison():
    """演示不同轨迹类型的比较"""
    print("=" * 60)
    print("摆动足轨迹规划方法比较演示")
    print("=" * 60)
    
    # 设置基本参数
    start_pos = np.array([0.0, 0.1, 0.0])  # 起始位置
    end_pos = np.array([0.15, 0.1, 0.0])   # 结束位置（步长15cm）
    step_height = 0.08                      # 抬脚高度8cm
    step_duration = 0.4                     # 摆动周期0.4s
    
    trajectory_types = ["polynomial", "bezier", "sinusoidal", "cycloid"]
    colors = ['red', 'blue', 'green', 'orange']
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('摆动足轨迹规划方法比较', fontsize=16)
    
    all_trajectories = {}
    
    for i, traj_type in enumerate(trajectory_types):
        print(f"\n{i+1}. {traj_type.upper()} 轨迹方法:")
        print("-" * 40)
        
        # 创建轨迹规划器
        planner = create_swing_trajectory_planner(
            trajectory_type=traj_type,
            step_height=step_height,
            step_duration=step_duration
        )
        planner.set_trajectory_parameters(start_pos, end_pos)
        
        # 生成轨迹采样点
        trajectory_data = planner.generate_trajectory_samples(num_samples=100)
        all_trajectories[traj_type] = trajectory_data
        
        positions = trajectory_data['positions']
        velocities = trajectory_data['velocities']
        
        # 分析轨迹特性
        max_height = np.max(positions[:, 2])
        max_horizontal_vel = np.max(np.linalg.norm(velocities[:, :2], axis=1))
        max_vertical_vel = np.max(np.abs(velocities[:, 2]))
        
        print(f"  最大高度: {max_height:.3f} m")
        print(f"  最大水平速度: {max_horizontal_vel:.3f} m/s")
        print(f"  最大垂直速度: {max_vertical_vel:.3f} m/s")
        
        # 绘制关键点信息
        print("  关键点位置:")
        key_phases = [0.0, 0.25, 0.5, 0.75, 1.0]
        for phase in key_phases:
            pos = planner.compute_trajectory_position(phase)
            print(f"    相位 {phase:.2f}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    # 绘制比较图
    plot_trajectory_comparison(all_trajectories, start_pos, end_pos, axes)
    
    plt.tight_layout()
    plt.show()
    
    return all_trajectories


def plot_trajectory_comparison(trajectories, start_pos, end_pos, axes):
    """绘制轨迹比较图"""
    colors = {'polynomial': 'red', 'bezier': 'blue', 
              'sinusoidal': 'green', 'cycloid': 'orange'}
    
    # 1. XZ平面轨迹图（侧视图）
    ax1 = axes[0, 0]
    ax1.set_title('XZ平面轨迹对比（侧视图）')
    ax1.set_xlabel('X位置 [m]')
    ax1.set_ylabel('Z位置 [m]')
    ax1.grid(True, alpha=0.3)
    
    for traj_type, data in trajectories.items():
        positions = data['positions']
        ax1.plot(positions[:, 0], positions[:, 2], 
                color=colors[traj_type], linewidth=2, label=traj_type)
    
    # 标记起点和终点
    ax1.plot(start_pos[0], start_pos[2], 'ko', markersize=8, label='起点')
    ax1.plot(end_pos[0], end_pos[2], 'ks', markersize=8, label='终点')
    ax1.legend()
    
    # 2. 垂直位置随时间变化
    ax2 = axes[0, 1]
    ax2.set_title('垂直位置随相位变化')
    ax2.set_xlabel('相位')
    ax2.set_ylabel('Z位置 [m]')
    ax2.grid(True, alpha=0.3)
    
    for traj_type, data in trajectories.items():
        phases = np.array(data['phases'])
        positions = data['positions']
        ax2.plot(phases, positions[:, 2], 
                color=colors[traj_type], linewidth=2, label=traj_type)
    
    ax2.legend()
    
    # 3. 水平速度随时间变化
    ax3 = axes[1, 0]
    ax3.set_title('水平速度随相位变化')
    ax3.set_xlabel('相位')
    ax3.set_ylabel('水平速度 [m/s]')
    ax3.grid(True, alpha=0.3)
    
    for traj_type, data in trajectories.items():
        phases = np.array(data['phases'])
        velocities = data['velocities']
        horizontal_speeds = np.linalg.norm(velocities[:, :2], axis=1)
        ax3.plot(phases, horizontal_speeds, 
                color=colors[traj_type], linewidth=2, label=traj_type)
    
    ax3.legend()
    
    # 4. 垂直速度随时间变化
    ax4 = axes[1, 1]
    ax4.set_title('垂直速度随相位变化')
    ax4.set_xlabel('相位')
    ax4.set_ylabel('垂直速度 [m/s]')
    ax4.grid(True, alpha=0.3)
    
    for traj_type, data in trajectories.items():
        phases = np.array(data['phases'])
        velocities = data['velocities']
        ax4.plot(phases, velocities[:, 2], 
                color=colors[traj_type], linewidth=2, label=traj_type)
    
    ax4.legend()


def demo_parameter_effects():
    """演示参数对轨迹的影响"""
    print("\n" + "=" * 60)
    print("轨迹参数影响演示")
    print("=" * 60)
    
    # 基础参数
    start_pos = np.array([0.0, 0.1, 0.0])
    end_pos = np.array([0.15, 0.1, 0.0])
    
    # 测试不同的抬脚高度
    print("\n1. 抬脚高度影响 (使用贝塞尔曲线):")
    print("-" * 30)
    
    step_heights = [0.05, 0.08, 0.12]  # 5cm, 8cm, 12cm
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('抬脚高度对轨迹的影响', fontsize=14)
    
    for i, height in enumerate(step_heights):
        planner = create_swing_trajectory_planner("bezier", height, 0.4)
        planner.set_trajectory_parameters(start_pos, end_pos)
        
        trajectory_data = planner.generate_trajectory_samples(50)
        positions = trajectory_data['positions']
        
        # 分析最大高度和位置
        max_height = np.max(positions[:, 2])
        max_height_idx = np.argmax(positions[:, 2])
        max_height_x = positions[max_height_idx, 0]
        
        print(f"  高度 {height*100:.0f}cm: 最大高度 {max_height:.3f}m，位置 x={max_height_x:.3f}m")
        
        # 绘制轨迹
        axes[0].plot(positions[:, 0], positions[:, 2], 
                    linewidth=2, label=f'{height*100:.0f}cm')
        
        # 绘制垂直位置
        phases = np.array(trajectory_data['phases'])
        axes[1].plot(phases, positions[:, 2], 
                    linewidth=2, label=f'{height*100:.0f}cm')
    
    axes[0].set_title('XZ平面轨迹')
    axes[0].set_xlabel('X位置 [m]')
    axes[0].set_ylabel('Z位置 [m]')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_title('垂直位置随相位变化')
    axes[1].set_xlabel('相位')
    axes[1].set_ylabel('Z位置 [m]')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def demo_trajectory_optimization():
    """演示轨迹参数优化"""
    print("\n" + "=" * 60)
    print("轨迹参数优化演示")
    print("=" * 60)
    
    optimizer = TrajectoryParameterOptimizer()
    
    # 测试不同地形和速度下的优化
    test_cases = [
        {"terrain": 0.0, "speed": 0.5, "name": "平地慢速"},
        {"terrain": 0.3, "speed": 1.0, "name": "轻微起伏中速"},
        {"terrain": 0.7, "speed": 0.8, "name": "复杂地形中速"},
        {"terrain": 1.0, "speed": 0.3, "name": "困难地形慢速"}
    ]
    
    print("\n抬脚高度优化结果:")
    print("-" * 40)
    print(f"{'场景':<15} {'地形难度':<8} {'速度[m/s]':<10} {'优化高度[cm]':<12}")
    print("-" * 40)
    
    for case in test_cases:
        optimized_height = optimizer.optimize_step_height(
            case["terrain"], case["speed"]
        )
        print(f"{case['name']:<15} {case['terrain']:<8.1f} {case['speed']:<10.1f} {optimized_height*100:<12.1f}")
    
    # 测试步态频率优化
    print("\n\n轨迹时序优化结果:")
    print("-" * 40)
    
    gait_frequencies = [0.5, 1.0, 1.5, 2.0]  # Hz
    
    print(f"{'频率[Hz]':<10} {'摆动周期[s]':<12} {'最高点相位':<12} {'触地相位':<12}")
    print("-" * 40)
    
    for freq in gait_frequencies:
        timing_params = optimizer.optimize_trajectory_timing(freq)
        print(f"{freq:<10.1f} {timing_params['step_duration']:<12.3f} "
              f"{timing_params['max_height_ratio']:<12.2f} {timing_params['touch_down_ratio']:<12.2f}")


def main():
    """主函数"""
    print("摆动足轨迹规划演示程序")
    print("实现以下轨迹规划方法的参数化描述：")
    print("1. 三次多项式插值")
    print("2. 贝塞尔曲线")
    print("3. 正弦函数")
    print("4. 摆线轨迹")
    print()
    
    try:
        # 轨迹方法比较
        trajectories = demo_trajectory_comparison()
        
        # 参数影响演示
        demo_parameter_effects()
        
        # 参数优化演示
        demo_trajectory_optimization()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("\n总结:")
        print("1. 三次多项式插值: 平滑，计算简单，适合基础应用")
        print("2. 贝塞尔曲线: 可控性好，形状优美，适合精确控制")
        print("3. 正弦函数: 自然，计算快速，适合实时应用")
        print("4. 摆线轨迹: 独特形状，适合特殊需求")
        print("\n关键参数:")
        print("- 抬脚高度 h: 建议 5-10cm，可根据地形调整")
        print("- 最大高度相位: 建议在摆动中点（0.5）或稍早（0.45）")
        print("- 轨迹函数类型: 根据性能需求和控制精度要求选择")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装 matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")


if __name__ == "__main__":
    main() 