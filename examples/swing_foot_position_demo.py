#!/usr/bin/env python3
"""
摆动足位置函数演示脚本

演示 get_swing_foot_position 函数的使用方法和效果，
包括不同的水平插值方法和垂直轨迹类型。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from gait_core.swing_trajectory import (
    get_swing_foot_position,
    get_swing_foot_velocity,
    create_swing_foot_trajectory
)


def demo_basic_usage():
    """演示基本使用方法"""
    print("=" * 60)
    print("get_swing_foot_position 基本使用演示")
    print("=" * 60)
    
    # 设置起始和目标位置
    start_pos = np.array([0.0, 0.1, 0.0])   # 起始位置 (x=0, y=0.1, z=0)
    target_pos = np.array([0.15, 0.1, 0.0]) # 目标位置 (x=0.15, y=0.1, z=0)，步长15cm
    step_height = 0.08                       # 抬脚高度8cm
    
    print(f"起始位置: {start_pos}")
    print(f"目标位置: {target_pos}")
    print(f"抬脚高度: {step_height}m ({step_height*100:.0f}cm)")
    print()
    
    # 测试关键相位点
    key_phases = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("使用线性插值 + 正弦垂直轨迹:")
    print("-" * 40)
    print(f"{'相位':<8} {'X位置[m]':<10} {'Y位置[m]':<10} {'Z位置[m]':<10} {'高度[cm]':<10}")
    print("-" * 40)
    
    for phase in key_phases:
        pos = get_swing_foot_position(phase, start_pos, target_pos, step_height, 
                                    "linear", "sine")
        print(f"{phase:<8.2f} {pos[0]:<10.3f} {pos[1]:<10.3f} {pos[2]:<10.3f} {pos[2]*100:<10.1f}")
    
    print("\n验证关键点:")
    print(f"✓ 起点 (phase=0.0): 位置应为起始位置 {start_pos}")
    print(f"✓ 终点 (phase=1.0): 位置应为目标位置 {target_pos}")
    print(f"✓ 中点 (phase=0.5): Z高度应为最大值 {step_height}m")


def demo_interpolation_types():
    """演示不同的水平插值方法"""
    print("\n" + "=" * 60)
    print("水平插值方法对比")
    print("=" * 60)
    
    start_pos = np.array([0.0, 0.1, 0.0])
    target_pos = np.array([0.20, 0.05, 0.0])  # 包含Y方向变化
    step_height = 0.06
    
    interpolation_types = ["linear", "cubic", "smooth"]
    phases = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    
    print(f"起始位置: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    print(f"目标位置: ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
    print()
    
    # 创建图形对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('不同水平插值方法对比', fontsize=14)
    
    for i, interp_type in enumerate(interpolation_types):
        print(f"{interp_type.upper()} 插值:")
        print("-" * 20)
        
        positions = []
        for phase in phases:
            pos = get_swing_foot_position(phase, start_pos, target_pos, step_height,
                                        interp_type, "sine")
            positions.append(pos)
            if phase in [0.0, 0.5, 1.0]:  # 只打印关键点
                print(f"  相位 {phase:.1f}: X={pos[0]:.3f}, Y={pos[1]:.3f}")
        
        positions = np.array(positions)
        
        # 绘制XY轨迹
        axes[i].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='轨迹')
        axes[i].plot(start_pos[0], start_pos[1], 'go', markersize=8, label='起点')
        axes[i].plot(target_pos[0], target_pos[1], 'ro', markersize=8, label='终点')
        axes[i].set_title(f'{interp_type.upper()} 插值')
        axes[i].set_xlabel('X位置 [m]')
        axes[i].set_ylabel('Y位置 [m]')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        axes[i].axis('equal')
        
        print()
    
    plt.tight_layout()
    plt.show()


def demo_vertical_trajectories():
    """演示不同的垂直轨迹类型"""
    print("\n" + "=" * 60)
    print("垂直轨迹类型对比")
    print("=" * 60)
    
    start_pos = np.array([0.0, 0.1, 0.0])
    target_pos = np.array([0.15, 0.1, 0.0])
    step_height = 0.08
    
    vertical_types = ["sine", "parabola", "smooth_parabola"]
    phases = np.linspace(0.0, 1.0, 101)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('不同垂直轨迹类型对比', fontsize=14)
    
    # 轨迹高度对比
    ax1 = axes[0, 0]
    ax1.set_title('垂直高度随相位变化')
    ax1.set_xlabel('相位')
    ax1.set_ylabel('Z高度 [m]')
    ax1.grid(True, alpha=0.3)
    
    # XZ轨迹对比
    ax2 = axes[0, 1]
    ax2.set_title('XZ平面轨迹对比')
    ax2.set_xlabel('X位置 [m]')
    ax2.set_ylabel('Z高度 [m]')
    ax2.grid(True, alpha=0.3)
    
    # 速度对比
    ax3 = axes[1, 0]
    ax3.set_title('垂直速度随相位变化')
    ax3.set_xlabel('相位')
    ax3.set_ylabel('垂直速度 [m/s]')
    ax3.grid(True, alpha=0.3)
    
    # 数值对比表
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    colors = ['blue', 'red', 'green']
    table_data = []
    
    for i, vert_type in enumerate(vertical_types):
        print(f"{vert_type.upper()} 垂直轨迹:")
        print("-" * 25)
        
        heights = []
        velocities = []
        positions_xz = []
        
        for phase in phases:
            pos = get_swing_foot_position(phase, start_pos, target_pos, step_height,
                                        "linear", vert_type)
            vel = get_swing_foot_velocity(phase, start_pos, target_pos, step_height, 0.4,
                                        "linear", vert_type)
            heights.append(pos[2])
            velocities.append(vel[2])
            positions_xz.append([pos[0], pos[2]])
        
        positions_xz = np.array(positions_xz)
        
        # 绘制图形
        ax1.plot(phases, heights, color=colors[i], linewidth=2, label=vert_type)
        ax2.plot(positions_xz[:, 0], positions_xz[:, 1], color=colors[i], 
                linewidth=2, label=vert_type)
        ax3.plot(phases, velocities, color=colors[i], linewidth=2, label=vert_type)
        
        # 分析关键指标
        max_height = max(heights)
        max_height_phase = phases[np.argmax(heights)]
        max_vel = max(np.abs(velocities))
        
        print(f"  最大高度: {max_height:.3f}m (相位 {max_height_phase:.2f})")
        print(f"  最大速度: {max_vel:.2f}m/s")
        
        # 为表格准备数据
        table_data.append([vert_type, f"{max_height:.3f}m", f"{max_height_phase:.2f}", f"{max_vel:.2f}m/s"])
        print()
    
    # 添加图例
    ax1.legend()
    ax2.legend()
    ax3.legend()
    
    # 创建对比表格
    table_headers = ['轨迹类型', '最大高度', '最大高度相位', '最大速度']
    table = ax4.table(cellText=table_data, colLabels=table_headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('性能对比表', pad=20)
    
    plt.tight_layout()
    plt.show()


def demo_complete_trajectory():
    """演示完整轨迹生成"""
    print("\n" + "=" * 60)
    print("完整轨迹生成演示")
    print("=" * 60)
    
    # 设置多个测试场景
    scenarios = [
        {
            "name": "标准前进步",
            "start": np.array([0.0, 0.1, 0.0]),
            "target": np.array([0.15, 0.1, 0.0]),
            "height": 0.08
        },
        {
            "name": "侧向步",
            "start": np.array([0.0, 0.0, 0.0]),
            "target": np.array([0.05, 0.15, 0.0]),
            "height": 0.06
        },
        {
            "name": "对角步",
            "start": np.array([0.0, 0.0, 0.0]),
            "target": np.array([0.12, 0.08, 0.0]),
            "height": 0.10
        }
    ]
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, scenario in enumerate(scenarios):
        print(f"{scenario['name']}:")
        print(f"  起始: {scenario['start']}")
        print(f"  目标: {scenario['target']}")
        print(f"  高度: {scenario['height']}m")
        
        # 生成轨迹
        trajectory = create_swing_foot_trajectory(
            scenario['start'], scenario['target'], scenario['height'],
            num_points=50, interpolation_type="cubic", vertical_trajectory_type="sine"
        )
        
        positions = trajectory['positions']
        
        # 3D轨迹图
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax.scatter(scenario['start'][0], scenario['start'][1], scenario['start'][2], 
                  color='green', s=100, label='起点')
        ax.scatter(scenario['target'][0], scenario['target'][1], scenario['target'][2], 
                  color='red', s=100, label='终点')
        
        ax.set_title(scenario['name'])
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend()
        
        # 分析统计
        max_height = np.max(positions[:, 2])
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        
        print(f"  实际最大高度: {max_height:.3f}m")
        print(f"  轨迹总长度: {total_distance:.3f}m")
        print()
    
    plt.tight_layout()
    plt.show()


def demo_real_time_usage():
    """演示实时使用场景"""
    print("\n" + "=" * 60)
    print("实时使用场景演示")
    print("=" * 60)
    
    # 模拟实时步态控制
    start_pos = np.array([0.0, 0.1, 0.0])
    target_pos = np.array([0.15, 0.1, 0.0])
    step_height = 0.08
    step_duration = 0.4  # 摆动周期400ms
    
    # 模拟控制频率
    control_frequency = 100  # 100Hz
    dt = 1.0 / control_frequency
    num_steps = int(step_duration / dt)
    
    print(f"模拟参数:")
    print(f"  控制频率: {control_frequency}Hz")
    print(f"  摆动周期: {step_duration}s")
    print(f"  控制步数: {num_steps}")
    print()
    
    print("实时轨迹生成 (每20ms显示一次):")
    print("-" * 50)
    print(f"{'时间[ms]':<10} {'相位':<8} {'X[m]':<8} {'Y[m]':<8} {'Z[m]':<8} {'速度[m/s]':<12}")
    print("-" * 50)
    
    positions = []
    velocities = []
    
    for step in range(0, num_steps + 1, 2):  # 每20ms (2个控制周期) 显示一次
        current_time = step * dt
        phase = current_time / step_duration
        
        if phase > 1.0:
            phase = 1.0
        
        # 计算位置和速度
        pos = get_swing_foot_position(phase, start_pos, target_pos, step_height,
                                    "cubic", "sine")
        vel = get_swing_foot_velocity(phase, start_pos, target_pos, step_height, 
                                    step_duration, "cubic", "sine")
        
        positions.append(pos)
        velocities.append(vel)
        
        vel_magnitude = np.linalg.norm(vel)
        
        print(f"{current_time*1000:<10.0f} {phase:<8.3f} {pos[0]:<8.3f} {pos[1]:<8.3f} "
              f"{pos[2]:<8.3f} {vel_magnitude:<12.3f}")
    
    # 性能分析
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    print("\n性能分析:")
    print(f"  平均水平速度: {np.mean(np.linalg.norm(velocities[:, :2], axis=1)):.3f} m/s")
    print(f"  最大垂直速度: {np.max(np.abs(velocities[:, 2])):.3f} m/s")
    print(f"  轨迹平滑度: {'优秀' if np.max(np.diff(positions[:, 2])) < 0.01 else '良好'}")


def main():
    """主函数"""
    print("摆动足位置函数 get_swing_foot_position 演示程序")
    print("实现水平插值和垂直轨迹的参数化描述")
    print()
    
    try:
        # 基本使用演示
        demo_basic_usage()
        
        # 插值方法对比
        demo_interpolation_types()
        
        # 垂直轨迹对比
        demo_vertical_trajectories()
        
        # 完整轨迹生成
        demo_complete_trajectory()
        
        # 实时使用场景
        demo_real_time_usage()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("\n总结:")
        print("✓ get_swing_foot_position 函数按要求实现了：")
        print("  - 水平插值：线性、三次曲线、平滑插值")
        print("  - 垂直轨迹：正弦函数、抛物线、平滑抛物线")
        print("  - 参数化相位：0.0(起点) → 0.5(最高点) → 1.0(终点)")
        print("  - 组合结果：返回3D位置坐标 [x, y, z]")
        print("\n推荐配置:")
        print("  - 基础应用：linear + sine")
        print("  - 平滑控制：cubic + sine") 
        print("  - 特殊需求：smooth + parabola")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装 matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")


if __name__ == "__main__":
    main() 