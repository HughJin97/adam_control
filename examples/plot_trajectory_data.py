#!/usr/bin/env python3
"""
轨迹数据可视化脚本
读取保存的轨迹数据并生成详细的图表分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from typing import Dict, List

def load_trajectory_data(filename: str) -> Dict:
    """加载轨迹数据"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # 转换列表回numpy数组
        for key in ['left_foot_pos', 'right_foot_pos', 'left_foot_target', 'right_foot_target']:
            if key in data:
                data[key] = [np.array(pos) for pos in data[key]]
        
        return data
    except Exception as e:
        print(f"❌ 加载轨迹数据失败: {e}")
        return None

def plot_3d_trajectory(data: Dict, save_path: str = None):
    """绘制3D轨迹图"""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    
    left_positions = np.array(data['left_foot_pos'])
    right_positions = np.array(data['right_foot_pos'])
    left_targets = np.array(data['left_foot_target'])
    right_targets = np.array(data['right_foot_target'])
    
    # 绘制轨迹
    ax1.plot(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2], 
             'g-', linewidth=2, label='左足轨迹', alpha=0.8)
    ax1.plot(right_positions[:, 0], right_positions[:, 1], right_positions[:, 2], 
             'r-', linewidth=2, label='右足轨迹', alpha=0.8)
    
    # 绘制目标点
    ax1.scatter(left_targets[:, 0], left_targets[:, 1], left_targets[:, 2], 
                c='yellow', s=50, marker='o', label='左足目标', alpha=0.7)
    ax1.scatter(right_targets[:, 0], right_targets[:, 1], right_targets[:, 2], 
                c='orange', s=50, marker='s', label='右足目标', alpha=0.7)
    
    # 标记起始和结束点
    ax1.scatter(left_positions[0, 0], left_positions[0, 1], left_positions[0, 2], 
                c='darkgreen', s=100, marker='^', label='左足起点')
    ax1.scatter(right_positions[0, 0], right_positions[0, 1], right_positions[0, 2], 
                c='darkred', s=100, marker='^', label='右足起点')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D足端轨迹')
    ax1.legend()
    ax1.grid(True)
    
    # XY平面投影
    ax2 = fig.add_subplot(222)
    ax2.plot(left_positions[:, 0], left_positions[:, 1], 'g-', linewidth=2, label='左足轨迹')
    ax2.plot(right_positions[:, 0], right_positions[:, 1], 'r-', linewidth=2, label='右足轨迹')
    ax2.scatter(left_targets[:, 0], left_targets[:, 1], c='yellow', s=30, alpha=0.7, label='左足目标')
    ax2.scatter(right_targets[:, 0], right_targets[:, 1], c='orange', s=30, alpha=0.7, label='右足目标')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY平面投影')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 高度变化
    ax3 = fig.add_subplot(223)
    time_data = data['time']
    ax3.plot(time_data, left_positions[:, 2], 'g-', linewidth=2, label='左足高度')
    ax3.plot(time_data, right_positions[:, 2], 'r-', linewidth=2, label='右足高度')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('高度 (m)')
    ax3.set_title('足端高度变化')
    ax3.legend()
    ax3.grid(True)
    
    # 相位变化
    ax4 = fig.add_subplot(224)
    ax4.plot(time_data, data['left_phase'], 'g-', linewidth=2, label='左足相位')
    ax4.plot(time_data, data['right_phase'], 'r-', linewidth=2, label='右足相位')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('相位')
    ax4.set_title('轨迹相位变化')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 3D轨迹图已保存到: {save_path}")
    
    plt.show()

def plot_velocity_analysis(data: Dict, save_path: str = None):
    """绘制速度分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    left_positions = np.array(data['left_foot_pos'])
    right_positions = np.array(data['right_foot_pos'])
    time_data = np.array(data['time'])
    
    # 计算速度（数值微分）
    dt = np.diff(time_data)
    left_velocities = np.diff(left_positions, axis=0) / dt[:, np.newaxis]
    right_velocities = np.diff(right_positions, axis=0) / dt[:, np.newaxis]
    time_vel = time_data[1:]  # 速度时间轴
    
    # X方向速度
    axes[0, 0].plot(time_vel, left_velocities[:, 0], 'g-', linewidth=2, label='左足X速度')
    axes[0, 0].plot(time_vel, right_velocities[:, 0], 'r-', linewidth=2, label='右足X速度')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('X速度 (m/s)')
    axes[0, 0].set_title('X方向速度')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Y方向速度
    axes[0, 1].plot(time_vel, left_velocities[:, 1], 'g-', linewidth=2, label='左足Y速度')
    axes[0, 1].plot(time_vel, right_velocities[:, 1], 'r-', linewidth=2, label='右足Y速度')
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('Y速度 (m/s)')
    axes[0, 1].set_title('Y方向速度')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Z方向速度
    axes[1, 0].plot(time_vel, left_velocities[:, 2], 'g-', linewidth=2, label='左足Z速度')
    axes[1, 0].plot(time_vel, right_velocities[:, 2], 'r-', linewidth=2, label='右足Z速度')
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('Z速度 (m/s)')
    axes[1, 0].set_title('Z方向速度')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 速度大小
    left_speed = np.linalg.norm(left_velocities, axis=1)
    right_speed = np.linalg.norm(right_velocities, axis=1)
    axes[1, 1].plot(time_vel, left_speed, 'g-', linewidth=2, label='左足速度大小')
    axes[1, 1].plot(time_vel, right_speed, 'r-', linewidth=2, label='右足速度大小')
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('速度大小 (m/s)')
    axes[1, 1].set_title('速度大小')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 速度分析图已保存到: {save_path}")
    
    plt.show()

def plot_phase_analysis(data: Dict, save_path: str = None):
    """绘制相位分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    time_data = data['time']
    left_phases = data['left_phase']
    right_phases = data['right_phase']
    left_states = data['left_state']
    right_states = data['right_state']
    
    # 相位变化
    axes[0, 0].plot(time_data, left_phases, 'g-', linewidth=2, label='左足相位')
    axes[0, 0].plot(time_data, right_phases, 'r-', linewidth=2, label='右足相位')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('相位')
    axes[0, 0].set_title('轨迹相位变化')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 相位差
    phase_diff = np.array(left_phases) - np.array(right_phases)
    axes[0, 1].plot(time_data, phase_diff, 'b-', linewidth=2, label='相位差 (左-右)')
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('相位差')
    axes[0, 1].set_title('左右足相位差')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 状态时间线
    state_colors = {'IDLE': 'gray', 'ACTIVE': 'blue', 'COMPLETED': 'green', 
                   'INTERRUPTED': 'orange', 'EMERGENCY_STOP': 'red'}
    
    # 左足状态
    for i, state in enumerate(left_states):
        color = state_colors.get(state, 'black')
        axes[1, 0].scatter(time_data[i], 0, c=color, s=20, alpha=0.7)
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('左足状态')
    axes[1, 0].set_title('左足状态时间线')
    axes[1, 0].set_ylim(-0.5, 0.5)
    
    # 右足状态
    for i, state in enumerate(right_states):
        color = state_colors.get(state, 'black')
        axes[1, 1].scatter(time_data[i], 0, c=color, s=20, alpha=0.7)
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('右足状态')
    axes[1, 1].set_title('右足状态时间线')
    axes[1, 1].set_ylim(-0.5, 0.5)
    
    # 添加状态图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=8, label=state)
                      for state, color in state_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 相位分析图已保存到: {save_path}")
    
    plt.show()

def print_trajectory_statistics(data: Dict):
    """打印详细的轨迹统计信息"""
    print("\n📊 详细轨迹统计分析")
    print("=" * 60)
    
    left_positions = np.array(data['left_foot_pos'])
    right_positions = np.array(data['right_foot_pos'])
    time_data = np.array(data['time'])
    
    # 基本统计
    total_time = time_data[-1] - time_data[0]
    data_points = len(time_data)
    avg_dt = total_time / (data_points - 1)
    
    print(f"仿真基本信息:")
    print(f"  - 总时间: {total_time:.2f}s")
    print(f"  - 数据点数: {data_points}")
    print(f"  - 平均时间步长: {avg_dt:.4f}s")
    print(f"  - 采样频率: {1/avg_dt:.1f}Hz")
    
    # 轨迹距离统计
    left_distances = np.linalg.norm(np.diff(left_positions, axis=0), axis=1)
    right_distances = np.linalg.norm(np.diff(right_positions, axis=0), axis=1)
    
    print(f"\n轨迹距离统计:")
    print(f"  左足:")
    print(f"    - 总移动距离: {np.sum(left_distances):.3f}m")
    print(f"    - 平均步长: {np.mean(left_distances):.4f}m")
    print(f"    - 最大单步距离: {np.max(left_distances):.4f}m")
    print(f"  右足:")
    print(f"    - 总移动距离: {np.sum(right_distances):.3f}m")
    print(f"    - 平均步长: {np.mean(right_distances):.4f}m")
    print(f"    - 最大单步距离: {np.max(right_distances):.4f}m")
    
    # 高度统计
    print(f"\n高度统计:")
    print(f"  左足:")
    print(f"    - 最小高度: {np.min(left_positions[:, 2]):.3f}m")
    print(f"    - 最大高度: {np.max(left_positions[:, 2]):.3f}m")
    print(f"    - 平均高度: {np.mean(left_positions[:, 2]):.3f}m")
    print(f"    - 高度变化范围: {np.max(left_positions[:, 2]) - np.min(left_positions[:, 2]):.3f}m")
    print(f"  右足:")
    print(f"    - 最小高度: {np.min(right_positions[:, 2]):.3f}m")
    print(f"    - 最大高度: {np.max(right_positions[:, 2]):.3f}m")
    print(f"    - 平均高度: {np.mean(right_positions[:, 2]):.3f}m")
    print(f"    - 高度变化范围: {np.max(right_positions[:, 2]) - np.min(right_positions[:, 2]):.3f}m")
    
    # 相位统计
    left_phases = data['left_phase']
    right_phases = data['right_phase']
    
    print(f"\n相位统计:")
    print(f"  左足:")
    print(f"    - 最大相位: {max(left_phases):.3f}")
    print(f"    - 平均相位: {np.mean(left_phases):.3f}")
    print(f"    - 相位完成次数: {left_phases.count(1.0)}")
    print(f"  右足:")
    print(f"    - 最大相位: {max(right_phases):.3f}")
    print(f"    - 平均相位: {np.mean(right_phases):.3f}")
    print(f"    - 相位完成次数: {right_phases.count(1.0)}")
    
    # 状态统计
    left_states = data['left_state']
    right_states = data['right_state']
    
    print(f"\n状态统计:")
    print(f"  左足状态分布:")
    for state in set(left_states):
        count = left_states.count(state)
        percentage = (count / len(left_states)) * 100
        print(f"    - {state}: {count}次 ({percentage:.1f}%)")
    
    print(f"  右足状态分布:")
    for state in set(right_states):
        count = right_states.count(state)
        percentage = (count / len(right_states)) * 100
        print(f"    - {state}: {count}次 ({percentage:.1f}%)")

def main():
    """主函数"""
    print("📊 AzureLoong轨迹数据可视化工具")
    print("=" * 50)
    
    # 查找最新的轨迹数据文件
    trajectory_files = [f for f in os.listdir('.') if f.startswith('trajectory_data_') and f.endswith('.json')]
    
    if not trajectory_files:
        print("❌ 未找到轨迹数据文件")
        print("请先运行 python run_trajectory_visualization.py 生成轨迹数据")
        return
    
    # 使用最新的文件
    latest_file = max(trajectory_files, key=lambda f: os.path.getctime(f))
    print(f"📂 加载轨迹数据文件: {latest_file}")
    
    # 加载数据
    data = load_trajectory_data(latest_file)
    if data is None:
        return
    
    print(f"✓ 成功加载 {len(data['time'])} 个数据点")
    
    # 打印统计信息
    print_trajectory_statistics(data)
    
    # 生成图表
    print("\n🎨 生成可视化图表...")
    
    base_name = latest_file.replace('.json', '')
    
    try:
        # 3D轨迹图
        plot_3d_trajectory(data, f"{base_name}_3d_trajectory.png")
        
        # 速度分析图
        plot_velocity_analysis(data, f"{base_name}_velocity_analysis.png")
        
        # 相位分析图
        plot_phase_analysis(data, f"{base_name}_phase_analysis.png")
        
        print("\n✅ 所有图表生成完成！")
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 