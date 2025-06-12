#!/usr/bin/env python3
"""
LIPM模型演示脚本
演示如何使用简化动力学模型进行质心动态预测和控制

作者: Adam Control Team
版本: 1.0
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from gait_core.simplified_dynamics import SimplifiedDynamicsModel, ControlMode
from gait_core.data_bus import DataBus, Vector3D, get_data_bus


def demo_basic_lipm_dynamics():
    """演示基本的LIPM动力学"""
    print("=== LIPM基本动力学演示 ===")
    
    # 创建数据总线
    data_bus = get_data_bus()
    
    # 设置初始质心状态
    initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
    initial_com_vel = Vector3D(x=0.1, y=0.0, z=0.0)  # 初始前向速度0.1 m/s
    
    data_bus.set_center_of_mass_position(initial_com_pos)
    data_bus.set_center_of_mass_velocity(initial_com_vel)
    
    # 创建LIPM模型（不使用URDF，使用默认参数）
    lipm_model = SimplifiedDynamicsModel()
    
    # 设置支撑足位置
    support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
    lipm_model.set_support_foot_position(support_foot)
    
    # 打印模型状态
    lipm_model.print_status()
    
    # 预测质心轨迹
    time_horizon = 2.0  # 预测2秒
    dt = 0.01          # 10ms时间步长
    
    trajectory = lipm_model.predict_com_trajectory(time_horizon, dt)
    
    # 提取轨迹数据用于绘图
    times = [i * dt for i in range(len(trajectory))]
    x_positions = [state.com_position.x for state in trajectory]
    y_positions = [state.com_position.y for state in trajectory]
    x_velocities = [state.com_velocity.x for state in trajectory]
    y_velocities = [state.com_velocity.y for state in trajectory]
    
    # 绘制结果
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 质心X位置
    ax1.plot(times, x_positions, 'b-', linewidth=2, label='X位置')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('位置 (m)')
    ax1.set_title('质心X方向位置')
    ax1.grid(True)
    ax1.legend()
    
    # 质心Y位置
    ax2.plot(times, y_positions, 'r-', linewidth=2, label='Y位置')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('位置 (m)')
    ax2.set_title('质心Y方向位置')
    ax2.grid(True)
    ax2.legend()
    
    # 质心X速度
    ax3.plot(times, x_velocities, 'b--', linewidth=2, label='X速度')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('速度 (m/s)')
    ax3.set_title('质心X方向速度')
    ax3.grid(True)
    ax3.legend()
    
    # 质心Y速度
    ax4.plot(times, y_velocities, 'r--', linewidth=2, label='Y速度')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('速度 (m/s)')
    ax4.set_title('质心Y方向速度')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('lipm_basic_dynamics.png', dpi=300, bbox_inches='tight')
    print("基本动力学图像已保存为 lipm_basic_dynamics.png")
    
    return lipm_model, trajectory


def demo_force_control():
    """演示力控制模式"""
    print("\n=== LIPM力控制模式演示 ===")
    
    # 创建数据总线和模型
    data_bus = get_data_bus()
    
    # 设置初始状态（质心偏离支撑足）
    initial_com_pos = Vector3D(x=0.1, y=0.05, z=0.8)  # 质心偏离中心
    initial_com_vel = Vector3D(x=0.0, y=0.0, z=0.0)   # 初始静止
    
    data_bus.set_center_of_mass_position(initial_com_pos)
    data_bus.set_center_of_mass_velocity(initial_com_vel)
    
    lipm_model = SimplifiedDynamicsModel()
    
    # 设置支撑足位置
    support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
    lipm_model.set_support_foot_position(support_foot)
    
    # 应用控制力来稳定质心（朝向支撑足的力）
    control_force = Vector3D(x=-20.0, y=-10.0, z=0.0)  # 向支撑足方向的力
    lipm_model.set_foot_force_command(control_force)
    
    print(f"初始质心位置: ({initial_com_pos.x:.3f}, {initial_com_pos.y:.3f}, {initial_com_pos.z:.3f})")
    print(f"支撑足位置: ({support_foot.x:.3f}, {support_foot.y:.3f}, {support_foot.z:.3f})")
    print(f"应用控制力: ({control_force.x:.1f}, {control_force.y:.1f}, {control_force.z:.1f}) N")
    
    # 预测有控制力的轨迹
    time_horizon = 3.0
    dt = 0.01
    
    controlled_trajectory = lipm_model.predict_com_trajectory(time_horizon, dt)
    
    # 比较：无控制力的情况
    lipm_model.set_foot_force_command(Vector3D(x=0.0, y=0.0, z=0.0))
    uncontrolled_trajectory = lipm_model.predict_com_trajectory(time_horizon, dt)
    
    # 绘制比较结果
    times = [i * dt for i in range(len(controlled_trajectory))]
    
    controlled_x = [state.com_position.x for state in controlled_trajectory]
    controlled_y = [state.com_position.y for state in controlled_trajectory]
    uncontrolled_x = [state.com_position.x for state in uncontrolled_trajectory]
    uncontrolled_y = [state.com_position.y for state in uncontrolled_trajectory]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # X方向比较
    ax1.plot(times, controlled_x, 'b-', linewidth=2, label='有控制力')
    ax1.plot(times, uncontrolled_x, 'r--', linewidth=2, label='无控制力')
    ax1.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='支撑足位置')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('X位置 (m)')
    ax1.set_title('质心X方向位置 - 力控制效果')
    ax1.grid(True)
    ax1.legend()
    
    # Y方向比较
    ax2.plot(times, controlled_y, 'b-', linewidth=2, label='有控制力')
    ax2.plot(times, uncontrolled_y, 'r--', linewidth=2, label='无控制力')
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='支撑足位置')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('Y位置 (m)')
    ax2.set_title('质心Y方向位置 - 力控制效果')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('lipm_force_control.png', dpi=300, bbox_inches='tight')
    print("力控制效果图像已保存为 lipm_force_control.png")


def demo_footstep_planning():
    """演示基于LIPM的落脚点规划"""
    print("\n=== LIPM落脚点规划演示 ===")
    
    # 创建数据总线和模型
    data_bus = get_data_bus()
    
    # 设置初始状态（机器人开始行走）
    initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
    initial_com_vel = Vector3D(x=0.2, y=0.0, z=0.0)  # 期望向前行走
    
    data_bus.set_center_of_mass_position(initial_com_pos)
    data_bus.set_center_of_mass_velocity(initial_com_vel)
    
    lipm_model = SimplifiedDynamicsModel()
    
    # 模拟步态序列
    step_duration = 0.6  # 每步0.6秒
    step_length = 0.15   # 步长15cm
    step_width = 0.1     # 步宽10cm
    
    time_horizon = 3.0
    dt = 0.01
    
    trajectory_states = []
    support_foot_positions = []
    times = []
    
    current_support_x = 0.0
    current_support_y = 0.0
    step_count = 0
    
    for t in np.arange(0, time_horizon, dt):
        # 确定当前支撑足位置（模拟步态切换）
        if t % step_duration < step_duration / 2:
            # 左脚支撑
            current_support_y = step_width / 2
        else:
            # 右脚支撑
            current_support_y = -step_width / 2
        
        # 向前移动支撑足
        if t > 0 and t % step_duration < dt:
            current_support_x += step_length
            step_count += 1
        
        # 设置支撑足位置
        support_foot = Vector3D(x=current_support_x, y=current_support_y, z=0.0)
        lipm_model.set_support_foot_position(support_foot)
        
        # 更新模型
        lipm_model.update(dt)
        
        # 记录状态
        trajectory_states.append(lipm_model.current_state)
        support_foot_positions.append((current_support_x, current_support_y))
        times.append(t)
    
    # 提取轨迹数据
    com_x = [state.com_position.x for state in trajectory_states]
    com_y = [state.com_position.y for state in trajectory_states]
    foot_x = [pos[0] for pos in support_foot_positions]
    foot_y = [pos[1] for pos in support_foot_positions]
    
    # 绘制步态轨迹
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 2D轨迹视图
    ax1.plot(com_x, com_y, 'b-', linewidth=2, label='质心轨迹')
    ax1.scatter(foot_x[::50], foot_y[::50], c='r', s=30, label='支撑足位置', marker='s')
    ax1.set_xlabel('X位置 (m)')
    ax1.set_ylabel('Y位置 (m)')
    ax1.set_title('机器人行走轨迹 - 俯视图')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    # 时间序列
    ax2.plot(times, com_x, 'b-', linewidth=2, label='质心X位置')
    ax2.plot(times, foot_x, 'r--', linewidth=1, label='支撑足X位置')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('X位置 (m)')
    ax2.set_title('行走过程中的X方向位置变化')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('lipm_walking_trajectory.png', dpi=300, bbox_inches='tight')
    print("行走轨迹图像已保存为 lipm_walking_trajectory.png")
    
    print(f"完成 {step_count} 步模拟")
    print(f"最终质心位置: ({com_x[-1]:.3f}, {com_y[-1]:.3f}) m")


def demo_with_urdf():
    """演示使用URDF模型参数"""
    print("\n=== 使用URDF模型参数演示 ===")
    
    # 尝试加载URDF文件
    urdf_path = "models/AzureLoong.urdf"
    
    if os.path.exists(urdf_path):
        # 使用URDF文件创建模型
        lipm_model = SimplifiedDynamicsModel(urdf_path=urdf_path)
        
        print("成功从URDF文件加载机器人参数:")
        lipm_model.print_status()
        
        # 显示模型信息
        model_info = lipm_model.get_model_info()
        print("模型详细信息:")
        print(f"  总质量: {model_info['parameters']['total_mass']:.2f} kg")
        print(f"  质心高度: {model_info['parameters']['com_height']:.3f} m")
        print(f"  自然频率: {model_info['parameters']['natural_frequency']:.3f} rad/s")
        print(f"  时间常数: {model_info['parameters']['time_constant']:.3f} s")
        
    else:
        print(f"URDF文件不存在: {urdf_path}")
        print("使用默认参数创建模型")
        lipm_model = SimplifiedDynamicsModel()
        lipm_model.print_status()


def demo_cop_calculation():
    """演示压力中心(COP)计算"""
    print("\n=== 压力中心(COP)计算演示 ===")
    
    # 创建模型
    lipm_model = SimplifiedDynamicsModel()
    
    # 设置质心状态
    data_bus = get_data_bus()
    com_position = Vector3D(x=0.1, y=0.05, z=0.8)
    data_bus.set_center_of_mass_position(com_position)
    lipm_model._update_state_from_data_bus()
    
    print(f"当前质心位置: ({com_position.x:.3f}, {com_position.y:.3f}, {com_position.z:.3f}) m")
    
    # 计算不同期望加速度下的所需COP位置
    desired_accelerations = [
        Vector3D(x=0.0, y=0.0, z=0.0),    # 保持静止
        Vector3D(x=-0.5, y=0.0, z=0.0),   # 向后减速
        Vector3D(x=0.3, y=0.2, z=0.0),    # 向前右加速
        Vector3D(x=0.0, y=-0.4, z=0.0),   # 向左加速
    ]
    
    print("\n期望加速度 -> 所需COP位置:")
    for i, desired_acc in enumerate(desired_accelerations):
        required_cop = lipm_model.compute_required_cop(desired_acc)
        print(f"  加速度 ({desired_acc.x:+.1f}, {desired_acc.y:+.1f}) m/s² -> "
              f"COP ({required_cop.x:+.3f}, {required_cop.y:+.3f}) m")


def main():
    """主函数"""
    print("LIPM (Linear Inverted Pendulum Model) 演示程序")
    print("=" * 50)
    
    try:
        # 基本动力学演示
        lipm_model, trajectory = demo_basic_lipm_dynamics()
        
        # 力控制演示
        demo_force_control()
        
        # 落脚点规划演示
        demo_footstep_planning()
        
        # URDF参数演示
        demo_with_urdf()
        
        # COP计算演示
        demo_cop_calculation()
        
        print("\n=" * 50)
        print("所有演示完成！")
        print("生成的图像文件:")
        print("  - lipm_basic_dynamics.png")
        print("  - lipm_force_control.png")
        print("  - lipm_walking_trajectory.png")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 