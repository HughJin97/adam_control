#!/usr/bin/env python3
"""
MPC控制器演示脚本
展示基于LIPM的模型预测控制器的使用方法

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

from gait_core.simplified_dynamics import SimplifiedDynamicsModel
from gait_core.mpc_controller import (
    MPCController, MPCParameters, MPCReference, MPCMode,
    create_mpc_controller
)
from gait_core.data_bus import DataBus, Vector3D, get_data_bus


def demo_basic_mpc_tracking():
    """演示基本的MPC轨迹跟踪"""
    print("=== MPC基本轨迹跟踪演示 ===")
    
    # 创建LIPM模型和MPC控制器
    data_bus = get_data_bus()
    
    # 设置初始状态
    initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
    initial_com_vel = Vector3D(x=0.0, y=0.0, z=0.0)
    
    data_bus.set_center_of_mass_position(initial_com_pos)
    data_bus.set_center_of_mass_velocity(initial_com_vel)
    
    # 创建LIPM模型
    lipm_model = SimplifiedDynamicsModel(data_bus=data_bus)
    
    # 创建MPC控制器
    mpc_params = MPCParameters()
    mpc_params.prediction_horizon = 20
    mpc_params.control_horizon = 10
    mpc_params.dt = 0.1
    
    mpc_controller = MPCController(lipm_model, mpc_params)
    mpc_controller.control_mode = MPCMode.FORCE_CONTROL
    
    # 生成简单的参考轨迹（正弦波）
    reference = MPCReference()
    time_horizon = mpc_params.prediction_horizon * mpc_params.dt
    times = np.arange(0, time_horizon, mpc_params.dt)
    
    for t in times:
        # 正弦波参考轨迹
        ref_pos = Vector3D(
            x=0.2 * np.sin(0.5 * t),
            y=0.1 * np.cos(0.3 * t),
            z=0.8
        )
        ref_vel = Vector3D(
            x=0.2 * 0.5 * np.cos(0.5 * t),
            y=-0.1 * 0.3 * np.sin(0.3 * t),
            z=0.0
        )
        
        reference.com_position_ref.append(ref_pos)
        reference.com_velocity_ref.append(ref_vel)
    
    mpc_controller.set_reference_trajectory(reference)
    
    # 设置支撑足位置
    support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
    lipm_model.set_support_foot_position(support_foot)
    
    # 运行MPC控制循环
    simulation_time = 5.0
    dt = 0.1
    steps = int(simulation_time / dt)
    
    # 记录数据
    actual_trajectory = []
    control_forces = []
    solve_times = []
    costs = []
    
    for i in range(steps):
        current_time = i * dt
        
        # MPC求解
        solution = mpc_controller.solve_mpc(lipm_model.current_state)
        
        if solution.success:
            # 应用控制
            if solution.optimal_forces:
                lipm_model.set_foot_force_command(solution.optimal_forces[0])
                control_forces.append(solution.optimal_forces[0])
            else:
                control_forces.append(Vector3D(x=0.0, y=0.0, z=0.0))
            
            solve_times.append(solution.solve_time)
            costs.append(solution.cost)
        else:
            control_forces.append(Vector3D(x=0.0, y=0.0, z=0.0))
            solve_times.append(0.0)
            costs.append(float('inf'))
        
        # 更新模型
        lipm_model.update(dt)
        actual_trajectory.append(lipm_model.current_state)
        
        if i % 10 == 0:
            print(f"时间: {current_time:.1f}s, 质心位置: ({lipm_model.current_state.com_position.x:.3f}, {lipm_model.current_state.com_position.y:.3f})")
    
    # 绘制结果
    plot_mpc_results(actual_trajectory, reference, control_forces, solve_times, costs, "基本MPC跟踪")
    
    return mpc_controller, actual_trajectory


def demo_walking_mpc():
    """演示行走步态的MPC控制"""
    print("\n=== MPC行走步态控制演示 ===")
    
    # 创建系统
    data_bus = get_data_bus()
    
    initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
    initial_com_vel = Vector3D(x=0.0, y=0.0, z=0.0)
    
    data_bus.set_center_of_mass_position(initial_com_pos)
    data_bus.set_center_of_mass_velocity(initial_com_vel)
    
    lipm_model = SimplifiedDynamicsModel(data_bus=data_bus)
    
    # MPC参数设置
    mpc_params = MPCParameters()
    mpc_params.prediction_horizon = 25
    mpc_params.control_horizon = 15
    mpc_params.dt = 0.08
    
    # 调整权重
    mpc_params.Q_position = np.diag([50.0, 100.0])  # Y方向更重要
    mpc_params.R_force = np.diag([2.0, 2.0])
    mpc_params.R_force_rate = 10.0
    
    mpc_controller = MPCController(lipm_model, mpc_params)
    mpc_controller.control_mode = MPCMode.COMBINED_CONTROL
    
    # 生成行走参考轨迹
    target_velocity = Vector3D(x=0.3, y=0.0, z=0.0)  # 前进速度0.3 m/s
    reference = mpc_controller.generate_walking_reference(
        target_velocity=target_velocity,
        step_length=0.2,
        step_width=0.15,
        step_duration=0.8
    )
    
    mpc_controller.set_reference_trajectory(reference)
    
    # 模拟行走
    simulation_time = 6.0
    dt = 0.08
    steps = int(simulation_time / dt)
    
    actual_trajectory = []
    control_forces = []
    footstep_positions = []
    solve_times = []
    costs = []
    
    current_step_index = 0
    step_switch_time = 0.8  # 步态切换时间
    
    for i in range(steps):
        current_time = i * dt
        
        # 更新支撑足位置（模拟步态切换）
        if current_time >= current_step_index * step_switch_time and current_step_index < len(reference.footstep_ref):
            current_footstep = reference.footstep_ref[current_step_index]
            lipm_model.set_support_foot_position(current_footstep)
            footstep_positions.append(current_footstep)
            current_step_index += 1
            print(f"切换到步态 {current_step_index}: 支撑足位置 ({current_footstep.x:.3f}, {current_footstep.y:.3f})")
        
        # MPC求解
        solution = mpc_controller.solve_mpc(lipm_model.current_state)
        
        if solution.success:
            if solution.optimal_forces:
                lipm_model.set_foot_force_command(solution.optimal_forces[0])
                control_forces.append(solution.optimal_forces[0])
            else:
                control_forces.append(Vector3D(x=0.0, y=0.0, z=0.0))
            
            solve_times.append(solution.solve_time)
            costs.append(solution.cost)
        else:
            control_forces.append(Vector3D(x=0.0, y=0.0, z=0.0))
            solve_times.append(0.0)
            costs.append(float('inf'))
        
        # 更新模型
        lipm_model.update(dt)
        actual_trajectory.append(lipm_model.current_state)
    
    # 绘制行走结果
    plot_walking_results(actual_trajectory, reference, footstep_positions, control_forces, "MPC行走控制")
    
    return mpc_controller, actual_trajectory


def demo_constraint_handling():
    """演示约束处理"""
    print("\n=== MPC约束处理演示 ===")
    
    # 创建系统
    data_bus = get_data_bus()
    
    initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
    initial_com_vel = Vector3D(x=0.0, y=0.0, z=0.0)
    
    data_bus.set_center_of_mass_position(initial_com_pos)
    data_bus.set_center_of_mass_velocity(initial_com_vel)
    
    lipm_model = SimplifiedDynamicsModel(data_bus=data_bus)
    
    # 设置严格的约束
    mpc_params = MPCParameters()
    mpc_params.prediction_horizon = 15
    mpc_params.control_horizon = 8
    mpc_params.dt = 0.1
    
    # 严格的约束
    mpc_params.max_force = 50.0  # 较小的力限制
    mpc_params.friction_coefficient = 0.5  # 较小的摩擦系数
    mpc_params.com_position_bounds = (-0.5, 0.5)  # 严格的位置约束
    mpc_params.com_velocity_bounds = (-1.0, 1.0)  # 严格的速度约束
    
    mpc_controller = MPCController(lipm_model, mpc_params)
    mpc_controller.control_mode = MPCMode.FORCE_CONTROL
    
    # 生成挑战性的参考轨迹（快速变化）
    reference = MPCReference()
    time_horizon = mpc_params.prediction_horizon * mpc_params.dt
    times = np.arange(0, time_horizon, mpc_params.dt)
    
    for t in times:
        # 快速变化的参考轨迹
        ref_pos = Vector3D(
            x=0.4 * np.sin(2.0 * t),  # 大幅度快速振荡
            y=0.3 * np.cos(1.5 * t),
            z=0.8
        )
        ref_vel = Vector3D(
            x=0.4 * 2.0 * np.cos(2.0 * t),
            y=-0.3 * 1.5 * np.sin(1.5 * t),
            z=0.0
        )
        
        reference.com_position_ref.append(ref_pos)
        reference.com_velocity_ref.append(ref_vel)
    
    mpc_controller.set_reference_trajectory(reference)
    
    # 设置支撑足
    support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
    lipm_model.set_support_foot_position(support_foot)
    
    # 运行约束测试
    simulation_time = 3.0
    dt = 0.1
    steps = int(simulation_time / dt)
    
    actual_trajectory = []
    control_forces = []
    constraint_violations = []
    solve_times = []
    
    for i in range(steps):
        current_time = i * dt
        
        # MPC求解
        solution = mpc_controller.solve_mpc(lipm_model.current_state)
        
        if solution.success:
            force = solution.optimal_forces[0] if solution.optimal_forces else Vector3D()
            control_forces.append(force)
            
            # 检查约束违反
            violations = check_constraints(lipm_model.current_state, force, mpc_params)
            constraint_violations.append(violations)
            
            lipm_model.set_foot_force_command(force)
            solve_times.append(solution.solve_time)
        else:
            control_forces.append(Vector3D(x=0.0, y=0.0, z=0.0))
            constraint_violations.append({'force': False, 'position': False, 'velocity': False})
            solve_times.append(0.0)
        
        # 更新模型
        lipm_model.update(dt)
        actual_trajectory.append(lipm_model.current_state)
    
    # 绘制约束处理结果
    plot_constraint_results(actual_trajectory, reference, control_forces, constraint_violations, mpc_params)
    
    return mpc_controller, actual_trajectory


def check_constraints(state, force: Vector3D, params: MPCParameters) -> dict:
    """检查约束违反情况"""
    violations = {}
    
    # 力约束
    force_magnitude = np.sqrt(force.x**2 + force.y**2)
    violations['force'] = force_magnitude > params.max_force
    
    # 位置约束
    pos_bounds = params.com_position_bounds
    violations['position'] = (
        state.com_position.x < pos_bounds[0] or state.com_position.x > pos_bounds[1] or
        state.com_position.y < pos_bounds[0] or state.com_position.y > pos_bounds[1]
    )
    
    # 速度约束
    vel_bounds = params.com_velocity_bounds
    violations['velocity'] = (
        state.com_velocity.x < vel_bounds[0] or state.com_velocity.x > vel_bounds[1] or
        state.com_velocity.y < vel_bounds[0] or state.com_velocity.y > vel_bounds[1]
    )
    
    return violations


def plot_mpc_results(trajectory, reference, forces, solve_times, costs, title):
    """绘制MPC控制结果"""
    times = [i * 0.1 for i in range(len(trajectory))]
    
    # 提取轨迹数据
    actual_x = [state.com_position.x for state in trajectory]
    actual_y = [state.com_position.y for state in trajectory]
    actual_vx = [state.com_velocity.x for state in trajectory]
    actual_vy = [state.com_velocity.y for state in trajectory]
    
    # 参考轨迹
    ref_times = [i * 0.1 for i in range(len(reference.com_position_ref))]
    ref_x = [pos.x for pos in reference.com_position_ref]
    ref_y = [pos.y for pos in reference.com_position_ref]
    
    # 控制力
    force_x = [f.x for f in forces]
    force_y = [f.y for f in forces]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 位置跟踪
    ax1.plot(times, actual_x, 'b-', linewidth=2, label='实际X')
    ax1.plot(times, actual_y, 'r-', linewidth=2, label='实际Y')
    ax1.plot(ref_times, ref_x, 'b--', alpha=0.7, label='参考X')
    ax1.plot(ref_times, ref_y, 'r--', alpha=0.7, label='参考Y')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('位置 (m)')
    ax1.set_title('质心位置跟踪')
    ax1.legend()
    ax1.grid(True)
    
    # 控制力
    ax2.plot(times, force_x, 'g-', linewidth=2, label='力X')
    ax2.plot(times, force_y, 'm-', linewidth=2, label='力Y')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('力 (N)')
    ax2.set_title('控制力')
    ax2.legend()
    ax2.grid(True)
    
    # 求解时间
    ax3.plot(times, solve_times, 'k-', linewidth=2)
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('求解时间 (s)')
    ax3.set_title('MPC求解时间')
    ax3.grid(True)
    
    # 代价函数
    ax4.plot(times, costs, 'orange', linewidth=2)
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('代价')
    ax4.set_title('优化代价')
    ax4.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'mpc_{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    print(f"{title}结果图已保存")


def plot_walking_results(trajectory, reference, footsteps, forces, title):
    """绘制行走控制结果"""
    times = [i * 0.08 for i in range(len(trajectory))]
    
    # 轨迹数据
    actual_x = [state.com_position.x for state in trajectory]
    actual_y = [state.com_position.y for state in trajectory]
    
    # 参考轨迹
    ref_x = [pos.x for pos in reference.com_position_ref]
    ref_y = [pos.y for pos in reference.com_position_ref]
    
    # 落脚点
    foot_x = [foot.x for foot in footsteps]
    foot_y = [foot.y for foot in footsteps]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 2D轨迹
    ax1.plot(actual_x, actual_y, 'b-', linewidth=3, label='实际轨迹')
    ax1.plot(ref_x, ref_y, 'r--', alpha=0.7, label='参考轨迹')
    ax1.scatter(foot_x, foot_y, c='red', s=100, marker='s', label='落脚点', zorder=5)
    ax1.set_xlabel('X位置 (m)')
    ax1.set_ylabel('Y位置 (m)')
    ax1.set_title('机器人行走轨迹 - 俯视图')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # 时间序列
    ax2.plot(times, actual_x, 'b-', linewidth=2, label='实际X')
    ax2.plot(times, actual_y, 'r-', linewidth=2, label='实际Y')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('位置 (m)')
    ax2.set_title('质心位置时间序列')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'mpc_walking.png', dpi=300, bbox_inches='tight')
    print("行走控制结果图已保存")


def plot_constraint_results(trajectory, reference, forces, violations, params):
    """绘制约束处理结果"""
    times = [i * 0.1 for i in range(len(trajectory))]
    
    # 轨迹数据
    actual_x = [state.com_position.x for state in trajectory]
    actual_y = [state.com_position.y for state in trajectory]
    
    # 控制力
    force_magnitudes = [np.sqrt(f.x**2 + f.y**2) for f in forces]
    
    # 约束违反
    force_violations = [v['force'] for v in violations]
    pos_violations = [v['position'] for v in violations]
    vel_violations = [v['velocity'] for v in violations]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 位置和约束
    ax1.plot(times, actual_x, 'b-', linewidth=2, label='X位置')
    ax1.plot(times, actual_y, 'r-', linewidth=2, label='Y位置')
    ax1.axhline(y=params.com_position_bounds[0], color='k', linestyle='--', alpha=0.5, label='位置约束')
    ax1.axhline(y=params.com_position_bounds[1], color='k', linestyle='--', alpha=0.5)
    
    # 标记违反点
    for i, violation in enumerate(pos_violations):
        if violation:
            ax1.plot(times[i], actual_x[i], 'ro', markersize=8)
            ax1.plot(times[i], actual_y[i], 'ro', markersize=8)
    
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('位置 (m)')
    ax1.set_title('位置约束')
    ax1.legend()
    ax1.grid(True)
    
    # 力约束
    ax2.plot(times, force_magnitudes, 'g-', linewidth=2, label='力大小')
    ax2.axhline(y=params.max_force, color='k', linestyle='--', alpha=0.5, label='力限制')
    
    # 标记违反点
    for i, violation in enumerate(force_violations):
        if violation:
            ax2.plot(times[i], force_magnitudes[i], 'ro', markersize=8)
    
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('力 (N)')
    ax2.set_title('力约束')
    ax2.legend()
    ax2.grid(True)
    
    # 约束违反统计
    violation_counts = {
        'Force': sum(force_violations),
        'Position': sum(pos_violations),
        'Velocity': sum(vel_violations)
    }
    
    ax3.bar(violation_counts.keys(), violation_counts.values())
    ax3.set_ylabel('违反次数')
    ax3.set_title('约束违反统计')
    ax3.grid(True)
    
    # 2D轨迹和边界
    ax4.plot(actual_x, actual_y, 'b-', linewidth=2, label='实际轨迹')
    
    # 绘制位置约束边界
    bounds = params.com_position_bounds
    ax4.axvline(x=bounds[0], color='k', linestyle='--', alpha=0.5)
    ax4.axvline(x=bounds[1], color='k', linestyle='--', alpha=0.5)
    ax4.axhline(y=bounds[0], color='k', linestyle='--', alpha=0.5)
    ax4.axhline(y=bounds[1], color='k', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('X位置 (m)')
    ax4.set_ylabel('Y位置 (m)')
    ax4.set_title('轨迹和位置约束')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')
    
    plt.suptitle('MPC约束处理')
    plt.tight_layout()
    plt.savefig('mpc_constraints.png', dpi=300, bbox_inches='tight')
    print("约束处理结果图已保存")


def main():
    """主函数"""
    print("MPC控制器演示程序")
    print("=" * 50)
    
    try:
        # 基本MPC跟踪演示
        mpc_controller, trajectory = demo_basic_mpc_tracking()
        
        # 行走MPC演示
        walking_controller, walking_trajectory = demo_walking_mpc()
        
        # 约束处理演示
        constraint_controller, constraint_trajectory = demo_constraint_handling()
        
        print("\n=" * 50)
        print("所有MPC演示完成！")
        print("生成的图像文件:")
        print("  - mpc_基本mpc跟踪.png")
        print("  - mpc_walking.png")
        print("  - mpc_constraints.png")
        
        # 显示性能统计
        print("\n性能统计:")
        for name, controller in [("基本跟踪", mpc_controller), 
                                ("行走控制", walking_controller), 
                                ("约束处理", constraint_controller)]:
            stats = controller.get_performance_stats()
            if stats:
                print(f"{name}: 平均求解时间 {stats['average_solve_time']:.3f}s, "
                      f"平均代价 {stats['average_cost']:.2f}")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 