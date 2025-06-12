"""
MPC数据总线集成演示
展示如何将MPC求解结果集成到机器人数据总线系统

功能演示：
1. MPC求解器与数据总线的集成
2. 支撑力输出到DataBus.desired_force[foot]
3. 质心轨迹输出供WBC跟踪
4. 智能落脚点规划输出到DataBus.target_foot_pos
5. 实时数据更新和监控

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List

# 导入核心模块
from gait_core.data_bus import DataBus, Vector3D, get_data_bus
from gait_core.mpc_solver import MPCSolver, QPSolver, GaitPlan, create_mpc_solver
from gait_core.mpc_data_bus_integration import (
    MPCDataBusIntegrator, MPCOutputMode, MPCDataBusConfig,
    create_mpc_databus_integrator
)
from gait_core.simplified_dynamics_model import (
    SimplifiedDynamicsModel, LIPMState, Vector3D as ModelVector3D
)


def create_walking_gait_plan(duration: float = 2.0, dt: float = 0.1) -> GaitPlan:
    """创建行走步态计划"""
    steps = int(duration / dt)
    
    gait_plan = GaitPlan()
    
    # 支撑序列：左脚支撑 -> 双脚支撑 -> 右脚支撑 -> 双脚支撑
    support_pattern = ['left', 'double', 'right', 'double']
    gait_plan.support_sequence = [support_pattern[i % 4] for i in range(steps)]
    
    # 接触状态矩阵
    contact_schedule = []
    for support in gait_plan.support_sequence:
        if support == 'left':
            contact_schedule.append([1, 0])  # 左脚接触，右脚摆动
        elif support == 'right':
            contact_schedule.append([0, 1])  # 左脚摆动，右脚接触
        else:  # double
            contact_schedule.append([1, 1])  # 双脚接触
    
    gait_plan.contact_schedule = np.array(contact_schedule)
    
    # 支撑足位置
    left_pos = Vector3D(0.0, 0.1, 0.0)
    right_pos = Vector3D(0.0, -0.1, 0.0)
    
    for i in range(steps):
        gait_plan.support_positions.append({
            'left': left_pos,
            'right': right_pos
        })
        # 模拟前进：每步前进0.05m
        if i % 20 == 0:  # 每2秒更新一次位置
            left_pos = Vector3D(left_pos.x + 0.1, left_pos.y, left_pos.z)
            right_pos = Vector3D(right_pos.x + 0.1, right_pos.y, right_pos.z)
    
    # 时间信息
    gait_plan.step_times = [i * dt for i in range(steps)]
    gait_plan.phase_durations = [0.5, 0.2, 0.5, 0.2]  # 支撑、双支撑、支撑、双支撑
    
    return gait_plan


def demo_basic_mpc_databus_integration():
    """基本MPC数据总线集成演示"""
    print("=== 基本MPC数据总线集成演示 ===")
    
    # 1. 创建数据总线
    data_bus = get_data_bus()
    print("数据总线创建完成")
    
    # 2. 创建LIPM模型
    limp_model = SimplifiedDynamicsModel()
    limp_model.set_com_height(0.8)
    print("LIPM模型创建完成")
    
    # 3. 创建MPC求解器
    mpc_solver = create_mpc_solver(
        limp_model,
        solver_type=QPSolver.OSQP,
        prediction_horizon=15,
        dt=0.1
    )
    print("MPC求解器创建完成")
    
    # 4. 创建MPC数据总线集成器
    integrator = create_mpc_databus_integrator(
        data_bus,
        output_mode=MPCOutputMode.COMBINED,
        update_frequency=125.0
    )
    print("MPC数据总线集成器创建完成")
    
    # 5. 创建步态计划
    gait_plan = create_walking_gait_plan(duration=3.0, dt=0.1)
    print("步态计划创建完成")
    
    # 6. 设置初始状态
    initial_state = LIPMState()
    initial_state.com_position = ModelVector3D(0.0, 0.0, 0.8)
    initial_state.com_velocity = ModelVector3D(0.1, 0.0, 0.0)
    limp_model.current_state = initial_state
    
    # 7. 运行MPC控制循环
    print("\n开始MPC控制循环...")
    simulation_time = 3.0
    dt = 0.1
    steps = int(simulation_time / dt)
    
    # 数据记录
    force_history = {'left_foot': [], 'right_foot': []}
    com_trajectory = []
    mpc_solve_times = []
    integration_success_rate = []
    
    for step in range(steps):
        current_time = step * dt
        print(f"\n--- 步骤 {step+1}/{steps}, 时间: {current_time:.1f}s ---")
        
        # MPC求解
        start_time = time.time()
        mpc_result = mpc_solver.solve(limp_model.current_state, gait_plan)
        solve_time = time.time() - start_time
        mpc_solve_times.append(solve_time)
        
        if mpc_result.success:
            print(f"MPC求解成功: 代价={mpc_result.cost:.3f}, 时间={solve_time:.3f}s")
            
            # 集成到数据总线
            integration_success = integrator.update_from_mpc_result(mpc_result, current_time)
            integration_success_rate.append(1.0 if integration_success else 0.0)
            
            # 记录数据
            if mpc_result.current_contact_forces:
                for foot, force in mpc_result.current_contact_forces.items():
                    if foot in force_history:
                        force_history[foot].append([force.x, force.y, force.z])
            
            if mpc_result.com_position_trajectory:
                pos = mpc_result.com_position_trajectory[0]
                com_trajectory.append([pos.x, pos.y, pos.z])
            
            # 更新机器人状态（简化的动力学积分）
            if mpc_result.com_velocity_trajectory:
                new_vel = mpc_result.com_velocity_trajectory[0]
                limp_model.current_state.com_velocity = ModelVector3D(new_vel.x, new_vel.y, new_vel.z)
            
            if mpc_result.com_position_trajectory and len(mpc_result.com_position_trajectory) > 1:
                next_pos = mpc_result.com_position_trajectory[1]
                limp_model.current_state.com_position = ModelVector3D(next_pos.x, next_pos.y, next_pos.z)
        
        else:
            print(f"MPC求解失败: {mpc_result.solver_info}")
            integration_success_rate.append(0.0)
        
        # 打印数据总线状态
        if step % 10 == 0:  # 每1秒打印一次
            print_databus_mpc_status(data_bus)
        
        # 模拟实时控制
        time.sleep(0.01)  # 10ms延迟
    
    # 8. 打印统计结果
    print_integration_statistics(integrator, mpc_solve_times, integration_success_rate)
    
    # 9. 可视化结果
    visualize_integration_results(force_history, com_trajectory, mpc_solve_times, integration_success_rate)
    
    return integrator, data_bus


def demo_different_output_modes():
    """不同输出模式演示"""
    print("\n=== 不同MPC输出模式演示 ===")
    
    modes = [
        MPCOutputMode.FORCE_ONLY,
        MPCOutputMode.TRAJECTORY_ONLY,
        MPCOutputMode.FOOTSTEP_ONLY,
        MPCOutputMode.COMBINED
    ]
    
    results = {}
    
    for mode in modes:
        print(f"\n--- 测试输出模式: {mode.value} ---")
        
        # 创建数据总线和集成器
        data_bus = DataBus()
        integrator = create_mpc_databus_integrator(
            data_bus,
            output_mode=mode,
            update_frequency=100.0
        )
        
        # 创建模拟MPC结果
        mock_result = create_mock_mpc_result()
        
        # 测试集成
        success = integrator.update_from_mpc_result(mock_result)
        
        # 检查数据总线状态
        mpc_summary = data_bus.get_mpc_data_summary()
        
        results[mode.value] = {
            'success': success,
            'desired_forces': mpc_summary['desired_forces_count'],
            'com_trajectory': mpc_summary['com_trajectory_length'],
            'target_positions': len(mpc_summary['target_foot_positions']),
            'mpc_status': mpc_summary['mpc_status']['solve_success']
        }
        
        print(f"集成结果: {results[mode.value]}")
    
    return results


def create_mock_mpc_result():
    """创建模拟MPC结果用于测试"""
    from gait_core.mpc_solver import MPCResult
    
    result = MPCResult()
    result.success = True
    result.solve_time = 0.015
    result.cost = 125.6
    result.solver_info = {'solver_type': 'osqp'}
    
    # 模拟接触力
    result.current_contact_forces = {
        'left_foot': Vector3D(10.0, 5.0, 80.0),
        'right_foot': Vector3D(-8.0, -3.0, 75.0)
    }
    
    # 模拟质心轨迹
    result.com_position_trajectory = [
        Vector3D(0.1, 0.0, 0.8),
        Vector3D(0.15, 0.02, 0.8),
        Vector3D(0.2, 0.01, 0.8)
    ]
    
    result.com_velocity_trajectory = [
        Vector3D(0.5, 0.1, 0.0),
        Vector3D(0.48, 0.05, 0.0),
        Vector3D(0.52, 0.08, 0.0)
    ]
    
    result.com_acceleration_trajectory = [
        Vector3D(0.1, 0.05, 0.0),
        Vector3D(0.08, 0.02, 0.0),
        Vector3D(0.12, 0.06, 0.0)
    ]
    
    # 模拟ZMP轨迹
    result.zmp_trajectory = [
        Vector3D(0.05, 0.0, 0.0),
        Vector3D(0.08, 0.01, 0.0),
        Vector3D(0.06, -0.01, 0.0)
    ]
    
    # 模拟下一步落脚点
    result.next_footstep = {
        'left_foot': Vector3D(0.3, 0.1, 0.0),
        'right_foot': Vector3D(0.25, -0.1, 0.0)
    }
    
    return result


def print_databus_mpc_status(data_bus: DataBus):
    """打印数据总线MPC状态"""
    print("\n--- 数据总线MPC状态 ---")
    
    # MPC状态
    mpc_status = data_bus.get_mpc_status()
    print(f"MPC求解状态: {'成功' if mpc_status['solve_success'] else '失败'}")
    print(f"求解时间: {mpc_status['last_solve_time']:.3f}s")
    print(f"优化代价: {mpc_status['cost']:.2f}")
    
    # 期望力
    desired_forces = data_bus.get_all_desired_contact_forces()
    print(f"期望接触力数量: {len(desired_forces)}")
    for foot, force in desired_forces.items():
        magnitude = np.sqrt(force.x**2 + force.y**2 + force.z**2)
        print(f"  {foot}: [{force.x:.1f}, {force.y:.1f}, {force.z:.1f}]N (|F|={magnitude:.1f}N)")
    
    # 质心状态
    com_pos = data_bus.get_center_of_mass_position()
    com_vel = data_bus.get_center_of_mass_velocity()
    print(f"质心位置: [{com_pos.x:.3f}, {com_pos.y:.3f}, {com_pos.z:.3f}]m")
    print(f"质心速度: [{com_vel.x:.3f}, {com_vel.y:.3f}, {com_vel.z:.3f}]m/s")
    
    # 目标足部位置
    for foot in ['left_foot', 'right_foot']:
        target_pos = data_bus.get_target_foot_position(foot)
        if target_pos:
            print(f"{foot}目标位置: [{target_pos['x']:.3f}, {target_pos['y']:.3f}, {target_pos['z']:.3f}]m")


def print_integration_statistics(integrator: MPCDataBusIntegrator, 
                                solve_times: List[float], 
                                success_rates: List[float]):
    """打印集成统计信息"""
    print("\n=== MPC数据总线集成统计 ===")
    
    # 集成器统计
    integrator.print_status()
    
    # MPC求解统计
    avg_solve_time = np.mean(solve_times) if solve_times else 0.0
    max_solve_time = np.max(solve_times) if solve_times else 0.0
    min_solve_time = np.min(solve_times) if solve_times else 0.0
    
    print(f"\n--- MPC求解性能 ---")
    print(f"平均求解时间: {avg_solve_time:.3f}s")
    print(f"最大求解时间: {max_solve_time:.3f}s")
    print(f"最小求解时间: {min_solve_time:.3f}s")
    
    # 集成成功率
    overall_success_rate = np.mean(success_rates) if success_rates else 0.0
    print(f"总体集成成功率: {overall_success_rate:.1%}")


def visualize_integration_results(force_history: Dict, 
                                com_trajectory: List, 
                                solve_times: List[float],
                                success_rates: List[float]):
    """可视化集成结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MPC数据总线集成结果', fontsize=16)
    
    # 1. 接触力历史
    ax1 = axes[0, 0]
    for foot, forces in force_history.items():
        if forces:
            forces_array = np.array(forces)
            force_magnitudes = np.sqrt(np.sum(forces_array**2, axis=1))
            ax1.plot(force_magnitudes, label=f'{foot} 力幅值')
    ax1.set_title('接触力历史')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('力幅值 [N]')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 质心轨迹
    ax2 = axes[0, 1]
    if com_trajectory:
        com_array = np.array(com_trajectory)
        ax2.plot(com_array[:, 0], com_array[:, 1], 'b-', label='质心轨迹')
        ax2.scatter(com_array[0, 0], com_array[0, 1], color='green', s=100, label='起点')
        ax2.scatter(com_array[-1, 0], com_array[-1, 1], color='red', s=100, label='终点')
    ax2.set_title('质心轨迹 (X-Y平面)')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. MPC求解时间
    ax3 = axes[1, 0]
    if solve_times:
        ax3.plot(solve_times, 'g-', linewidth=2)
        ax3.axhline(y=np.mean(solve_times), color='r', linestyle='--', label=f'平均: {np.mean(solve_times):.3f}s')
    ax3.set_title('MPC求解时间')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('求解时间 [s]')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 集成成功率
    ax4 = axes[1, 1]
    if success_rates:
        # 计算滑动窗口成功率
        window_size = 10
        windowed_success = []
        for i in range(len(success_rates)):
            start_idx = max(0, i - window_size + 1)
            windowed_success.append(np.mean(success_rates[start_idx:i+1]))
        
        ax4.plot(windowed_success, 'purple', linewidth=2)
        ax4.axhline(y=np.mean(success_rates), color='orange', linestyle='--', 
                   label=f'总体: {np.mean(success_rates):.1%}')
    ax4.set_title('集成成功率 (滑动窗口)')
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('成功率')
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('mpc_databus_integration_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("MPC数据总线集成演示程序")
    print("=" * 50)
    
    try:
        # 1. 基本集成演示
        integrator, data_bus = demo_basic_mpc_databus_integration()
        
        # 2. 不同输出模式演示
        mode_results = demo_different_output_modes()
        
        print("\n=== 不同输出模式对比 ===")
        for mode, result in mode_results.items():
            print(f"{mode}: 成功={result['success']}, "
                  f"力={result['desired_forces']}, "
                  f"轨迹={result['com_trajectory']}, "
                  f"位置={result['target_positions']}")
        
        print("\n=== 演示完成 ===")
        print("生成的文件:")
        print("  - mpc_databus_integration_results.png: 集成结果可视化")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 