#!/usr/bin/env python3
"""
专业MPC求解器演示
展示QP求解器集成、步态规划输入和支撑力轨迹输出

参考OpenLoong框架的设计理念
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

from gait_core.simplified_dynamics import SimplifiedDynamicsModel, LIPMState
from gait_core.mpc_solver import (
    MPCSolver, GaitPlan, MPCResult, QPSolver, create_mpc_solver
)
from gait_core.data_bus import DataBus, Vector3D, get_data_bus


def create_walking_gait_plan(duration: float = 2.0, step_duration: float = 0.6) -> GaitPlan:
    """
    创建行走步态计划
    
    Args:
        duration: 总时长
        step_duration: 每步持续时间
        
    Returns:
        GaitPlan: 步态计划
    """
    gait_plan = GaitPlan()
    
    dt = 0.1  # 时间步长
    n_steps = int(duration / dt)
    
    # 1. 生成支撑序列
    for i in range(n_steps):
        t = i * dt
        step_phase = (t % step_duration) / step_duration
        
        if step_phase < 0.3:
            # 双支撑期
            gait_plan.support_sequence.append('double')
        elif step_phase < 0.8:
            # 单支撑期（交替左右）
            step_index = int(t / step_duration)
            if step_index % 2 == 0:
                gait_plan.support_sequence.append('left')
            else:
                gait_plan.support_sequence.append('right')
        else:
            # 摆动期过渡
            gait_plan.support_sequence.append('double')
    
    # 2. 生成接触调度矩阵 [time_steps x n_contacts]
    gait_plan.contact_schedule = np.zeros((n_steps, 2))  # 2个接触点（左右脚）
    
    for i, support_type in enumerate(gait_plan.support_sequence):
        if support_type == 'double':
            gait_plan.contact_schedule[i, :] = [1.0, 1.0]  # 双脚接触
        elif support_type == 'left':
            gait_plan.contact_schedule[i, :] = [1.0, 0.0]  # 仅左脚接触
        elif support_type == 'right':
            gait_plan.contact_schedule[i, :] = [0.0, 1.0]  # 仅右脚接触
        else:  # flight
            gait_plan.contact_schedule[i, :] = [0.0, 0.0]  # 无接触
    
    # 3. 生成支撑足位置轨迹
    step_length = 0.15
    step_width = 0.1
    
    for i in range(n_steps):
        t = i * dt
        step_index = int(t / step_duration)
        
        # 左脚位置
        left_x = step_index * step_length
        left_y = step_width / 2
        
        # 右脚位置  
        right_x = step_index * step_length
        right_y = -step_width / 2
        
        # 如果是摆动腿，则向前移动
        if gait_plan.support_sequence[i] == 'right':
            # 左脚摆动
            left_x += step_length * 0.5
        elif gait_plan.support_sequence[i] == 'left':
            # 右脚摆动
            right_x += step_length * 0.5
        
        foot_positions = {
            'left': Vector3D(x=left_x, y=left_y, z=0.0),
            'right': Vector3D(x=right_x, y=right_y, z=0.0)
        }
        
        gait_plan.support_positions.append(foot_positions)
    
    # 4. 设置步态时间信息
    gait_plan.step_times = list(np.arange(0, duration, step_duration))
    gait_plan.phase_durations = [step_duration] * len(gait_plan.step_times)
    
    print(f"生成步态计划: {len(gait_plan.support_sequence)} 时间步, {len(gait_plan.step_times)} 步")
    
    return gait_plan


def demo_osqp_solver():
    """演示OSQP求解器"""
    print("=== OSQP求解器演示 ===")
    
    # 创建系统
    data_bus = get_data_bus()
    
    # 初始状态
    initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
    initial_com_vel = Vector3D(x=0.2, y=0.0, z=0.0)  # 前进速度
    
    data_bus.set_center_of_mass_position(initial_com_pos)
    data_bus.set_center_of_mass_velocity(initial_com_vel)
    
    lipm_model = SimplifiedDynamicsModel(data_bus=data_bus)
    
    # 创建MPC求解器
    mpc_solver = create_mpc_solver(
        lipm_model=lipm_model,
        solver_type=QPSolver.OSQP,
        prediction_horizon=15,
        dt=0.1
    )
    
    # 创建步态计划
    gait_plan = create_walking_gait_plan(duration=1.5, step_duration=0.6)
    
    # 设置当前状态
    current_state = LIPMState()
    current_state.com_position = initial_com_pos
    current_state.com_velocity = initial_com_vel
    current_state.com_acceleration = Vector3D(x=0.0, y=0.0, z=0.0)
    current_state.timestamp = 0.0
    
    # 求解MPC
    print("开始MPC求解...")
    result = mpc_solver.solve(current_state, gait_plan)
    
    if result.success:
        print(f"✅ OSQP求解成功!")
        print(f"   代价: {result.cost:.3f}")
        print(f"   求解时间: {result.solve_time:.3f}s")
        print(f"   预测轨迹长度: {len(result.predicted_com_states)}")
        print(f"   接触力矩阵形状: {result.contact_forces.shape}")
        
        # 显示当前控制命令
        print(f"   当前左脚力: ({result.current_contact_forces['left'].x:.2f}, "
              f"{result.current_contact_forces['left'].y:.2f}, "
              f"{result.current_contact_forces['left'].z:.2f}) N")
        print(f"   当前右脚力: ({result.current_contact_forces['right'].x:.2f}, "
              f"{result.current_contact_forces['right'].y:.2f}, "
              f"{result.current_contact_forces['right'].z:.2f}) N")
    else:
        print(f"❌ OSQP求解失败: {result.solver_info}")
    
    return mpc_solver, result, gait_plan


def demo_multi_solver_comparison():
    """演示多个求解器性能比较"""
    print("\n=== 多求解器性能比较 ===")
    
    # 创建系统
    data_bus = get_data_bus()
    initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
    initial_com_vel = Vector3D(x=0.1, y=0.0, z=0.0)
    
    data_bus.set_center_of_mass_position(initial_com_pos)
    data_bus.set_center_of_mass_velocity(initial_com_vel)
    
    lipm_model = SimplifiedDynamicsModel(data_bus=data_bus)
    
    # 测试不同求解器
    solvers_to_test = [
        QPSolver.OSQP,
        QPSolver.SCIPY,  # 后备求解器
    ]
    
    # 添加CVXPY求解器（如果可用）
    try:
        import cvxpy as cp
        solvers_to_test.append(QPSolver.CVXPY_OSQP)
    except ImportError:
        pass
    
    # 创建测试问题
    gait_plan = create_walking_gait_plan(duration=1.0, step_duration=0.5)
    
    current_state = LIPMState()
    current_state.com_position = initial_com_pos
    current_state.com_velocity = initial_com_vel
    current_state.com_acceleration = Vector3D(x=0.0, y=0.0, z=0.0)
    current_state.timestamp = 0.0
    
    results = {}
    
    for solver_type in solvers_to_test:
        print(f"\n测试求解器: {solver_type.value}")
        
        try:
            # 创建求解器
            mpc_solver = create_mpc_solver(
                lipm_model=lipm_model,
                solver_type=solver_type,
                prediction_horizon=10,  # 较小的问题用于比较
                dt=0.1
            )
            
            # 多次求解取平均
            solve_times = []
            success_count = 0
            
            for i in range(3):
                result = mpc_solver.solve(current_state, gait_plan)
                solve_times.append(result.solve_time)
                if result.success:
                    success_count += 1
            
            results[solver_type.value] = {
                'avg_time': np.mean(solve_times),
                'max_time': np.max(solve_times),
                'min_time': np.min(solve_times),
                'success_rate': success_count / 3,
                'solver_info': result.solver_info if 'result' in locals() else {}
            }
            
            print(f"   平均求解时间: {results[solver_type.value]['avg_time']:.4f}s")
            print(f"   成功率: {results[solver_type.value]['success_rate']:.1%}")
            
        except Exception as e:
            print(f"   ❌ 求解器 {solver_type.value} 测试失败: {e}")
            results[solver_type.value] = {
                'avg_time': float('inf'),
                'success_rate': 0.0,
                'error': str(e)
            }
    
    # 输出比较结果
    print(f"\n{'求解器':<15} {'平均时间(ms)':<12} {'成功率':<8} {'状态'}")
    print("-" * 50)
    
    for solver_name, stats in results.items():
        avg_time_ms = stats['avg_time'] * 1000 if stats['avg_time'] != float('inf') else float('inf')
        success_rate = f"{stats['success_rate']:.1%}"
        status = "✅" if stats['success_rate'] > 0.5 else "❌"
        
        print(f"{solver_name:<15} {avg_time_ms:<12.2f} {success_rate:<8} {status}")
    
    return results


def demo_real_time_control_loop():
    """演示实时控制循环"""
    print("\n=== 实时MPC控制循环演示 ===")
    
    # 创建系统
    data_bus = get_data_bus()
    initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
    initial_com_vel = Vector3D(x=0.15, y=0.0, z=0.0)
    
    data_bus.set_center_of_mass_position(initial_com_pos)
    data_bus.set_center_of_mass_velocity(initial_com_vel)
    
    lipm_model = SimplifiedDynamicsModel(data_bus=data_bus)
    
    # 创建高效的MPC求解器
    mpc_solver = create_mpc_solver(
        lipm_model=lipm_model,
        solver_type=QPSolver.OSQP,
        prediction_horizon=12,  # 优化后的时域
        dt=0.08
    )
    
    # 创建长期步态计划
    total_duration = 3.0
    gait_plan = create_walking_gait_plan(duration=total_duration, step_duration=0.6)
    
    # 控制循环
    dt = 0.08  # 125Hz控制频率
    simulation_steps = int(total_duration / dt)
    
    # 记录数据
    time_history = []
    com_trajectory = []
    force_trajectory = []
    zmp_trajectory = []
    solve_time_history = []
    cost_history = []
    
    print(f"开始实时控制循环: {simulation_steps} 步, {1/dt:.0f}Hz")
    
    for step in range(simulation_steps):
        current_time = step * dt
        
        # 获取当前状态
        current_state = lipm_model.current_state
        current_state.timestamp = current_time
        
        # 更新步态计划（滚动窗口）
        gait_window = extract_gait_window(gait_plan, current_time, mpc_solver.N * dt)
        
        # MPC求解
        result = mpc_solver.solve(current_state, gait_window)
        
        if result.success:
            # 应用控制命令
            apply_control_commands(lipm_model, result)
            
            # 记录数据
            time_history.append(current_time)
            com_trajectory.append(current_state.com_position)
            
            # 记录当前时刻的接触力
            current_forces = {
                'left': result.current_contact_forces['left'],
                'right': result.current_contact_forces['right']
            }
            force_trajectory.append(current_forces)
            
            # 记录ZMP
            if result.zmp_trajectory:
                zmp_trajectory.append(result.zmp_trajectory[0])
            else:
                zmp_trajectory.append(Vector3D(x=0.0, y=0.0, z=0.0))
                
            solve_time_history.append(result.solve_time)
            cost_history.append(result.cost)
            
            # 更新系统状态
            lipm_model.update(dt)
            
            if step % 10 == 0:
                print(f"步骤 {step:3d}: t={current_time:.2f}s, "
                      f"COM=({current_state.com_position.x:.3f}, {current_state.com_position.y:.3f}), "
                      f"求解时间={result.solve_time:.3f}s")
        else:
            print(f"⚠️  步骤 {step}: MPC求解失败")
            # 使用后备控制
            apply_fallback_control(lipm_model)
            lipm_model.update(dt)
    
    # 性能统计
    stats = mpc_solver.get_solver_statistics()
    print(f"\n实时控制性能统计:")
    print(f"  平均求解时间: {stats['average_solve_time']:.3f}s ({stats['average_solve_time']*1000:.1f}ms)")
    print(f"  最大求解时间: {stats['max_solve_time']:.3f}s")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  求解器: {stats['solver_type']}")
    
    # 绘制结果
    plot_real_time_results(time_history, com_trajectory, force_trajectory, 
                          zmp_trajectory, solve_time_history, cost_history)
    
    return mpc_solver, {
        'time': time_history,
        'com_trajectory': com_trajectory, 
        'force_trajectory': force_trajectory,
        'solve_times': solve_time_history
    }


def extract_gait_window(full_gait_plan: GaitPlan, current_time: float, window_duration: float) -> GaitPlan:
    """从完整步态计划中提取时间窗口"""
    dt = 0.08  # 假设时间步长
    start_idx = max(0, int(current_time / dt))
    window_steps = int(window_duration / dt)
    end_idx = min(len(full_gait_plan.support_sequence), start_idx + window_steps)
    
    windowed_plan = GaitPlan()
    
    # 提取支撑序列
    windowed_plan.support_sequence = full_gait_plan.support_sequence[start_idx:end_idx]
    
    # 提取接触调度
    if len(full_gait_plan.contact_schedule) > 0:
        windowed_plan.contact_schedule = full_gait_plan.contact_schedule[start_idx:end_idx, :]
    
    # 提取支撑位置
    windowed_plan.support_positions = full_gait_plan.support_positions[start_idx:end_idx]
    
    # 继承约束参数
    windowed_plan.min_contact_duration = full_gait_plan.min_contact_duration
    windowed_plan.max_step_length = full_gait_plan.max_step_length
    windowed_plan.step_height = full_gait_plan.step_height
    
    return windowed_plan


def apply_control_commands(lipm_model: SimplifiedDynamicsModel, result: MPCResult):
    """应用MPC控制命令到LIPM模型"""
    # 设置期望的质心加速度
    desired_acc = result.current_desired_com_acceleration
    
    # 计算等效的足底力（简化处理）
    total_force_x = desired_acc.x * lipm_model.parameters.total_mass
    total_force_y = desired_acc.y * lipm_model.parameters.total_mass
    
    # 将总力分配到接触足（简化为平均分配）
    left_force = result.current_contact_forces['left']
    right_force = result.current_contact_forces['right']
    
    # 使用优化得到的接触力
    equivalent_force = Vector3D(
        x=(left_force.x + right_force.x),
        y=(left_force.y + right_force.y),
        z=0.0
    )
    
    lipm_model.set_foot_force_command(equivalent_force)


def apply_fallback_control(lipm_model: SimplifiedDynamicsModel):
    """应用后备控制"""
    # 简单的平衡控制
    fallback_force = Vector3D(x=0.0, y=0.0, z=0.0)
    lipm_model.set_foot_force_command(fallback_force)


def plot_real_time_results(times, com_trajectory, force_trajectory, zmp_trajectory, 
                         solve_times, costs):
    """绘制实时控制结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 质心轨迹
    com_x = [pos.x for pos in com_trajectory]
    com_y = [pos.y for pos in com_trajectory]
    
    axes[0, 0].plot(times, com_x, 'b-', linewidth=2, label='COM X')
    axes[0, 0].plot(times, com_y, 'r-', linewidth=2, label='COM Y')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('位置 (m)')
    axes[0, 0].set_title('质心轨迹')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2D轨迹图
    axes[0, 1].plot(com_x, com_y, 'b-', linewidth=3, label='COM轨迹')
    axes[0, 1].scatter(com_x[0], com_y[0], c='green', s=100, marker='o', label='起点')
    axes[0, 1].scatter(com_x[-1], com_y[-1], c='red', s=100, marker='s', label='终点')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_title('质心2D轨迹')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].axis('equal')
    
    # 接触力
    left_forces_x = [f['left'].x for f in force_trajectory]
    left_forces_y = [f['left'].y for f in force_trajectory]
    left_forces_z = [f['left'].z for f in force_trajectory]
    
    right_forces_x = [f['right'].x for f in force_trajectory]
    right_forces_y = [f['right'].y for f in force_trajectory]
    right_forces_z = [f['right'].z for f in force_trajectory]
    
    axes[0, 2].plot(times, left_forces_z, 'b-', linewidth=2, label='左脚 Fz')
    axes[0, 2].plot(times, right_forces_z, 'r-', linewidth=2, label='右脚 Fz')
    axes[0, 2].set_xlabel('时间 (s)')
    axes[0, 2].set_ylabel('垂直力 (N)')
    axes[0, 2].set_title('垂直接触力')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 水平接触力
    axes[1, 0].plot(times, left_forces_x, 'b-', linewidth=2, label='左脚 Fx')
    axes[1, 0].plot(times, right_forces_x, 'r-', linewidth=2, label='右脚 Fx')
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('水平力 (N)')
    axes[1, 0].set_title('前向接触力')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # ZMP轨迹
    if zmp_trajectory:
        zmp_x = [zmp.x for zmp in zmp_trajectory]
        zmp_y = [zmp.y for zmp in zmp_trajectory]
        
        axes[1, 1].plot(times, zmp_x, 'g-', linewidth=2, label='ZMP X')
        axes[1, 1].plot(times, zmp_y, 'm-', linewidth=2, label='ZMP Y')
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('ZMP位置 (m)')
        axes[1, 1].set_title('ZMP轨迹')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # 求解性能
    solve_times_ms = [t * 1000 for t in solve_times]  # 转换为毫秒
    
    axes[1, 2].plot(times, solve_times_ms, 'k-', linewidth=2)
    axes[1, 2].axhline(y=np.mean(solve_times_ms), color='r', linestyle='--', 
                       label=f'平均: {np.mean(solve_times_ms):.1f}ms')
    axes[1, 2].set_xlabel('时间 (s)')
    axes[1, 2].set_ylabel('求解时间 (ms)')
    axes[1, 2].set_title('MPC求解性能')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.suptitle('专业MPC控制器实时性能', fontsize=16)
    plt.tight_layout()
    plt.savefig('professional_mpc_results.png', dpi=300, bbox_inches='tight')
    print("专业MPC结果图已保存: professional_mpc_results.png")


def main():
    """主函数"""
    print("专业MPC求解器演示程序")
    print("=" * 60)
    
    try:
        # 1. OSQP求解器演示
        osqp_solver, osqp_result, gait_plan = demo_osqp_solver()
        
        # 2. 多求解器性能比较
        comparison_results = demo_multi_solver_comparison()
        
        # 3. 实时控制循环演示
        realtime_solver, realtime_data = demo_real_time_control_loop()
        
        print("\n" + "=" * 60)
        print("🎉 所有专业MPC演示完成!")
        
        print(f"\n📊 性能总结:")
        print(f"   OSQP求解器成功率: {'✅' if osqp_result.success else '❌'}")
        
        if comparison_results:
            best_solver = min(comparison_results.items(), 
                            key=lambda x: x[1]['avg_time'] if x[1]['success_rate'] > 0 else float('inf'))
            print(f"   最佳求解器: {best_solver[0]} ({best_solver[1]['avg_time']*1000:.1f}ms)")
        
        if realtime_data['solve_times']:
            avg_realtime = np.mean(realtime_data['solve_times']) * 1000
            print(f"   实时控制平均求解时间: {avg_realtime:.1f}ms")
            print(f"   实时控制成功率: {len([t for t in realtime_data['solve_times'] if t > 0]) / len(realtime_data['solve_times']):.1%}")
        
        print(f"\n📁 生成文件:")
        print(f"   - professional_mpc_results.png: 控制结果可视化")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 