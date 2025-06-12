#!/usr/bin/env python3
"""
FootTrajectory类演示程序

展示如何使用FootTrajectory类进行实时轨迹生成：
- 数据总线集成
- 主循环更新
- 步态调度器协调
- 触地检测和重置
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import matplotlib.pyplot as plt
from gait_core.foot_trajectory import (
    FootTrajectory, 
    QuadrupedTrajectoryManager, 
    TrajectoryConfig,
    DataBusInterface
)


class MockGaitScheduler:
    """模拟步态调度器"""
    
    def __init__(self):
        self.swing_completed_callbacks = []
        self.current_gait = "trot"
        self.step_length = 0.10  # 10cm步长
        
    def on_swing_completed(self, foot_name: str):
        """摆动完成回调"""
        print(f"Gait Scheduler: {foot_name} swing completed")
        self.swing_completed_callbacks.append((foot_name, time.time()))
    
    def get_next_target(self, foot_name: str, current_pos: np.ndarray) -> np.ndarray:
        """获取下一个目标位置"""
        # 简单前进步态
        return current_pos + np.array([self.step_length, 0, 0])


def demo_single_foot_trajectory():
    """演示单个足部轨迹"""
    print("=" * 60)
    print("单个足部轨迹演示")
    print("=" * 60)
    
    # 创建轨迹配置
    config = TrajectoryConfig(
        step_height=0.08,
        swing_duration=0.4,
        interpolation_type="cubic",
        vertical_trajectory_type="sine"
    )
    
    # 创建足部轨迹生成器
    foot_traj = FootTrajectory("RF", config)
    
    # 创建数据总线
    data_bus = DataBusInterface()
    foot_traj.connect_data_bus(data_bus)
    
    # 设置轨迹
    start_pos = np.array([0.15, -0.15, 0.0])
    target_pos = np.array([0.25, -0.15, 0.0])
    foot_traj.start_swing(start_pos, target_pos)
    
    print(f"起始位置: {start_pos}")
    print(f"目标位置: {target_pos}")
    print(f"步长: {np.linalg.norm(target_pos - start_pos):.3f}m")
    print()
    
    # 模拟主控制循环
    print("主控制循环 (100Hz, 显示每50ms):")
    print("-" * 50)
    print(f"{'时间[ms]':<10} {'相位':<8} {'位置[m]':<25} {'状态':<12}")
    print("-" * 50)
    
    dt = 0.01  # 100Hz控制频率
    positions_log = []
    times_log = []
    
    loop_count = 0
    while foot_traj.is_active():
        # 主循环更新
        current_pos = foot_traj.update(dt)
        positions_log.append(current_pos.copy())
        times_log.append(loop_count * dt)
        
        # 显示进度（每50ms）
        if loop_count % 5 == 0:
            progress = foot_traj.get_progress_info()
            pos_str = f"({current_pos[0]:.3f},{current_pos[1]:.3f},{current_pos[2]:.3f})"
            print(f"{loop_count*10:<10} {progress['phase']:.3f}    {pos_str:<25} {progress['state']}")
        
        loop_count += 1
        
        # 安全退出
        if loop_count > 1000:
            break
    
    print("-" * 50)
    
    # 最终统计
    final_stats = foot_traj.get_progress_info()
    print(f"\n最终统计:")
    print(f"  总轨迹数: {final_stats['trajectory_count']}")
    print(f"  总距离: {final_stats['total_distance']:.3f}m")
    print(f"  最大高度: {final_stats['max_height_achieved']:.3f}m")
    print(f"  实际时长: {final_stats['elapsed_time']:.3f}s")
    
    return times_log, positions_log


def demo_quadruped_coordination():
    """演示四足协调控制"""
    print("\n" + "=" * 60)
    print("四足协调控制演示 - Trot步态")
    print("=" * 60)
    
    # 创建轨迹管理器
    config = TrajectoryConfig(
        step_height=0.06,
        swing_duration=0.3,
        interpolation_type="cubic",
        vertical_trajectory_type="sine"
    )
    traj_manager = QuadrupedTrajectoryManager(config)
    
    # 创建并连接数据总线和步态调度器
    data_bus = DataBusInterface()
    gait_scheduler = MockGaitScheduler()
    
    traj_manager.connect_data_bus(data_bus)
    traj_manager.connect_gait_scheduler(gait_scheduler)
    
    # 四足初始位置
    foot_positions = {
        "RF": np.array([0.15, -0.15, 0.0]),
        "LF": np.array([0.15, 0.15, 0.0]),
        "RH": np.array([-0.15, -0.15, 0.0]),
        "LH": np.array([-0.15, 0.15, 0.0])
    }
    
    print("初始足部位置:")
    for name, pos in foot_positions.items():
        print(f"  {name}: {pos}")
    print()
    
    # Trot步态模拟：对角足同时摆动
    print("开始Trot步态演示...")
    
    # 第一阶段：RF + LH 摆动
    print("\n第一阶段 - RF和LH同时摆动:")
    step_length = 0.08
    
    rf_target = foot_positions["RF"] + np.array([step_length, 0, 0])
    lh_target = foot_positions["LH"] + np.array([step_length, 0, 0])
    
    traj_manager.start_swing("RF", foot_positions["RF"], rf_target)
    traj_manager.start_swing("LH", foot_positions["LH"], lh_target)
    
    # 运行第一阶段
    dt = 0.01
    phase1_steps = int(0.3 / dt)  # 300ms
    
    for i in range(phase1_steps):
        positions = traj_manager.update_all(dt)
        
        # 每100ms显示进度
        if i % 10 == 0:
            progress = traj_manager.get_all_progress()
            rf_phase = progress["RF"]["phase"]
            lh_phase = progress["LH"]["phase"]
            print(f"  t={i*10:3d}ms: RF phase={rf_phase:.2f}, LH phase={lh_phase:.2f}")
        
        # 检查是否完成
        if (not traj_manager.foot_trajectories["RF"].is_active() and 
            not traj_manager.foot_trajectories["LH"].is_active()):
            break
    
    # 更新足部位置
    foot_positions["RF"] = rf_target
    foot_positions["LH"] = lh_target
    
    print(f"第一阶段完成，回调次数: {len(gait_scheduler.swing_completed_callbacks)}")
    
    # 第二阶段：LF + RH 摆动
    print("\n第二阶段 - LF和RH同时摆动:")
    
    lf_target = foot_positions["LF"] + np.array([step_length, 0, 0])
    rh_target = foot_positions["RH"] + np.array([step_length, 0, 0])
    
    traj_manager.start_swing("LF", foot_positions["LF"], lf_target)
    traj_manager.start_swing("RH", foot_positions["RH"], rh_target)
    
    # 运行第二阶段
    for i in range(phase1_steps):
        positions = traj_manager.update_all(dt)
        
        if i % 10 == 0:
            progress = traj_manager.get_all_progress()
            lf_phase = progress["LF"]["phase"]
            rh_phase = progress["RH"]["phase"]
            print(f"  t={i*10:3d}ms: LF phase={lf_phase:.2f}, RH phase={rh_phase:.2f}")
        
        if (not traj_manager.foot_trajectories["LF"].is_active() and 
            not traj_manager.foot_trajectories["RH"].is_active()):
            break
    
    foot_positions["LF"] = lf_target
    foot_positions["RH"] = rh_target
    
    print(f"第二阶段完成，总回调次数: {len(gait_scheduler.swing_completed_callbacks)}")
    
    print("\n最终足部位置:")
    for name, pos in foot_positions.items():
        print(f"  {name}: {pos}")
    
    print(f"\n✓ 一个完整Trot步态周期完成，机器人前进 {step_length:.2f}m")


def demo_terrain_adaptation():
    """演示地形自适应"""
    print("\n" + "=" * 60)
    print("地形自适应演示")
    print("=" * 60)
    
    foot_traj = FootTrajectory("RF")
    
    # 不同地形类型
    terrains = ["flat", "grass", "rough", "stairs"]
    start_pos = np.array([0.0, -0.15, 0.0])
    target_pos = np.array([0.12, -0.15, 0.0])
    
    print("地形自适应轨迹对比:")
    print("-" * 40)
    
    terrain_results = {}
    
    for terrain in terrains:
        print(f"\n{terrain.upper()} 地形:")
        
        # 设置地形参数
        foot_traj.set_terrain_adaptive_params(terrain)
        
        # 开始摆动
        foot_traj.start_swing(start_pos, target_pos)
        
        # 计算关键点
        key_phases = [0.0, 0.25, 0.5, 0.75, 1.0]
        max_height = 0.0
        
        for phase in key_phases:
            foot_traj.phase = phase
            pos = foot_traj.current_position = foot_traj.config.__dict__.copy()
            # 手动计算位置以显示
            from gait_core.swing_trajectory import get_swing_foot_position
            pos = get_swing_foot_position(
                phase, start_pos, target_pos,
                foot_traj.config.step_height,
                foot_traj.config.interpolation_type,
                foot_traj.config.vertical_trajectory_type
            )
            if pos[2] > max_height:
                max_height = pos[2]
            
            if phase in [0.0, 0.5, 1.0]:
                print(f"  相位 {phase:.1f}: Z高度 = {pos[2]:.3f}m")
        
        terrain_results[terrain] = {
            'max_height': max_height,
            'step_height': foot_traj.config.step_height,
            'interpolation': foot_traj.config.interpolation_type,
            'vertical': foot_traj.config.vertical_trajectory_type
        }
        
        foot_traj.reset()
    
    # 对比总结
    print("\n地形参数对比:")
    print("-" * 50)
    print(f"{'地形':<10} {'抬脚高度':<10} {'最大高度':<10} {'插值类型':<10}")
    print("-" * 50)
    for terrain, data in terrain_results.items():
        print(f"{terrain:<10} {data['step_height']:<10.2f} {data['max_height']:<10.3f} {data['interpolation']:<10}")


def demo_data_bus_integration():
    """演示数据总线集成"""
    print("\n" + "=" * 60)
    print("数据总线集成演示")
    print("=" * 60)
    
    # 创建组件
    foot_traj = FootTrajectory("RF")
    data_bus = DataBusInterface()
    foot_traj.connect_data_bus(data_bus)
    
    # 初始轨迹
    start_pos = np.array([0.0, -0.15, 0.0])
    initial_target = np.array([0.10, -0.15, 0.0])
    foot_traj.start_swing(start_pos, initial_target)
    
    print(f"初始目标位置: {initial_target}")
    
    # 模拟主循环中的目标更新
    dt = 0.01
    update_count = 0
    
    print("\n模拟目标位置动态更新:")
    print("-" * 40)
    
    while foot_traj.is_active() and update_count < 20:
        # 模拟每200ms更新一次目标
        if update_count == 10:  # 在100ms时更新目标
            new_target = np.array([0.15, -0.10, 0.0])  # 改变Y坐标
            data_bus.set_foot_target("RF", new_target)
            print(f"t={update_count*10}ms: 数据总线更新目标位置到 {new_target}")
        
        # 从数据总线更新
        current_pos = foot_traj.update_from_data_bus(dt)
        
        if update_count % 5 == 0:
            progress = foot_traj.get_progress_info()
            print(f"t={update_count*10:3d}ms: phase={progress['phase']:.2f}, "
                  f"pos=({current_pos[0]:.3f},{current_pos[1]:.3f},{current_pos[2]:.3f})")
        
        update_count += 1
    
    print(f"\n最终位置: {foot_traj.current_position}")
    print("✓ 数据总线集成测试完成")


def visualize_trajectory_comparison(times_log, positions_log):
    """可视化轨迹对比"""
    if not positions_log:
        return
    
    positions = np.array(positions_log)
    times = np.array(times_log)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('FootTrajectory 轨迹分析', fontsize=14)
    
    # X-Y轨迹
    ax1 = axes[0, 0]
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='起点')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='终点')
    ax1.set_title('XY平面轨迹')
    ax1.set_xlabel('X位置 [m]')
    ax1.set_ylabel('Y位置 [m]')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # 高度随时间变化
    ax2 = axes[0, 1]
    ax2.plot(times, positions[:, 2], 'r-', linewidth=2)
    ax2.set_title('高度随时间变化')
    ax2.set_xlabel('时间 [s]')
    ax2.set_ylabel('Z高度 [m]')
    ax2.grid(True, alpha=0.3)
    
    # 3D轨迹
    ax3 = axes[1, 0]
    ax3.remove()
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax3.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                color='green', s=100, label='起点')
    ax3.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                color='red', s=100, label='终点')
    ax3.set_title('3D轨迹')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_zlabel('Z [m]')
    ax3.legend()
    
    # 速度分析
    ax4 = axes[1, 1]
    if len(positions) > 1:
        velocities = np.diff(positions, axis=0) / np.diff(times)[:, np.newaxis]
        vel_magnitude = np.linalg.norm(velocities, axis=1)
        ax4.plot(times[1:], vel_magnitude, 'g-', linewidth=2)
        ax4.set_title('速度大小随时间变化')
        ax4.set_xlabel('时间 [s]')
        ax4.set_ylabel('速度 [m/s]')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """主演示函数"""
    print("FootTrajectory 轨迹生成模块演示")
    print("展示数据总线集成、实时更新和协调控制")
    print()
    
    try:
        # 单个足部轨迹演示
        times_log, positions_log = demo_single_foot_trajectory()
        
        # 四足协调控制演示
        demo_quadruped_coordination()
        
        # 地形自适应演示
        demo_terrain_adaptation()
        
        # 数据总线集成演示
        demo_data_bus_integration()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        
        print("\nFootTrajectory类的核心特性:")
        print("✓ 实时轨迹更新：支持foot_traj.update(dt)调用")
        print("✓ 自动相位管理：内部维护phase = elapsed_time / tSwing")
        print("✓ 数据总线集成：从总线获取起始和目标位置")
        print("✓ 状态监控：检测完成、重置和停止条件")
        print("✓ 步态调度器协调：轨迹完成时自动通知")
        print("✓ 地形自适应：根据环境自动调整参数")
        
        print("\n推荐使用模式:")
        print("• 主循环集成：while robot.running: positions = traj_manager.update_all(dt)")
        print("• 数据总线获取：start_pos = data_bus.get_current_position(foot)")
        print("• 步态调度：target_pos = gait_scheduler.get_next_target(foot)")
        print("• 完成检测：if foot_traj.is_completed(): start_new_trajectory()")
        
        # 可视化
        print(f"\n是否显示轨迹可视化? (matplotlib)")
        try:
            visualize_trajectory_comparison(times_log, positions_log)
        except Exception as e:
            print(f"可视化跳过: {e}")
            
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 