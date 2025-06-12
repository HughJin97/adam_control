#!/usr/bin/env python3
"""
MuJoCo边界情况处理演示程序

展示FootTrajectory类在MuJoCo环境中的边界情况处理：
- 提前落地检测和轨迹中止
- 地面穿透防护
- 触觉传感器集成
- 步态调度器协调
- AzureLoong机器人集成
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# 导入轨迹生成模块
from gait_core.foot_trajectory import (
    FootTrajectory, 
    QuadrupedTrajectoryManager, 
    TrajectoryConfig,
    DataBusInterface,
    GroundContactEvent,
    TrajectoryState
)

# 模拟MuJoCo环境（如果没有安装MuJoCo）
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("MuJoCo not available, using simulation mode")


class MockMuJoCoModel:
    """模拟MuJoCo模型（用于演示）"""
    
    def __init__(self):
        self.nsensor = 2
        self.sensors = [
            type('Sensor', (), {'name': 'rf-touch'}),
            type('Sensor', (), {'name': 'lf-touch'})
        ]
    
    def sensor(self, i):
        return self.sensors[i]


class MockMuJoCoData:
    """模拟MuJoCo数据（用于演示）"""
    
    def __init__(self):
        self.sensordata = np.array([0.0, 0.0])  # 两个触觉传感器
        self.time = 0.0
        self.contact_forces = {}  # 模拟接触力
    
    def update_sensor_data(self, rf_force: float, lf_force: float):
        """更新传感器数据"""
        self.sensordata[0] = rf_force  # rf-touch
        self.sensordata[1] = lf_force  # lf-touch


class EnhancedGaitScheduler:
    """增强版步态调度器 - 支持边界情况处理"""
    
    def __init__(self):
        self.swing_completed_callbacks = []
        self.trajectory_interrupted_callbacks = []
        self.emergency_stop_callbacks = []
        self.current_gait = "trot"
        self.step_length = 0.10
        
    def on_swing_completed(self, foot_name: str):
        """摆动完成回调"""
        callback_info = {
            "foot_name": foot_name,
            "event": "swing_completed",
            "time": time.time()
        }
        self.swing_completed_callbacks.append(callback_info)
        print(f"Gait Scheduler: {foot_name} swing completed normally")
    
    def on_trajectory_interrupted(self, foot_name: str, contact_info):
        """轨迹中断回调"""
        callback_info = {
            "foot_name": foot_name,
            "event": "trajectory_interrupted",
            "contact_info": contact_info,
            "time": time.time()
        }
        self.trajectory_interrupted_callbacks.append(callback_info)
        print(f"Gait Scheduler: {foot_name} trajectory interrupted - {contact_info.event_type.value}")
        
        # 根据中断类型采取不同策略
        if contact_info.event_type == GroundContactEvent.EARLY_CONTACT:
            print(f"  -> Early contact detected at phase {contact_info.contact_phase:.3f}")
            print(f"  -> Adjusting gait timing for {foot_name}")
        elif contact_info.event_type == GroundContactEvent.SENSOR_CONTACT:
            print(f"  -> Sensor contact detected, force: {contact_info.contact_force:.1f}N")
    
    def on_emergency_stop(self, foot_name: str, data: Dict[str, Any]):
        """紧急停止回调"""
        callback_info = {
            "foot_name": foot_name,
            "event": "emergency_stop",
            "reason": data.get("reason", "Unknown"),
            "time": time.time()
        }
        self.emergency_stop_callbacks.append(callback_info)
        print(f"Gait Scheduler: EMERGENCY STOP for {foot_name} - {data.get('reason')}")


class TerrainSimulator:
    """地形模拟器"""
    
    def __init__(self):
        self.terrain_type = "flat"
        self.obstacles = []
        self.ground_height_map = {}
    
    def add_obstacle(self, position: np.ndarray, size: np.ndarray):
        """添加障碍物"""
        self.obstacles.append({
            "position": position,
            "size": size,
            "height": position[2] + size[2]
        })
        print(f"Added obstacle at {position} with size {size}")
    
    def get_ground_height(self, xy_pos: np.ndarray) -> float:
        """获取地面高度"""
        # 检查是否在障碍物上
        for obstacle in self.obstacles:
            obs_pos = obstacle["position"]
            obs_size = obstacle["size"]
            
            if (abs(xy_pos[0] - obs_pos[0]) < obs_size[0] and 
                abs(xy_pos[1] - obs_pos[1]) < obs_size[1]):
                return obstacle["height"]
        
        return 0.0  # 默认地面高度
    
    def simulate_contact_force(self, foot_pos: np.ndarray, foot_name: str) -> float:
        """模拟接触力"""
        ground_height = self.get_ground_height(foot_pos[:2])
        
        if foot_pos[2] <= ground_height + 0.01:  # 接触阈值
            # 模拟接触力（基于穿透深度）
            penetration = max(0, ground_height - foot_pos[2])
            contact_force = penetration * 1000  # 简单的弹簧模型
            return contact_force
        
        return 0.0


def demo_early_contact_detection():
    """演示提前接触检测"""
    print("=" * 60)
    print("提前接触检测演示")
    print("=" * 60)
    
    # 创建配置
    config = TrajectoryConfig(
        step_height=0.08,
        swing_duration=0.4,
        early_contact_phase_threshold=0.6,  # 60%相位前为提前接触
        contact_force_threshold=15.0,
        enable_sensor_feedback=True
    )
    
    # 创建足部轨迹和MuJoCo模拟
    foot_traj = FootTrajectory("RF", config)
    mock_model = MockMuJoCoModel()
    mock_data = MockMuJoCoData()
    foot_traj.connect_mujoco(mock_model, mock_data)
    
    # 创建步态调度器
    gait_scheduler = EnhancedGaitScheduler()
    foot_traj.gait_scheduler = gait_scheduler
    
    # 创建地形模拟器
    terrain = TerrainSimulator()
    terrain.add_obstacle(np.array([0.18, -0.15, 0.0]), np.array([0.05, 0.05, 0.03]))  # 3cm高障碍物
    
    # 开始摆动
    start_pos = np.array([0.15, -0.15, 0.0])
    target_pos = np.array([0.25, -0.15, 0.0])
    foot_traj.start_swing(start_pos, target_pos)
    
    print(f"开始摆动: {start_pos} -> {target_pos}")
    print(f"障碍物位置: [0.18, -0.15, 0.03] (高度3cm)")
    print()
    
    # 模拟主循环
    dt = 0.01
    positions_log = []
    forces_log = []
    states_log = []
    
    print("主循环进度:")
    print("-" * 50)
    print(f"{'时间[ms]':<8} {'相位':<6} {'Z高度[cm]':<10} {'接触力[N]':<10} {'状态':<12}")
    print("-" * 50)
    
    loop_count = 0
    while foot_traj.is_active() and loop_count < 100:
        # 更新轨迹
        current_pos = foot_traj.update(dt)
        
        # 模拟传感器数据
        contact_force = terrain.simulate_contact_force(current_pos, "RF")
        mock_data.update_sensor_data(contact_force, 0.0)
        
        # 记录数据
        positions_log.append(current_pos.copy())
        forces_log.append(contact_force)
        states_log.append(foot_traj.state.value)
        
        # 显示进度（每50ms）
        if loop_count % 5 == 0:
            print(f"{loop_count*10:<8} {foot_traj.phase:.3f}  {current_pos[2]*100:.1f}       "
                  f"{contact_force:.1f}        {foot_traj.state.value}")
        
        loop_count += 1
    
    print("-" * 50)
    
    # 分析结果
    final_state = foot_traj.state
    interruption_count = len(gait_scheduler.trajectory_interrupted_callbacks)
    
    print(f"\n结果分析:")
    print(f"  最终状态: {final_state.value}")
    print(f"  中断次数: {interruption_count}")
    print(f"  最终位置: {foot_traj.current_position}")
    
    if interruption_count > 0:
        interrupt_info = gait_scheduler.trajectory_interrupted_callbacks[0]
        contact_info = interrupt_info["contact_info"]
        print(f"  中断类型: {contact_info.event_type.value}")
        print(f"  中断相位: {contact_info.contact_phase:.3f}")
        print(f"  接触力: {contact_info.contact_force:.1f}N")
        print("  ✓ 提前接触成功检测并处理")
    else:
        print("  ✗ 未检测到提前接触")
    
    return positions_log, forces_log, states_log


def demo_ground_penetration_protection():
    """演示地面穿透防护"""
    print("\n" + "=" * 60)
    print("地面穿透防护演示")
    print("=" * 60)
    
    # 创建配置
    config = TrajectoryConfig(
        step_height=0.06,
        swing_duration=0.3,
        enable_penetration_protection=True,
        ground_penetration_limit=0.002  # 2mm穿透限制
    )
    
    # 创建足部轨迹
    foot_traj = FootTrajectory("LF", config)
    
    # 创建数据总线（模拟不平整地面）
    class UnEvenDataBus(DataBusInterface):
        def get_ground_height(self, position_xy: np.ndarray) -> Optional[float]:
            # 模拟不平整地面：在x=0.2处有一个凸起
            if 0.18 <= position_xy[0] <= 0.22:
                return 0.02  # 2cm凸起
            return 0.0
    
    data_bus = UnEvenDataBus()
    foot_traj.connect_data_bus(data_bus)
    
    # 开始摆动（目标位置会经过凸起）
    start_pos = np.array([0.15, 0.15, 0.0])
    target_pos = np.array([0.25, 0.15, 0.0])
    foot_traj.start_swing(start_pos, target_pos)
    
    print(f"开始摆动: {start_pos} -> {target_pos}")
    print(f"地面凸起: x=0.18-0.22, 高度=2cm")
    print()
    
    # 模拟主循环
    dt = 0.01
    positions_log = []
    ground_heights_log = []
    violations_log = []
    
    print("穿透防护监控:")
    print("-" * 60)
    print(f"{'时间[ms]':<8} {'X位置[m]':<10} {'Z高度[cm]':<10} {'地面[cm]':<8} {'防护':<8}")
    print("-" * 60)
    
    loop_count = 0
    while foot_traj.is_active() and loop_count < 50:
        current_pos = foot_traj.update(dt)
        ground_height = data_bus.get_ground_height(current_pos[:2])
        
        positions_log.append(current_pos.copy())
        ground_heights_log.append(ground_height)
        violations_log.append(len(foot_traj.safety_violations))
        
        # 显示进度
        if loop_count % 3 == 0:
            protection_status = "✓" if len(foot_traj.safety_violations) > violations_log[max(0, loop_count-3)] else " "
            print(f"{loop_count*10:<8} {current_pos[0]:.3f}      {current_pos[2]*100:.1f}       "
                  f"{ground_height*100:.1f}      {protection_status}")
        
        loop_count += 1
    
    print("-" * 60)
    
    # 分析结果
    total_violations = len(foot_traj.safety_violations)
    print(f"\n穿透防护结果:")
    print(f"  安全违规次数: {total_violations}")
    print(f"  最终位置: {foot_traj.current_position}")
    
    if total_violations > 0:
        print(f"  违规记录: {foot_traj.safety_violations}")
        print("  ✓ 地面穿透防护正常工作")
    else:
        print("  ✓ 无穿透发生，轨迹安全")
    
    return positions_log, ground_heights_log


def demo_emergency_stop_scenarios():
    """演示紧急停止场景"""
    print("\n" + "=" * 60)
    print("紧急停止场景演示")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "传感器异常",
            "config": TrajectoryConfig(enable_sensor_feedback=True),
            "trigger": "sensor_anomaly"
        },
        {
            "name": "轨迹偏差过大", 
            "config": TrajectoryConfig(max_trajectory_deviation=0.02),
            "trigger": "trajectory_deviation"
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n场景 {i+1}: {scenario['name']}")
        print("-" * 30)
        
        # 创建足部轨迹
        foot_traj = FootTrajectory("RH", scenario["config"])
        mock_model = MockMuJoCoModel()
        mock_data = MockMuJoCoData()
        foot_traj.connect_mujoco(mock_model, mock_data)
        
        # 创建步态调度器
        gait_scheduler = EnhancedGaitScheduler()
        foot_traj.gait_scheduler = gait_scheduler
        
        # 开始摆动
        start_pos = np.array([-0.15, -0.15, 0.0])
        target_pos = np.array([-0.05, -0.15, 0.0])
        foot_traj.start_swing(start_pos, target_pos)
        
        # 模拟异常情况
        dt = 0.01
        loop_count = 0
        emergency_triggered = False
        
        while foot_traj.is_active() and loop_count < 30:
            # 在特定时间触发异常
            if loop_count == 15 and not emergency_triggered:
                if scenario["trigger"] == "sensor_anomaly":
                    # 模拟传感器异常（过大读数）
                    mock_data.update_sensor_data(1500.0, 0.0)  # 异常大的力值
                    print(f"  t={loop_count*10}ms: 触发传感器异常 (1500N)")
                elif scenario["trigger"] == "trajectory_deviation":
                    # 模拟轨迹偏差（强制改变位置）
                    foot_traj.current_position += np.array([0.05, 0, 0])  # 5cm偏差
                    print(f"  t={loop_count*10}ms: 触发轨迹偏差 (5cm)")
                
                emergency_triggered = True
            
            current_pos = foot_traj.update(dt)
            
            if foot_traj.is_emergency_stopped():
                print(f"  t={loop_count*10}ms: 紧急停止触发")
                break
            
            loop_count += 1
        
        # 检查结果
        emergency_count = len(gait_scheduler.emergency_stop_callbacks)
        if emergency_count > 0:
            emergency_info = gait_scheduler.emergency_stop_callbacks[0]
            print(f"  结果: ✓ 紧急停止成功 - {emergency_info['reason']}")
        else:
            print(f"  结果: ✗ 紧急停止未触发")


def demo_mujoco_integration():
    """演示完整MuJoCo集成"""
    print("\n" + "=" * 60)
    print("完整MuJoCo集成演示")
    print("=" * 60)
    
    # 创建轨迹管理器
    config = TrajectoryConfig(
        step_height=0.08,
        swing_duration=0.35,
        enable_sensor_feedback=True,
        enable_penetration_protection=True,
        contact_force_threshold=20.0
    )
    
    traj_manager = QuadrupedTrajectoryManager(config)
    
    # 模拟MuJoCo环境
    mock_model = MockMuJoCoModel()
    mock_data = MockMuJoCoData()
    
    # 连接MuJoCo到所有足部轨迹
    for foot_name in ["RF", "LF"]:  # AzureLoong的两只脚
        traj_manager.foot_trajectories[foot_name].connect_mujoco(mock_model, mock_data)
    
    # 创建增强步态调度器
    gait_scheduler = EnhancedGaitScheduler()
    traj_manager.connect_gait_scheduler(gait_scheduler)
    
    # 创建地形
    terrain = TerrainSimulator()
    terrain.add_obstacle(np.array([0.20, -0.15, 0.0]), np.array([0.03, 0.03, 0.025]))  # RF路径上的障碍
    terrain.add_obstacle(np.array([0.22, 0.15, 0.0]), np.array([0.03, 0.03, 0.015]))   # LF路径上的障碍
    
    # 初始足部位置
    foot_positions = {
        "RF": np.array([0.15, -0.15, 0.0]),
        "LF": np.array([0.15, 0.15, 0.0])
    }
    
    print("AzureLoong机器人双足协调摆动:")
    print(f"RF障碍物: [0.20, -0.15, 0.025] (2.5cm高)")
    print(f"LF障碍物: [0.22, 0.15, 0.015] (1.5cm高)")
    print()
    
    # 开始双足摆动
    rf_target = foot_positions["RF"] + np.array([0.10, 0, 0])
    lf_target = foot_positions["LF"] + np.array([0.10, 0, 0])
    
    traj_manager.start_swing("RF", foot_positions["RF"], rf_target)
    traj_manager.start_swing("LF", foot_positions["LF"], lf_target)
    
    # 主控制循环
    dt = 0.01
    loop_count = 0
    
    print("双足协调控制:")
    print("-" * 70)
    print(f"{'时间[ms]':<8} {'RF状态':<12} {'LF状态':<12} {'RF接触力':<10} {'LF接触力':<10}")
    print("-" * 70)
    
    while (traj_manager.foot_trajectories["RF"].is_active() or 
           traj_manager.foot_trajectories["LF"].is_active()) and loop_count < 60:
        
        # 更新所有轨迹
        positions = traj_manager.update_all(dt)
        
        # 模拟传感器数据
        rf_force = terrain.simulate_contact_force(positions["RF"], "RF")
        lf_force = terrain.simulate_contact_force(positions["LF"], "LF")
        mock_data.update_sensor_data(rf_force, lf_force)
        
        # 显示进度
        if loop_count % 5 == 0:
            rf_state = traj_manager.foot_trajectories["RF"].state.value[:8]
            lf_state = traj_manager.foot_trajectories["LF"].state.value[:8]
            print(f"{loop_count*10:<8} {rf_state:<12} {lf_state:<12} {rf_force:<10.1f} {lf_force:<10.1f}")
        
        loop_count += 1
    
    print("-" * 70)
    
    # 分析结果
    rf_progress = traj_manager.foot_trajectories["RF"].get_progress_info()
    lf_progress = traj_manager.foot_trajectories["LF"].get_progress_info()
    
    print(f"\n双足协调结果:")
    print(f"RF足部:")
    print(f"  最终状态: {rf_progress['state']}")
    print(f"  中断次数: {rf_progress['interruption_count']}")
    print(f"  安全违规: {rf_progress['safety_violations']}")
    
    print(f"LF足部:")
    print(f"  最终状态: {lf_progress['state']}")
    print(f"  中断次数: {lf_progress['interruption_count']}")
    print(f"  安全违规: {lf_progress['safety_violations']}")
    
    print(f"\n步态调度器回调统计:")
    print(f"  正常完成: {len(gait_scheduler.swing_completed_callbacks)}")
    print(f"  轨迹中断: {len(gait_scheduler.trajectory_interrupted_callbacks)}")
    print(f"  紧急停止: {len(gait_scheduler.emergency_stop_callbacks)}")


def visualize_boundary_cases(positions_log, forces_log, states_log):
    """可视化边界情况处理"""
    if not positions_log:
        return
    
    positions = np.array(positions_log)
    forces = np.array(forces_log)
    times = np.arange(len(positions)) * 0.01  # 10ms间隔
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 轨迹3D图
    ax1.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='足部轨迹')
    ax1.axhline(y=0.03, color='r', linestyle='--', alpha=0.7, label='障碍物高度')
    ax1.set_xlabel('X位置 [m]')
    ax1.set_ylabel('Z高度 [m]')
    ax1.set_title('足部轨迹 - 侧视图')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 高度随时间变化
    ax2.plot(times, positions[:, 2] * 100, 'g-', linewidth=2, label='Z高度')
    ax2.axhline(y=3.0, color='r', linestyle='--', alpha=0.7, label='障碍物')
    ax2.set_xlabel('时间 [s]')
    ax2.set_ylabel('高度 [cm]')
    ax2.set_title('足部高度变化')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 接触力
    ax3.plot(times, forces, 'r-', linewidth=2, label='接触力')
    ax3.axhline(y=15.0, color='orange', linestyle='--', alpha=0.7, label='接触阈值')
    ax3.set_xlabel('时间 [s]')
    ax3.set_ylabel('接触力 [N]')
    ax3.set_title('接触力检测')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 状态变化
    state_mapping = {"idle": 0, "active": 1, "completed": 2, "interrupted": 3, "emergency_stop": 4}
    state_values = [state_mapping.get(state, 0) for state in states_log]
    ax4.plot(times, state_values, 'mo-', linewidth=2, markersize=4, label='轨迹状态')
    ax4.set_xlabel('时间 [s]')
    ax4.set_ylabel('状态编号')
    ax4.set_title('轨迹状态变化')
    ax4.set_yticks([0, 1, 2, 3, 4])
    ax4.set_yticklabels(['idle', 'active', 'completed', 'interrupted', 'emergency'])
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    """主演示程序"""
    print("MuJoCo边界情况处理演示")
    print("AzureLoong机器人足部轨迹生成")
    print("=" * 60)
    
    # 演示1：提前接触检测
    positions_log, forces_log, states_log = demo_early_contact_detection()
    
    # 演示2：地面穿透防护
    demo_ground_penetration_protection()
    
    # 演示3：紧急停止场景
    demo_emergency_stop_scenarios()
    
    # 演示4：完整MuJoCo集成
    demo_mujoco_integration()
    
    print("\n" + "=" * 60)
    print("所有边界情况处理演示完成！")
    print("=" * 60)
    
    print("\n核心边界情况处理特性:")
    print("✓ 提前落地检测：相位<70%时检测到接触则中断轨迹")
    print("✓ 地面穿透防护：防止足部位置低于地面高度")
    print("✓ 触觉传感器集成：MuJoCo rf-touch/lf-touch传感器")
    print("✓ 轨迹中止机制：phase跳到1.0，固定在接触点")
    print("✓ 步态调度器通知：中断、完成、紧急停止事件")
    print("✓ 安全违规记录：穿透防护、偏差检测记录")
    print("✓ 多种接触检测：位置、传感器、力反馈")
    
    print("\n推荐使用方式:")
    print("• 连接MuJoCo: foot_traj.connect_mujoco(model, data)")
    print("• 注册回调: gait_scheduler.on_trajectory_interrupted()")
    print("• 强制接触: foot_traj.force_ground_contact(contact_pos)")
    print("• 状态检查: foot_traj.is_interrupted()")
    
    # 可视化结果
    try:
        visualize_boundary_cases(positions_log, forces_log, states_log)
    except Exception as e:
        print(f"\n可视化跳过: {e}")


if __name__ == "__main__":
    main() 