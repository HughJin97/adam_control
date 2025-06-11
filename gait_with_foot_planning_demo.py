#!/usr/bin/env python3
"""
步态调度器与足步规划集成演示

演示当步态状态转换时自动触发足步规划，
计算摆动腿的目标着地位置。

作者: Adam Control Team
版本: 1.0
"""

import sys
import time
import numpy as np

sys.path.append('.')

from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler
from foot_placement import Vector3D


def demo_gait_with_foot_planning():
    """演示步态调度器与足步规划的集成"""
    print("=" * 60)
    print("步态调度器与足步规划集成演示")
    print("=" * 60)
    
    # 初始化系统
    data_bus = get_data_bus()
    scheduler = get_gait_scheduler()
    
    # 设置初始关节状态
    initial_positions = {
        "left_hip_pitch": -0.1, "left_knee_pitch": 0.2, "left_ankle_pitch": -0.1,
        "right_hip_pitch": -0.1, "right_knee_pitch": 0.2, "right_ankle_pitch": -0.1,
        "left_hip_roll": 0.05, "right_hip_roll": -0.05,
        "left_ankle_roll": -0.05, "right_ankle_roll": 0.05
    }
    
    for joint, position in initial_positions.items():
        data_bus.set_joint_position(joint, position)
    
    # 设置运动意图
    data_bus.set_body_motion_intent(0.3, 0.0, 0.0)  # 前进0.3m/s
    
    print(f"✓ 系统初始化完成")
    print(f"  数据总线: {'可用' if data_bus else '不可用'}")
    print(f"  步态调度器: {'可用' if scheduler else '不可用'}")
    print(f"  足步规划器: {'可用' if data_bus._has_foot_planner else '不可用'}")
    
    # 重置步态调度器
    scheduler.reset()
    
    # 设置传感器数据
    left_velocity = np.array([0.0, 0.0, 0.0])
    right_velocity = np.array([0.0, 0.0, 0.0])
    
    print(f"\n开始步态演示...")
    print(f"运动意图: 前进 0.3 m/s")
    
    # 开始行走
    scheduler.start_walking()
    print(f"✓ 开始行走")
    
    # 模拟步态过程
    dt = 0.02  # 50Hz
    total_time = 5.0  # 5秒演示
    steps = int(total_time / dt)
    
    foot_planning_events = []
    state_changes = []
    
    print(f"\n{'='*80}")
    print(f"{'时间':<8} {'状态':<20} {'摆动腿':<10} {'支撑腿':<10} {'足步规划':<20}")
    print(f"{'='*80}")
    
    for i in range(steps):
        current_time = i * dt
        
        # 更新传感器数据 (添加一些随机变化模拟真实情况)
        force_noise = np.random.normal(0, 2)
        left_force = 25.0 + force_noise
        right_force = 25.0 + force_noise
        
        scheduler.update_sensor_data(left_force, right_force, left_velocity, right_velocity)
        
        # 更新步态调度器 (同时会触发足步规划)
        state_changed = data_bus.update_gait_scheduler_with_foot_planning(dt)
        
        if state_changed:
            # 记录状态变化
            current_state = scheduler.current_state.value
            swing_leg = scheduler.swing_leg
            support_leg = scheduler.support_leg
            
            state_changes.append({
                'time': current_time,
                'state': current_state,
                'swing_leg': swing_leg,
                'support_leg': support_leg
            })
            
            # 检查是否触发了足步规划
            foot_planning_info = ""
            if swing_leg in ["left", "right"]:
                target_pos = data_bus.get_target_foot_position(swing_leg)
                foot_planning_info = f"目标({target_pos['x']:.3f}, {target_pos['y']:.3f}, {target_pos['z']:.3f})"
                
                foot_planning_events.append({
                    'time': current_time,
                    'swing_leg': swing_leg,
                    'target_position': target_pos
                })
            
            print(f"{current_time:<8.2f} {current_state:<20} {swing_leg:<10} {support_leg:<10} {foot_planning_info:<20}")
        
        # 每隔一段时间打印足部位移信息
        if i % 50 == 0 and i > 0:  # 每1秒
            print(f"\n--- t={current_time:.1f}s 足部状态 ---")
            for foot in ["left", "right"]:
                current_pos = data_bus.get_current_foot_position(foot)
                target_pos = data_bus.get_target_foot_position(foot)
                distance = data_bus.get_foot_distance_to_target(foot)
                
                print(f"{foot}脚: 当前({current_pos['x']:.3f}, {current_pos['y']:.3f}, {current_pos['z']:.3f}) "
                      f"目标({target_pos['x']:.3f}, {target_pos['y']:.3f}, {target_pos['z']:.3f}) "
                      f"距离{distance:.3f}m")
            print()
    
    print(f"{'='*80}")
    
    # 打印统计信息
    print(f"\n=== 演示统计 ===")
    print(f"总时间: {total_time:.1f}s")
    print(f"状态转换次数: {len(state_changes)}")
    print(f"足步规划触发次数: {len(foot_planning_events)}")
    
    if len(state_changes) > 0:
        # 计算步频
        swing_events = [event for event in foot_planning_events if event['swing_leg'] in ['left', 'right']]
        if len(swing_events) > 1:
            step_frequency = len(swing_events) / total_time
            print(f"平均步频: {step_frequency:.2f} 步/秒")
        
        # 分析状态分布
        states = [change['state'] for change in state_changes]
        unique_states = list(set(states))
        print(f"经历状态: {', '.join(unique_states)}")
    
    # 获取步态统计
    gait_stats = scheduler.get_state_statistics()
    print(f"\n=== 步态状态统计 ===")
    for state_name, data in gait_stats['state_stats'].items():
        print(f"{state_name}: {data['count']}次, "
              f"平均{data['avg_duration']:.3f}s, "
              f"占比{data['percentage']:.1f}%")
    
    # 获取足步规划统计
    print(f"\n=== 足步规划统计 ===")
    foot_info = data_bus.get_foot_planning_info()
    print(f"规划策略: {foot_info.get('planning_strategy', 'N/A')}")
    print(f"地形类型: {foot_info.get('terrain_type', 'N/A')}")
    print(f"数据总线规划次数: {foot_info['foot_planning_count']}")
    print(f"规划器总次数: {foot_info.get('planner_count', 0)}")
    
    # 打印最终足部状态
    print(f"\n=== 最终足部状态 ===")
    data_bus.print_foot_planning_status()
    
    print(f"\n{'='*60}")
    print(f"演示完成！")
    print(f"{'='*60}")
    
    return {
        'state_changes': state_changes,
        'foot_planning_events': foot_planning_events,
        'gait_stats': gait_stats,
        'foot_info': foot_info
    }


def main():
    """主函数"""
    try:
        results = demo_gait_with_foot_planning()
        
        print(f"\n总结:")
        if len(results['foot_planning_events']) > 0:
            print(f"✓ 足步规划系统工作正常")
            print(f"✓ 步态调度器集成成功")
            print(f"✓ 在状态转换时自动触发足步规划")
        else:
            print(f"⚠ 未检测到足步规划事件")
        
    except Exception as e:
        print(f"演示出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 