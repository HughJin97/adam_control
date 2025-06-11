#!/usr/bin/env python3
"""
简化版步态逻辑测试

专注于核心功能验证：
1. 推进状态机（update_gait_state）
2. 检查legState交替
3. 验证target_foot_pos更新
4. 手动触发条件测试
"""

import time
import numpy as np
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig


def test_basic_state_machine():
    """测试基本状态机逻辑"""
    print("="*60)
    print("基本状态机测试")
    print("="*60)
    
    # 初始化
    config = GaitSchedulerConfig()
    config.swing_time = 0.2  # 缩短到200ms便于测试
    config.double_support_time = 0.05  # 缩短到50ms
    config.use_sensor_trigger = True
    config.require_both_triggers = False  # 允许单一触发方式
    config.enable_logging = False
    
    scheduler = get_gait_scheduler(config)
    data_bus = get_data_bus()
    
    print(f"初始状态: {scheduler.current_state.value}")
    print(f"初始legState: {scheduler.leg_state.value}")
    
    # 设置初始传感器状态
    scheduler.left_foot_force = 100.0
    scheduler.right_foot_force = 100.0
    scheduler.left_foot_contact = True
    scheduler.right_foot_contact = True
    
    # 开始行走
    scheduler.start_walking()
    print(f"开始行走后状态: {scheduler.current_state.value}")
    print(f"legState: {scheduler.leg_state.value}")
    print(f"摆动腿: {scheduler.swing_leg}")
    print(f"支撑腿: {scheduler.support_leg}")
    
    return scheduler, data_bus


def test_manual_state_progression():
    """测试手动推进状态"""
    print("\n" + "="*60)
    print("手动状态推进测试")
    print("="*60)
    
    scheduler, data_bus = test_basic_state_machine()
    
    # 记录初始状态
    initial_state = scheduler.current_state.value
    initial_leg_state = scheduler.leg_state.value
    
    print(f"\n步骤1: 推进状态机 (摆动时间{scheduler.config.swing_time}s)")
    
    dt = 0.01
    steps = int(scheduler.config.swing_time / dt) + 10  # 超过摆动时间
    
    for i in range(steps):
        current_time = i * dt
        
        # 在摆动快结束时模拟着地
        if current_time >= scheduler.config.swing_time * 0.9:
            if scheduler.swing_leg == "right":
                scheduler.right_foot_force = 200.0  # 增加右脚力
                scheduler.right_foot_velocity = np.array([0.0, 0.0, 0.0])
            elif scheduler.swing_leg == "left":
                scheduler.left_foot_force = 200.0  # 增加左脚力
                scheduler.left_foot_velocity = np.array([0.0, 0.0, 0.0])
        
        # 推进状态机
        state_changed = scheduler.update_gait_state(dt)
        
        if state_changed:
            print(f"[{current_time:.3f}s] 状态变化: {scheduler.current_state.value}")
            print(f"  legState: {scheduler.leg_state.value}")
            print(f"  摆动腿: {scheduler.swing_leg}")
            print(f"  支撑腿: {scheduler.support_leg}")
            
            # 检查数据总线同步
            print(f"  DataBus legState: {data_bus.legState}")
            print(f"  DataBus swing_leg: {data_bus.swing_leg}")
    
    # 验证状态是否改变
    final_state = scheduler.current_state.value
    final_leg_state = scheduler.leg_state.value
    
    print(f"\n结果验证:")
    print(f"  初始状态: {initial_state} -> {final_state}")
    print(f"  初始legState: {initial_leg_state} -> {final_leg_state}")
    
    if final_state != initial_state:
        print("✅ 状态成功切换")
    else:
        print("❌ 状态未切换")
    
    return scheduler, data_bus


def test_target_position_updates():
    """测试目标位置更新"""
    print("\n" + "="*60)
    print("目标位置更新测试")
    print("="*60)
    
    scheduler, data_bus = test_basic_state_machine()
    
    # 获取初始目标位置
    initial_targets = scheduler.get_target_foot_positions()
    print("初始目标位置:")
    for foot, pos in initial_targets.items():
        print(f"  {foot}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
    
    # 设置运动意图
    print(f"\n设置前进运动意图...")
    scheduler.set_motion_command(forward_velocity=0.3, lateral_velocity=0.0, turning_rate=0.0)
    
    # 触发足步规划
    try:
        data_bus.trigger_foot_placement_planning()
        print("✅ 足步规划触发成功")
    except Exception as e:
        print(f"⚠️ 足步规划触发失败: {e}")
    
    # 获取更新后的目标位置
    updated_targets = scheduler.get_target_foot_positions()
    print("更新后目标位置:")
    for foot, pos in updated_targets.items():
        print(f"  {foot}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
    
    # 验证位置变化
    position_changed = False
    for foot in ['left_foot', 'right_foot']:
        initial_pos = initial_targets[foot]
        updated_pos = updated_targets[foot]
        
        for coord in ['x', 'y', 'z']:
            if abs(initial_pos[coord] - updated_pos[coord]) > 1e-6:
                position_changed = True
                print(f"  {foot} {coord}: {initial_pos[coord]:.3f} -> {updated_pos[coord]:.3f}")
    
    if position_changed:
        print("✅ 目标位置有更新")
    else:
        print("ℹ️ 目标位置无变化（可能是正常的）")
    
    # 验证位置合理性
    left_pos = updated_targets['left_foot']
    right_pos = updated_targets['right_foot']
    
    y_distance = abs(left_pos['y'] - right_pos['y'])
    print(f"\n位置合理性检查:")
    print(f"  左右脚间距: {y_distance:.3f}m")
    
    if 0.08 <= y_distance <= 0.4:
        print("✅ 左右脚间距合理")
    else:
        print("❌ 左右脚间距异常")
    
    for foot, pos in updated_targets.items():
        if abs(pos['x']) <= 1.0 and abs(pos['y']) <= 0.5 and abs(pos['z']) <= 0.2:
            print(f"✅ {foot}位置在合理范围内")
        else:
            print(f"❌ {foot}位置超出合理范围")


def test_state_alternation():
    """测试状态交替逻辑"""
    print("\n" + "="*60)
    print("状态交替逻辑测试")
    print("="*60)
    
    scheduler, data_bus = test_basic_state_machine()
    
    states_history = []
    dt = 0.01
    test_duration = 1.0  # 1秒测试
    steps = int(test_duration / dt)
    
    print(f"运行{test_duration}s测试，手动触发状态切换...")
    
    for i in range(steps):
        current_time = i * dt
        
        # 每200ms手动触发一次状态切换
        if i % 20 == 0 and i > 0:  # 每200ms
            print(f"\n[{current_time:.3f}s] 手动触发状态切换")
            
            # 强制设置摆动时间完成
            scheduler.swing_elapsed_time = scheduler.config.swing_time + 0.01
            
            # 设置着地条件
            if scheduler.swing_leg == "right":
                scheduler.right_foot_force = 300.0
                scheduler.right_foot_contact = True
                print("  模拟右脚着地")
            elif scheduler.swing_leg == "left":
                scheduler.left_foot_force = 300.0
                scheduler.left_foot_contact = True
                print("  模拟左脚着地")
        
        # 推进状态机
        state_changed = scheduler.update_gait_state(dt)
        
        # 记录状态
        state_record = {
            'time': current_time,
            'state': scheduler.current_state.value,
            'leg_state': scheduler.leg_state.value,
            'swing_leg': scheduler.swing_leg,
            'support_leg': scheduler.support_leg
        }
        states_history.append(state_record)
        
        if state_changed:
            print(f"[{current_time:.3f}s] ✓ 状态切换: {state_record['state']}")
            print(f"  legState: {state_record['leg_state']}")
            print(f"  摆动腿: {state_record['swing_leg']}")
    
    # 分析状态交替
    print(f"\n状态交替分析:")
    unique_states = []
    for record in states_history:
        if not unique_states or unique_states[-1]['leg_state'] != record['leg_state']:
            unique_states.append(record)
    
    print(f"检测到{len(unique_states)}个不同状态:")
    for i, state in enumerate(unique_states):
        print(f"  {i+1}. {state['time']:.3f}s: {state['leg_state']} (摆动腿: {state['swing_leg']})")
    
    # 验证交替逻辑
    if len(unique_states) >= 2:
        alternation_correct = True
        for i in range(1, len(unique_states)):
            prev_state = unique_states[i-1]['leg_state']
            curr_state = unique_states[i]['leg_state']
            
            # 检查是否是合理的状态转换
            valid_transitions = [
                ("double_support", "left_support"),
                ("double_support", "right_support"),
                ("left_support", "double_support"),
                ("right_support", "double_support")
            ]
            
            if (prev_state, curr_state) not in valid_transitions:
                print(f"❌ 无效状态转换: {prev_state} -> {curr_state}")
                alternation_correct = False
        
        if alternation_correct:
            print("✅ 状态交替逻辑正确")
        else:
            print("❌ 状态交替逻辑有问题")
    else:
        print("⚠️ 状态变化不足，无法验证交替逻辑")


def main():
    """主测试函数"""
    print("AzureLoong机器人简化步态逻辑测试")
    print("专注于核心功能验证\n")
    
    try:
        # 1. 基本状态机测试
        test_basic_state_machine()
        
        # 2. 手动状态推进测试
        test_manual_state_progression()
        
        # 3. 目标位置更新测试
        test_target_position_updates()
        
        # 4. 状态交替逻辑测试
        test_state_alternation()
        
        print("\n" + "="*60)
        print("✅ 简化步态逻辑测试完成")
        print("主要验证了:")
        print("  • update_gait_state() 状态机推进")
        print("  • legState 状态管理")
        print("  • target_foot_pos 位置更新")
        print("  • 手动触发条件响应")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 