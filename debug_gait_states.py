#!/usr/bin/env python3
"""
步态状态切换调试脚本

直接测试状态切换逻辑，不依赖复杂的触发条件
"""

import time
import numpy as np
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig, GaitState


def debug_state_machine_directly():
    """直接调试状态机逻辑"""
    print("="*70)
    print("直接状态机调试")
    print("="*70)
    
    # 创建新的调度器实例
    config = GaitSchedulerConfig()
    config.swing_time = 0.1  # 很短的摆动时间便于测试
    config.double_support_time = 0.05
    config.use_time_trigger = True
    config.use_sensor_trigger = False  # 禁用传感器触发
    config.enable_logging = True
    
    scheduler = get_gait_scheduler(config)
    data_bus = get_data_bus()
    
    print(f"初始状态: {scheduler.current_state}")
    print(f"初始legState: {scheduler.leg_state}")
    
    # 强制开始行走
    scheduler.start_walking()
    print(f"开始行走状态: {scheduler.current_state}")
    
    # 检查状态转换设置
    print(f"\n状态转换配置:")
    print(f"  摆动时间: {config.swing_time}s")
    print(f"  双支撑时间: {config.double_support_time}s")
    print(f"  仅使用时间触发: {config.use_time_trigger}")
    print(f"  禁用传感器触发: {not config.use_sensor_trigger}")
    
    return scheduler, data_bus


def test_forced_state_transitions():
    """测试强制状态转换"""
    print("\n" + "="*70)
    print("强制状态转换测试")
    print("="*70)
    
    scheduler, data_bus = debug_state_machine_directly()
    
    # 记录初始状态
    initial_state = scheduler.current_state.value
    print(f"测试前状态: {initial_state}")
    print(f"摆动时间: {scheduler.swing_elapsed_time:.3f}s")
    print(f"状态开始时间: {scheduler.state_start_time:.3f}s")
    
    # 方法1: 直接调用状态转换
    print(f"\n方法1: 直接调用状态转换")
    try:
        if scheduler.current_state == GaitState.LEFT_SUPPORT:
            target_state = GaitState.DOUBLE_SUPPORT_LR
        else:
            target_state = GaitState.LEFT_SUPPORT
        
        print(f"尝试转换到: {target_state}")
        scheduler._transition_to_state(target_state)
        print(f"转换后状态: {scheduler.current_state}")
        print(f"转换后legState: {scheduler.leg_state}")
        
    except Exception as e:
        print(f"直接转换失败: {e}")
    
    # 方法2: 通过时间推进
    print(f"\n方法2: 通过时间推进强制触发")
    
    # 重置到初始状态
    scheduler.start_walking()
    
    # 设置时间条件满足
    scheduler.swing_elapsed_time = scheduler.config.swing_time + 0.01
    scheduler.total_time += scheduler.config.swing_time + 0.01
    
    print(f"设置摆动时间: {scheduler.swing_elapsed_time:.3f}s (阈值: {scheduler.config.swing_time}s)")
    
    # 推进状态机
    dt = 0.01
    for i in range(10):
        print(f"\n推进步骤 {i+1}:")
        print(f"  当前状态: {scheduler.current_state.value}")
        print(f"  摆动时间: {scheduler.swing_elapsed_time:.3f}s")
        
        state_changed = scheduler.update_gait_state(dt)
        
        print(f"  状态是否改变: {state_changed}")
        print(f"  新状态: {scheduler.current_state.value}")
        print(f"  新legState: {scheduler.leg_state.value}")
        
        if state_changed:
            print(f"✅ 状态成功切换!")
            break
    else:
        print(f"❌ 10次推进后状态仍未切换")


def test_manual_step_completion():
    """测试手动步完成"""
    print("\n" + "="*70)
    print("手动步完成测试")
    print("="*70)
    
    scheduler, data_bus = debug_state_machine_directly()
    
    print(f"当前状态: {scheduler.current_state.value}")
    print(f"摆动腿: {scheduler.swing_leg}")
    
    # 手动触发步完成
    print(f"\n手动标记步完成...")
    
    # 方法1: 直接设置步完成标志
    try:
        data_bus.mark_step_completion(
            swing_leg=scheduler.swing_leg,
            swing_duration=scheduler.swing_elapsed_time,
            step_count=1
        )
        print(f"✅ 步完成标记成功")
        print(f"步完成状态: {data_bus.step_finished}")
    except Exception as e:
        print(f"步完成标记失败: {e}")
    
    # 方法2: 通过调度器内部方法
    try:
        scheduler._handle_step_completion()
        print(f"✅ 内部步完成处理成功")
    except Exception as e:
        print(f"内部步完成处理失败: {e}")
    
    # 推进状态机看是否响应
    state_changed = scheduler.update_gait_state(0.01)
    print(f"推进后状态改变: {state_changed}")
    print(f"当前状态: {scheduler.current_state.value}")


def test_target_foot_positions_debug():
    """调试目标足部位置"""
    print("\n" + "="*70)
    print("目标足部位置调试")
    print("="*70)
    
    scheduler, data_bus = debug_state_machine_directly()
    
    # 显示初始位置
    initial_targets = scheduler.get_target_foot_positions()
    print("初始目标位置:")
    for foot, pos in initial_targets.items():
        print(f"  {foot}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
    
    # 检查数据总线中的位置
    db_targets = data_bus.target_foot_pos
    print(f"\n数据总线中的目标位置:")
    for foot, pos in db_targets.items():
        print(f"  {foot}: ({pos.get('x', 0):.3f}, {pos.get('y', 0):.3f}, {pos.get('z', 0):.3f})")
    
    # 直接修改目标位置测试
    print(f"\n直接修改目标位置测试:")
    test_positions = {
        "left_foot": {"x": 0.1, "y": 0.12, "z": 0.0},
        "right_foot": {"x": 0.1, "y": -0.12, "z": 0.0}
    }
    
    data_bus.target_foot_pos.update(test_positions)
    print(f"设置测试位置完成")
    
    # 验证更新
    updated_targets = scheduler.get_target_foot_positions()
    print("更新后目标位置:")
    for foot, pos in updated_targets.items():
        print(f"  {foot}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
    
    # 计算并显示间距
    left_pos = updated_targets['left_foot']
    right_pos = updated_targets['right_foot']
    y_distance = abs(left_pos['y'] - right_pos['y'])
    print(f"\n左右脚间距: {y_distance:.3f}m")
    
    if y_distance >= 0.08:
        print("✅ 间距合理")
    else:
        print("❌ 间距过小")


def test_leg_state_logic():
    """测试腿部状态逻辑"""
    print("\n" + "="*70)
    print("腿部状态逻辑测试")
    print("="*70)
    
    scheduler, data_bus = debug_state_machine_directly()
    
    print(f"当前状态机状态: {scheduler.current_state.value}")
    print(f"当前腿部状态: {scheduler.leg_state.value}")
    print(f"摆动腿: {scheduler.swing_leg}")
    print(f"支撑腿: {scheduler.support_leg}")
    
    # 手动测试不同的步态状态
    test_states = [
        GaitState.LEFT_SUPPORT,
        GaitState.DOUBLE_SUPPORT_LR,
        GaitState.RIGHT_SUPPORT,
        GaitState.DOUBLE_SUPPORT_RL
    ]
    
    print(f"\n测试各种步态状态的腿部状态映射:")
    
    for test_state in test_states:
        # 强制设置状态
        scheduler.current_state = test_state
        
        # 更新腿部状态
        scheduler._update_leg_states()
        
        print(f"  {test_state.value}:")
        print(f"    legState: {scheduler.leg_state.value}")
        print(f"    摆动腿: {scheduler.swing_leg}")
        print(f"    支撑腿: {scheduler.support_leg}")
        
        # 检查数据总线同步
        data_bus.current_gait_state = test_state.value
        data_bus.legState = scheduler.leg_state.value
        data_bus.swing_leg = scheduler.swing_leg
        data_bus.support_leg = scheduler.support_leg
        
        print(f"    数据总线已同步")


def main():
    """主调试函数"""
    print("AzureLoong机器人步态状态调试")
    print("专门调试状态切换逻辑问题\n")
    
    try:
        # 1. 直接状态机调试
        debug_state_machine_directly()
        
        # 2. 强制状态转换测试
        test_forced_state_transitions()
        
        # 3. 手动步完成测试
        test_manual_step_completion()
        
        # 4. 目标位置调试
        test_target_foot_positions_debug()
        
        # 5. 腿部状态逻辑测试
        test_leg_state_logic()
        
        print("\n" + "="*70)
        print("✅ 步态状态调试完成")
        print("主要调试了:")
        print("  • 状态机基本功能")
        print("  • 强制状态转换机制")
        print("  • 步完成触发逻辑")
        print("  • 目标位置更新机制")
        print("  • 腿部状态映射逻辑")
        print("="*70)
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 