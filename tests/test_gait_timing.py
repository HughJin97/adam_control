#!/usr/bin/env python3
"""
步态计时与过渡功能测试脚本

测试内容:
1. 步态计时器的初始化和更新
2. 步完成条件的检测
3. 步态切换和计数
4. 数据总线中的步态事件管理
"""

import time
import numpy as np
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig, GaitState


def setup_test_environment():
    """设置测试环境"""
    print("=" * 60)
    print("步态计时与过渡功能测试")
    print("=" * 60)
    
    # 获取数据总线和步态调度器
    data_bus = get_data_bus()
    
    # 配置步态调度器
    config = GaitSchedulerConfig()
    config.swing_time = 0.3                    # 缩短摆动时间用于测试
    config.double_support_time = 0.1           # 双支撑时间
    config.use_time_trigger = True             # 启用时间触发
    config.use_sensor_trigger = False          # 禁用传感器触发（简化测试）
    config.enable_logging = True               # 启用日志
    
    gait_scheduler = get_gait_scheduler(config)
    
    # 重置计数器
    data_bus.reset_step_counters()
    gait_scheduler.reset()
    
    return data_bus, gait_scheduler


def test_gait_timers(data_bus, gait_scheduler):
    """测试步态计时器"""
    print("\n[测试] 步态计时器初始化和更新")
    
    # 检查初始状态
    print(f"初始步态状态: {gait_scheduler.current_state}")
    print(f"初始摆动计时: {gait_scheduler.swing_elapsed_time:.3f}s")
    print(f"初始步相位: {gait_scheduler.current_step_phase}")
    
    # 开始行走
    gait_scheduler.start_walking()
    print(f"开始行走后状态: {gait_scheduler.current_state}")
    
    # 模拟时间更新
    dt = 0.01  # 10ms时间步长
    total_time = 0.0
    max_test_time = 2.0  # 测试2秒
    
    print(f"\n开始时间循环更新 (dt={dt}s)...")
    
    while total_time < max_test_time:
        # 更新步态状态
        state_changed = gait_scheduler.update_gait_state(dt)
        
        if state_changed:
            print(f"[{total_time:.3f}s] 状态变化: {gait_scheduler.current_state}, "
                  f"摆动腿: {gait_scheduler.swing_leg}, "
                  f"摆动时间: {gait_scheduler.swing_elapsed_time:.3f}s")
        
        # 每0.1秒打印一次计时器状态
        if int(total_time * 10) % 10 == 0:  # 每100ms
            print(f"[{total_time:.3f}s] 计时器状态: "
                  f"摆动={gait_scheduler.swing_elapsed_time:.3f}s, "
                  f"步时间={gait_scheduler.step_elapsed_time:.3f}s, "
                  f"相位={gait_scheduler.current_step_phase}")
        
        total_time += dt
        time.sleep(dt * 0.1)  # 轻微延迟以便观察
    
    print(f"时间循环测试完成，总耗时: {total_time:.3f}s")


def test_step_completion_detection(data_bus, gait_scheduler):
    """测试步完成检测"""
    print("\n[测试] 步完成检测")
    
    # 确保处于行走状态
    if gait_scheduler.current_state == GaitState.IDLE:
        gait_scheduler.start_walking()
    
    # 添加步完成回调
    def step_completion_callback(event):
        print(f"步完成回调触发: 步数{event['step_number']}, "
              f"{event['completed_swing_leg']}腿完成摆动 "
              f"({event['swing_duration']:.3f}s)")
    
    data_bus.add_step_completion_callback(step_completion_callback)
    
    # 模拟多个步态周期
    dt = 0.01
    total_time = 0.0
    max_cycles = 3  # 测试3个完整周期
    completed_steps = 0
    
    print(f"监测步完成事件...")
    
    while completed_steps < max_cycles * 2:  # 每个周期包含左右两步
        # 更新步态
        state_changed = gait_scheduler.update_gait_state(dt)
        
        # 检查步完成事件
        if data_bus.is_step_completed():
            completed_steps += 1
            stats = data_bus.get_step_statistics()
            
            print(f"[{total_time:.3f}s] 步完成事件 #{completed_steps}:")
            print(f"  当前步数: {stats['step_count']}")
            print(f"  完成摆动腿: {stats['current_swing_leg']}")
            print(f"  摆动持续时间: {stats['swing_duration']:.3f}s")
            
            # 重置步完成标志
            data_bus.reset_step_completion_flag()
        
        total_time += dt
        time.sleep(dt * 0.1)
        
        # 防止无限循环
        if total_time > 5.0:
            break
    
    print(f"步完成检测测试完成，检测到 {completed_steps} 步")


def test_leg_state_switching(data_bus, gait_scheduler):
    """测试腿部状态切换"""
    print("\n[测试] 腿部状态切换")
    
    # 记录状态变化
    state_changes = []
    
    dt = 0.008  # 8ms
    total_time = 0.0
    test_duration = 1.5
    
    last_swing_leg = gait_scheduler.swing_leg
    last_state = gait_scheduler.current_state
    
    print("监测腿部状态切换...")
    
    while total_time < test_duration:
        gait_scheduler.update_gait_state(dt)
        
        # 检测状态变化
        if (gait_scheduler.swing_leg != last_swing_leg or 
            gait_scheduler.current_state != last_state):
            
            change_record = {
                "time": total_time,
                "old_swing_leg": last_swing_leg,
                "new_swing_leg": gait_scheduler.swing_leg,
                "old_state": last_state,
                "new_state": gait_scheduler.current_state,
                "support_leg": gait_scheduler.support_leg
            }
            
            state_changes.append(change_record)
            
            print(f"[{total_time:.3f}s] 状态切换:")
            print(f"  摆动腿: {last_swing_leg} -> {gait_scheduler.swing_leg}")
            print(f"  步态状态: {last_state} -> {gait_scheduler.current_state}")
            print(f"  支撑腿: {gait_scheduler.support_leg}")
            
            last_swing_leg = gait_scheduler.swing_leg
            last_state = gait_scheduler.current_state
        
        total_time += dt
        time.sleep(dt * 0.1)
    
    print(f"\n检测到 {len(state_changes)} 次状态切换")
    print("状态切换序列:")
    for i, change in enumerate(state_changes):
        print(f"  #{i+1} [{change['time']:.3f}s]: "
              f"{change['old_swing_leg']} -> {change['new_swing_leg']} "
              f"({change['old_state']} -> {change['new_state']})")


def test_step_statistics(data_bus, gait_scheduler):
    """测试步数统计"""
    print("\n[测试] 步数统计")
    
    # 确保有一些步数数据
    if data_bus.get_step_count() == 0:
        print("运行短时间以产生步数数据...")
        dt = 0.01
        for _ in range(int(1.0 / dt)):  # 运行1秒
            gait_scheduler.update_gait_state(dt)
            
            if data_bus.is_step_completed():
                data_bus.reset_step_completion_flag()
    
    # 打印详细统计
    data_bus.print_step_status()
    
    # 测试历史记录
    print("\n步态事件历史:")
    recent_events = data_bus.get_recent_step_events(3)
    for event in recent_events:
        print(f"  {event['timestamp']:.3f}: 步数{event['step_number']} "
              f"{event['completed_swing_leg']}腿 ({event['swing_duration']:.3f}s)")
    
    print("\n步态相位历史:")
    phase_history = data_bus.get_gait_phase_history(3)
    for record in phase_history:
        print(f"  步数{record['step_count']}: {record['swing_leg']}腿 "
              f"({record['duration']:.3f}s)")


def test_sensor_based_completion():
    """测试传感器触发的步完成检测"""
    print("\n[测试] 传感器触发步完成检测")
    
    # 创建带传感器触发的配置
    config = GaitSchedulerConfig()
    config.swing_time = 0.4
    config.use_time_trigger = True
    config.use_sensor_trigger = True
    config.require_both_triggers = False  # 只需满足一个条件
    config.touchdown_force_threshold = 20.0
    config.contact_velocity_threshold = 0.1
    config.enable_logging = True
    
    gait_scheduler = get_gait_scheduler(config)
    data_bus = get_data_bus()
    
    gait_scheduler.reset()
    data_bus.reset_step_counters()
    gait_scheduler.start_walking()
    
    dt = 0.01
    total_time = 0.0
    
    print("模拟传感器数据...")
    
    for i in range(100):  # 1秒模拟
        # 更新传感器数据
        if gait_scheduler.swing_leg == "left":
            # 左腿摆动，在0.2秒后模拟触地
            if gait_scheduler.swing_elapsed_time > 0.2:
                gait_scheduler.left_foot_force = 30.0  # 超过阈值
                gait_scheduler.left_foot_velocity = np.array([0.05, 0.0, 0.0])  # 低速度
        elif gait_scheduler.swing_leg == "right":
            # 右腿摆动，在0.2秒后模拟触地
            if gait_scheduler.swing_elapsed_time > 0.2:
                gait_scheduler.right_foot_force = 25.0
                gait_scheduler.right_foot_velocity = np.array([0.03, 0.0, 0.0])
        
        # 更新步态
        state_changed = gait_scheduler.update_gait_state(dt)
        
        if state_changed:
            print(f"[{total_time:.3f}s] 传感器触发状态切换: {gait_scheduler.current_state}, "
                  f"摆动腿: {gait_scheduler.swing_leg}")
        
        # 检查步完成
        if data_bus.is_step_completed():
            stats = data_bus.get_step_statistics()
            print(f"[{total_time:.3f}s] 传感器触发步完成: "
                  f"步数{stats['step_count']}, 摆动时间{stats['swing_duration']:.3f}s")
            data_bus.reset_step_completion_flag()
        
        total_time += dt
        time.sleep(dt * 0.1)
    
    print("传感器触发测试完成")


def main():
    """主测试函数"""
    try:
        # 设置测试环境
        data_bus, gait_scheduler = setup_test_environment()
        
        # 测试1: 步态计时器
        test_gait_timers(data_bus, gait_scheduler)
        
        # 测试2: 步完成检测
        test_step_completion_detection(data_bus, gait_scheduler)
        
        # 测试3: 腿部状态切换
        test_leg_state_switching(data_bus, gait_scheduler)
        
        # 测试4: 步数统计
        test_step_statistics(data_bus, gait_scheduler)
        
        # 测试5: 传感器触发
        test_sensor_based_completion()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
        # 最终状态报告
        print("\n最终状态报告:")
        gait_scheduler.print_status()
        data_bus.print_step_status()
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 