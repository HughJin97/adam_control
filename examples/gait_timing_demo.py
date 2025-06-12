#!/usr/bin/env python3
"""
步态计时与过渡演示脚本

演示AzureLoong机器人的步态计时器和步数计数功能：
- 摆动相和支撑相的时间管理
- 步完成条件检测 (时间 + 传感器)
- 步数计数和状态切换
- 数据总线中的事件管理
"""

import time
import numpy as np
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig


def demo_gait_timing():
    """演示步态计时功能"""
    print("="*60)
    print("AzureLoong机器人步态计时与过渡演示")
    print("="*60)
    
    # 初始化系统
    data_bus = get_data_bus()
    
    # 配置步态调度器
    config = GaitSchedulerConfig()
    config.swing_time = 0.4                 # 摆动时间400ms
    config.double_support_time = 0.1        # 双支撑时间100ms
    config.use_time_trigger = True          # 启用时间触发
    config.use_sensor_trigger = True        # 启用传感器触发
    config.require_both_triggers = False    # 只需满足一个条件
    config.touchdown_force_threshold = 30.0 # 触地力阈值30N
    config.contact_velocity_threshold = 0.05 # 接触速度阈值0.05m/s
    config.enable_logging = True
    
    gait_scheduler = get_gait_scheduler(config)
    
    # 重置系统
    data_bus.reset_step_counters()
    gait_scheduler.reset()
    
    print(f"配置参数:")
    print(f"  摆动时间: {config.swing_time}s")
    print(f"  双支撑时间: {config.double_support_time}s")
    print(f"  触地力阈值: {config.touchdown_force_threshold}N")
    print(f"  接触速度阈值: {config.contact_velocity_threshold}m/s")
    
    # 添加步完成事件回调
    def on_step_completed(event):
        print(f"✓ 步完成事件: 步数{event['step_number']}, "
              f"{event['completed_swing_leg']}腿摆动完成 ({event['swing_duration']:.3f}s)")
    
    data_bus.add_step_completion_callback(on_step_completed)
    
    print(f"\n开始步态演示...")
    print(f"初始状态: {gait_scheduler.current_state}")
    
    # 开始行走
    gait_scheduler.start_walking()
    print(f"开始行走，当前状态: {gait_scheduler.current_state}")
    
    # 模拟步态循环
    dt = 0.01  # 10ms控制周期
    total_time = 0.0
    max_demo_time = 3.0  # 演示3秒
    step_target = 4  # 目标完成4步
    
    print(f"\n步态时间循环 (控制周期: {dt*1000:.0f}ms):")
    print("-" * 80)
    
    completed_steps = 0
    last_print_time = 0.0
    
    while total_time < max_demo_time and completed_steps < step_target:
        # 模拟传感器数据
        simulate_sensor_data(gait_scheduler, total_time)
        
        # 更新步态状态
        state_changed = gait_scheduler.update_gait_state(dt)
        
        # 检查状态变化
        if state_changed:
            print(f"[{total_time:.3f}s] 状态转换: {gait_scheduler.current_state}")
            print(f"         摆动腿: {gait_scheduler.swing_leg}, "
                  f"支撑腿: {gait_scheduler.support_leg}")
        
        # 检查步完成事件
        if data_bus.is_step_completed():
            completed_steps += 1
            data_bus.reset_step_completion_flag()
            
            # 打印步完成信息
            stats = data_bus.get_step_statistics()
            print(f"[{total_time:.3f}s] 📊 步数统计: "
                  f"总步数={stats['total_steps']}, 左腿={stats['left_step_count']}, "
                  f"右腿={stats['right_step_count']}")
        
        # 每200ms打印计时器状态
        if total_time - last_print_time >= 0.2:
            print(f"[{total_time:.3f}s] ⏱️  计时器: "
                  f"摆动={gait_scheduler.swing_elapsed_time:.3f}s, "
                  f"相位={gait_scheduler.current_step_phase}, "
                  f"腿={gait_scheduler.swing_leg}")
            last_print_time = total_time
        
        total_time += dt
        time.sleep(dt * 0.1)  # 稍微减慢演示速度
    
    print("-" * 80)
    print(f"演示完成! 总时间: {total_time:.3f}s, 完成步数: {completed_steps}")
    
    # 最终状态报告
    print("\n" + "="*60)
    print("最终状态报告")
    print("="*60)
    
    # 步态调度器状态
    print(f"步态调度器状态:")
    print(f"  当前状态: {gait_scheduler.current_state}")
    print(f"  摆动腿: {gait_scheduler.swing_leg}")
    print(f"  支撑腿: {gait_scheduler.support_leg}")
    print(f"  摆动计时: {gait_scheduler.swing_elapsed_time:.3f}s")
    print(f"  步计时: {gait_scheduler.step_elapsed_time:.3f}s")
    print(f"  当前相位: {gait_scheduler.current_step_phase}")
    
    # 步数统计
    stats = data_bus.get_step_statistics()
    print(f"\n步数统计:")
    print(f"  当前步数: {stats['step_count']}")
    print(f"  总步数: {stats['total_steps']}")
    print(f"  左腿步数: {stats['left_step_count']}")
    print(f"  右腿步数: {stats['right_step_count']}")
    print(f"  平均摆动时间: {stats['swing_duration']:.3f}s")
    
    # 最近事件历史
    print(f"\n最近步态事件:")
    recent_events = data_bus.get_recent_step_events(5)
    for i, event in enumerate(recent_events, 1):
        print(f"  {i}. [{event['timestamp']:.3f}] 步数{event['step_number']}: "
              f"{event['completed_swing_leg']}腿 ({event['swing_duration']:.3f}s)")
    
    # 计时器性能
    print(f"\n计时器性能:")
    if completed_steps > 0:
        avg_swing_time = sum(event['swing_duration'] for event in recent_events) / len(recent_events)
        print(f"  平均摆动时间: {avg_swing_time:.3f}s")
        print(f"  配置摆动时间: {config.swing_time:.3f}s")
        print(f"  时间精度: {abs(avg_swing_time - config.swing_time):.3f}s")
    
    print("="*60)


def simulate_sensor_data(gait_scheduler, current_time):
    """模拟足部传感器数据"""
    # 基础力值
    left_force = 0.0
    right_force = 0.0
    left_velocity = np.array([0.0, 0.0, 0.0])
    right_velocity = np.array([0.0, 0.0, 0.0])
    
    # 根据当前摆动腿模拟传感器数据
    if gait_scheduler.swing_leg == "left":
        # 左腿摆动，右腿支撑
        right_force = 150.0  # 支撑腿有较大力（但不超过紧急停止阈值）
        right_velocity = np.array([0.01, 0.0, 0.0])
        
        # 左腿在摆动后期模拟触地
        if gait_scheduler.swing_elapsed_time > 0.25:
            left_force = min(50.0, (gait_scheduler.swing_elapsed_time - 0.25) * 200.0)
            left_velocity = np.array([0.03, 0.0, -0.02])
            
    elif gait_scheduler.swing_leg == "right":
        # 右腿摆动，左腿支撑
        left_force = 150.0  # 支撑腿有较大力（但不超过紧急停止阈值）
        left_velocity = np.array([0.01, 0.0, 0.0])
        
        # 右腿在摆动后期模拟触地
        if gait_scheduler.swing_elapsed_time > 0.25:
            right_force = min(50.0, (gait_scheduler.swing_elapsed_time - 0.25) * 200.0)
            right_velocity = np.array([0.03, 0.0, -0.02])
    
    else:
        # 双支撑相，两腿都有力
        left_force = 100.0
        right_force = 100.0
        left_velocity = np.array([0.01, 0.0, 0.0])
        right_velocity = np.array([0.01, 0.0, 0.0])
    
    # 更新传感器数据
    gait_scheduler.left_foot_force = left_force
    gait_scheduler.right_foot_force = right_force
    gait_scheduler.left_foot_velocity = left_velocity
    gait_scheduler.right_foot_velocity = right_velocity


def demonstrate_timing_precision():
    """演示计时精度"""
    print("\n" + "="*60)
    print("步态计时精度测试")
    print("="*60)
    
    data_bus = get_data_bus()
    config = GaitSchedulerConfig()
    config.swing_time = 0.3  # 短摆动时间用于精度测试
    config.enable_logging = False  # 减少输出
    
    gait_scheduler = get_gait_scheduler(config)
    data_bus.reset_step_counters()
    gait_scheduler.reset()
    gait_scheduler.start_walking()
    
    print(f"目标摆动时间: {config.swing_time:.3f}s")
    print(f"测试步数: 10步")
    
    dt = 0.005  # 5ms高精度控制
    measured_times = []
    
    for step in range(10):
        # 重置摆动计时器
        gait_scheduler._reset_swing_timer()
        
        # 模拟摆动相
        while gait_scheduler.swing_elapsed_time < config.swing_time:
            gait_scheduler._update_gait_timers(dt)
            
        # 记录实际时间
        actual_time = gait_scheduler.swing_elapsed_time
        measured_times.append(actual_time)
        
        print(f"步{step+1:2d}: 目标={config.swing_time:.3f}s, "
              f"实际={actual_time:.3f}s, "
              f"误差={abs(actual_time - config.swing_time):.3f}s")
    
    # 统计精度
    avg_time = np.mean(measured_times)
    std_time = np.std(measured_times)
    max_error = max(abs(t - config.swing_time) for t in measured_times)
    
    print(f"\n精度统计:")
    print(f"  平均时间: {avg_time:.6f}s")
    print(f"  标准差: {std_time:.6f}s")
    print(f"  最大误差: {max_error:.6f}s")
    print(f"  相对精度: {(max_error/config.swing_time)*100:.3f}%")


if __name__ == "__main__":
    # 主演示
    demo_gait_timing()
    
    # 精度测试
    demonstrate_timing_precision()
    
    print("\n🎉 步态计时与过渡演示完成!") 