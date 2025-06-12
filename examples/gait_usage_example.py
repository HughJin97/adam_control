#!/usr/bin/env python3
"""
步态参数系统使用示例

演示如何使用步态参数管理器和数据总线来：
1. 设置步态参数
2. 控制机器人步态
3. 实时获取足部轨迹
4. 保存和加载参数配置

作者: Adam Control Team
版本: 1.0
"""

import time
from gait_parameters import get_gait_manager, GaitType
from data_bus import get_data_bus


def basic_usage_example():
    """基础使用示例"""
    print("=== 步态参数系统基础使用示例 ===\n")
    
    # 获取管理器实例
    gait_manager = get_gait_manager()
    data_bus = get_data_bus()
    
    # 1. 设置基本步态参数
    print("1. 设置基本参数:")
    data_bus.set_walking_speed(0.15)  # 设置行走速度 0.15 m/s
    data_bus.set_step_frequency(1.0)  # 设置步频 1.0 Hz
    
    print(f"   行走速度: {data_bus.get_walking_speed():.3f} m/s")
    print(f"   步频: {data_bus.get_step_frequency():.2f} Hz")
    print(f"   步长: {gait_manager.spatial.step_length:.3f} m")
    print(f"   步宽: {gait_manager.spatial.step_width:.3f} m")
    
    # 2. 加载预设参数
    print("\n2. 加载预设参数:")
    available_presets = data_bus.get_available_gait_presets()
    print(f"   可用预设: {available_presets}")
    
    # 加载正常步行预设
    data_bus.load_gait_preset("normal_walk")
    print(f"   已加载预设: normal_walk")
    print(f"   新的周期时间: {gait_manager.timing.gait_cycle_time:.2f}s")
    
    # 3. 控制步态状态
    print("\n3. 控制步态状态:")
    data_bus.start_walking()
    print(f"   开始行走，机器人模式: {data_bus.get_robot_mode()}")
    
    # 4. 实时相位更新和足部轨迹获取
    print("\n4. 实时相位和轨迹:")
    current_time = 0.0
    dt = 0.1  # 时间步长
    
    for i in range(5):  # 演示5个时间步
        # 更新相位
        data_bus.update_gait_phase(current_time)
        
        # 获取相位信息
        phase_info = data_bus.get_gait_phase_info()
        
        # 获取足部目标位置
        left_target = data_bus.get_foot_target_position("left")
        right_target = data_bus.get_foot_target_position("right")
        
        print(f"   t={current_time:.1f}s: 左腿相位={phase_info['left_phase']:.2f}, "
              f"右腿相位={phase_info['right_phase']:.2f}")
        print(f"           左脚目标=(X:{left_target.x:.3f}, Y:{left_target.y:.3f}, Z:{left_target.z:.3f})")
        print(f"           右脚目标=(X:{right_target.x:.3f}, Y:{right_target.y:.3f}, Z:{right_target.z:.3f})")
        
        current_time += dt
    
    # 5. 停止行走
    print("\n5. 停止步态:")
    data_bus.stop_walking()
    print(f"   停止行走，机器人模式: {data_bus.get_robot_mode()}")


def advanced_usage_example():
    """高级使用示例"""
    print("\n=== 步态参数系统高级使用示例 ===\n")
    
    gait_manager = get_gait_manager()
    data_bus = get_data_bus()
    
    # 1. 自定义步态参数
    print("1. 自定义步态参数:")
    
    # 直接修改管理器参数
    gait_manager.timing.gait_cycle_time = 1.5  # 1.5秒周期
    gait_manager.timing.swing_ratio = 0.35     # 35% 摆动相
    gait_manager.timing.stance_ratio = 0.65    # 65% 支撑相
    gait_manager.timing.update_from_cycle_time()  # 更新相关参数
    
    gait_manager.spatial.step_length = 0.12    # 12cm 步长
    gait_manager.spatial.step_height = 0.06    # 6cm 抬脚高度
    
    # 同步到数据总线
    data_bus.update_gait_from_manager()
    
    print(f"   自定义周期时间: {gait_manager.timing.gait_cycle_time:.2f}s")
    print(f"   摆动/支撑比例: {gait_manager.timing.swing_ratio:.1%}/{gait_manager.timing.stance_ratio:.1%}")
    print(f"   步长: {gait_manager.spatial.step_length:.3f}m")
    print(f"   抬脚高度: {gait_manager.spatial.step_height:.3f}m")
    
    # 2. 保存自定义预设
    print("\n2. 保存自定义预设:")
    gait_manager.save_current_as_preset("custom_slow")
    print("   已保存为 'custom_slow' 预设")
    
    # 3. 导出配置到文件
    print("\n3. 导出配置:")
    export_file = "my_gait_config.yaml"
    data_bus.export_gait_parameters(export_file)
    print(f"   配置已导出到: {export_file}")
    
    # 4. 动态调整参数
    print("\n4. 动态调整参数:")
    
    # 模拟加速过程
    speeds = [0.05, 0.1, 0.15, 0.2]
    for speed in speeds:
        data_bus.set_walking_speed(speed)
        print(f"   速度: {speed:.2f} m/s -> 步长: {gait_manager.spatial.step_length:.3f} m")
    
    # 5. 步态类型切换
    print("\n5. 步态类型切换:")
    gait_types = [GaitType.STATIC_WALK, GaitType.DYNAMIC_WALK, GaitType.TROT]
    
    for gait_type in gait_types:
        gait_manager.set_gait_type(gait_type)
        print(f"   步态类型: {gait_type.value}")
        print(f"     双支撑比例: {gait_manager.timing.double_support_ratio:.1%}")
        print(f"     摆动比例: {gait_manager.timing.swing_ratio:.1%}")


def walking_control_example():
    """行走控制示例"""
    print("\n=== 行走控制示例 ===\n")
    
    data_bus = get_data_bus()
    
    # 1. 基本行走控制
    print("1. 基本行走控制序列:")
    
    # 准备阶段
    data_bus.load_gait_preset("normal_walk")
    print("   加载正常步行预设")
    
    # 开始行走
    data_bus.start_walking()
    data_bus.set_walking_speed(0.1)
    print(f"   开始行走，速度: {data_bus.get_walking_speed():.2f} m/s")
    
    # 加速
    data_bus.set_walking_speed(0.2)
    print(f"   加速到: {data_bus.get_walking_speed():.2f} m/s")
    
    # 转向
    data_bus.start_turning(0.2)  # 0.2 rad/s
    print(f"   开始转向: {data_bus.gait.turning_rate:.2f} rad/s")
    
    # 直行
    data_bus.stop_turning()
    print("   停止转向，继续直行")
    
    # 减速
    data_bus.set_walking_speed(0.05)
    print(f"   减速到: {data_bus.get_walking_speed():.2f} m/s")
    
    # 停止
    data_bus.stop_walking()
    print("   停止行走")
    
    # 2. 原地踏步
    print("\n2. 原地踏步:")
    data_bus.load_gait_preset("march_in_place")
    data_bus.start_walking()
    print(f"   原地踏步，步长: {data_bus.gait.step_length:.3f} m")
    print(f"   抬脚高度: {data_bus.gait.step_height:.3f} m")
    data_bus.stop_walking()


def monitoring_example():
    """监控示例"""
    print("\n=== 步态监控示例 ===\n")
    
    data_bus = get_data_bus()
    
    # 设置步态
    data_bus.load_gait_preset("normal_walk")
    data_bus.start_walking()
    
    print("监控一个步态周期的状态变化:")
    print("时间\t左脚状态\t右脚状态\t双支撑\t左脚高度\t右脚高度")
    print("-" * 70)
    
    cycle_time = data_bus.gait.gait_cycle_time
    for i in range(11):
        t = i * cycle_time / 10
        
        # 更新相位
        data_bus.update_gait_phase(t)
        
        # 获取状态
        left_in_swing = data_bus.is_foot_in_swing("left")
        right_in_swing = data_bus.is_foot_in_swing("right")
        double_support = data_bus.gait.double_support
        
        # 获取足部位置
        left_target = data_bus.get_foot_target_position("left")
        right_target = data_bus.get_foot_target_position("right")
        
        left_state = "摆动" if left_in_swing else "支撑"
        right_state = "摆动" if right_in_swing else "支撑"
        ds_state = "是" if double_support else "否"
        
        print(f"{t:.2f}\t{left_state}\t\t{right_state}\t\t{ds_state}\t"
              f"{left_target.z:.3f}\t\t{right_target.z:.3f}")
    
    data_bus.stop_walking()


def main():
    """主函数"""
    print("步态参数系统使用示例")
    print("=" * 50)
    
    try:
        # 运行各种示例
        basic_usage_example()
        advanced_usage_example()
        walking_control_example()
        monitoring_example()
        
        print("\n=" * 50)
        print("所有示例运行完成！")
        print("步态参数系统功能正常，可以集成到机器人控制系统中。")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 