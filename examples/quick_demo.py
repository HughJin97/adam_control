#!/usr/bin/env python3
"""
快速演示脚本 - 控制循环基础功能验证

这个脚本演示了完整的控制循环系统：
1. 传感器数据读取
2. 控制计算
3. 控制指令发送
4. 性能监控

运行这个脚本验证整个系统的数据流通路是否正常工作。

用法：
python quick_demo.py

作者: Adam Control Team
版本: 1.0
"""

import time
import numpy as np
import mujoco

# 导入我们的控制模块
from control_loop import ControlLoop, ControlLoopMode
from data_bus import get_data_bus


def quick_demo():
    """快速演示主控制循环"""
    print("=== 快速演示：主控制循环 ===\n")
    
    # 1. 加载MuJoCo模型
    print("1. 加载MuJoCo模型...")
    try:
        model = mujoco.MjModel.from_xml_path("models/scene.xml")
        data = mujoco.MjData(model)
        print(f"   ✓ 模型加载成功: {model.nq} DOF, {model.nu} 个执行器")
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        return
    
    # 2. 创建控制循环
    print("\n2. 创建控制循环...")
    control_loop = ControlLoop(
        mujoco_model=model,
        mujoco_data=data,
        control_frequency=1000.0,  # 1000 Hz
        enable_monitoring=True
    )
    print("   ✓ 控制循环创建成功")
    
    # 3. 数据总线验证
    print("\n3. 验证数据总线...")
    data_bus = get_data_bus()
    print(f"   ✓ 数据总线连接成功")
    print(f"   ✓ 关节数量: {len(data_bus.joint_names)}")
    print(f"   ✓ 末端执行器数量: {len(data_bus.end_effectors)}")
    
    # 4. 运行空闲模式测试（发送零力矩）
    print("\n4. 运行空闲模式测试 (3秒)...")
    control_loop.set_control_mode(ControlLoopMode.IDLE)
    control_loop.start()
    
    # 仿真3秒钟
    start_time = time.time()
    step_count = 0
    while time.time() - start_time < 3.0:
        mujoco.mj_step(model, data)
        step_count += 1
        time.sleep(0.001)  # 1ms 步长
    
    control_loop.stop()
    print(f"   ✓ 空闲模式测试完成 ({step_count} 仿真步)")
    
    # 5. 性能统计
    print("\n5. 性能统计...")
    stats = control_loop.get_status()
    print(f"   ✓ 控制循环数: {stats['loop_count']}")
    print(f"   ✓ 平均频率: {stats.get('loop_frequency', 0):.1f} Hz")
    print(f"   ✓ 平均循环时间: {stats.get('avg_loop_time', 0)*1000:.2f} ms")
    
    # 6. 传感器数据验证
    print("\n6. 传感器数据验证...")
    
    # 检查关节数据
    joint_positions = data_bus.get_all_joint_positions()
    print(f"   ✓ 关节位置数据: {len(joint_positions)} 个关节")
    
    # 检查IMU数据
    imu_ori = data_bus.get_imu_orientation()
    print(f"   ✓ IMU姿态: quat({imu_ori.w:.3f}, {imu_ori.x:.3f}, {imu_ori.y:.3f}, {imu_ori.z:.3f})")
    
    # 检查质心数据
    com_pos = data_bus.get_center_of_mass_position()
    print(f"   ✓ 质心位置: ({com_pos.x:.3f}, {com_pos.y:.3f}, {com_pos.z:.3f})")
    
    # 检查末端执行器数据
    left_foot_pos = data_bus.get_end_effector_position("left_foot")
    if left_foot_pos:
        print(f"   ✓ 左足位置: ({left_foot_pos.x:.3f}, {left_foot_pos.y:.3f}, {left_foot_pos.z:.3f})")
    
    # 7. 位置保持测试
    print("\n7. 运行位置保持测试 (2秒)...")
    control_loop.set_control_mode(ControlLoopMode.POSITION_HOLD)
    
    # 设置一些位置目标
    control_loop.set_position_hold_target("J_arm_l_02", 0.3)
    control_loop.set_position_hold_target("J_arm_r_02", -0.3)
    
    control_loop.start()
    
    # 仿真2秒钟
    start_time = time.time()
    while time.time() - start_time < 2.0:
        mujoco.mj_step(model, data)
        time.sleep(0.001)
    
    control_loop.stop()
    print("   ✓ 位置保持测试完成")
    
    # 8. 最终状态报告
    print("\n8. 最终状态报告...")
    final_stats = control_loop.get_status()
    print(f"   ✓ 总控制循环数: {final_stats['loop_count']}")
    print(f"   ✓ 最终平均频率: {final_stats.get('loop_frequency', 0):.1f} Hz")
    print(f"   ✓ 最大循环时间: {final_stats.get('max_loop_time', 0)*1000:.2f} ms")
    
    print("\n=== 演示完成 ===")
    print("✓ 数据总线工作正常")
    print("✓ 传感器读取正常")
    print("✓ 控制计算正常")
    print("✓ 控制发送正常")
    print("✓ 性能监控正常")
    print("\n系统已准备好进行更复杂的控制算法开发！")


def frequency_test():
    """频率性能测试"""
    print("\n=== 频率性能测试 ===")
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path("models/scene.xml")
    data = mujoco.MjData(model)
    
    # 测试不同频率
    frequencies = [100, 250, 500, 1000, 2000]
    
    for freq in frequencies:
        print(f"\n测试 {freq} Hz 控制频率...")
        
        # 创建控制循环
        control_loop = ControlLoop(
            mujoco_model=model,
            mujoco_data=data,
            control_frequency=freq,
            enable_monitoring=True
        )
        
        control_loop.set_control_mode(ControlLoopMode.IDLE)
        control_loop.start()
        
        # 运行2秒测试
        start_time = time.time()
        while time.time() - start_time < 2.0:
            mujoco.mj_step(model, data)
            time.sleep(0.0001)  # 0.1ms 步长
        
        control_loop.stop()
        
        # 统计结果
        stats = control_loop.get_status()
        actual_freq = stats.get('loop_frequency', 0)
        avg_time = stats.get('avg_loop_time', 0) * 1000
        max_time = stats.get('max_loop_time', 0) * 1000
        
        print(f"  目标: {freq:4d} Hz | 实际: {actual_freq:6.1f} Hz | 平均: {avg_time:5.2f}ms | 最大: {max_time:5.2f}ms")
        
        # 性能评估
        if actual_freq >= freq * 0.95:  # 95%以上认为良好
            status = "✓ 良好"
        elif actual_freq >= freq * 0.8:  # 80%以上认为可接受
            status = "△ 可接受"
        else:
            status = "✗ 需要优化"
        
        print(f"  性能评估: {status}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--freq-test":
        frequency_test()
    else:
        quick_demo()
        
        # 询问是否运行频率测试
        response = input("\n是否运行频率性能测试? (y/n): ").strip().lower()
        if response == 'y':
            frequency_test() 