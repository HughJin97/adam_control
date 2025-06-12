#!/usr/bin/env python3
"""
足步规划计算调试脚本

验证步长计算的具体逻辑和参数影响
"""

import numpy as np
from foot_placement import FootPlacementPlanner, FootPlacementConfig, Vector3D

def debug_step_calculation():
    """调试步长计算"""
    print("=== 足步规划计算调试 ===")
    
    # 创建配置
    config = FootPlacementConfig()
    config.nominal_step_length = 0.15  # 标准步长 15cm
    config.nominal_step_width = 0.12   # 标准步宽 12cm
    config.lateral_stability_margin = 0.03   # 横向稳定性余量 3cm
    config.longitudinal_stability_margin = 0.02  # 纵向稳定性余量 2cm
    config.speed_step_gain = 0.8  # 速度-步长增益
    
    print(f"配置参数:")
    print(f"  基础步长: {config.nominal_step_length:.3f}m")
    print(f"  基础步宽: {config.nominal_step_width:.3f}m")
    print(f"  速度增益: {config.speed_step_gain:.3f}")
    print(f"  纵向稳定性余量: {config.longitudinal_stability_margin:.3f}m")
    print(f"  横向稳定性余量: {config.lateral_stability_margin:.3f}m")
    
    # 创建足步规划器
    planner = FootPlacementPlanner(config)
    
    # 测试不同速度下的步长计算
    test_velocities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    print(f"\n=== 不同速度下的步长计算 ===")
    print(f"{'速度(m/s)':<10} {'计算步长(m)':<15} {'期望步长(m)':<15} {'误差(m)':<10}")
    print("-" * 60)
    
    for velocity in test_velocities:
        # 设置运动意图
        planner.set_body_motion_intent(Vector3D(velocity, 0.0, 0.0), 0.0)
        
        # 计算步态参数
        step_params = planner.compute_step_parameters("left", "right")
        calculated_step_length = step_params['step_length']
        
        # 理论计算
        expected_step_length = config.nominal_step_length + velocity * config.speed_step_gain
        error = abs(calculated_step_length - expected_step_length)
        
        print(f"{velocity:<10.1f} {calculated_step_length:<15.3f} {expected_step_length:<15.3f} {error:<10.6f}")
    
    # 测试完整的落脚点计算过程
    print(f"\n=== 完整落脚点计算过程 ===")
    
    # 设置测试条件
    test_velocity = 0.3  # 0.3m/s前进速度
    planner.set_body_motion_intent(Vector3D(test_velocity, 0.0, 0.0), 0.0)
    
    # 设置支撑足位置（右脚）
    planner.right_foot.position = Vector3D(0.0, -0.102, -0.649)
    
    print(f"测试条件:")
    print(f"  前进速度: {test_velocity}m/s")
    print(f"  支撑足位置: ({planner.right_foot.position.x:.3f}, {planner.right_foot.position.y:.3f}, {planner.right_foot.position.z:.3f})")
    
    # 步骤1: 计算步态参数
    step_params = planner.compute_step_parameters("left", "right")
    print(f"\n步骤1 - 步态参数:")
    print(f"  计算步长: {step_params['step_length']:.3f}m")
    print(f"  计算步宽: {step_params['step_width']:.3f}m")
    
    # 步骤2: 计算目标位置（不含稳定性调整）
    support_pos = planner.right_foot.position
    step_length = step_params['step_length']
    step_width = step_params['step_width']
    
    # 前进方向位移
    forward_displacement = Vector3D(
        step_length * np.cos(planner.body_heading),  # body_heading = 0
        step_length * np.sin(planner.body_heading),
        0.0
    )
    
    # 横向位移（左脚向左）
    lateral_displacement = Vector3D(
        -step_width * 0.5 * np.sin(planner.body_heading),  # = 0
        step_width * 0.5 * np.cos(planner.body_heading),   # = step_width * 0.5
        0.0
    )
    
    # 基础目标位置
    basic_target = support_pos + forward_displacement + lateral_displacement
    
    print(f"\n步骤2 - 位移计算:")
    print(f"  前进位移: ({forward_displacement.x:.3f}, {forward_displacement.y:.3f}, {forward_displacement.z:.3f})")
    print(f"  横向位移: ({lateral_displacement.x:.3f}, {lateral_displacement.y:.3f}, {lateral_displacement.z:.3f})")
    print(f"  基础目标: ({basic_target.x:.3f}, {basic_target.y:.3f}, {basic_target.z:.3f})")
    
    # 步骤3: 应用稳定性调整
    final_target = planner._apply_stability_adjustments(basic_target, "left", "right")
    
    print(f"\n步骤3 - 稳定性调整:")
    print(f"  纵向调整: +{config.longitudinal_stability_margin:.3f}m")
    print(f"  横向调整: -{config.lateral_stability_margin:.3f}m (左脚)")
    print(f"  最终目标: ({final_target.x:.3f}, {final_target.y:.3f}, {final_target.z:.3f})")
    
    # 步骤4: 验证实际规划结果
    planned_target = planner.plan_foot_placement("left", "right")
    
    print(f"\n步骤4 - 实际规划结果:")
    print(f"  规划目标: ({planned_target.x:.3f}, {planned_target.y:.3f}, {planned_target.z:.3f})")
    
    # 计算相对于支撑足的位移
    actual_delta_x = planned_target.x - support_pos.x
    actual_delta_y = planned_target.y - support_pos.y
    
    print(f"\n=== 位移分析 ===")
    print(f"实际位移: Δx={actual_delta_x:.3f}m, Δy={actual_delta_y:.3f}m")
    
    # 期望位移（如果只考虑基础步长）
    expected_basic_delta_x = config.nominal_step_length + config.longitudinal_stability_margin
    expected_basic_delta_y = -config.nominal_step_width / 2 - config.lateral_stability_margin
    
    print(f"基础期望位移: Δx={expected_basic_delta_x:.3f}m, Δy={expected_basic_delta_y:.3f}m")
    
    # 动态调整后的期望位移
    expected_dynamic_delta_x = step_length + config.longitudinal_stability_margin
    expected_dynamic_delta_y = -config.nominal_step_width / 2 - config.lateral_stability_margin
    
    print(f"动态期望位移: Δx={expected_dynamic_delta_x:.3f}m, Δy={expected_dynamic_delta_y:.3f}m")
    
    # 误差分析
    error_basic_x = abs(actual_delta_x - expected_basic_delta_x)
    error_basic_y = abs(actual_delta_y - expected_basic_delta_y)
    error_basic_total = np.sqrt(error_basic_x**2 + error_basic_y**2)
    
    error_dynamic_x = abs(actual_delta_x - expected_dynamic_delta_x)
    error_dynamic_y = abs(actual_delta_y - expected_dynamic_delta_y)
    error_dynamic_total = np.sqrt(error_dynamic_x**2 + error_dynamic_y**2)
    
    print(f"\n=== 误差分析 ===")
    print(f"相对基础步长误差: {error_basic_total*1000:.1f}mm")
    print(f"相对动态步长误差: {error_dynamic_total*1000:.1f}mm")
    
    if error_dynamic_total < 0.001:  # < 1mm
        print("✅ 动态步长计算正确")
    else:
        print("❌ 动态步长计算有误")
    
    if error_basic_total < 0.02:  # < 2cm
        print("✅ 基础步长误差在可接受范围内")
    else:
        print("❌ 基础步长误差过大")

def test_zero_velocity_case():
    """测试零速度情况"""
    print(f"\n=== 零速度测试 ===")
    
    config = FootPlacementConfig()
    config.nominal_step_length = 0.15
    config.nominal_step_width = 0.12
    config.lateral_stability_margin = 0.03
    config.longitudinal_stability_margin = 0.02
    
    planner = FootPlacementPlanner(config)
    
    # 设置零速度
    planner.set_body_motion_intent(Vector3D(0.0, 0.0, 0.0), 0.0)
    
    # 设置支撑足位置
    planner.right_foot.position = Vector3D(0.0, -0.102, -0.649)
    
    # 计算目标位置
    target = planner.plan_foot_placement("left", "right")
    
    # 计算位移
    delta_x = target.x - planner.right_foot.position.x
    delta_y = target.y - planner.right_foot.position.y
    
    # 期望位移
    expected_delta_x = config.nominal_step_length + config.longitudinal_stability_margin
    expected_delta_y = -config.nominal_step_width / 2 - config.lateral_stability_margin
    
    # 误差
    error_x = abs(delta_x - expected_delta_x)
    error_y = abs(delta_y - expected_delta_y)
    error_total = np.sqrt(error_x**2 + error_y**2)
    
    print(f"零速度下:")
    print(f"  实际位移: Δx={delta_x:.3f}m, Δy={delta_y:.3f}m")
    print(f"  期望位移: Δx={expected_delta_x:.3f}m, Δy={expected_delta_y:.3f}m")
    print(f"  总误差: {error_total*1000:.1f}mm")
    
    if error_total < 0.02:  # < 2cm
        print("✅ 零速度情况下步长精度符合要求")
    else:
        print("❌ 零速度情况下步长精度不符合要求")

if __name__ == "__main__":
    debug_step_calculation()
    test_zero_velocity_case() 