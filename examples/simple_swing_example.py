#!/usr/bin/env python3
"""
简单的摆动足位置控制示例

这个示例展示了如何在实际的四足机器人控制中使用 get_swing_foot_position 函数。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from gait_core.swing_trajectory import get_swing_foot_position, get_swing_foot_velocity


def simple_swing_control_example():
    """简单的摆动足控制示例"""
    print("=" * 60)
    print("简单摆动足控制示例")
    print("=" * 60)
    
    # 步态参数
    start_pos = np.array([0.0, 0.15, 0.0])    # 起始位置（机器人右前足）
    target_pos = np.array([0.20, 0.15, 0.0])  # 目标位置（向前20cm）
    step_height = 0.08                         # 抬脚高度8cm
    
    # 时间参数
    swing_duration = 0.4    # 摆动周期400ms
    control_freq = 100      # 控制频率100Hz
    dt = 1.0 / control_freq
    
    print(f"控制参数:")
    print(f"  起始位置: {start_pos} [m]")
    print(f"  目标位置: {target_pos} [m]")
    print(f"  步长: {np.linalg.norm(target_pos - start_pos):.2f} [m]")
    print(f"  抬脚高度: {step_height} [m]")
    print(f"  摆动周期: {swing_duration} [s]")
    print(f"  控制频率: {control_freq} [Hz]")
    print()
    
    # 模拟控制循环
    print("摆动控制过程 (显示关键时刻):")
    print("-" * 70)
    print(f"{'时间[ms]':<10} {'相位':<8} {'位置[m]':<25} {'速度[m/s]':<15} {'状态':<10}")
    print("-" * 70)
    
    current_time = 0.0
    while current_time <= swing_duration:
        # 计算当前相位
        phase = current_time / swing_duration
        if phase > 1.0:
            phase = 1.0
        
        # 计算期望位置和速度
        desired_pos = get_swing_foot_position(
            phase, start_pos, target_pos, step_height, 
            interpolation_type="cubic",      # 使用三次插值确保平滑
            vertical_trajectory_type="sine"  # 使用正弦轨迹
        )
        
        desired_vel = get_swing_foot_velocity(
            phase, start_pos, target_pos, step_height, swing_duration,
            interpolation_type="cubic",
            vertical_trajectory_type="sine"
        )
        
        # 确定当前状态
        if phase <= 0.1:
            status = "起步"
        elif phase <= 0.4:
            status = "上升"
        elif phase <= 0.6:
            status = "顶峰"
        elif phase <= 0.9:
            status = "下降"
        else:
            status = "着地"
        
        # 显示关键时刻的数据
        if current_time == 0.0 or abs(phase - 0.25) < 0.01 or abs(phase - 0.5) < 0.01 or \
           abs(phase - 0.75) < 0.01 or phase >= 0.99:
            vel_mag = np.linalg.norm(desired_vel)
            pos_str = f"({desired_pos[0]:.3f},{desired_pos[1]:.3f},{desired_pos[2]:.3f})"
            print(f"{current_time*1000:<10.0f} {phase:<8.3f} {pos_str:<25} {vel_mag:<15.3f} {status:<10}")
        
        # 在这里可以发送控制命令给机器人...
        # robot.set_foot_position("RF", desired_pos)
        # robot.set_foot_velocity("RF", desired_vel)
        
        current_time += dt
    
    print("-" * 70)
    print("✓ 摆动控制完成")
    print()


def quadruped_gait_example():
    """四足步态控制示例"""
    print("=" * 60)
    print("四足步态控制示例 - Trot步态")
    print("=" * 60)
    
    # 四足初始位置（站立姿态）
    foot_positions = {
        "RF": np.array([0.15, -0.15, 0.0]),   # 右前
        "LF": np.array([0.15,  0.15, 0.0]),   # 左前
        "RH": np.array([-0.15, -0.15, 0.0]),  # 右后
        "LH": np.array([-0.15,  0.15, 0.0])   # 左后
    }
    
    # 步长设置
    step_length = 0.10  # 10cm步长
    step_height = 0.06  # 6cm抬脚高度
    step_duration = 0.3 # 300ms摆动周期
    
    # Trot步态：对角足同时摆动
    # 第一阶段：RF + LH 摆动，LF + RH 支撑
    print("第一阶段 - RF和LH摆动：")
    print("-" * 40)
    
    # 计算目标位置
    rf_target = foot_positions["RF"] + np.array([step_length, 0, 0])
    lh_target = foot_positions["LH"] + np.array([step_length, 0, 0])
    
    # 显示几个关键相位的位置
    key_phases = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for phase in key_phases:
        rf_pos = get_swing_foot_position(phase, foot_positions["RF"], rf_target, 
                                        step_height, "cubic", "sine")
        lh_pos = get_swing_foot_position(phase, foot_positions["LH"], lh_target,
                                        step_height, "cubic", "sine")
        
        print(f"  相位 {phase:.2f}: RF({rf_pos[0]:.3f},{rf_pos[1]:.3f},{rf_pos[2]:.3f}) "
              f"LH({lh_pos[0]:.3f},{lh_pos[1]:.3f},{lh_pos[2]:.3f})")
    
    print("\n第二阶段 - LF和RH摆动：")
    print("-" * 40)
    
    # 更新足部位置（第一阶段结束后）
    foot_positions["RF"] = rf_target
    foot_positions["LH"] = lh_target
    
    # 计算第二阶段目标位置
    lf_target = foot_positions["LF"] + np.array([step_length, 0, 0])
    rh_target = foot_positions["RH"] + np.array([step_length, 0, 0])
    
    for phase in key_phases:
        lf_pos = get_swing_foot_position(phase, foot_positions["LF"], lf_target,
                                        step_height, "cubic", "sine")
        rh_pos = get_swing_foot_position(phase, foot_positions["RH"], rh_target,
                                        step_height, "cubic", "sine")
        
        print(f"  相位 {phase:.2f}: LF({lf_pos[0]:.3f},{lf_pos[1]:.3f},{lf_pos[2]:.3f}) "
              f"RH({rh_pos[0]:.3f},{rh_pos[1]:.3f},{rh_pos[2]:.3f})")
    
    print(f"\n✓ 一个完整步态周期完成，机器人前进 {step_length:.2f}m")


def terrain_adaptive_example():
    """地形自适应示例"""
    print("\n" + "=" * 60)
    print("地形自适应摆动轨迹示例")
    print("=" * 60)
    
    # 不同地形的参数设置
    terrains = {
        "平地": {
            "step_height": 0.05,
            "interpolation": "linear",
            "vertical": "sine",
            "description": "节能模式，最小抬脚高度"
        },
        "草地": {
            "step_height": 0.08,
            "interpolation": "cubic",
            "vertical": "sine",
            "description": "标准模式，中等抬脚高度"
        },
        "石块地": {
            "step_height": 0.12,
            "interpolation": "smooth",
            "vertical": "parabola",
            "description": "高抬腿模式，避障优先"
        }
    }
    
    start_pos = np.array([0.0, 0.15, 0.0])
    target_pos = np.array([0.15, 0.15, 0.0])
    
    print(f"起始位置: {start_pos}")
    print(f"目标位置: {target_pos}")
    print()
    
    for terrain_name, params in terrains.items():
        print(f"{terrain_name}地形:")
        print(f"  {params['description']}")
        print(f"  参数: 高度={params['step_height']}m, "
              f"插值={params['interpolation']}, 轨迹={params['vertical']}")
        
        # 计算关键点位置
        mid_pos = get_swing_foot_position(
            0.5, start_pos, target_pos, params['step_height'],
            params['interpolation'], params['vertical']
        )
        
        print(f"  最高点位置: ({mid_pos[0]:.3f}, {mid_pos[1]:.3f}, {mid_pos[2]:.3f})")
        print(f"  最大抬脚高度: {mid_pos[2]:.3f}m ({mid_pos[2]*100:.0f}cm)")
        print()
    
    print("✓ 地形自适应配置完成")


def real_time_integration_example():
    """实时集成示例"""
    print("\n" + "=" * 60)
    print("实时控制集成示例")
    print("=" * 60)
    
    print("示例代码结构:")
    print("""
# 在实际机器人控制循环中的使用方式

import numpy as np
from gait_core.swing_trajectory import get_swing_foot_position

class QuadrupedController:
    def __init__(self):
        self.control_frequency = 200  # 200Hz高频控制
        self.dt = 1.0 / self.control_frequency
        
    def update_swing_foot(self, foot_name, phase, start_pos, target_pos):
        \"\"\"更新摆动足位置\"\"\"
        # 根据地形和速度选择参数
        if self.terrain_type == "rough":
            step_height = 0.12
            interp_type = "smooth"
            vert_type = "parabola"
        else:
            step_height = 0.08
            interp_type = "cubic"
            vert_type = "sine"
        
        # 计算期望位置
        desired_pos = get_swing_foot_position(
            phase, start_pos, target_pos, step_height,
            interp_type, vert_type
        )
        
        # 发送到底层控制器
        self.robot.set_foot_position(foot_name, desired_pos)
        return desired_pos
        
    def control_loop(self):
        \"\"\"主控制循环\"\"\"
        while self.running:
            current_time = time.time()
            
            # 更新步态相位
            for foot_name in ["RF", "LF", "RH", "LH"]:
                if self.is_swing_phase(foot_name):
                    phase = self.get_swing_phase(foot_name)
                    start_pos = self.swing_start_positions[foot_name]
                    target_pos = self.swing_target_positions[foot_name]
                    
                    # 更新摆动足位置
                    self.update_swing_foot(foot_name, phase, start_pos, target_pos)
            
            time.sleep(self.dt)
    """)
    
    print("关键特性:")
    print("✓ 高频控制（200Hz）支持")
    print("✓ 实时参数调整")
    print("✓ 地形自适应")
    print("✓ 多足协调控制")
    print("✓ 平滑轨迹生成")


def main():
    """主函数"""
    print("摆动足位置函数使用示例")
    print("展示 get_swing_foot_position 在实际控制中的应用")
    print()
    
    # 运行各种示例
    simple_swing_control_example()
    quadruped_gait_example()
    terrain_adaptive_example()
    real_time_integration_example()
    
    print("\n" + "=" * 60)
    print("所有示例演示完成！")
    print("=" * 60)
    print("\n核心函数 get_swing_foot_position 的优势:")
    print("✓ 参数化相位控制：phase ∈ [0,1]")
    print("✓ 多种插值方法：linear, cubic, smooth")
    print("✓ 多种垂直轨迹：sine, parabola, smooth_parabola")
    print("✓ 精确的边界条件：起终点位置保证")
    print("✓ 实时计算性能：支持高频控制")
    print("✓ 易于集成：简单的函数接口")
    print("\n使用建议:")
    print("• 标准行走：get_swing_foot_position(phase, start, target, 0.08, 'cubic', 'sine')")
    print("• 障碍环境：get_swing_foot_position(phase, start, target, 0.12, 'smooth', 'parabola')")
    print("• 节能模式：get_swing_foot_position(phase, start, target, 0.05, 'linear', 'sine')")


if __name__ == "__main__":
    main() 