#!/usr/bin/env python3
"""
步态节拍调试测试 - 详细版本

检查状态转换条件和时序控制的详细信息
"""

import time
import numpy as np
from typing import Dict
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig, GaitState, LegState


class GaitTimingDebugger:
    """步态时序调试器"""
    
    def __init__(self):
        # 测试配置
        self.dt = 0.005  # 5ms 控制周期
        
        # 配置步态参数  
        config = GaitSchedulerConfig()
        config.swing_time = 0.4         # 400ms 摆动时间
        config.stance_time = 0.6        # 600ms 支撑时间
        config.double_support_time = 0.1 # 100ms 双支撑时间
        config.use_time_trigger = True
        config.use_sensor_trigger = True
        config.require_both_triggers = False  # 只要时间或传感器条件满足即可
        config.enable_logging = True
        config.touchdown_force_threshold = 30.0
        
        self.scheduler = get_gait_scheduler(config)
        self.data_bus = get_data_bus()
        
        self.current_time = 0.0
        self.debug_log = []
        
        print("="*80)
        print("步态时序调试器")
        print("="*80)
        print(f"摆动时间: {config.swing_time*1000:.1f}ms")
        print(f"双支撑时间: {config.double_support_time*1000:.1f}ms")
        print(f"控制周期: {self.dt*1000:.1f}ms")
    
    def setup_initial_state(self):
        """设置初始状态"""
        # 确保传感器状态正确
        self.scheduler.left_foot_force = 100.0
        self.scheduler.right_foot_force = 100.0
        self.scheduler.left_foot_contact = True
        self.scheduler.right_foot_contact = True
        
        # 启动步态
        self.scheduler.start_walking()
        print(f"\n初始状态: {self.scheduler.current_state.value}")
        print(f"初始摆动腿: {self.scheduler.swing_leg}")
    
    def simulate_perfect_sensors(self):
        """模拟完美的传感器响应"""
        current_state = self.scheduler.current_state
        swing_progress = 0.0
        
        if self.scheduler.swing_leg != "none" and self.scheduler.config.swing_time > 0:
            swing_progress = self.scheduler.swing_elapsed_time / self.scheduler.config.swing_time
        
        # 根据状态和进度设置传感器
        if current_state == GaitState.LEFT_SUPPORT:
            # 左腿支撑，右腿摆动
            self.scheduler.left_foot_contact = True
            self.scheduler.left_foot_force = 100.0
            
            # 在摆动95%时右腿着地
            if swing_progress >= 0.95:
                self.scheduler.right_foot_contact = True
                self.scheduler.right_foot_force = 100.0
            else:
                self.scheduler.right_foot_contact = False
                self.scheduler.right_foot_force = 0.0
                
        elif current_state == GaitState.RIGHT_SUPPORT:
            # 右腿支撑，左腿摆动
            self.scheduler.right_foot_contact = True
            self.scheduler.right_foot_force = 100.0
            
            # 在摆动95%时左腿着地
            if swing_progress >= 0.95:
                self.scheduler.left_foot_contact = True
                self.scheduler.left_foot_force = 100.0
            else:
                self.scheduler.left_foot_contact = False
                self.scheduler.left_foot_force = 0.0
        else:
            # 双支撑或其他状态
            self.scheduler.left_foot_contact = True
            self.scheduler.right_foot_contact = True
            self.scheduler.left_foot_force = 100.0
            self.scheduler.right_foot_force = 100.0
    
    def debug_transition_conditions(self):
        """调试状态转换条件"""
        current_state = self.scheduler.current_state
        
        # 检查当前状态的所有可能转换
        if current_state in self.scheduler.transitions:
            print(f"\n[{self.current_time:.3f}s] 状态: {current_state.value}")
            print(f"  摆动腿: {self.scheduler.swing_leg}")
            print(f"  摆动时间: {self.scheduler.swing_elapsed_time:.3f}s / {self.scheduler.config.swing_time:.3f}s")
            print(f"  传感器: L={self.scheduler.left_foot_force:.1f}N/{self.scheduler.left_foot_contact}, R={self.scheduler.right_foot_force:.1f}N/{self.scheduler.right_foot_contact}")
            
            # 检查每个可能的转换
            for i, transition in enumerate(self.scheduler.transitions[current_state]):
                print(f"  转换{i+1}: → {transition.to_state.value}")
                print(f"    触发类型: {transition.trigger_type.value}")
                print(f"    时间条件: {transition.time_condition:.3f}s")
                print(f"    力阈值: {transition.force_threshold:.1f}N")
                
                # 计算当前持续时间
                current_duration = time.time() - self.scheduler.state_start_time
                print(f"    当前持续时间: {current_duration:.3f}s")
                
                # 检查时间条件
                time_satisfied = current_duration >= transition.time_condition
                print(f"    时间满足: {time_satisfied}")
                
                # 检查传感器条件
                sensor_satisfied = self.scheduler._check_sensor_condition(transition)
                print(f"    传感器满足: {sensor_satisfied}")
                
                # 检查总体条件
                basic_condition = self.scheduler._check_transition_condition(transition, current_duration)
                print(f"    基本条件: {basic_condition}")
                
                # 检查步完成条件
                step_condition = True
                if (current_state in [GaitState.LEFT_SUPPORT, GaitState.RIGHT_SUPPORT] and
                    transition.to_state in [GaitState.DOUBLE_SUPPORT_LR, GaitState.DOUBLE_SUPPORT_RL]):
                    step_condition = self.scheduler._check_step_completion()
                    print(f"    步完成条件: {step_condition}")
                
                final_condition = basic_condition and step_condition
                print(f"    最终条件: {final_condition}")
                
                if final_condition:
                    print(f"    >>> 应该转换到 {transition.to_state.value} <<<")
                
                print()
    
    def run_debug_test(self):
        """运行调试测试"""
        print(f"\n开始调试测试...")
        
        self.setup_initial_state()
        
        # 测试参数
        max_test_time = 3.0  # 最大测试时间
        test_steps = int(max_test_time / self.dt)
        
        print(f"测试参数:")
        print(f"  最大测试时间: {max_test_time}s")
        print(f"  控制周期: {self.dt*1000:.1f}ms")
        print(f"  总步数: {test_steps}")
        print("-" * 80)
        
        last_state = self.scheduler.current_state
        
        # 主测试循环
        for step in range(test_steps):
            self.current_time = step * self.dt
            
            # 模拟传感器
            self.simulate_perfect_sensors()
            
            # 调试转换条件（每100ms检查一次）
            if step % int(0.1 / self.dt) == 0:
                self.debug_transition_conditions()
            
            # 推进状态机
            state_changed = self.scheduler.update_gait_state(self.dt)
            
            # 检查状态变化
            current_state = self.scheduler.current_state
            if current_state != last_state:
                print(f"*** 状态变化: {last_state.value} → {current_state.value} ***")
                last_state = current_state
            
            # 如果有状态变化，提前结束一轮调试
            if state_changed:
                print(f"检测到状态变化，进入下一轮...")
                time.sleep(0.1)  # 短暂暂停
        
        print(f"\n调试测试完成")


def main():
    """主函数"""
    try:
        debugger = GaitTimingDebugger()
        debugger.run_debug_test()
        return True
    except Exception as e:
        print(f"❌ 调试测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 