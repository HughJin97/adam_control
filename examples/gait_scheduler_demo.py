#!/usr/bin/env python3
"""
步态调度器演示脚本

展示步态有限状态机的实际使用场景和最佳实践。
包括实时步态控制、传感器反馈处理、与数据总线的集成等。

功能演示：
1. 基础步态控制循环
2. 实时传感器数据处理
3. 状态转换监控
4. 步态参数动态调整
5. 紧急情况处理

作者: Adam Control Team
版本: 1.0
"""

import time
import numpy as np
import threading
from typing import Dict, List, Optional
from gait_scheduler import (
    get_gait_scheduler, GaitSchedulerConfig, GaitState, LegState, 
    StateInfo, TransitionTrigger
)
from data_bus import get_data_bus
from gait_parameters import get_gait_manager


class GaitControlDemo:
    """步态控制演示类"""
    
    def __init__(self):
        """初始化演示系统"""
        # 配置步态调度器
        self.scheduler_config = GaitSchedulerConfig(
            swing_time=0.4,
            stance_time=0.6,
            double_support_time=0.1,
            touchdown_force_threshold=30.0,
            liftoff_force_threshold=10.0,
            contact_velocity_threshold=0.02,
            use_time_trigger=True,
            use_sensor_trigger=True,
            require_both_triggers=False,
            enable_logging=True,
            log_transitions=True
        )
        
        # 初始化系统组件
        self.scheduler = get_gait_scheduler(self.scheduler_config)
        self.data_bus = get_data_bus()
        self.gait_params = get_gait_manager()
        
        # 运行控制
        self.running = False
        self.demo_thread = None
        
        # 监控数据
        self.step_count = 0
        self.cycle_count = 0
        self.start_time = 0.0
        
        # 回调注册
        self.scheduler.add_state_change_callback(self.on_state_change)
        
        print("步态控制演示系统初始化完成")
        print(f"调度器可用: {self.data_bus._has_gait_scheduler}")
        print(f"参数管理器可用: {self.data_bus._has_gait_manager}")
    
    def on_state_change(self, old_state: GaitState, new_state: GaitState, duration: float):
        """状态变化回调函数"""
        print(f"[{time.time() - self.start_time:.2f}s] 状态转换: "
              f"{old_state.value} -> {new_state.value} (持续: {duration:.3f}s)")
        
        # 统计步数和周期
        if new_state in [GaitState.LEFT_SUPPORT, GaitState.RIGHT_SUPPORT]:
            self.step_count += 1
        
        if (old_state == GaitState.DOUBLE_SUPPORT_RL and 
            new_state == GaitState.LEFT_SUPPORT):
            self.cycle_count += 1
    
    def simulate_sensor_data(self) -> tuple:
        """
        模拟传感器数据
        
        返回:
            tuple: (left_force, right_force, left_velocity, right_velocity)
        """
        current_time = time.time() - self.start_time
        
        # 根据当前步态状态生成传感器数据
        if self.scheduler.current_state == GaitState.LEFT_SUPPORT:
            # 左腿支撑，右腿摆动
            left_force = 70.0 + 10.0 * np.sin(current_time * 2)  # 支撑腿有变化的力
            
            # 模拟右腿摆动过程中的力变化
            swing_progress = (current_time % self.scheduler.config.swing_time) / self.scheduler.config.swing_time
            if swing_progress > 0.8:  # 摆动后期，准备着地
                right_force = 15.0 + 20.0 * (swing_progress - 0.8) / 0.2
            else:
                right_force = 5.0 + 5.0 * np.sin(swing_progress * np.pi)  # 摆动中的小力变化
            
            left_velocity = np.array([0.0, 0.0, 0.01 + 0.005 * np.sin(current_time * 3)])
            right_velocity = np.array([0.0, 0.0, 0.1 * (1 - swing_progress)])
            
        elif self.scheduler.current_state == GaitState.RIGHT_SUPPORT:
            # 右腿支撑，左腿摆动
            right_force = 70.0 + 10.0 * np.sin(current_time * 2)
            
            swing_progress = (current_time % self.scheduler.config.swing_time) / self.scheduler.config.swing_time
            if swing_progress > 0.8:  # 摆动后期，准备着地
                left_force = 15.0 + 20.0 * (swing_progress - 0.8) / 0.2
            else:
                left_force = 5.0 + 5.0 * np.sin(swing_progress * np.pi)
            
            right_velocity = np.array([0.0, 0.0, 0.01 + 0.005 * np.sin(current_time * 3)])
            left_velocity = np.array([0.0, 0.0, 0.1 * (1 - swing_progress)])
            
        elif self.scheduler.current_state in [GaitState.DOUBLE_SUPPORT_LR, 
                                             GaitState.DOUBLE_SUPPORT_RL]:
            # 双支撑阶段
            left_force = 45.0 + 5.0 * np.sin(current_time * 4)
            right_force = 45.0 + 5.0 * np.cos(current_time * 4)
            left_velocity = np.array([0.0, 0.0, 0.005])
            right_velocity = np.array([0.0, 0.0, 0.005])
            
        else:  # 站立或其他状态
            left_force = right_force = 40.0
            left_velocity = right_velocity = np.array([0.0, 0.0, 0.0])
        
        # 添加一些噪声模拟真实传感器
        noise_scale = 2.0
        left_force += np.random.normal(0, noise_scale)
        right_force += np.random.normal(0, noise_scale)
        
        return left_force, right_force, left_velocity, right_velocity
    
    def update_data_bus_sensors(self, left_force: float, right_force: float,
                               left_vel: np.ndarray, right_vel: np.ndarray):
        """更新数据总线中的传感器数据"""
        # 更新足部力传感器
        self.data_bus.end_effectors["left_foot"].contact_force_magnitude = left_force
        self.data_bus.end_effectors["right_foot"].contact_force_magnitude = right_force
        
        # 更新足部速度
        self.data_bus.end_effectors["left_foot"].velocity.x = left_vel[0]
        self.data_bus.end_effectors["left_foot"].velocity.y = left_vel[1]
        self.data_bus.end_effectors["left_foot"].velocity.z = left_vel[2]
        
        self.data_bus.end_effectors["right_foot"].velocity.x = right_vel[0]
        self.data_bus.end_effectors["right_foot"].velocity.y = right_vel[1]
        self.data_bus.end_effectors["right_foot"].velocity.z = right_vel[2]
        
        # 更新接触状态
        left_contact = "CONTACT" if left_force > 20.0 else "NO_CONTACT"
        right_contact = "CONTACT" if right_force > 20.0 else "NO_CONTACT"
        
        self.data_bus.set_end_effector_contact_state("left_foot", left_contact)
        self.data_bus.set_end_effector_contact_state("right_foot", right_contact)
    
    def control_loop(self):
        """主控制循环"""
        print("\n=== 开始步态控制循环 ===")
        self.start_time = time.time()
        
        dt = 0.02  # 50Hz 控制频率
        
        # 启动步态
        print("启动步态调度器...")
        self.data_bus.start_gait_scheduler_walking()
        
        loop_count = 0
        last_print_time = 0.0
        
        while self.running:
            loop_start = time.time()
            current_time = loop_start - self.start_time
            
            # 1. 获取传感器数据
            left_force, right_force, left_vel, right_vel = self.simulate_sensor_data()
            
            # 2. 更新数据总线传感器数据
            self.update_data_bus_sensors(left_force, right_force, left_vel, right_vel)
            
            # 3. 通过数据总线更新步态调度器
            state_changed = self.data_bus.update_gait_scheduler(dt)
            
            # 4. 检查紧急情况
            if max(left_force, right_force) > 150.0:  # 异常大力
                print(f"[{current_time:.2f}s] 检测到异常力: {max(left_force, right_force):.1f}N")
                self.data_bus.emergency_stop_gait()
                break
            
            # 5. 定期打印状态信息
            if current_time - last_print_time >= 2.0:  # 每2秒打印一次
                self.print_status_summary(current_time)
                last_print_time = current_time
            
            # 6. 控制频率
            loop_count += 1
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
        
        print(f"\n控制循环结束，共运行 {loop_count} 次迭代")
    
    def print_status_summary(self, current_time: float):
        """打印状态摘要"""
        state_info = self.data_bus.get_gait_state_info()
        
        print(f"\n--- 步态状态摘要 ({current_time:.1f}s) ---")
        print(f"当前状态: {state_info['current_gait_state']}")
        print(f"支撑腿: {state_info['leg_state']} ({state_info['support_leg']})")
        print(f"摆动腿: {state_info['swing_leg']}")
        print(f"状态持续: {state_info.get('current_state_duration', 0):.2f}s")
        print(f"步数: {self.step_count}, 周期: {self.cycle_count}")
        print(f"状态转换次数: {state_info['state_transition_count']}")
        print(f"左脚力: {state_info.get('left_foot_force', 0):.1f}N, "
              f"右脚力: {state_info.get('right_foot_force', 0):.1f}N")
        print(f"左脚接触: {state_info.get('left_foot_contact', False)}, "
              f"右脚接触: {state_info.get('right_foot_contact', False)}")
    
    def demo_basic_walking(self, duration: float = 10.0):
        """演示基础行走"""
        print(f"\n=== 演示基础行走 ({duration}s) ===")
        
        self.running = True
        self.demo_thread = threading.Thread(target=self.control_loop)
        self.demo_thread.start()
        
        # 运行指定时间
        time.sleep(duration)
        
        # 停止
        self.running = False
        if self.demo_thread:
            self.demo_thread.join()
        
        # 打印最终统计
        self.print_final_statistics()
    
    def demo_dynamic_parameter_adjustment(self):
        """演示动态参数调整"""
        print("\n=== 演示动态参数调整 ===")
        
        # 启动控制循环
        self.running = True
        self.demo_thread = threading.Thread(target=self.control_loop)
        self.demo_thread.start()
        
        try:
            # 阶段1: 正常行走
            print("阶段1: 正常步态 (3秒)")
            time.sleep(3.0)
            
            # 阶段2: 加快步频
            print("阶段2: 加快步频")
            fast_config = {
                'swing_time': 0.3,
                'double_support_time': 0.08
            }
            self.data_bus.set_gait_scheduler_config(fast_config)
            time.sleep(3.0)
            
            # 阶段3: 放慢步频
            print("阶段3: 放慢步频")
            slow_config = {
                'swing_time': 0.6,
                'double_support_time': 0.15
            }
            self.data_bus.set_gait_scheduler_config(slow_config)
            time.sleep(3.0)
            
            # 阶段4: 调整传感器敏感度
            print("阶段4: 提高传感器敏感度")
            sensitive_config = {
                'touchdown_force_threshold': 20.0,
                'contact_velocity_threshold': 0.01
            }
            self.data_bus.set_gait_scheduler_config(sensitive_config)
            time.sleep(3.0)
            
        finally:
            # 停止演示
            self.running = False
            if self.demo_thread:
                self.demo_thread.join()
        
        print("动态参数调整演示完成")
    
    def demo_emergency_scenarios(self):
        """演示紧急场景处理"""
        print("\n=== 演示紧急场景处理 ===")
        
        # 启动控制循环
        self.running = True
        self.demo_thread = threading.Thread(target=self.control_loop)
        self.demo_thread.start()
        
        try:
            # 正常行走一段时间
            print("正常行走...")
            time.sleep(2.0)
            
            # 模拟紧急停止 - 手动触发
            print("模拟手动紧急停止...")
            self.data_bus.emergency_stop_gait()
            time.sleep(1.0)
            
            # 恢复
            print("恢复到站立状态...")
            self.data_bus.reset_gait_scheduler()
            time.sleep(0.5)
            
            # 重新开始
            print("重新开始行走...")
            self.data_bus.start_gait_scheduler_walking()
            time.sleep(2.0)
            
        finally:
            self.running = False
            if self.demo_thread:
                self.demo_thread.join()
        
        print("紧急场景演示完成")
    
    def demo_sensor_fusion(self):
        """演示传感器融合"""
        print("\n=== 演示传感器融合模式 ===")
        
        # 配置为需要时间和传感器双重确认
        fusion_config = {
            'require_both_triggers': True,
            'use_time_trigger': True,
            'use_sensor_trigger': True
        }
        self.data_bus.set_gait_scheduler_config(fusion_config)
        
        print("配置双触发模式（时间 + 传感器）")
        
        # 运行演示
        self.demo_basic_walking(8.0)
        
        # 恢复单触发模式
        single_config = {
            'require_both_triggers': False
        }
        self.data_bus.set_gait_scheduler_config(single_config)
        
        print("传感器融合演示完成")
    
    def print_final_statistics(self):
        """打印最终统计信息"""
        print("\n=== 最终统计信息 ===")
        
        # 从调度器获取详细统计
        stats = self.data_bus.get_gait_state_statistics()
        
        if 'error' not in stats:
            print(f"总运行时间: {stats['total_duration']:.2f}s")
            print(f"总状态转换: {stats['total_transitions']} 次")
            print(f"步数统计: {self.step_count} 步")
            print(f"步态周期: {self.cycle_count} 个完整周期")
            
            if stats['total_duration'] > 0:
                print(f"平均步频: {self.step_count / stats['total_duration']:.2f} 步/秒")
                print(f"平均周期频率: {self.cycle_count / stats['total_duration']:.2f} 周期/秒")
            
            print("\n各状态时间分布:")
            for state, data in stats['state_stats'].items():
                print(f"  {state:20s}: {data['percentage']:5.1f}% "
                      f"({data['avg_duration']:.3f}s 平均)")
        
        # 打印数据总线状态
        self.data_bus.print_gait_status()
    
    def run_interactive_demo(self):
        """运行交互式演示"""
        print("\n=== 步态调度器交互式演示 ===")
        
        while True:
            print("\n请选择演示模式:")
            print("1. 基础行走演示")
            print("2. 动态参数调整演示")
            print("3. 紧急场景演示")
            print("4. 传感器融合演示")
            print("5. 查看当前状态")
            print("6. 重置系统")
            print("0. 退出")
            
            choice = input("\n请输入选择 (0-6): ").strip()
            
            if choice == '1':
                duration = float(input("请输入行走时间(秒，默认10): ") or "10")
                self.demo_basic_walking(duration)
                
            elif choice == '2':
                self.demo_dynamic_parameter_adjustment()
                
            elif choice == '3':
                self.demo_emergency_scenarios()
                
            elif choice == '4':
                self.demo_sensor_fusion()
                
            elif choice == '5':
                self.data_bus.print_gait_status()
                
            elif choice == '6':
                print("重置系统...")
                self.data_bus.reset_gait_scheduler()
                self.step_count = 0
                self.cycle_count = 0
                print("系统已重置")
                
            elif choice == '0':
                print("退出演示")
                break
                
            else:
                print("无效选择，请重新输入")


def main():
    """主函数"""
    print("步态调度器演示系统")
    print("=" * 50)
    
    try:
        demo = GaitControlDemo()
        
        # 可以选择运行特定演示或交互式演示
        import sys
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
            if mode == 'basic':
                demo.demo_basic_walking(10.0)
            elif mode == 'dynamic':
                demo.demo_dynamic_parameter_adjustment()
            elif mode == 'emergency':
                demo.demo_emergency_scenarios()
            elif mode == 'fusion':
                demo.demo_sensor_fusion()
            else:
                print(f"未知模式: {mode}")
                print("可用模式: basic, dynamic, emergency, fusion")
        else:
            # 默认运行交互式演示
            demo.run_interactive_demo()
    
    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 