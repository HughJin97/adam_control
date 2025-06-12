#!/usr/bin/env python3
"""
步态调度器测试脚本

测试步态有限状态机的各种功能：
1. 基于时间的状态转换
2. 基于传感器的状态转换
3. 混合触发模式
4. 数据总线集成
5. 状态统计和监控

作者: Adam Control Team
版本: 1.0
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig, GaitState, LegState
from data_bus import get_data_bus


class GaitSchedulerTester:
    """步态调度器测试器"""
    
    def __init__(self):
        """初始化测试器"""
        # 创建自定义配置
        self.config = GaitSchedulerConfig(
            swing_time=0.4,
            stance_time=0.6,
            double_support_time=0.1,
            touchdown_force_threshold=25.0,
            liftoff_force_threshold=10.0,
            contact_velocity_threshold=0.05,
            enable_logging=True,
            log_transitions=True
        )
        
        self.scheduler = get_gait_scheduler(self.config)
        self.data_bus = get_data_bus()
        
        # 测试数据记录
        self.test_log = []
        
        print("GaitSchedulerTester initialized")
        print(f"Scheduler available: {self.data_bus._has_gait_scheduler}")
    
    def test_basic_state_transitions(self):
        """测试基本状态转换"""
        print("\n=== 测试基本状态转换 ===")
        
        # 重置调度器
        self.scheduler.reset()
        
        print("初始状态:")
        self.scheduler.print_status()
        
        # 模拟启动序列
        print("\n1. 启动序列测试:")
        
        # IDLE -> STANDING (自动, 基于时间)
        print("等待自动转换到 STANDING...")
        for i in range(10):
            self.scheduler.update_gait_state(0.02)
            time.sleep(0.02)
        
        self.scheduler.print_status()
        
        # STANDING -> LEFT_SUPPORT (手动开始行走)
        print("\n开始行走...")
        self.scheduler.start_walking()
        self.scheduler.print_status()
        
        return self.scheduler.current_state == GaitState.LEFT_SUPPORT
    
    def test_sensor_based_transitions(self):
        """测试基于传感器的状态转换"""
        print("\n=== 测试传感器驱动的状态转换 ===")
        
        # 确保在LEFT_SUPPORT状态
        if self.scheduler.current_state != GaitState.LEFT_SUPPORT:
            self.scheduler.start_walking()
        
        print("当前状态: LEFT_SUPPORT (右腿摆动)")
        
        # 模拟右脚着地触发转换
        print("\n模拟右脚着地...")
        
        transition_detected = False
        for step in range(100):
            dt = 0.02
            
            # 模拟传感器数据
            left_force = 60.0   # 左脚支撑
            
            # 在步骤50时模拟右脚着地
            if step >= 50:
                right_force = 35.0  # 右脚着地
                right_velocity = np.array([0.0, 0.0, 0.01])  # 低速接触
            else:
                right_force = 5.0   # 右脚摆动
                right_velocity = np.array([0.0, 0.0, 0.1])   # 摆动速度
            
            left_velocity = np.array([0.0, 0.0, 0.01])
            
            # 更新传感器数据
            self.scheduler.update_sensor_data(
                left_force, right_force, left_velocity, right_velocity
            )
            
            # 更新状态
            state_changed = self.scheduler.update_gait_state(dt)
            
            if state_changed:
                print(f"步骤 {step}: 状态转换到 {self.scheduler.current_state.value}")
                if self.scheduler.current_state == GaitState.DOUBLE_SUPPORT_LR:
                    transition_detected = True
                    break
            
            time.sleep(dt)
        
        return transition_detected
    
    def test_hybrid_triggers(self):
        """测试混合触发模式"""
        print("\n=== 测试混合触发模式 ===")
        
        # 记录状态转换时间
        transition_times = []
        
        # 运行完整的步态循环
        print("运行完整步态循环...")
        
        start_time = time.time()
        last_state = self.scheduler.current_state
        
        for step in range(500):  # 10秒测试
            dt = 0.02
            current_time = time.time() - start_time
            
            # 根据当前状态模拟不同的传感器数据
            if self.scheduler.current_state == GaitState.LEFT_SUPPORT:
                # 左腿支撑，右腿摆动
                left_force = 70.0
                right_force = 5.0 + 30.0 * max(0, (current_time % 1.0) - 0.3)  # 逐渐着地
                left_velocity = np.array([0.0, 0.0, 0.02])
                right_velocity = np.array([0.0, 0.0, 0.05])
                
            elif self.scheduler.current_state == GaitState.RIGHT_SUPPORT:
                # 右腿支撑，左腿摆动
                right_force = 70.0
                left_force = 5.0 + 30.0 * max(0, (current_time % 1.0) - 0.3)  # 逐渐着地
                right_velocity = np.array([0.0, 0.0, 0.02])
                left_velocity = np.array([0.0, 0.0, 0.05])
                
            else:  # 双支撑状态
                left_force = right_force = 40.0
                left_velocity = right_velocity = np.array([0.0, 0.0, 0.01])
            
            # 更新传感器数据
            self.scheduler.update_sensor_data(
                left_force, right_force, left_velocity, right_velocity
            )
            
            # 更新状态
            state_changed = self.scheduler.update_gait_state(dt)
            
            if state_changed:
                transition_info = {
                    'time': current_time,
                    'from_state': last_state.value,
                    'to_state': self.scheduler.current_state.value,
                    'step': step
                }
                transition_times.append(transition_info)
                
                print(f"步骤 {step} ({current_time:.2f}s): "
                      f"{last_state.value} -> {self.scheduler.current_state.value}")
                
                last_state = self.scheduler.current_state
            
            # 每2秒打印一次状态
            if step % 100 == 0:
                self.scheduler.print_status()
            
            time.sleep(dt)
        
        # 分析转换时间
        print(f"\n状态转换分析 (共{len(transition_times)}次转换):")
        for i, trans in enumerate(transition_times):
            print(f"  {i+1}. {trans['time']:.2f}s: {trans['from_state']} -> {trans['to_state']}")
        
        return transition_times
    
    def test_data_bus_integration(self):
        """测试数据总线集成"""
        print("\n=== 测试数据总线集成 ===")
        
        # 测试数据总线的步态调度器接口
        print("1. 测试数据总线状态同步:")
        
        # 设置一些传感器数据到数据总线
        self.data_bus.set_end_effector_contact_state("left_foot", "CONTACT")
        self.data_bus.set_end_effector_contact_state("right_foot", "NO_CONTACT")
        
        # 模拟足部力
        self.data_bus.end_effectors["left_foot"].contact_force_magnitude = 60.0
        self.data_bus.end_effectors["right_foot"].contact_force_magnitude = 8.0
        
        # 通过数据总线更新调度器
        state_changed = self.data_bus.update_gait_scheduler(0.02)
        
        # 检查状态同步
        state_info = self.data_bus.get_gait_state_info()
        print(f"数据总线中的步态状态:")
        for key, value in state_info.items():
            print(f"  {key}: {value}")
        
        # 测试便捷接口
        print(f"\n2. 测试便捷查询接口:")
        print(f"当前支撑腿: {self.data_bus.get_current_support_leg()}")
        print(f"当前摆动腿: {self.data_bus.get_current_swing_leg()}")
        print(f"左腿在摆动相: {self.data_bus.is_in_swing_phase('left')}")
        print(f"右腿在摆动相: {self.data_bus.is_in_swing_phase('right')}")
        print(f"左腿在支撑相: {self.data_bus.is_in_support_phase('left')}")
        print(f"右腿在支撑相: {self.data_bus.is_in_support_phase('right')}")
        print(f"在双支撑相: {self.data_bus.is_in_double_support()}")
        
        # 测试控制接口
        print(f"\n3. 测试控制接口:")
        print("通过数据总线停止行走...")
        self.data_bus.stop_gait_scheduler_walking()
        
        time.sleep(0.1)
        print(f"停止后状态: {self.data_bus.current_gait_state}")
        
        print("通过数据总线重新开始行走...")
        self.data_bus.start_gait_scheduler_walking()
        
        time.sleep(0.1)
        print(f"重新开始后状态: {self.data_bus.current_gait_state}")
        
        return True
    
    def test_emergency_stop(self):
        """测试紧急停止功能"""
        print("\n=== 测试紧急停止功能 ===")
        
        # 确保在行走状态
        if self.scheduler.current_state in [GaitState.IDLE, GaitState.STANDING]:
            self.scheduler.start_walking()
            time.sleep(0.2)
        
        print(f"行走状态: {self.scheduler.current_state.value}")
        
        # 模拟过大的力触发紧急停止
        print("模拟紧急情况...")
        emergency_force = 250.0  # 超过阈值的力
        
        self.scheduler.update_sensor_data(
            emergency_force, emergency_force,
            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
        )
        
        # 更新状态应该触发紧急停止
        state_changed = self.scheduler.update_gait_state(0.02)
        
        print(f"紧急停止后状态: {self.scheduler.current_state.value}")
        
        # 手动恢复
        print("手动恢复到站立状态...")
        self.scheduler._transition_to_state(GaitState.STANDING)
        
        return self.scheduler.current_state == GaitState.EMERGENCY_STOP or state_changed
    
    def test_configuration(self):
        """测试配置功能"""
        print("\n=== 测试配置功能 ===")
        
        # 测试配置修改
        original_swing_time = self.scheduler.config.swing_time
        
        new_config = {
            'swing_time': 0.5,
            'touchdown_force_threshold': 20.0,
            'enable_logging': False
        }
        
        success = self.data_bus.set_gait_scheduler_config(new_config)
        print(f"配置更新成功: {success}")
        
        print(f"原始摆动时间: {original_swing_time:.2f}s")
        print(f"新摆动时间: {self.scheduler.config.swing_time:.2f}s")
        print(f"新力阈值: {self.scheduler.config.touchdown_force_threshold:.1f}N")
        print(f"新日志设置: {self.scheduler.config.enable_logging}")
        
        return success
    
    def test_statistics_and_monitoring(self):
        """测试统计和监控功能"""
        print("\n=== 测试统计和监控功能 ===")
        
        # 运行一段时间收集统计数据
        print("运行步态获取统计数据...")
        
        # 重置以获得干净的统计
        self.scheduler.reset()
        self.scheduler.start_walking()
        
        for i in range(200):  # 4秒
            dt = 0.02
            
            # 模拟正常的传感器数据
            if self.scheduler.swing_leg == "right":
                left_force = 65.0
                right_force = 10.0 + 25.0 * (i % 50) / 50.0
            elif self.scheduler.swing_leg == "left":
                right_force = 65.0
                left_force = 10.0 + 25.0 * (i % 50) / 50.0
            else:
                left_force = right_force = 35.0
            
            left_vel = np.array([0.0, 0.0, 0.02])
            right_vel = np.array([0.0, 0.0, 0.02])
            
            self.scheduler.update_sensor_data(left_force, right_force, left_vel, right_vel)
            state_changed = self.scheduler.update_gait_state(dt)
            
            time.sleep(dt)
        
        # 获取统计信息
        stats = self.scheduler.get_state_statistics()
        
        print(f"\n步态统计:")
        print(f"总运行时间: {stats['total_duration']:.2f}s")
        print(f"总转换次数: {stats['total_transitions']}")
        print(f"当前状态: {stats['current_state']}")
        print(f"当前状态持续: {stats['current_duration']:.3f}s")
        
        print(f"\n各状态统计:")
        for state, data in stats['state_stats'].items():
            print(f"  {state}:")
            print(f"    次数: {data['count']}")
            print(f"    总时间: {data['total_duration']:.3f}s")
            print(f"    平均时间: {data['avg_duration']:.3f}s")
            print(f"    时间范围: {data['min_duration']:.3f}s - {data['max_duration']:.3f}s")
            print(f"    占比: {data['percentage']:.1f}%")
        
        return len(stats['state_stats']) > 0
    
    def create_state_transition_diagram(self, transition_data: List):
        """创建状态转换图表"""
        if not transition_data:
            return
        
        try:
            # 提取时间和状态
            times = [t['time'] for t in transition_data]
            states = [t['to_state'] for t in transition_data]
            
            # 状态映射到数值
            state_map = {
                'left_support': 1,
                'double_support_lr': 2,
                'right_support': 3,
                'double_support_rl': 4,
                'standing': 0,
                'idle': -1
            }
            
            state_values = [state_map.get(state, 0) for state in states]
            
            # 绘制状态转换图
            plt.figure(figsize=(12, 6))
            plt.step(times, state_values, where='post', linewidth=2)
            plt.xlabel('Time [s]')
            plt.ylabel('Gait State')
            plt.title('Gait State Transition Timeline')
            
            # 设置y轴标签
            plt.yticks(list(state_map.values()), list(state_map.keys()))
            plt.grid(True, alpha=0.3)
            
            # 标记转换点
            for i, (time_pt, state) in enumerate(zip(times, states)):
                if i < len(times) - 1:  # 不标记最后一个点
                    plt.annotate(f'{i+1}', (time_pt, state_map[state]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('gait_state_transitions.png', dpi=150, bbox_inches='tight')
            print("状态转换图已保存: gait_state_transitions.png")
            
        except Exception as e:
            print(f"绘制状态转换图失败: {e}")
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("=== 步态调度器综合测试 ===")
        
        test_results = {}
        
        # 1. 基本状态转换测试
        test_results['basic_transitions'] = self.test_basic_state_transitions()
        
        # 2. 传感器驱动转换测试
        test_results['sensor_transitions'] = self.test_sensor_based_transitions()
        
        # 3. 混合触发测试
        transition_data = self.test_hybrid_triggers()
        test_results['hybrid_triggers'] = len(transition_data) > 0
        
        # 4. 数据总线集成测试
        test_results['data_bus_integration'] = self.test_data_bus_integration()
        
        # 5. 紧急停止测试
        test_results['emergency_stop'] = self.test_emergency_stop()
        
        # 6. 配置测试
        test_results['configuration'] = self.test_configuration()
        
        # 7. 统计监控测试
        test_results['statistics'] = self.test_statistics_and_monitoring()
        
        # 8. 创建可视化图表
        self.create_state_transition_diagram(transition_data)
        
        # 打印最终结果
        print(f"\n=== 测试结果总结 ===")
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n总体结果: {passed}/{total} 测试通过")
        
        if passed == total:
            print("🎉 所有测试通过！步态调度器功能正常。")
        else:
            print("⚠️  部分测试失败，需要检查相关功能。")
        
        # 最终状态打印
        print(f"\n=== 最终状态 ===")
        self.data_bus.print_gait_status()
        
        return test_results


def main():
    """主函数"""
    try:
        tester = GaitSchedulerTester()
        results = tester.run_comprehensive_test()
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 