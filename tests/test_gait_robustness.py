#!/usr/bin/env python3
"""
步态系统鲁棒性测试

测试内容:
1. 摆动脚提前触地检测和状态切换
2. 摆动被打断后的恢复能力
3. 后续步态周期的正常运行（无锁死）
4. 足底力阈值检测的准确性

作者: Adam Control Team
"""

import numpy as np
import time
import sys
from typing import Dict, List, Tuple
import threading

# 导入必要模块
try:
    from data_bus import DataBus
    from foot_placement import FootPlacementPlanner, FootPlacementConfig, Vector3D
    from gait_scheduler import GaitScheduler, GaitState, GaitSchedulerConfig
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)


class GaitRobustnessTester:
    """步态系统鲁棒性测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.data_bus = DataBus()
        
        # 测试参数
        self.control_frequency = 1000.0  # 控制频率 1kHz
        self.dt = 1.0 / self.control_frequency
        
        # 鲁棒性测试参数
        self.contact_force_threshold = 50.0  # 足底力阈值 [N]
        self.normal_ground_force = 100.0     # 正常地面反力 [N]
        self.early_contact_force = 80.0      # 提前接触力 [N]
        
        # 配置足步规划参数
        config = FootPlacementConfig()
        self.foot_planner = FootPlacementPlanner(config)
        
        # 配置步态调度参数
        gait_config = GaitSchedulerConfig()
        gait_config.touchdown_force_threshold = self.contact_force_threshold
        self.gait_scheduler = GaitScheduler(gait_config)
        
        # 测试记录
        self.test_results = []
        self.state_transitions = []
        self.timing_records = []
        
        # 测试状态
        self.is_testing = False
        self.test_start_time = 0.0
        self.early_contact_injected = False
        self.recovery_verified = False
        
    def setup_initial_conditions(self):
        """设置初始条件"""
        # 设置初始关节角度
        initial_angles = {
            "left_hip_yaw": 0.0, "left_hip_roll": 0.02, "left_hip_pitch": -0.05,
            "left_knee_pitch": 0.1, "left_ankle_pitch": -0.05, "left_ankle_roll": -0.02,
            "right_hip_yaw": 0.0, "right_hip_roll": -0.02, "right_hip_pitch": -0.05,
            "right_knee_pitch": 0.1, "right_ankle_pitch": -0.05, "right_ankle_roll": 0.02
        }
        
        # 设置关节角度
        for joint_name, angle in initial_angles.items():
            self.data_bus.set_joint_position(joint_name, angle)
        
        # 更新足部位置
        self.foot_planner.update_foot_states_from_kinematics(initial_angles)
        
        # 设置初始足底力（正常支撑）
        self.data_bus.set_end_effector_contact_force("left_foot", 0.0)   # 左脚无接触
        self.data_bus.set_end_effector_contact_force("right_foot", self.normal_ground_force)  # 右脚支撑
        
        # 设置运动意图
        self.foot_planner.set_body_motion_intent(Vector3D(0.2, 0.0, 0.0), 0.0)
        
        print("=== 鲁棒性测试初始条件设置完成 ===")
        print(f"足底力阈值: {self.contact_force_threshold}N")
        print(f"正常地面反力: {self.normal_ground_force}N")
        print(f"提前接触测试力: {self.early_contact_force}N")
        
    def inject_early_contact(self, foot_name: str, contact_force: float):
        """注入提前接触事件"""
        print(f"  🚨 注入提前接触事件: {foot_name} 接触力 {contact_force}N")
        self.data_bus.set_end_effector_contact_force(foot_name, contact_force)
        self.early_contact_injected = True
    
    def update_scheduler_with_sensors(self):
        """更新步态调度器的传感器数据"""
        left_force = self.data_bus.get_end_effector_contact_force("left_foot") or 0.0
        right_force = self.data_bus.get_end_effector_contact_force("right_foot") or 0.0
        left_vel = np.zeros(3)  # 简化处理
        right_vel = np.zeros(3)
        
        self.gait_scheduler.update_sensor_data(left_force, right_force, left_vel, right_vel)
        
    def monitor_state_transitions(self):
        """监控状态转换"""
        current_state = self.gait_scheduler.current_state
        current_time = time.time()
        
        # 记录状态转换
        if hasattr(self, 'last_state') and self.last_state != current_state:
            transition = {
                'time': current_time,
                'from_state': self.last_state,
                'to_state': current_state,
                'elapsed_time': current_time - self.test_start_time
            }
            self.state_transitions.append(transition)
            print(f"  状态转换: {self.last_state} -> {current_state} (时间: {transition['elapsed_time']:.3f}s)")
        
        self.last_state = current_state
        return current_state
    
    def test_early_contact_detection(self) -> Dict:
        """测试提前接触检测"""
        print("\n=== 测试1: 提前接触检测 ===")
        
        self.test_start_time = time.time()
        self.state_transitions = []
        self.early_contact_injected = False
        
        # 初始化状态
        self.gait_scheduler.current_state = GaitState.RIGHT_SUPPORT
        self.last_state = GaitState.RIGHT_SUPPORT
        
        early_contact_detected = False
        state_switched = False
        switch_time = 0.0
        injection_time = 0.0
        
        print("  开始步态循环...")
        
        # 运行步态循环
        for cycle in range(2000):  # 2秒测试
            current_time = time.time()
            elapsed_time = current_time - self.test_start_time
            
            # 监控状态转换
            current_state = self.monitor_state_transitions()
            
            # 在左脚进入摆动相时注入提前接触
            if (current_state == GaitState.LEFT_SUPPORT and 
                not self.early_contact_injected and 
                elapsed_time > 0.1):  # 摆动开始100ms后注入
                
                injection_time = elapsed_time
                self.inject_early_contact("left_foot", self.early_contact_force)
            
            # 更新传感器数据并执行步态调度
            self.update_scheduler_with_sensors()
            self.gait_scheduler.update_gait_state(self.dt)
            
            # 检测是否因提前接触而切换状态
            if (self.early_contact_injected and not state_switched):
                new_state = self.gait_scheduler.current_state
                if new_state != GaitState.LEFT_SUPPORT:
                    early_contact_detected = True
                    state_switched = True
                    switch_time = elapsed_time
                    print(f"  ✅ 检测到提前接触，状态切换: {current_state} -> {new_state}")
                    break
            
            time.sleep(self.dt)
            
            # 超时检测
            if elapsed_time > 1.5 and self.early_contact_injected and not state_switched:
                print(f"  ❌ 超时：注入提前接触后1.5秒内未检测到状态切换")
                break
        
        # 计算响应时间
        if state_switched and injection_time > 0:
            response_time = switch_time - injection_time
        else:
            response_time = float('inf')
        
        # 要求：响应时间 < 10ms
        response_requirement = 0.01
        
        result = {
            'passed': early_contact_detected and response_time < response_requirement,
            'early_contact_detected': early_contact_detected,
            'state_switched': state_switched,
            'injection_time': injection_time,
            'switch_time': switch_time,
            'response_time': response_time,
            'requirement': f'< {response_requirement*1000:.0f}ms',
            'actual': f'{response_time*1000:.1f}ms' if response_time != float('inf') else 'N/A',
            'state_transitions': self.state_transitions
        }
        
        print(f"  注入时间: {injection_time:.3f}s")
        print(f"  切换时间: {switch_time:.3f}s")
        print(f"  响应时间: {response_time*1000:.1f}ms" if response_time != float('inf') else "  响应时间: 超时")
        print(f"  测试结果: {'✅ 通过' if result['passed'] else '❌ 失败'}")
        
        return result
    
    def test_swing_interruption_recovery(self) -> Dict:
        """测试摆动被打断后的恢复能力"""
        print("\n=== 测试2: 摆动中断恢复 ===")
        
        # 重置系统状态
        self.setup_initial_conditions()
        time.sleep(0.1)
        
        self.test_start_time = time.time()
        self.state_transitions = []
        self.early_contact_injected = False
        
        # 初始化状态
        self.gait_scheduler.current_state = GaitState.RIGHT_SUPPORT
        self.last_state = GaitState.RIGHT_SUPPORT
        
        interruption_injected = False
        recovery_detected = False
        normal_cycles_after_recovery = 0
        recovery_time = 0.0
        
        print("  测试摆动中断恢复能力...")
        
        # 运行测试
        for cycle in range(3000):  # 3秒测试
            current_time = time.time()
            elapsed_time = current_time - self.test_start_time
            
            current_state = self.monitor_state_transitions()
            
            # 在左脚摆动相中期注入中断
            if (current_state == GaitState.LEFT_SUPPORT and 
                not interruption_injected and 
                elapsed_time > 0.2):  # 摆动中期
                
                print(f"  🚨 注入摆动中断: 左脚提前触地")
                self.inject_early_contact("left_foot", self.early_contact_force)
                interruption_injected = True
                interrupt_time = elapsed_time
            
            # 更新传感器数据并执行步态调度
            self.update_scheduler_with_sensors()
            self.gait_scheduler.update_gait_state(self.dt)
            
            # 检测恢复情况
            if interruption_injected and not recovery_detected:
                # 如果系统从中断状态恢复到正常步态
                if (current_state in [GaitState.LEFT_SUPPORT, GaitState.RIGHT_SUPPORT] and
                    elapsed_time > interrupt_time + 0.1):
                    recovery_detected = True
                    recovery_time = elapsed_time
                    print(f"  ✅ 检测到系统恢复，当前状态: {current_state}")
            
            # 统计恢复后的正常周期数
            if recovery_detected:
                if (current_state in [GaitState.LEFT_SUPPORT, GaitState.RIGHT_SUPPORT,
                                    GaitState.DOUBLE_SUPPORT_LR, GaitState.DOUBLE_SUPPORT_RL]):
                    normal_cycles_after_recovery += 1
                    
                # 如果恢复后能正常运行500个周期，认为测试通过
                if normal_cycles_after_recovery >= 500:
                    print(f"  ✅ 恢复后正常运行 {normal_cycles_after_recovery} 个周期")
                    break
            
            time.sleep(self.dt)
            
            # 超时检测
            if elapsed_time > 2.5:
                print(f"  ⚠️  测试超时，检查恢复情况...")
                break
        
        # 分析恢复能力
        recovery_time_limit = 0.5  # 要求在500ms内恢复
        stable_cycles_requirement = 500  # 要求恢复后稳定运行500个周期
        
        if recovery_detected:
            recovery_delay = recovery_time - interrupt_time if 'interrupt_time' in locals() else 0
        else:
            recovery_delay = float('inf')
        
        result = {
            'passed': (recovery_detected and 
                      recovery_delay < recovery_time_limit and 
                      normal_cycles_after_recovery >= stable_cycles_requirement),
            'interruption_injected': interruption_injected,
            'recovery_detected': recovery_detected,
            'recovery_delay': recovery_delay,
            'normal_cycles_after_recovery': normal_cycles_after_recovery,
            'requirement': f'恢复时间<{recovery_time_limit*1000:.0f}ms, 稳定运行>{stable_cycles_requirement}周期',
            'actual': f'恢复时间{recovery_delay*1000:.1f}ms, 稳定运行{normal_cycles_after_recovery}周期' if recovery_delay != float('inf') else 'N/A',
            'state_transitions': self.state_transitions
        }
        
        print(f"  中断注入: {'✅' if interruption_injected else '❌'}")
        print(f"  系统恢复: {'✅' if recovery_detected else '❌'}")
        print(f"  恢复延迟: {recovery_delay*1000:.1f}ms" if recovery_delay != float('inf') else "  恢复延迟: 超时")
        print(f"  稳定周期: {normal_cycles_after_recovery}")
        print(f"  测试结果: {'✅ 通过' if result['passed'] else '❌ 失败'}")
        
        return result
    
    def test_force_threshold_accuracy(self) -> Dict:
        """测试足底力阈值检测准确性"""
        print("\n=== 测试3: 力阈值检测准确性 ===")
        
        threshold_tests = []
        test_forces = [
            (30.0, False, "低于阈值"),
            (45.0, False, "接近阈值"),
            (55.0, True, "超过阈值"),
            (75.0, True, "明显超过"),
            (100.0, True, "大幅超过")
        ]
        
        for test_force, should_trigger, description in test_forces:
            print(f"\n  测试力值: {test_force}N ({description})")
            
            # 重置测试环境
            self.setup_initial_conditions()
            self.gait_scheduler.current_state = GaitState.LEFT_SUPPORT
            
            # 设置测试力
            self.data_bus.set_end_effector_contact_force("left_foot", test_force)
            
            # 检测是否触发状态切换
            initial_state = self.gait_scheduler.current_state
            
            # 运行几个周期看是否触发
            triggered = False
            for _ in range(10):
                self.gait_scheduler.update_sensor_data(test_force, 0.0, np.zeros(3), np.zeros(3))
                self.gait_scheduler.update_gait_state(self.dt)
                if self.gait_scheduler.current_state != initial_state:
                    triggered = True
                    break
                time.sleep(self.dt)
            
            test_result = {
                'force': test_force,
                'should_trigger': should_trigger,
                'actually_triggered': triggered,
                'correct': triggered == should_trigger,
                'description': description
            }
            
            threshold_tests.append(test_result)
            
            status = "✅ 正确" if test_result['correct'] else "❌ 错误"
            print(f"    期望触发: {should_trigger}, 实际触发: {triggered} - {status}")
        
        # 计算准确率
        correct_count = sum(1 for test in threshold_tests if test['correct'])
        accuracy = correct_count / len(threshold_tests)
        
        # 要求：准确率 >= 90%
        accuracy_requirement = 0.9
        
        result = {
            'passed': accuracy >= accuracy_requirement,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_tests': len(threshold_tests),
            'threshold_tests': threshold_tests,
            'requirement': f'>= {accuracy_requirement*100:.0f}%',
            'actual': f'{accuracy*100:.1f}%'
        }
        
        print(f"\n  测试总数: {len(threshold_tests)}")
        print(f"  正确判断: {correct_count}")
        print(f"  准确率: {accuracy*100:.1f}%")
        print(f"  测试结果: {'✅ 通过' if result['passed'] else '❌ 失败'}")
        
        return result
    
    def test_continuous_operation(self) -> Dict:
        """测试连续运行无锁死"""
        print("\n=== 测试4: 连续运行稳定性 ===")
        
        self.setup_initial_conditions()
        time.sleep(0.1)
        
        self.test_start_time = time.time()
        self.state_transitions = []
        
        # 初始化
        self.gait_scheduler.current_state = GaitState.RIGHT_SUPPORT
        self.last_state = GaitState.RIGHT_SUPPORT
        
        state_counts = {}
        total_cycles = 0
        stuck_detection_limit = 100  # 连续100个周期相同状态认为锁死
        consecutive_same_state = 0
        last_state_for_stuck = None
        
        # 在测试过程中随机注入干扰
        disturbance_injected = False
        
        print("  开始连续运行测试 (5秒)...")
        
        for cycle in range(5000):  # 5秒连续运行
            current_time = time.time()
            elapsed_time = current_time - self.test_start_time
            
            current_state = self.monitor_state_transitions()
            
            # 统计状态
            if current_state not in state_counts:
                state_counts[current_state] = 0
            state_counts[current_state] += 1
            total_cycles += 1
            
            # 检测锁死
            if current_state == last_state_for_stuck:
                consecutive_same_state += 1
            else:
                consecutive_same_state = 0
                last_state_for_stuck = current_state
            
            if consecutive_same_state >= stuck_detection_limit:
                print(f"  ❌ 检测到锁死: 状态 {current_state} 持续 {consecutive_same_state} 个周期")
                break
            
            # 在中期注入一次干扰
            if elapsed_time > 2.0 and not disturbance_injected:
                if current_state == GaitState.LEFT_SUPPORT:
                    print(f"  🚨 注入中期干扰")
                    self.inject_early_contact("left_foot", self.early_contact_force)
                    disturbance_injected = True
            
            # 更新传感器数据并执行步态调度
            self.update_scheduler_with_sensors()
            self.gait_scheduler.update_gait_state(self.dt)
            
            time.sleep(self.dt)
        
        # 分析连续运行结果
        total_transitions = len(self.state_transitions)
        unique_states = len(state_counts)
        
        # 检查是否有足够的状态转换（表明正常运行）
        min_transitions_required = 10  # 至少10次状态转换
        min_unique_states = 2  # 至少访问过2种状态
        
        # 检查是否没有锁死
        no_deadlock = consecutive_same_state < stuck_detection_limit
        
        result = {
            'passed': (total_transitions >= min_transitions_required and 
                      unique_states >= min_unique_states and 
                      no_deadlock),
            'total_cycles': total_cycles,
            'total_transitions': total_transitions,
            'unique_states': unique_states,
            'state_counts': state_counts,
            'no_deadlock': no_deadlock,
            'max_consecutive_same_state': consecutive_same_state,
            'disturbance_injected': disturbance_injected,
            'requirement': f'转换>{min_transitions_required}, 状态>{min_unique_states}, 无锁死',
            'actual': f'转换{total_transitions}, 状态{unique_states}, 连续{consecutive_same_state}'
        }
        
        print(f"  总周期数: {total_cycles}")
        print(f"  状态转换: {total_transitions}")
        print(f"  访问状态: {unique_states}")
        print(f"  最大连续相同状态: {consecutive_same_state}")
        print(f"  状态分布: {dict(state_counts)}")
        print(f"  测试结果: {'✅ 通过' if result['passed'] else '❌ 失败'}")
        
        return result
    
    def run_all_tests(self):
        """运行所有鲁棒性测试"""
        print("开始步态系统鲁棒性测试")
        print("=" * 60)
        
        # 设置初始条件
        self.setup_initial_conditions()
        time.sleep(0.2)  # 等待系统稳定
        
        # 运行测试
        results = {}
        
        # 测试1: 提前接触检测
        results['early_contact'] = self.test_early_contact_detection()
        
        # 测试2: 摆动中断恢复
        results['swing_recovery'] = self.test_swing_interruption_recovery()
        
        # 测试3: 力阈值检测准确性
        results['threshold_accuracy'] = self.test_force_threshold_accuracy()
        
        # 测试4: 连续运行稳定性
        results['continuous_operation'] = self.test_continuous_operation()
        
        # 汇总结果
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("鲁棒性测试结果总结")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get('passed', False))
        
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        print("\n详细结果:")
        
        # 测试1结果
        early_result = results.get('early_contact', {})
        print(f"1. 提前接触检测: {'✅ 通过' if early_result.get('passed') else '❌ 失败'}")
        if 'actual' in early_result:
            print(f"   要求: {early_result.get('requirement', 'N/A')}")
            print(f"   实际: {early_result.get('actual', 'N/A')}")
        
        # 测试2结果
        recovery_result = results.get('swing_recovery', {})
        print(f"2. 摆动中断恢复: {'✅ 通过' if recovery_result.get('passed') else '❌ 失败'}")
        if 'actual' in recovery_result:
            print(f"   要求: {recovery_result.get('requirement', 'N/A')}")
            print(f"   实际: {recovery_result.get('actual', 'N/A')}")
        
        # 测试3结果
        threshold_result = results.get('threshold_accuracy', {})
        print(f"3. 力阈值检测准确性: {'✅ 通过' if threshold_result.get('passed') else '❌ 失败'}")
        if 'actual' in threshold_result:
            print(f"   要求: {threshold_result.get('requirement', 'N/A')}")
            print(f"   实际: {threshold_result.get('actual', 'N/A')}")
        
        # 测试4结果
        continuous_result = results.get('continuous_operation', {})
        print(f"4. 连续运行稳定性: {'✅ 通过' if continuous_result.get('passed') else '❌ 失败'}")
        if 'actual' in continuous_result:
            print(f"   要求: {continuous_result.get('requirement', 'N/A')}")
            print(f"   实际: {continuous_result.get('actual', 'N/A')}")
        
        print("\n" + "=" * 60)
        
        if passed_tests == total_tests:
            print("🎉 所有鲁棒性测试通过！步态系统具备优秀的抗干扰能力：")
            print("   ✅ 检测到足底力 > 阈值即切换状态")
            print("   ✅ 摆动被打断后能快速恢复")
            print("   ✅ 后续周期继续正常运行（无锁死）")
            print("   ✅ 力阈值检测准确可靠")
        else:
            print("⚠️  部分鲁棒性测试失败，需要改进异常处理机制。")


def main():
    """主函数"""
    try:
        tester = GaitRobustnessTester()
        results = tester.run_all_tests()
        
        # 保存测试结果
        import json
        
        # 递归转换所有数据为JSON兼容格式
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                # 处理字典，确保键都是字符串
                return {str(k): make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return make_json_serializable(obj.__dict__)
            elif hasattr(obj, 'value'):
                return obj.value
            elif hasattr(obj, 'name'):
                return obj.name
            else:
                return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
        
        serializable_results = make_json_serializable(results)
        
        with open('gait_robustness_results.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n测试结果已保存到: gait_robustness_results.json")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 