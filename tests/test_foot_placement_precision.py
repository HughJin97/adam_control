#!/usr/bin/env python3
"""
落脚点计算精度测试

测试内容:
1. 进入摆动相 ≤ 1个控制周期内写入 target_foot_pos
2. 落脚点相对支撑足的平移符合设定步长 (Δx, Δy)  
3. 落脚点与期望偏差 ‖error‖ < 2 cm（水平向）

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
    from gait_scheduler import GaitScheduler, GaitState
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)


class FootPlacementPrecisionTester:
    """落脚点计算精度测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.data_bus = DataBus()
        
        # 配置足步规划参数
        config = FootPlacementConfig()
        config.nominal_step_length = 0.15  # 标准步长 15cm
        config.nominal_step_width = 0.12   # 标准步宽 12cm
        config.lateral_stability_margin = 0.03   # 横向稳定性余量 3cm
        config.longitudinal_stability_margin = 0.02  # 纵向稳定性余量 2cm
        
        self.foot_planner = FootPlacementPlanner(config)
        self.gait_scheduler = GaitScheduler(self.data_bus)
        
        # 测试参数
        self.control_frequency = 1000.0  # 控制频率 1kHz
        self.dt = 1.0 / self.control_frequency
        
        # 测试记录
        self.test_results = []
        self.timing_records = []
        self.position_errors = []
        
        # 测试状态
        self.is_testing = False
        self.test_start_time = 0.0
        self.swing_transition_detected = False
        self.target_pos_written = False
        self.target_write_time = 0.0
        
    def setup_initial_conditions(self):
        """设置初始条件"""
        # 设置初始关节角度（双腿支撑站立状态）
        initial_angles = {
            "left_hip_yaw": 0.0, "left_hip_roll": 0.02, "left_hip_pitch": -0.05,
            "left_knee_pitch": 0.1, "left_ankle_pitch": -0.05, "left_ankle_roll": -0.02,
            "right_hip_yaw": 0.0, "right_hip_roll": -0.02, "right_hip_pitch": -0.05,
            "right_knee_pitch": 0.1, "right_ankle_pitch": -0.05, "right_ankle_roll": 0.02
        }
        
        self.data_bus.joint_angles.update(initial_angles)
        
        # 更新足部位置
        self.foot_planner.update_foot_states_from_kinematics(initial_angles)
        
        # 设置运动意图（前进0.3m/s）
        self.foot_planner.set_body_motion_intent(Vector3D(0.3, 0.0, 0.0), 0.0)
        
        # 初始状态为右腿支撑
        self.data_bus.gait_state = GaitState.RIGHT_SUPPORT_PHASE
        
        print("=== 初始条件设置完成 ===")
        current_positions = self.foot_planner.get_current_foot_positions()
        print(f"左脚初始位置: ({current_positions['left_foot'].x:.3f}, {current_positions['left_foot'].y:.3f}, {current_positions['left_foot'].z:.3f})")
        print(f"右脚初始位置: ({current_positions['right_foot'].x:.3f}, {current_positions['right_foot'].y:.3f}, {current_positions['right_foot'].z:.3f})")
        
    def test_swing_phase_timing(self) -> Dict:
        """测试摆动相开始时的落脚点计算时机"""
        print("\n=== 测试1: 摆动相计算时机 ===")
        
        self.is_testing = True
        self.swing_transition_detected = False
        self.target_pos_written = False
        
        # 记录测试开始时间
        start_time = time.time()
        transition_time = 0.0
        write_time = 0.0
        
        # 模拟步态循环
        for cycle in range(1000):  # 1秒测试
            current_time = time.time()
            
            # 检测状态转换（从支撑相到摆动相）
            if not self.swing_transition_detected:
                if cycle == 100:  # 在第100个周期触发状态转换
                    self.data_bus.gait_state = GaitState.LEFT_SWING_PHASE
                    self.swing_transition_detected = True
                    transition_time = current_time
                    print(f"  状态转换检测: 周期 {cycle}, 时间 {current_time:.6f}s")
            
            # 执行步态调度
            self.gait_scheduler.update_gait_state(self.dt)
            
            # 检查是否写入了target_foot_pos
            if self.swing_transition_detected and not self.target_pos_written:
                target_positions = self.data_bus.target_foot_pos
                if target_positions and 'left_foot' in target_positions:
                    if target_positions['left_foot'] is not None:
                        self.target_pos_written = True
                        write_time = current_time
                        print(f"  目标位置写入: 周期 {cycle}, 时间 {current_time:.6f}s")
                        break
            
            time.sleep(self.dt)
        
        # 计算时间差
        if self.target_pos_written and self.swing_transition_detected:
            time_diff = write_time - transition_time
            cycles_diff = time_diff / self.dt
            
            result = {
                'passed': cycles_diff <= 1.0,
                'time_diff': time_diff,
                'cycles_diff': cycles_diff,
                'requirement': '≤ 1个控制周期',
                'actual': f'{cycles_diff:.2f}个周期'
            }
            
            print(f"  时间差: {time_diff*1000:.3f}ms ({cycles_diff:.2f}个周期)")
            print(f"  测试结果: {'通过' if result['passed'] else '失败'}")
            
            return result
        else:
            print("  测试失败: 未检测到目标位置写入")
            return {'passed': False, 'error': '未检测到目标位置写入'}
    
    def test_step_length_accuracy(self) -> Dict:
        """测试步长精度"""
        print("\n=== 测试2: 步长精度 ===")
        
        # 获取当前足部位置
        current_positions = self.foot_planner.get_current_foot_positions()
        support_foot_pos = current_positions['right_foot']  # 右腿支撑
        
        # 执行足步规划
        target_pos = self.foot_planner.plan_foot_placement("left", "right")
        
        # 计算相对位移
        actual_delta_x = target_pos.x - support_foot_pos.x
        actual_delta_y = target_pos.y - support_foot_pos.y
        
        # 期望步长（考虑稳定性调整）
        config = self.foot_planner.config
        expected_delta_x = config.nominal_step_length + config.longitudinal_stability_margin
        expected_delta_y = -config.nominal_step_width / 2 - config.lateral_stability_margin  # 左脚向左
        
        # 计算误差
        error_x = abs(actual_delta_x - expected_delta_x)
        error_y = abs(actual_delta_y - expected_delta_y)
        horizontal_error = np.sqrt(error_x**2 + error_y**2)
        
        # 精度要求: < 2cm
        accuracy_requirement = 0.02
        
        result = {
            'passed': horizontal_error < accuracy_requirement,
            'horizontal_error': horizontal_error,
            'error_x': error_x,
            'error_y': error_y,
            'actual_delta_x': actual_delta_x,
            'actual_delta_y': actual_delta_y,
            'expected_delta_x': expected_delta_x,
            'expected_delta_y': expected_delta_y,
            'requirement': '< 2cm',
            'actual': f'{horizontal_error*100:.2f}cm'
        }
        
        print(f"  支撑足位置: ({support_foot_pos.x:.3f}, {support_foot_pos.y:.3f}, {support_foot_pos.z:.3f})")
        print(f"  目标足位置: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
        print(f"  期望位移: Δx={expected_delta_x:.3f}m, Δy={expected_delta_y:.3f}m")
        print(f"  实际位移: Δx={actual_delta_x:.3f}m, Δy={actual_delta_y:.3f}m")
        print(f"  位移误差: Δx={error_x*1000:.1f}mm, Δy={error_y*1000:.1f}mm")
        print(f"  水平误差: {horizontal_error*1000:.1f}mm")
        print(f"  测试结果: {'通过' if result['passed'] else '失败'}")
        
        return result
    
    def test_multiple_step_consistency(self) -> Dict:
        """测试多步步长一致性"""
        print("\n=== 测试3: 多步一致性 ===")
        
        step_errors = []
        step_results = []
        
        # 设置不同的初始位置进行多次测试
        test_positions = [
            (0.0, 0.09, 0.0),   # 右脚初始位置
            (0.15, 0.09, 0.0),  # 右脚前移
            (0.30, 0.09, 0.0),  # 右脚更远前移
            (0.0, 0.15, 0.0),   # 右脚侧移
            (0.0, 0.03, 0.0),   # 右脚内移
        ]
        
        for i, (x, y, z) in enumerate(test_positions):
            print(f"\n  测试位置 {i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
            
            # 设置右脚位置
            self.foot_planner.right_foot.position = Vector3D(x, y, z)
            
            # 计算左脚目标位置
            target_pos = self.foot_planner.plan_foot_placement("left", "right")
            
            # 计算实际步长
            actual_delta_x = target_pos.x - x
            actual_delta_y = target_pos.y - y
            
            # 期望步长
            config = self.foot_planner.config
            expected_delta_x = config.nominal_step_length + config.longitudinal_stability_margin
            expected_delta_y = -config.nominal_step_width / 2 - config.lateral_stability_margin
            
            # 计算误差
            error_x = abs(actual_delta_x - expected_delta_x)
            error_y = abs(actual_delta_y - expected_delta_y)
            horizontal_error = np.sqrt(error_x**2 + error_y**2)
            
            step_errors.append(horizontal_error)
            step_results.append({
                'position': (x, y, z),
                'target': (target_pos.x, target_pos.y, target_pos.z),
                'delta': (actual_delta_x, actual_delta_y),
                'error': horizontal_error
            })
            
            print(f"    目标位置: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
            print(f"    步长: Δx={actual_delta_x:.3f}m, Δy={actual_delta_y:.3f}m")
            print(f"    误差: {horizontal_error*1000:.1f}mm")
        
        # 计算统计信息
        max_error = max(step_errors)
        avg_error = np.mean(step_errors)
        std_error = np.std(step_errors)
        
        # 一致性要求：最大误差 < 2cm，标准差 < 1cm
        consistency_passed = max_error < 0.02 and std_error < 0.01
        
        result = {
            'passed': consistency_passed,
            'max_error': max_error,
            'avg_error': avg_error,
            'std_error': std_error,
            'step_results': step_results,
            'requirement': '最大误差<2cm, 标准差<1cm',
            'actual': f'最大误差{max_error*1000:.1f}mm, 标准差{std_error*1000:.1f}mm'
        }
        
        print(f"\n  统计结果:")
        print(f"    最大误差: {max_error*1000:.1f}mm")
        print(f"    平均误差: {avg_error*1000:.1f}mm")
        print(f"    误差标准差: {std_error*1000:.1f}mm")
        print(f"    一致性测试: {'通过' if consistency_passed else '失败'}")
        
        return result
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始落脚点计算精度测试")
        print("=" * 50)
        
        # 设置初始条件
        self.setup_initial_conditions()
        
        # 运行测试
        results = {}
        
        # 测试1: 摆动相计算时机
        results['timing'] = self.test_swing_phase_timing()
        
        # 测试2: 步长精度
        results['accuracy'] = self.test_step_length_accuracy()
        
        # 测试3: 多步一致性
        results['consistency'] = self.test_multiple_step_consistency()
        
        # 汇总结果
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """打印测试总结"""
        print("\n" + "=" * 50)
        print("测试结果总结")
        print("=" * 50)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get('passed', False))
        
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        print("\n详细结果:")
        
        # 测试1结果
        timing_result = results.get('timing', {})
        print(f"1. 摆动相计算时机: {'✓ 通过' if timing_result.get('passed') else '✗ 失败'}")
        if 'actual' in timing_result:
            print(f"   要求: {timing_result.get('requirement', 'N/A')}")
            print(f"   实际: {timing_result.get('actual', 'N/A')}")
        
        # 测试2结果
        accuracy_result = results.get('accuracy', {})
        print(f"2. 步长精度: {'✓ 通过' if accuracy_result.get('passed') else '✗ 失败'}")
        if 'actual' in accuracy_result:
            print(f"   要求: {accuracy_result.get('requirement', 'N/A')}")
            print(f"   实际: {accuracy_result.get('actual', 'N/A')}")
        
        # 测试3结果
        consistency_result = results.get('consistency', {})
        print(f"3. 多步一致性: {'✓ 通过' if consistency_result.get('passed') else '✗ 失败'}")
        if 'actual' in consistency_result:
            print(f"   要求: {consistency_result.get('requirement', 'N/A')}")
            print(f"   实际: {consistency_result.get('actual', 'N/A')}")
        
        print("\n" + "=" * 50)
        
        if passed_tests == total_tests:
            print("🎉 所有测试通过！落脚点计算精度符合要求。")
        else:
            print("⚠️  部分测试失败，需要检查落脚点计算算法。")


def main():
    """主函数"""
    try:
        tester = FootPlacementPrecisionTester()
        results = tester.run_all_tests()
        
        # 保存测试结果
        import json
        with open('foot_placement_precision_results.json', 'w', encoding='utf-8') as f:
            # 转换Vector3D对象为字典以便JSON序列化
            def convert_for_json(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)
            
            json.dump(results, f, indent=2, default=convert_for_json, ensure_ascii=False)
        
        print(f"\n测试结果已保存到: foot_placement_precision_results.json")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 