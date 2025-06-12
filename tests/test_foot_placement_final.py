#!/usr/bin/env python3
"""
落脚点计算精度测试 (最终版本)

验证用户要求：
1. 进入摆动相 ≤ 1个控制周期内写入 target_foot_pos
2. 落脚点相对支撑足的平移符合设定步长 (Δx, Δy)  
3. 落脚点与期望偏差 ‖error‖ < 2 cm（水平向）

使用零速度测试基础步长精度，以及不同速度下的动态调整精度
"""

import numpy as np
import time
import sys
from typing import Dict, List, Tuple

# 导入必要模块
try:
    from data_bus import DataBus
    from foot_placement import FootPlacementPlanner, FootPlacementConfig, Vector3D
    from gait_scheduler import GaitState
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)


class FootPlacementFinalTester:
    """落脚点计算精度最终测试器"""
    
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
        
        # 测试参数
        self.control_frequency = 1000.0  # 控制频率 1kHz
        self.dt = 1.0 / self.control_frequency
        
    def setup_initial_conditions(self):
        """设置初始条件"""
        # 设置初始关节角度（双腿支撑站立状态）
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
        
        print("=== 初始条件设置完成 ===")
        current_positions = self.foot_planner.get_current_foot_positions()
        print(f"左脚初始位置: ({current_positions['left_foot'].x:.3f}, {current_positions['left_foot'].y:.3f}, {current_positions['left_foot'].z:.3f})")
        print(f"右脚初始位置: ({current_positions['right_foot'].x:.3f}, {current_positions['right_foot'].y:.3f}, {current_positions['right_foot'].z:.3f})")
        
    def test_computation_timing(self) -> Dict:
        """测试计算时机（≤ 1个控制周期）"""
        print("\n=== 测试1: 计算时机 ===")
        
        # 设置零速度（使用基础步长）
        self.foot_planner.set_body_motion_intent(Vector3D(0.0, 0.0, 0.0), 0.0)
        
        # 记录计算时间
        start_time = time.time()
        target_pos = self.foot_planner.plan_foot_placement("left", "right")
        end_time = time.time()
        
        calculation_time = end_time - start_time
        one_cycle_time = self.dt  # 1ms
        
        result = {
            'passed': calculation_time <= one_cycle_time,
            'calculation_time': calculation_time,
            'one_cycle_time': one_cycle_time,
            'requirement': f'≤ {one_cycle_time*1000:.1f}ms (1个控制周期)',
            'actual': f'{calculation_time*1000:.3f}ms'
        }
        
        print(f"  计算时间: {calculation_time*1000:.3f}ms")
        print(f"  控制周期: {one_cycle_time*1000:.1f}ms")
        print(f"  测试结果: {'✅ 通过' if result['passed'] else '❌ 失败'}")
        
        return result
    
    def test_basic_step_accuracy(self) -> Dict:
        """测试基础步长精度（零速度情况）"""
        print("\n=== 测试2: 基础步长精度 ===")
        
        # 设置零速度
        self.foot_planner.set_body_motion_intent(Vector3D(0.0, 0.0, 0.0), 0.0)
        
        # 获取当前足部位置
        current_positions = self.foot_planner.get_current_foot_positions()
        support_foot_pos = current_positions['right_foot']  # 右腿支撑
        
        # 执行足步规划
        target_pos = self.foot_planner.plan_foot_placement("left", "right")
        
        # 计算相对位移
        actual_delta_x = target_pos.x - support_foot_pos.x
        actual_delta_y = target_pos.y - support_foot_pos.y
        
        # 期望位移（基础步长）
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
        
        print(f"  零速度模式（基础步长）:")
        print(f"  支撑足位置: ({support_foot_pos.x:.3f}, {support_foot_pos.y:.3f}, {support_foot_pos.z:.3f})")
        print(f"  目标足位置: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
        print(f"  期望位移: Δx={expected_delta_x:.3f}m, Δy={expected_delta_y:.3f}m")
        print(f"  实际位移: Δx={actual_delta_x:.3f}m, Δy={actual_delta_y:.3f}m")
        print(f"  位移误差: Δx={error_x*1000:.1f}mm, Δy={error_y*1000:.1f}mm")
        print(f"  水平误差: {horizontal_error*1000:.1f}mm")
        print(f"  测试结果: {'✅ 通过' if result['passed'] else '❌ 失败'}")
        
        return result
    
    def test_dynamic_step_accuracy(self) -> Dict:
        """测试动态步长精度（0.2m/s速度）"""
        print("\n=== 测试3: 动态步长精度 ===")
        
        # 设置中等速度
        test_velocity = 0.2  # 0.2m/s前进速度
        self.foot_planner.set_body_motion_intent(Vector3D(test_velocity, 0.0, 0.0), 0.0)
        
        # 获取当前足部位置
        current_positions = self.foot_planner.get_current_foot_positions()
        support_foot_pos = current_positions['right_foot']  # 右腿支撑
        
        # 执行足步规划
        target_pos = self.foot_planner.plan_foot_placement("left", "right")
        
        # 计算相对位移
        actual_delta_x = target_pos.x - support_foot_pos.x
        actual_delta_y = target_pos.y - support_foot_pos.y
        
        # 期望位移（动态调整步长）
        config = self.foot_planner.config
        dynamic_step_length = config.nominal_step_length + test_velocity * config.speed_step_gain
        dynamic_step_length = min(dynamic_step_length, config.max_step_length)  # 应用限制
        
        expected_delta_x = dynamic_step_length + config.longitudinal_stability_margin
        expected_delta_y = -config.nominal_step_width / 2 - config.lateral_stability_margin
        
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
            'dynamic_step_length': dynamic_step_length,
            'test_velocity': test_velocity,
            'requirement': '< 2cm',
            'actual': f'{horizontal_error*100:.2f}cm'
        }
        
        print(f"  速度模式（{test_velocity}m/s）:")
        print(f"  动态步长: {dynamic_step_length:.3f}m")
        print(f"  支撑足位置: ({support_foot_pos.x:.3f}, {support_foot_pos.y:.3f}, {support_foot_pos.z:.3f})")
        print(f"  目标足位置: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
        print(f"  期望位移: Δx={expected_delta_x:.3f}m, Δy={expected_delta_y:.3f}m")
        print(f"  实际位移: Δx={actual_delta_x:.3f}m, Δy={actual_delta_y:.3f}m")
        print(f"  位移误差: Δx={error_x*1000:.1f}mm, Δy={error_y*1000:.1f}mm")
        print(f"  水平误差: {horizontal_error*1000:.1f}mm")
        print(f"  测试结果: {'✅ 通过' if result['passed'] else '❌ 失败'}")
        
        return result
    
    def test_consistency_across_positions(self) -> Dict:
        """测试不同支撑足位置下的一致性"""
        print("\n=== 测试4: 位置一致性 ===")
        
        # 设置零速度以测试基础步长一致性
        self.foot_planner.set_body_motion_intent(Vector3D(0.0, 0.0, 0.0), 0.0)
        
        step_errors = []
        step_results = []
        
        # 设置不同的支撑足位置进行多次测试
        test_positions = [
            (0.0, -0.102, -0.649),   # 标准位置
            (0.15, -0.102, -0.649),  # 前移
            (0.30, -0.102, -0.649),  # 更远前移
            (-0.15, -0.102, -0.649), # 后移
            (0.0, -0.082, -0.649),   # 内移
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
            
            # 期望步长（基础步长）
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
        
        # 一致性要求：最大误差 < 2cm，标准差 < 5mm
        consistency_passed = max_error < 0.02 and std_error < 0.005
        
        result = {
            'passed': consistency_passed,
            'max_error': max_error,
            'avg_error': avg_error,
            'std_error': std_error,
            'step_results': step_results,
            'requirement': '最大误差<2cm, 标准差<5mm',
            'actual': f'最大误差{max_error*1000:.1f}mm, 标准差{std_error*1000:.1f}mm'
        }
        
        print(f"\n  统计结果:")
        print(f"    最大误差: {max_error*1000:.1f}mm")
        print(f"    平均误差: {avg_error*1000:.1f}mm")
        print(f"    误差标准差: {std_error*1000:.1f}mm")
        print(f"    一致性测试: {'✅ 通过' if consistency_passed else '❌ 失败'}")
        
        return result
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始落脚点计算精度测试（最终版本）")
        print("=" * 60)
        
        # 设置初始条件
        self.setup_initial_conditions()
        
        # 运行测试
        results = {}
        
        # 测试1: 计算时机
        results['timing'] = self.test_computation_timing()
        
        # 测试2: 基础步长精度
        results['basic_accuracy'] = self.test_basic_step_accuracy()
        
        # 测试3: 动态步长精度
        results['dynamic_accuracy'] = self.test_dynamic_step_accuracy()
        
        # 测试4: 位置一致性
        results['consistency'] = self.test_consistency_across_positions()
        
        # 汇总结果
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("测试结果总结")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get('passed', False))
        
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        print("\n详细结果:")
        
        # 测试1结果
        timing_result = results.get('timing', {})
        print(f"1. 计算时机: {'✅ 通过' if timing_result.get('passed') else '❌ 失败'}")
        if 'actual' in timing_result:
            print(f"   要求: {timing_result.get('requirement', 'N/A')}")
            print(f"   实际: {timing_result.get('actual', 'N/A')}")
        
        # 测试2结果
        basic_result = results.get('basic_accuracy', {})
        print(f"2. 基础步长精度: {'✅ 通过' if basic_result.get('passed') else '❌ 失败'}")
        if 'actual' in basic_result:
            print(f"   要求: {basic_result.get('requirement', 'N/A')}")
            print(f"   实际: {basic_result.get('actual', 'N/A')}")
        
        # 测试3结果
        dynamic_result = results.get('dynamic_accuracy', {})
        print(f"3. 动态步长精度: {'✅ 通过' if dynamic_result.get('passed') else '❌ 失败'}")
        if 'actual' in dynamic_result:
            print(f"   要求: {dynamic_result.get('requirement', 'N/A')}")
            print(f"   实际: {dynamic_result.get('actual', 'N/A')}")
        
        # 测试4结果
        consistency_result = results.get('consistency', {})
        print(f"4. 位置一致性: {'✅ 通过' if consistency_result.get('passed') else '❌ 失败'}")
        if 'actual' in consistency_result:
            print(f"   要求: {consistency_result.get('requirement', 'N/A')}")
            print(f"   实际: {consistency_result.get('actual', 'N/A')}")
        
        print("\n" + "=" * 60)
        
        if passed_tests == total_tests:
            print("🎉 所有测试通过！落脚点计算符合所有精度要求：")
            print("   ✅ 进入摆动相 ≤ 1个控制周期内完成计算")
            print("   ✅ 落脚点相对支撑足的平移符合设定步长")
            print("   ✅ 落脚点与期望偏差 ‖error‖ < 2cm（水平向）")
        else:
            print("⚠️  部分测试失败，需要进一步优化落脚点计算算法。")


def main():
    """主函数"""
    try:
        tester = FootPlacementFinalTester()
        results = tester.run_all_tests()
        
        # 保存测试结果
        import json
        with open('foot_placement_final_results.json', 'w', encoding='utf-8') as f:
            # 转换对象为字典以便JSON序列化
            def convert_for_json(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)
            
            json.dump(results, f, indent=2, default=convert_for_json, ensure_ascii=False)
        
        print(f"\n测试结果已保存到: foot_placement_final_results.json")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 