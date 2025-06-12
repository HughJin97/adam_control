#!/usr/bin/env python3
"""
摆动足轨迹规划模块单元测试

测试各种轨迹规划方法的正确性和一致性
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from gait_core.swing_trajectory import (
    SwingTrajectoryPlanner,
    SwingTrajectoryConfig,
    TrajectoryType,
    TrajectoryParameterOptimizer,
    create_swing_trajectory_planner
)


class TestSwingTrajectory(unittest.TestCase):
    """摆动轨迹规划测试类"""
    
    def setUp(self):
        """测试设置"""
        self.start_pos = np.array([0.0, 0.1, 0.0])
        self.end_pos = np.array([0.15, 0.1, 0.0])
        self.step_height = 0.08
        self.step_duration = 0.4
    
    def test_trajectory_types(self):
        """测试不同轨迹类型"""
        trajectory_types = ["polynomial", "bezier", "sinusoidal", "cycloid"]
        
        for traj_type in trajectory_types:
            with self.subTest(trajectory_type=traj_type):
                planner = create_swing_trajectory_planner(
                    trajectory_type=traj_type,
                    step_height=self.step_height,
                    step_duration=self.step_duration
                )
                planner.set_trajectory_parameters(self.start_pos, self.end_pos)
                
                # 测试起点和终点
                start_computed = planner.compute_trajectory_position(0.0)
                end_computed = planner.compute_trajectory_position(1.0)
                
                np.testing.assert_array_almost_equal(
                    start_computed[:2], self.start_pos[:2], decimal=3,
                    err_msg=f"{traj_type} 轨迹起点不正确"
                )
                np.testing.assert_array_almost_equal(
                    end_computed[:2], self.end_pos[:2], decimal=3,
                    err_msg=f"{traj_type} 轨迹终点不正确"
                )
                
                # 测试中点高度
                mid_pos = planner.compute_trajectory_position(0.5)
                self.assertGreater(
                    mid_pos[2], max(self.start_pos[2], self.end_pos[2]),
                    f"{traj_type} 轨迹中点高度应该大于起终点"
                )
    
    def test_trajectory_continuity(self):
        """测试轨迹连续性"""
        planner = create_swing_trajectory_planner("bezier", self.step_height)
        planner.set_trajectory_parameters(self.start_pos, self.end_pos)
        
        # 测试位置连续性
        phases = np.linspace(0.0, 1.0, 101)
        positions = []
        for phase in phases:
            pos = planner.compute_trajectory_position(phase)
            positions.append(pos)
        
        positions = np.array(positions)
        
        # 检查位置变化是否平滑（相邻点距离不能太大）
        for i in range(1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[i-1])
            self.assertLess(distance, 0.05, "轨迹位置变化过大，不够平滑")
    
    def test_velocity_calculation(self):
        """测试速度计算"""
        planner = create_swing_trajectory_planner("polynomial", self.step_height)
        planner.set_trajectory_parameters(self.start_pos, self.end_pos)
        
        # 测试起点和终点速度应该较小（边界条件）
        start_vel = planner.compute_trajectory_velocity(0.0)
        end_vel = planner.compute_trajectory_velocity(1.0)
        
        # 水平速度不应该太大
        self.assertLess(np.linalg.norm(start_vel[:2]), 1.0, "起点水平速度过大")
        self.assertLess(np.linalg.norm(end_vel[:2]), 1.0, "终点水平速度过大")
    
    def test_parameter_validation(self):
        """测试参数验证"""
        planner = SwingTrajectoryPlanner()
        
        # 测试设置有效参数
        planner.set_trajectory_parameters(
            self.start_pos, self.end_pos, 
            step_height=0.05, step_duration=0.3
        )
        self.assertEqual(planner.config.step_height, 0.05)
        self.assertEqual(planner.config.step_duration, 0.3)
        
        # 测试轨迹信息获取
        info = planner.get_trajectory_info()
        self.assertIn('trajectory_type', info)
        self.assertIn('step_height', info)
        self.assertIn('is_active', info)
    
    def test_trajectory_sampling(self):
        """测试轨迹采样"""
        planner = create_swing_trajectory_planner("sinusoidal", self.step_height)
        planner.set_trajectory_parameters(self.start_pos, self.end_pos)
        
        # 生成采样点
        trajectory_data = planner.generate_trajectory_samples(50)
        
        self.assertIn('phases', trajectory_data)
        self.assertIn('positions', trajectory_data)
        self.assertIn('velocities', trajectory_data)
        self.assertIn('config', trajectory_data)
        
        # 检查采样点数量
        self.assertEqual(len(trajectory_data['phases']), 50)
        self.assertEqual(trajectory_data['positions'].shape[0], 50)
        self.assertEqual(trajectory_data['velocities'].shape[0], 50)


class TestTrajectoryOptimizer(unittest.TestCase):
    """轨迹参数优化器测试类"""
    
    def setUp(self):
        """测试设置"""
        self.optimizer = TrajectoryParameterOptimizer()
    
    def test_step_height_optimization(self):
        """测试抬脚高度优化"""
        # 测试不同地形和速度组合
        test_cases = [
            (0.0, 0.5),  # 平地慢速
            (0.5, 1.0),  # 中等地形中速
            (1.0, 0.3),  # 困难地形慢速
        ]
        
        for terrain, speed in test_cases:
            height = self.optimizer.optimize_step_height(terrain, speed)
            
            # 检查高度在合理范围内
            self.assertGreaterEqual(height, 0.03, "优化高度不能小于3cm")
            self.assertLessEqual(height, 0.12, "优化高度不能大于12cm")
            
            # 困难地形应该有更高的抬脚高度
            if terrain > 0.5:
                self.assertGreater(height, 0.06, "困难地形抬脚高度应该更高")
    
    def test_timing_optimization(self):
        """测试时序优化"""
        frequencies = [0.5, 1.0, 2.0]
        
        for freq in frequencies:
            params = self.optimizer.optimize_trajectory_timing(freq)
            
            self.assertIn('step_duration', params)
            self.assertIn('lift_off_ratio', params)
            self.assertIn('max_height_ratio', params)
            self.assertIn('touch_down_ratio', params)
            
            # 检查时序参数合理性
            self.assertGreater(params['step_duration'], 0, "摆动周期必须为正")
            self.assertGreaterEqual(params['lift_off_ratio'], 0.0, "离地时间比例不能为负")
            self.assertLessEqual(params['touch_down_ratio'], 1.0, "触地时间比例不能超过1")
            self.assertLess(params['lift_off_ratio'], params['max_height_ratio'], 
                           "离地时间应该在最高点之前")
            self.assertLess(params['max_height_ratio'], params['touch_down_ratio'], 
                           "最高点时间应该在触地之前")


class TestTrajectoryComparison(unittest.TestCase):
    """轨迹方法比较测试类"""
    
    def setUp(self):
        """测试设置"""
        self.start_pos = np.array([0.0, 0.1, 0.0])
        self.end_pos = np.array([0.15, 0.1, 0.0])
        self.step_height = 0.08
        self.trajectory_types = ["polynomial", "bezier", "sinusoidal", "cycloid"]
    
    def test_trajectory_characteristics(self):
        """测试各轨迹特性"""
        planners = {}
        trajectories = {}
        
        # 创建所有类型的规划器
        for traj_type in self.trajectory_types:
            planner = create_swing_trajectory_planner(traj_type, self.step_height)
            planner.set_trajectory_parameters(self.start_pos, self.end_pos)
            planners[traj_type] = planner
            
            # 生成轨迹数据
            data = planner.generate_trajectory_samples(100)
            trajectories[traj_type] = data
        
        # 比较关键特性
        for traj_type, data in trajectories.items():
            positions = data['positions']
            velocities = data['velocities']
            
            # 检查最大高度
            max_height = np.max(positions[:, 2])
            self.assertGreater(max_height, 0.02, f"{traj_type} 最大高度太小")
            
            # 检查速度合理性
            max_horizontal_speed = np.max(np.linalg.norm(velocities[:, :2], axis=1))
            max_vertical_speed = np.max(np.abs(velocities[:, 2]))
            
            self.assertLess(max_horizontal_speed, 3.0, f"{traj_type} 水平速度过大")
            self.assertLess(max_vertical_speed, 2.0, f"{traj_type} 垂直速度过大")
    
    def test_trajectory_efficiency(self):
        """测试轨迹效率（能耗评估）"""
        for traj_type in self.trajectory_types:
            planner = create_swing_trajectory_planner(traj_type, self.step_height)
            planner.set_trajectory_parameters(self.start_pos, self.end_pos)
            
            # 计算轨迹的能耗估算（基于速度和加速度）
            phases = np.linspace(0.0, 1.0, 50)
            total_energy = 0.0
            
            for i, phase in enumerate(phases[:-1]):
                vel1 = planner.compute_trajectory_velocity(phases[i])
                vel2 = planner.compute_trajectory_velocity(phases[i+1])
                
                # 简化的能耗计算：动能变化
                kinetic_energy = 0.5 * (np.linalg.norm(vel2)**2 - np.linalg.norm(vel1)**2)
                total_energy += abs(kinetic_energy)
            
            # 能耗不应该过大（具体数值根据实际需求调整）
            self.assertLess(total_energy, 10.0, f"{traj_type} 轨迹能耗过高")


class TestGetSwingFootPosition(unittest.TestCase):
    """get_swing_foot_position 函数测试类"""
    
    def setUp(self):
        """测试设置"""
        self.start_pos = np.array([0.0, 0.1, 0.0])
        self.target_pos = np.array([0.15, 0.1, 0.0])
        self.step_height = 0.08
    
    def test_basic_functionality(self):
        """测试基本功能"""
        from gait_core.swing_trajectory import get_swing_foot_position
        
        # 测试起点
        pos_start = get_swing_foot_position(0.0, self.start_pos, self.target_pos, self.step_height)
        np.testing.assert_array_almost_equal(pos_start, self.start_pos, decimal=6,
                                           err_msg="起点位置不正确")
        
        # 测试终点
        pos_end = get_swing_foot_position(1.0, self.start_pos, self.target_pos, self.step_height)
        np.testing.assert_array_almost_equal(pos_end, self.target_pos, decimal=6,
                                           err_msg="终点位置不正确")
        
        # 测试中点高度
        pos_mid = get_swing_foot_position(0.5, self.start_pos, self.target_pos, self.step_height)
        self.assertAlmostEqual(pos_mid[2], self.step_height, places=6,
                              msg="中点高度不正确")
    
    def test_interpolation_types(self):
        """测试不同插值类型"""
        from gait_core.swing_trajectory import get_swing_foot_position
        
        interpolation_types = ["linear", "cubic", "smooth"]
        
        for interp_type in interpolation_types:
            with self.subTest(interpolation_type=interp_type):
                # 测试关键点
                pos_start = get_swing_foot_position(0.0, self.start_pos, self.target_pos, 
                                                  self.step_height, interp_type, "sine")
                pos_end = get_swing_foot_position(1.0, self.start_pos, self.target_pos, 
                                                self.step_height, interp_type, "sine")
                
                # 起终点应该一致
                np.testing.assert_array_almost_equal(pos_start[:2], self.start_pos[:2], decimal=3)
                np.testing.assert_array_almost_equal(pos_end[:2], self.target_pos[:2], decimal=3)
    
    def test_vertical_trajectory_types(self):
        """测试不同垂直轨迹类型"""
        from gait_core.swing_trajectory import get_swing_foot_position
        
        vertical_types = ["sine", "parabola", "smooth_parabola"]
        
        for vert_type in vertical_types:
            with self.subTest(vertical_type=vert_type):
                # 测试起终点高度
                pos_start = get_swing_foot_position(0.0, self.start_pos, self.target_pos, 
                                                  self.step_height, "linear", vert_type)
                pos_end = get_swing_foot_position(1.0, self.start_pos, self.target_pos, 
                                                self.step_height, "linear", vert_type)
                
                self.assertAlmostEqual(pos_start[2], self.start_pos[2], places=3,
                                     msg=f"{vert_type} 起点高度不正确")
                self.assertAlmostEqual(pos_end[2], self.target_pos[2], places=3,
                                     msg=f"{vert_type} 终点高度不正确")
                
                # 测试中点高度应该大于起终点
                pos_mid = get_swing_foot_position(0.5, self.start_pos, self.target_pos, 
                                                self.step_height, "linear", vert_type)
                self.assertGreater(pos_mid[2], max(self.start_pos[2], self.target_pos[2]),
                                 msg=f"{vert_type} 中点高度应该大于起终点")
    
    def test_phase_limits(self):
        """测试相位限制"""
        from gait_core.swing_trajectory import get_swing_foot_position
        
        # 测试相位超出范围时的处理
        pos_negative = get_swing_foot_position(-0.1, self.start_pos, self.target_pos, self.step_height)
        pos_zero = get_swing_foot_position(0.0, self.start_pos, self.target_pos, self.step_height)
        np.testing.assert_array_almost_equal(pos_negative, pos_zero, decimal=6,
                                           err_msg="负相位应该被限制为0")
        
        pos_over_one = get_swing_foot_position(1.1, self.start_pos, self.target_pos, self.step_height)
        pos_one = get_swing_foot_position(1.0, self.start_pos, self.target_pos, self.step_height)
        np.testing.assert_array_almost_equal(pos_over_one, pos_one, decimal=6,
                                           err_msg="超过1的相位应该被限制为1")
    
    def test_trajectory_continuity(self):
        """测试轨迹连续性"""
        from gait_core.swing_trajectory import get_swing_foot_position
        
        # 测试相邻相位点的连续性
        phases = np.linspace(0.0, 1.0, 51)
        positions = []
        
        for phase in phases:
            pos = get_swing_foot_position(phase, self.start_pos, self.target_pos, 
                                        self.step_height, "cubic", "sine")
            positions.append(pos)
        
        positions = np.array(positions)
        
        # 检查相邻点之间的距离
        for i in range(1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[i-1])
            self.assertLess(distance, 0.01, "轨迹不够连续")
    
    def test_velocity_calculation(self):
        """测试速度计算"""
        from gait_core.swing_trajectory import get_swing_foot_velocity
        
        # 测试速度计算的合理性
        vel_start = get_swing_foot_velocity(0.0, self.start_pos, self.target_pos, 
                                          self.step_height, 0.4, "cubic", "sine")
        vel_end = get_swing_foot_velocity(1.0, self.start_pos, self.target_pos, 
                                        self.step_height, 0.4, "cubic", "sine")
        
        # 起终点的垂直速度应该较小
        self.assertLess(abs(vel_start[2]), 1.0, "起点垂直速度过大")
        self.assertLess(abs(vel_end[2]), 1.0, "终点垂直速度过大")
        
        # 中点的水平速度应该较大
        vel_mid = get_swing_foot_velocity(0.5, self.start_pos, self.target_pos, 
                                        self.step_height, 0.4, "linear", "sine")
        horizontal_speed = np.linalg.norm(vel_mid[:2])
        self.assertGreater(horizontal_speed, 0.1, "中点水平速度过小")
    
    def test_trajectory_creation(self):
        """测试轨迹创建"""
        from gait_core.swing_trajectory import create_swing_foot_trajectory
        
        trajectory = create_swing_foot_trajectory(
            self.start_pos, self.target_pos, self.step_height, 
            num_points=50, interpolation_type="cubic", vertical_trajectory_type="sine"
        )
        
        # 检查返回的数据结构
        self.assertIn('phases', trajectory)
        self.assertIn('positions', trajectory)
        self.assertIn('start_pos', trajectory)
        self.assertIn('target_pos', trajectory)
        
        # 检查数据一致性
        self.assertEqual(len(trajectory['phases']), 50)
        self.assertEqual(trajectory['positions'].shape, (50, 3))
        
        # 检查起终点
        np.testing.assert_array_almost_equal(
            trajectory['positions'][0], self.start_pos, decimal=3
        )
        np.testing.assert_array_almost_equal(
            trajectory['positions'][-1], self.target_pos, decimal=3
        )


def run_trajectory_tests():
    """运行所有轨迹测试"""
    print("开始运行摆动轨迹规划模块测试...")
    print("=" * 50)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # 添加测试
    test_suite.addTest(loader.loadTestsFromTestCase(TestSwingTrajectory))
    test_suite.addTest(loader.loadTestsFromTestCase(TestTrajectoryOptimizer))
    test_suite.addTest(loader.loadTestsFromTestCase(TestTrajectoryComparison))
    test_suite.addTest(loader.loadTestsFromTestCase(TestGetSwingFootPosition))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 报告结果
    print("\n" + "=" * 50)
    print("测试结果统计:")
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n测试通过率: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_trajectory_tests()
    sys.exit(0 if success else 1) 