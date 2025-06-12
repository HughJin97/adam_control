#!/usr/bin/env python3
"""
FootTrajectory模块测试

测试足部轨迹生成的各种功能，包括：
- 基本轨迹生成
- 数据总线集成
- 四足协调管理
- 步态调度器集成
- 边界情况处理（在专门的测试中）
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import time

from gait_core.foot_trajectory import (
    FootTrajectory, 
    QuadrupedTrajectoryManager, 
    TrajectoryConfig,
    TrajectoryState,
    DataBusInterface
)


class TestFootTrajectory(unittest.TestCase):
    """FootTrajectory类的基本功能测试"""
    
    def setUp(self):
        """测试前设置"""
        self.config = TrajectoryConfig(
            step_height=0.08,
            swing_duration=0.4,
            enable_ground_contact_detection=False,  # 测试时禁用
            enable_penetration_protection=False,    # 测试时禁用
            enable_sensor_feedback=False            # 测试时禁用
        )
        self.foot_traj = FootTrajectory("RF", self.config)
        self.foot_traj.enable_test_mode()  # 启用测试模式
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.foot_traj.foot_name, "RF")
        self.assertEqual(self.foot_traj.state, TrajectoryState.IDLE)
        self.assertEqual(self.foot_traj.phase, 0.0)
        self.assertTrue(np.allclose(self.foot_traj.current_position, np.zeros(3)))
    
    def test_start_swing(self):
        """测试开始摆动"""
        start_pos = np.array([0.15, -0.15, 0.0])
        target_pos = np.array([0.25, -0.15, 0.0])
        
        self.foot_traj.start_swing(start_pos, target_pos)
        
        self.assertEqual(self.foot_traj.state, TrajectoryState.ACTIVE)
        self.assertTrue(np.allclose(self.foot_traj.start_position, start_pos))
        self.assertTrue(np.allclose(self.foot_traj.target_position, target_pos))
    
    def test_update_basic(self):
        """测试基本更新功能"""
        start_pos = np.array([0.15, -0.15, 0.0])
        target_pos = np.array([0.25, -0.15, 0.0])
        
        self.foot_traj.start_swing(start_pos, target_pos)
        
        # 更新到中点
        dt = 0.01
        for _ in range(20):  # 0.2秒，相位应该是0.5
            self.foot_traj.update(dt)
        
        self.assertAlmostEqual(self.foot_traj.phase, 0.5, places=2)
        self.assertEqual(self.foot_traj.state, TrajectoryState.ACTIVE)
        
        # 检查位置在起始和目标之间
        self.assertGreater(self.foot_traj.current_position[0], start_pos[0])
        self.assertLess(self.foot_traj.current_position[0], target_pos[0])
    
    def test_completion(self):
        """测试轨迹完成"""
        start_pos = np.array([0.15, -0.15, 0.0])
        target_pos = np.array([0.25, -0.15, 0.0])
        
        self.foot_traj.start_swing(start_pos, target_pos)
        
        # 更新到完成
        dt = 0.01
        for _ in range(50):  # 0.5秒，超过0.4秒的摆动周期
            self.foot_traj.update(dt)
        
        self.assertTrue(self.foot_traj.is_completed())
        self.assertAlmostEqual(self.foot_traj.phase, 1.0, places=1)
    
    def test_reset(self):
        """测试重置功能"""
        start_pos = np.array([0.15, -0.15, 0.0])
        target_pos = np.array([0.25, -0.15, 0.0])
        
        self.foot_traj.start_swing(start_pos, target_pos)
        self.foot_traj.update(0.1)  # 更新一下
        
        self.foot_traj.reset()
        self.assertEqual(self.foot_traj.state, TrajectoryState.RESET)
        
        # 下一次更新后应该变为IDLE
        self.foot_traj.update(0.01)
        self.assertEqual(self.foot_traj.state, TrajectoryState.IDLE)
    
    def test_config_update(self):
        """测试配置更新"""
        original_height = self.foot_traj.config.step_height
        self.foot_traj.update_config(step_height=0.12)
        self.assertEqual(self.foot_traj.config.step_height, 0.12)
        self.assertNotEqual(self.foot_traj.config.step_height, original_height)
    
    def test_terrain_adaptation(self):
        """测试地形自适应"""
        original_height = self.foot_traj.config.step_height
        self.foot_traj.set_terrain_adaptive_params("rough")
        self.assertNotEqual(self.foot_traj.config.step_height, original_height)
        self.assertEqual(self.foot_traj.config.step_height, 0.12)
    
    def test_trajectory_data(self):
        """测试轨迹数据获取"""
        start_pos = np.array([0.15, -0.15, 0.0])
        target_pos = np.array([0.25, -0.15, 0.0])
        
        self.foot_traj.start_swing(start_pos, target_pos)
        self.foot_traj.update(0.1)
        
        data = self.foot_traj.get_trajectory_data()
        
        self.assertEqual(data.state, TrajectoryState.ACTIVE)
        self.assertGreater(data.phase, 0)
        self.assertGreater(data.elapsed_time, 0)
        self.assertIsInstance(data.current_position, np.ndarray)
    
    def test_progress_info(self):
        """测试进度信息"""
        start_pos = np.array([0.15, -0.15, 0.0])
        target_pos = np.array([0.25, -0.15, 0.0])
        
        self.foot_traj.start_swing(start_pos, target_pos)
        self.foot_traj.update(0.1)
        
        progress = self.foot_traj.get_progress_info()
        
        self.assertEqual(progress['state'], "active")
        self.assertEqual(progress['foot_name'], "RF")
        self.assertGreater(progress['phase'], 0)
        self.assertIsInstance(progress['current_position'], list)


class TestDataBusIntegration(unittest.TestCase):
    """数据总线集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.config = TrajectoryConfig(
            enable_ground_contact_detection=False,
            enable_penetration_protection=False,
            enable_sensor_feedback=False
        )
        self.foot_traj = FootTrajectory("RF", self.config)
        self.foot_traj.enable_test_mode()
        self.data_bus = DataBusInterface()
    
    def test_data_bus_connection(self):
        """测试数据总线连接"""
        self.foot_traj.connect_data_bus(self.data_bus)
        self.assertIsNotNone(self.foot_traj.data_bus)
    
    def test_ground_height_from_bus(self):
        """测试从数据总线获取地面高度"""
        self.foot_traj.connect_data_bus(self.data_bus)
        
        start_pos = np.array([0.0, -0.15, 0.0])
        target_pos = np.array([0.1, -0.15, 0.0])
        
        self.foot_traj.start_swing(start_pos, target_pos)
        
        # 模拟数据总线返回地面高度
        self.data_bus.ground_height_map = {(0.05, -0.15): 0.02}
        
        position = self.foot_traj.update(0.1)
        self.assertIsInstance(position, np.ndarray)
    
    def test_target_update_from_bus(self):
        """测试从数据总线更新目标"""
        self.foot_traj.connect_data_bus(self.data_bus)
        
        start_pos = np.array([0.0, -0.15, 0.0])
        target_pos = np.array([0.1, -0.15, 0.0])
        
        self.foot_traj.start_swing(start_pos, target_pos)
        
        # 模拟数据总线更新目标
        new_target = np.array([0.15, -0.15, 0.0])
        self.data_bus.set_foot_target("RF", new_target)
        
        position = self.foot_traj.update_from_data_bus(0.1)
        self.assertIsInstance(position, np.ndarray)


class TestQuadrupedTrajectoryManager(unittest.TestCase):
    """四足轨迹管理器测试"""
    
    def setUp(self):
        """测试前设置"""
        self.config = TrajectoryConfig(
            swing_duration=0.3,
            enable_ground_contact_detection=False,
            enable_penetration_protection=False,
            enable_sensor_feedback=False
        )
        self.manager = QuadrupedTrajectoryManager(self.config)
        
        # 为所有足部启用测试模式
        for foot_traj in self.manager.foot_trajectories.values():
            foot_traj.enable_test_mode()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(len(self.manager.foot_trajectories), 4)
        self.assertIn("RF", self.manager.foot_trajectories)
        self.assertIn("LF", self.manager.foot_trajectories)
        self.assertIn("RH", self.manager.foot_trajectories)
        self.assertIn("LH", self.manager.foot_trajectories)
    
    def test_start_swing(self):
        """测试开始摆动"""
        start_pos = np.array([0.15, -0.15, 0.0])
        target_pos = np.array([0.25, -0.15, 0.0])
        
        self.manager.start_swing("RF", start_pos, target_pos)
        
        rf_traj = self.manager.foot_trajectories["RF"]
        self.assertEqual(rf_traj.state, TrajectoryState.ACTIVE)
        self.assertTrue(np.allclose(rf_traj.start_position, start_pos))
    
    def test_update_all(self):
        """测试更新所有足部"""
        # 开始两个足部的摆动
        self.manager.start_swing("RF", np.array([0.15, -0.15, 0.0]), np.array([0.25, -0.15, 0.0]))
        self.manager.start_swing("LH", np.array([-0.15, 0.15, 0.0]), np.array([-0.05, 0.15, 0.0]))
        
        positions = self.manager.update_all(0.1)
        
        self.assertIn("RF", positions)
        self.assertIn("LH", positions)
        self.assertIsInstance(positions["RF"], np.ndarray)
        self.assertIsInstance(positions["LH"], np.ndarray)
    
    def test_progress_tracking(self):
        """测试进度跟踪"""
        self.manager.start_swing("RF", np.array([0.15, -0.15, 0.0]), np.array([0.25, -0.15, 0.0]))
        
        # 更新几次
        for _ in range(10):
            self.manager.update_all(0.01)
        
        progress = self.manager.get_all_progress()
        self.assertIn("RF", progress)
        self.assertEqual(progress["RF"]["state"], "active")
    
    def test_terrain_setting(self):
        """测试地形设置"""
        self.manager.set_terrain_type("rough")
        
        # 检查所有足部的配置是否更新
        for foot_traj in self.manager.foot_trajectories.values():
            self.assertEqual(foot_traj.config.step_height, 0.12)
    
    def test_emergency_stop(self):
        """测试紧急停止"""
        self.manager.start_swing("RF", np.array([0.15, -0.15, 0.0]), np.array([0.25, -0.15, 0.0]))
        self.manager.start_swing("LH", np.array([-0.15, 0.15, 0.0]), np.array([-0.05, 0.15, 0.0]))
        
        self.manager.emergency_stop()
        
        for foot_traj in self.manager.foot_trajectories.values():
            self.assertEqual(foot_traj.state, TrajectoryState.IDLE)
    
    def test_reset_all(self):
        """测试重置所有"""
        self.manager.start_swing("RF", np.array([0.15, -0.15, 0.0]), np.array([0.25, -0.15, 0.0]))
        
        # 更新一下
        for _ in range(10):
            self.manager.update_all(0.01)
        
        self.manager.reset_all()
        
        for foot_traj in self.manager.foot_trajectories.values():
            self.assertEqual(foot_traj.state, TrajectoryState.RESET)


class TestGaitSchedulerIntegration(unittest.TestCase):
    """步态调度器集成测试"""
    
    def setUp(self):
        """测试前设置"""
        self.config = TrajectoryConfig(
            swing_duration=0.1,  # 短周期便于测试
            enable_ground_contact_detection=False,
            enable_penetration_protection=False,
            enable_sensor_feedback=False
        )
        self.manager = QuadrupedTrajectoryManager(self.config)
        
        # 为所有足部启用测试模式
        for foot_traj in self.manager.foot_trajectories.values():
            foot_traj.enable_test_mode()
        
        # 模拟步态调度器
        self.gait_scheduler = type('GaitScheduler', (), {
            'swing_completed_count': 0,
            'on_swing_completed': lambda self, foot_name: setattr(self, 'swing_completed_count', self.swing_completed_count + 1)
        })()
        
        self.manager.connect_gait_scheduler(self.gait_scheduler)
    
    def test_completion_callback(self):
        """测试完成回调"""
        self.manager.start_swing("RF", np.array([0.15, -0.15, 0.0]), np.array([0.25, -0.15, 0.0]))
        
        # 更新到完成
        for _ in range(15):  # 0.15秒，超过0.1秒的摆动周期
            self.manager.update_all(0.01)
        
        # 检查回调是否被调用
        self.assertGreater(self.gait_scheduler.swing_completed_count, 0)


class TestConfigurationClasses(unittest.TestCase):
    """配置类测试"""
    
    def test_trajectory_config(self):
        """测试轨迹配置"""
        config = TrajectoryConfig()
        
        # 检查默认值
        self.assertEqual(config.step_height, 0.08)
        self.assertEqual(config.swing_duration, 0.4)
        self.assertEqual(config.interpolation_type, "cubic")
        
        # 检查边界情况处理参数
        self.assertTrue(config.enable_ground_contact_detection)
        self.assertTrue(config.enable_penetration_protection)
        self.assertGreater(config.contact_force_threshold, 0)
    
    def test_trajectory_state_enum(self):
        """测试轨迹状态枚举"""
        self.assertEqual(TrajectoryState.IDLE.value, "idle")
        self.assertEqual(TrajectoryState.ACTIVE.value, "active")
        self.assertEqual(TrajectoryState.COMPLETED.value, "completed")
        self.assertEqual(TrajectoryState.INTERRUPTED.value, "interrupted")
        self.assertEqual(TrajectoryState.EMERGENCY_STOP.value, "emergency_stop")


def run_tests():
    """运行所有测试"""
    print("开始运行FootTrajectory模块测试...")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestFootTrajectory,
        TestDataBusIntegration, 
        TestQuadrupedTrajectoryManager,
        TestGaitSchedulerIntegration,
        TestConfigurationClasses
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 统计结果
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total_tests - failures - errors
    
    print("=" * 50)
    print("测试结果统计:")
    print(f"总测试数: {total_tests}")
    print(f"成功: {success}")
    print(f"失败: {failures}")
    print(f"错误: {errors}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success_rate = (success / total_tests) * 100 if total_tests > 0 else 0
    print(f"\n测试通过率: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 