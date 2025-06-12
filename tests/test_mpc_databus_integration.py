"""
MPC数据总线集成测试
测试MPC求解结果与数据总线的集成功能

测试内容：
1. 基本集成功能测试
2. 不同输出模式测试
3. 数据验证测试
4. 性能测试
5. 错误处理测试

作者: Adam Control Team
版本: 1.0
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch

# 导入测试模块
from gait_core.data_bus import DataBus, Vector3D
from gait_core.mpc_data_bus_integration import (
    MPCDataBusIntegrator, MPCOutputMode, MPCDataBusConfig,
    create_mpc_databus_integrator
)
from gait_core.mpc_solver import MPCResult


class TestMPCDataBusIntegration(unittest.TestCase):
    """MPC数据总线集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.data_bus = DataBus()
        self.config = MPCDataBusConfig()
        self.integrator = MPCDataBusIntegrator(self.data_bus, self.config)
        
    def create_test_mpc_result(self, success=True):
        """创建测试用MPC结果"""
        result = MPCResult()
        result.success = success
        result.solve_time = 0.015
        result.cost = 125.6
        result.solver_info = {'solver_type': 'osqp'}
        
        if success:
            # 模拟接触力
            result.current_contact_forces = {
                'left_foot': Vector3D(10.0, 5.0, 80.0),
                'right_foot': Vector3D(-8.0, -3.0, 75.0)
            }
            
            # 模拟质心轨迹
            result.com_position_trajectory = [
                Vector3D(0.1, 0.0, 0.8),
                Vector3D(0.15, 0.02, 0.8)
            ]
            
            result.com_velocity_trajectory = [
                Vector3D(0.5, 0.1, 0.0),
                Vector3D(0.48, 0.05, 0.0)
            ]
            
            result.com_acceleration_trajectory = [
                Vector3D(0.1, 0.05, 0.0),
                Vector3D(0.08, 0.02, 0.0)
            ]
            
            # 模拟ZMP轨迹
            result.zmp_trajectory = [
                Vector3D(0.05, 0.0, 0.0),
                Vector3D(0.08, 0.01, 0.0)
            ]
            
            # 模拟下一步落脚点
            result.next_footstep = {
                'left_foot': Vector3D(0.3, 0.1, 0.0),
                'right_foot': Vector3D(0.25, -0.1, 0.0)
            }
        
        return result
    
    def test_basic_integration(self):
        """测试基本集成功能"""
        print("测试基本MPC数据总线集成...")
        
        # 创建测试MPC结果
        mpc_result = self.create_test_mpc_result()
        
        # 执行集成
        success = self.integrator.update_from_mpc_result(mpc_result)
        
        # 验证集成成功
        self.assertTrue(success, "MPC数据总线集成应该成功")
        
        # 验证期望力已写入
        desired_forces = self.data_bus.get_all_desired_contact_forces()
        self.assertEqual(len(desired_forces), 2, "应该有两个足部的期望力")
        self.assertIn('left_foot', desired_forces, "应该包含左脚期望力")
        self.assertIn('right_foot', desired_forces, "应该包含右脚期望力")
        
        # 验证质心状态已更新
        com_pos = self.data_bus.get_center_of_mass_position()
        self.assertAlmostEqual(com_pos.x, 0.1, places=3, msg="质心X位置应该正确")
        self.assertAlmostEqual(com_pos.z, 0.8, places=3, msg="质心Z位置应该正确")
        
        com_vel = self.data_bus.get_center_of_mass_velocity()
        self.assertAlmostEqual(com_vel.x, 0.5, places=3, msg="质心X速度应该正确")
        
        # 验证MPC状态已更新
        mpc_status = self.data_bus.get_mpc_status()
        self.assertTrue(mpc_status['solve_success'], "MPC状态应该显示求解成功")
        self.assertAlmostEqual(mpc_status['last_solve_time'], 0.015, places=3)
        
        print("✓ 基本集成功能测试通过")
    
    def test_force_only_mode(self):
        """测试仅力输出模式"""
        print("测试仅力输出模式...")
        
        # 配置为仅力输出模式
        config = MPCDataBusConfig(output_mode=MPCOutputMode.FORCE_ONLY)
        integrator = MPCDataBusIntegrator(self.data_bus, config)
        
        # 创建测试MPC结果
        mpc_result = self.create_test_mpc_result()
        
        # 执行集成
        success = integrator.update_from_mpc_result(mpc_result)
        self.assertTrue(success, "仅力输出模式集成应该成功")
        
        # 验证期望力已写入
        desired_forces = self.data_bus.get_all_desired_contact_forces()
        self.assertEqual(len(desired_forces), 2, "应该有两个足部的期望力")
        
        # 验证质心状态未更新（因为是仅力模式）
        # 注意：这里需要检查质心状态是否为初始值
        
        print("✓ 仅力输出模式测试通过")
    
    def test_trajectory_only_mode(self):
        """测试仅轨迹输出模式"""
        print("测试仅轨迹输出模式...")
        
        # 配置为仅轨迹输出模式
        config = MPCDataBusConfig(output_mode=MPCOutputMode.TRAJECTORY_ONLY)
        integrator = MPCDataBusIntegrator(self.data_bus, config)
        
        # 创建测试MPC结果
        mpc_result = self.create_test_mpc_result()
        
        # 执行集成
        success = integrator.update_from_mpc_result(mpc_result)
        self.assertTrue(success, "仅轨迹输出模式集成应该成功")
        
        # 验证质心状态已更新
        com_pos = self.data_bus.get_center_of_mass_position()
        self.assertAlmostEqual(com_pos.x, 0.1, places=3)
        
        # 验证轨迹缓存已更新
        com_trajectory = self.data_bus.get_mpc_com_trajectory()
        self.assertEqual(len(com_trajectory), 2, "质心轨迹缓存应该有2个点")
        
        print("✓ 仅轨迹输出模式测试通过")
    
    def test_footstep_only_mode(self):
        """测试仅落脚点输出模式"""
        print("测试仅落脚点输出模式...")
        
        # 配置为仅落脚点输出模式
        config = MPCDataBusConfig(output_mode=MPCOutputMode.FOOTSTEP_ONLY)
        integrator = MPCDataBusIntegrator(self.data_bus, config)
        
        # 创建测试MPC结果
        mpc_result = self.create_test_mpc_result()
        
        # 执行集成
        success = integrator.update_from_mpc_result(mpc_result)
        self.assertTrue(success, "仅落脚点输出模式集成应该成功")
        
        # 验证目标足部位置已更新
        left_target = self.data_bus.get_target_foot_position('left_foot')
        self.assertIsNotNone(left_target, "左脚目标位置应该存在")
        self.assertAlmostEqual(left_target['x'], 0.3, places=3)
        
        right_target = self.data_bus.get_target_foot_position('right_foot')
        self.assertIsNotNone(right_target, "右脚目标位置应该存在")
        self.assertAlmostEqual(right_target['x'], 0.25, places=3)
        
        print("✓ 仅落脚点输出模式测试通过")
    
    def test_data_validation(self):
        """测试数据验证功能"""
        print("测试数据验证功能...")
        
        # 启用数据验证
        config = MPCDataBusConfig(
            enable_data_validation=True,
            max_force_magnitude=100.0,  # 设置较小的力限制
            max_com_velocity=1.0        # 设置较小的速度限制
        )
        integrator = MPCDataBusIntegrator(self.data_bus, config)
        
        # 创建超出限制的MPC结果
        mpc_result = self.create_test_mpc_result()
        mpc_result.current_contact_forces['left_foot'] = Vector3D(200.0, 0.0, 0.0)  # 超出力限制
        
        # 执行集成，应该失败
        success = integrator.update_from_mpc_result(mpc_result)
        self.assertFalse(success, "超出力限制的结果应该验证失败")
        
        # 检查验证错误计数
        stats = integrator.get_integration_statistics()
        self.assertGreater(stats['validation_errors'], 0, "应该有验证错误记录")
        
        print("✓ 数据验证功能测试通过")
    
    def test_force_smoothing(self):
        """测试力平滑功能"""
        print("测试力平滑功能...")
        
        # 配置力平滑
        config = MPCDataBusConfig(force_smoothing_factor=0.5)
        integrator = MPCDataBusIntegrator(self.data_bus, config)
        
        # 第一次更新
        mpc_result1 = self.create_test_mpc_result()
        mpc_result1.current_contact_forces['left_foot'] = Vector3D(100.0, 0.0, 0.0)
        integrator.update_from_mpc_result(mpc_result1)
        
        # 第二次更新，力值变化较大
        mpc_result2 = self.create_test_mpc_result()
        mpc_result2.current_contact_forces['left_foot'] = Vector3D(0.0, 0.0, 0.0)
        integrator.update_from_mpc_result(mpc_result2)
        
        # 检查平滑后的力值
        smoothed_force = self.data_bus.get_desired_contact_force('left_foot')
        self.assertIsNotNone(smoothed_force, "平滑后的力应该存在")
        
        # 平滑后的力应该在原始值之间
        self.assertGreater(smoothed_force.x, 0.0, "平滑后的力应该大于0")
        self.assertLess(smoothed_force.x, 100.0, "平滑后的力应该小于原始值")
        
        print("✓ 力平滑功能测试通过")
    
    def test_update_frequency_control(self):
        """测试更新频率控制"""
        print("测试更新频率控制...")
        
        # 配置较低的更新频率
        config = MPCDataBusConfig(update_frequency=10.0)  # 10Hz
        integrator = MPCDataBusIntegrator(self.data_bus, config)
        
        mpc_result = self.create_test_mpc_result()
        
        # 第一次更新应该成功
        success1 = integrator.update_from_mpc_result(mpc_result, time.time())
        self.assertTrue(success1, "第一次更新应该成功")
        
        # 立即第二次更新应该被跳过（频率限制）
        success2 = integrator.update_from_mpc_result(mpc_result, time.time())
        self.assertTrue(success2, "频率限制下的更新应该返回True但被跳过")
        
        # 等待足够时间后更新应该成功
        future_time = time.time() + 0.2  # 200ms后
        success3 = integrator.update_from_mpc_result(mpc_result, future_time)
        self.assertTrue(success3, "等待足够时间后的更新应该成功")
        
        print("✓ 更新频率控制测试通过")
    
    def test_failed_mpc_result(self):
        """测试MPC求解失败的情况"""
        print("测试MPC求解失败处理...")
        
        # 创建失败的MPC结果
        failed_result = self.create_test_mpc_result(success=False)
        
        # 执行集成
        success = self.integrator.update_from_mpc_result(failed_result)
        self.assertFalse(success, "失败的MPC结果应该导致集成失败")
        
        # 检查验证错误计数
        stats = self.integrator.get_integration_statistics()
        self.assertGreater(stats['validation_errors'], 0, "应该有验证错误记录")
        
        print("✓ MPC求解失败处理测试通过")
    
    def test_integration_statistics(self):
        """测试集成统计功能"""
        print("测试集成统计功能...")
        
        # 执行多次集成
        for i in range(5):
            mpc_result = self.create_test_mpc_result()
            self.integrator.update_from_mpc_result(mpc_result)
        
        # 获取统计信息
        stats = self.integrator.get_integration_statistics()
        
        # 验证统计信息
        self.assertEqual(stats['update_count'], 5, "更新次数应该正确")
        self.assertGreater(stats['average_update_time'], 0, "平均更新时间应该大于0")
        self.assertGreaterEqual(stats['success_rate'], 0.8, "成功率应该较高")
        
        # 验证配置信息
        self.assertEqual(stats['config']['output_mode'], 'combined')
        self.assertTrue(stats['config']['force_enabled'])
        self.assertTrue(stats['config']['trajectory_enabled'])
        self.assertTrue(stats['config']['footstep_enabled'])
        
        print("✓ 集成统计功能测试通过")
    
    def test_create_integrator_function(self):
        """测试集成器创建函数"""
        print("测试集成器创建函数...")
        
        data_bus = DataBus()
        integrator = create_mpc_databus_integrator(
            data_bus,
            output_mode=MPCOutputMode.FORCE_ONLY,
            update_frequency=200.0
        )
        
        # 验证集成器配置
        self.assertEqual(integrator.config.output_mode, MPCOutputMode.FORCE_ONLY)
        self.assertEqual(integrator.config.update_frequency, 200.0)
        self.assertEqual(integrator.data_bus, data_bus)
        
        print("✓ 集成器创建函数测试通过")
    
    def test_databus_mpc_interfaces(self):
        """测试数据总线MPC专用接口"""
        print("测试数据总线MPC专用接口...")
        
        # 测试期望力接口
        test_force = Vector3D(10.0, 5.0, 80.0)
        success = self.data_bus.set_desired_contact_force('left_foot', test_force)
        self.assertTrue(success, "设置期望接触力应该成功")
        
        retrieved_force = self.data_bus.get_desired_contact_force('left_foot')
        self.assertIsNotNone(retrieved_force, "获取期望接触力应该成功")
        self.assertAlmostEqual(retrieved_force.x, 10.0, places=3)
        
        # 测试质心加速度接口
        test_acc = Vector3D(0.1, 0.05, 0.0)
        self.data_bus.set_center_of_mass_acceleration(test_acc)
        
        retrieved_acc = self.data_bus.get_center_of_mass_acceleration()
        self.assertAlmostEqual(retrieved_acc.x, 0.1, places=3)
        
        # 测试MPC状态接口
        self.data_bus.update_mpc_status(0.015, True, 125.6, 'osqp')
        mpc_status = self.data_bus.get_mpc_status()
        self.assertTrue(mpc_status['solve_success'])
        self.assertAlmostEqual(mpc_status['last_solve_time'], 0.015, places=3)
        
        # 测试轨迹缓存接口
        test_trajectory = [Vector3D(0.1, 0.0, 0.8), Vector3D(0.15, 0.02, 0.8)]
        self.data_bus.set_mpc_com_trajectory(test_trajectory)
        
        retrieved_trajectory = self.data_bus.get_mpc_com_trajectory()
        self.assertEqual(len(retrieved_trajectory), 2)
        self.assertAlmostEqual(retrieved_trajectory[0].x, 0.1, places=3)
        
        print("✓ 数据总线MPC专用接口测试通过")


def run_integration_tests():
    """运行所有集成测试"""
    print("开始MPC数据总线集成测试...")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMPCDataBusIntegration)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 打印测试结果
    print("\n" + "=" * 60)
    print(f"测试完成: 运行 {result.testsRun} 个测试")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1) 