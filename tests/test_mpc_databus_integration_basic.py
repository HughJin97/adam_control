"""
MPC数据总线集成基础测试
测试MPC数据总线集成的核心功能，不依赖外部库

测试内容：
1. 数据总线MPC接口测试
2. 基本数据结构测试
3. 集成配置测试

作者: Adam Control Team
版本: 1.0
"""

import unittest
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

# 导入核心模块
from gait_core.data_bus import DataBus, Vector3D


# 模拟MPC相关数据结构（简化版本）
@dataclass
class MockMPCResult:
    """模拟MPC结果"""
    success: bool = False
    solve_time: float = 0.0
    cost: float = float('inf')
    solver_info: Dict = field(default_factory=dict)
    
    # 支撑力轨迹
    current_contact_forces: Dict[str, Vector3D] = field(default_factory=dict)
    
    # 质心轨迹
    com_position_trajectory: List[Vector3D] = field(default_factory=list)
    com_velocity_trajectory: List[Vector3D] = field(default_factory=list)
    com_acceleration_trajectory: List[Vector3D] = field(default_factory=list)
    
    # ZMP轨迹
    zmp_trajectory: List[Vector3D] = field(default_factory=list)
    
    # 下一步最优落脚位置
    next_footstep: Dict[str, Vector3D] = field(default_factory=dict)


class MockMPCOutputMode(Enum):
    """模拟MPC输出模式"""
    FORCE_ONLY = "force_only"
    TRAJECTORY_ONLY = "trajectory_only"
    FOOTSTEP_ONLY = "footstep_only"
    COMBINED = "combined"


@dataclass
class MockMPCDataBusConfig:
    """模拟MPC数据总线配置"""
    output_mode: MockMPCOutputMode = MockMPCOutputMode.COMBINED
    force_output_enabled: bool = True
    trajectory_output_enabled: bool = True
    footstep_output_enabled: bool = True
    update_frequency: float = 125.0
    enable_data_validation: bool = True
    max_force_magnitude: float = 300.0
    max_com_velocity: float = 2.0
    force_smoothing_factor: float = 0.8
    force_threshold: float = 1.0


class MockMPCDataBusIntegrator:
    """模拟MPC数据总线集成器"""
    
    def __init__(self, data_bus: DataBus, config: Optional[MockMPCDataBusConfig] = None):
        self.data_bus = data_bus
        self.config = config if config is not None else MockMPCDataBusConfig()
        self.last_update_time = 0.0
        self.update_count = 0
        self.update_times = []
        self.data_validation_errors = 0
        
    def update_from_mpc_result(self, mpc_result: MockMPCResult, current_time: float = None) -> bool:
        """从MPC结果更新数据总线"""
        if current_time is None:
            current_time = time.time()
            
        start_time = time.time()
        
        try:
            # 检查更新频率
            if not self._should_update(current_time):
                return True
                
            # 数据验证
            if self.config.enable_data_validation:
                if not self._validate_mpc_result(mpc_result):
                    self.data_validation_errors += 1
                    return False
            
            # 根据配置模式更新数据总线
            success = True
            
            if self.config.output_mode in [MockMPCOutputMode.FORCE_ONLY, MockMPCOutputMode.COMBINED]:
                if self.config.force_output_enabled:
                    success &= self._update_contact_forces(mpc_result)
            
            if self.config.output_mode in [MockMPCOutputMode.TRAJECTORY_ONLY, MockMPCOutputMode.COMBINED]:
                if self.config.trajectory_output_enabled:
                    success &= self._update_com_trajectory(mpc_result)
            
            if self.config.output_mode in [MockMPCOutputMode.FOOTSTEP_ONLY, MockMPCOutputMode.COMBINED]:
                if self.config.footstep_output_enabled:
                    success &= self._update_footstep_targets(mpc_result)
            
            # 更新MPC状态
            self._update_mpc_status(mpc_result)
            
            # 更新内部状态
            self.last_update_time = current_time
            self.update_count += 1
            
            # 性能统计
            update_time = time.time() - start_time
            self.update_times.append(update_time)
            
            return success
            
        except Exception as e:
            print(f"MPC数据总线更新异常: {e}")
            return False
    
    def _should_update(self, current_time: float) -> bool:
        """检查是否应该更新"""
        if self.last_update_time == 0.0:
            return True
            
        dt = current_time - self.last_update_time
        min_interval = 1.0 / self.config.update_frequency
        
        return dt >= min_interval
    
    def _validate_mpc_result(self, mpc_result: MockMPCResult) -> bool:
        """验证MPC结果的有效性"""
        if not mpc_result.success:
            return False
            
        # 验证接触力
        if len(mpc_result.current_contact_forces) > 0:
            for foot, force in mpc_result.current_contact_forces.items():
                force_magnitude = (force.x**2 + force.y**2 + force.z**2)**0.5
                if force_magnitude > self.config.max_force_magnitude:
                    return False
        
        # 验证质心速度
        if len(mpc_result.com_velocity_trajectory) > 0:
            current_vel = mpc_result.com_velocity_trajectory[0]
            vel_magnitude = (current_vel.x**2 + current_vel.y**2)**0.5
            if vel_magnitude > self.config.max_com_velocity:
                return False
        
        return True
    
    def _update_contact_forces(self, mpc_result: MockMPCResult) -> bool:
        """更新接触力到数据总线"""
        try:
            current_forces = mpc_result.current_contact_forces
            
            for foot_name, force_vector in current_forces.items():
                # 应用力阈值
                force_magnitude = (force_vector.x**2 + force_vector.y**2 + force_vector.z**2)**0.5
                if force_magnitude < self.config.force_threshold:
                    force_vector = Vector3D(0.0, 0.0, 0.0)
                
                # 写入数据总线
                success = self.data_bus.set_desired_contact_force(foot_name, force_vector)
                if not success:
                    return False
            
            return True
            
        except Exception as e:
            print(f"更新接触力异常: {e}")
            return False
    
    def _update_com_trajectory(self, mpc_result: MockMPCResult) -> bool:
        """更新质心轨迹到数据总线"""
        try:
            if not mpc_result.com_position_trajectory or not mpc_result.com_velocity_trajectory:
                return True
            
            # 当前时刻的质心状态
            current_pos = mpc_result.com_position_trajectory[0]
            current_vel = mpc_result.com_velocity_trajectory[0]
            
            # 写入数据总线
            self.data_bus.set_center_of_mass_position(current_pos)
            self.data_bus.set_center_of_mass_velocity(current_vel)
            
            # 如果有加速度数据，也更新
            if mpc_result.com_acceleration_trajectory:
                current_acc = mpc_result.com_acceleration_trajectory[0]
                self.data_bus.set_center_of_mass_acceleration(current_acc)
            
            return True
            
        except Exception as e:
            print(f"更新质心轨迹异常: {e}")
            return False
    
    def _update_footstep_targets(self, mpc_result: MockMPCResult) -> bool:
        """更新落脚点目标到数据总线"""
        try:
            next_footsteps = mpc_result.next_footstep
            
            for foot_name, target_pos in next_footsteps.items():
                # 写入数据总线
                success = self.data_bus.set_target_foot_position_vector3d(foot_name, target_pos)
                if not success:
                    return False
            
            return True
            
        except Exception as e:
            print(f"更新落脚点目标异常: {e}")
            return False
    
    def _update_mpc_status(self, mpc_result: MockMPCResult):
        """更新MPC状态信息到数据总线"""
        try:
            self.data_bus.update_mpc_status(
                solve_time=mpc_result.solve_time,
                success=mpc_result.success,
                cost=mpc_result.cost,
                solver_type=mpc_result.solver_info.get('solver_type', 'unknown')
            )
        except Exception as e:
            print(f"更新MPC状态异常: {e}")
    
    def get_integration_statistics(self) -> Dict:
        """获取集成统计信息"""
        avg_update_time = sum(self.update_times) / len(self.update_times) if self.update_times else 0.0
        success_rate = (self.update_count - self.data_validation_errors) / max(self.update_count, 1)
        
        return {
            'update_count': self.update_count,
            'average_update_time': avg_update_time,
            'success_rate': success_rate,
            'validation_errors': self.data_validation_errors,
            'last_update_time': self.last_update_time,
            'config': {
                'output_mode': self.config.output_mode.value,
                'update_frequency': self.config.update_frequency,
                'force_enabled': self.config.force_output_enabled,
                'trajectory_enabled': self.config.trajectory_output_enabled,
                'footstep_enabled': self.config.footstep_output_enabled
            }
        }


class TestMPCDataBusIntegrationBasic(unittest.TestCase):
    """MPC数据总线集成基础测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.data_bus = DataBus()
        self.config = MockMPCDataBusConfig()
        self.integrator = MockMPCDataBusIntegrator(self.data_bus, self.config)
        
    def create_test_mpc_result(self, success=True):
        """创建测试用MPC结果"""
        result = MockMPCResult()
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
        self.assertAlmostEqual(retrieved_force.y, 5.0, places=3)
        self.assertAlmostEqual(retrieved_force.z, 80.0, places=3)
        
        # 测试获取所有期望力
        all_forces = self.data_bus.get_all_desired_contact_forces()
        self.assertEqual(len(all_forces), 1, "应该有一个足部的期望力")
        self.assertIn('left_foot', all_forces, "应该包含左脚期望力")
        
        # 测试质心加速度接口
        test_acc = Vector3D(0.1, 0.05, 0.0)
        self.data_bus.set_center_of_mass_acceleration(test_acc)
        
        retrieved_acc = self.data_bus.get_center_of_mass_acceleration()
        self.assertAlmostEqual(retrieved_acc.x, 0.1, places=3)
        self.assertAlmostEqual(retrieved_acc.y, 0.05, places=3)
        
        # 测试MPC状态接口
        self.data_bus.update_mpc_status(0.015, True, 125.6, 'osqp')
        mpc_status = self.data_bus.get_mpc_status()
        self.assertTrue(mpc_status['solve_success'])
        self.assertAlmostEqual(mpc_status['last_solve_time'], 0.015, places=3)
        self.assertAlmostEqual(mpc_status['cost'], 125.6, places=1)
        self.assertEqual(mpc_status['solver_type'], 'osqp')
        
        # 测试轨迹缓存接口
        test_trajectory = [Vector3D(0.1, 0.0, 0.8), Vector3D(0.15, 0.02, 0.8)]
        self.data_bus.set_mpc_com_trajectory(test_trajectory)
        
        retrieved_trajectory = self.data_bus.get_mpc_com_trajectory()
        self.assertEqual(len(retrieved_trajectory), 2)
        self.assertAlmostEqual(retrieved_trajectory[0].x, 0.1, places=3)
        self.assertAlmostEqual(retrieved_trajectory[1].x, 0.15, places=3)
        
        # 测试ZMP轨迹接口
        test_zmp = [Vector3D(0.05, 0.0, 0.0), Vector3D(0.08, 0.01, 0.0)]
        self.data_bus.set_mpc_zmp_trajectory(test_zmp)
        
        retrieved_zmp = self.data_bus.get_mpc_zmp_trajectory()
        self.assertEqual(len(retrieved_zmp), 2)
        self.assertAlmostEqual(retrieved_zmp[0].x, 0.05, places=3)
        
        # 测试目标足部位置接口
        test_pos = Vector3D(0.3, 0.1, 0.0)
        success = self.data_bus.set_target_foot_position_vector3d('left_foot', test_pos)
        self.assertTrue(success, "设置目标足部位置应该成功")
        
        retrieved_pos = self.data_bus.get_target_foot_position('left_foot')
        self.assertIsNotNone(retrieved_pos, "获取目标足部位置应该成功")
        self.assertAlmostEqual(retrieved_pos['x'], 0.3, places=3)
        self.assertAlmostEqual(retrieved_pos['y'], 0.1, places=3)
        
        print("✓ 数据总线MPC专用接口测试通过")
    
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
        
        # 验证质心加速度已更新
        com_acc = self.data_bus.get_center_of_mass_acceleration()
        self.assertAlmostEqual(com_acc.x, 0.1, places=3, msg="质心X加速度应该正确")
        
        # 验证MPC状态已更新
        mpc_status = self.data_bus.get_mpc_status()
        self.assertTrue(mpc_status['solve_success'], "MPC状态应该显示求解成功")
        self.assertAlmostEqual(mpc_status['last_solve_time'], 0.015, places=3)
        
        # 验证目标足部位置已更新
        left_target = self.data_bus.get_target_foot_position('left_foot')
        self.assertIsNotNone(left_target, "左脚目标位置应该存在")
        self.assertAlmostEqual(left_target['x'], 0.3, places=3)
        
        print("✓ 基本集成功能测试通过")
    
    def test_different_output_modes(self):
        """测试不同输出模式"""
        print("测试不同输出模式...")
        
        # 测试仅力输出模式
        config_force = MockMPCDataBusConfig(output_mode=MockMPCOutputMode.FORCE_ONLY)
        integrator_force = MockMPCDataBusIntegrator(self.data_bus, config_force)
        
        mpc_result = self.create_test_mpc_result()
        success = integrator_force.update_from_mpc_result(mpc_result)
        self.assertTrue(success, "仅力输出模式应该成功")
        
        # 验证期望力已写入
        desired_forces = self.data_bus.get_all_desired_contact_forces()
        self.assertEqual(len(desired_forces), 2, "仅力模式应该有期望力")
        
        # 测试仅轨迹输出模式
        data_bus_traj = DataBus()
        config_traj = MockMPCDataBusConfig(output_mode=MockMPCOutputMode.TRAJECTORY_ONLY)
        integrator_traj = MockMPCDataBusIntegrator(data_bus_traj, config_traj)
        
        success = integrator_traj.update_from_mpc_result(mpc_result)
        self.assertTrue(success, "仅轨迹输出模式应该成功")
        
        # 验证质心状态已更新
        com_pos = data_bus_traj.get_center_of_mass_position()
        self.assertAlmostEqual(com_pos.x, 0.1, places=3)
        
        # 测试仅落脚点输出模式
        data_bus_foot = DataBus()
        config_foot = MockMPCDataBusConfig(output_mode=MockMPCOutputMode.FOOTSTEP_ONLY)
        integrator_foot = MockMPCDataBusIntegrator(data_bus_foot, config_foot)
        
        success = integrator_foot.update_from_mpc_result(mpc_result)
        self.assertTrue(success, "仅落脚点输出模式应该成功")
        
        # 验证目标足部位置已更新
        left_target = data_bus_foot.get_target_foot_position('left_foot')
        self.assertIsNotNone(left_target, "仅落脚点模式应该有目标位置")
        
        print("✓ 不同输出模式测试通过")
    
    def test_data_validation(self):
        """测试数据验证功能"""
        print("测试数据验证功能...")
        
        # 启用数据验证，设置较小的限制
        config = MockMPCDataBusConfig(
            enable_data_validation=True,
            max_force_magnitude=100.0,  # 设置较小的力限制
            max_com_velocity=1.0        # 设置较小的速度限制
        )
        integrator = MockMPCDataBusIntegrator(self.data_bus, config)
        
        # 创建超出力限制的MPC结果
        mpc_result = self.create_test_mpc_result()
        mpc_result.current_contact_forces['left_foot'] = Vector3D(200.0, 0.0, 0.0)  # 超出力限制
        
        # 执行集成，应该失败
        success = integrator.update_from_mpc_result(mpc_result)
        self.assertFalse(success, "超出力限制的结果应该验证失败")
        
        # 检查验证错误计数
        stats = integrator.get_integration_statistics()
        self.assertGreater(stats['validation_errors'], 0, "应该有验证错误记录")
        
        # 测试超出速度限制
        mpc_result2 = self.create_test_mpc_result()
        mpc_result2.com_velocity_trajectory[0] = Vector3D(2.0, 0.0, 0.0)  # 超出速度限制
        
        success2 = integrator.update_from_mpc_result(mpc_result2)
        self.assertFalse(success2, "超出速度限制的结果应该验证失败")
        
        print("✓ 数据验证功能测试通过")
    
    def test_update_frequency_control(self):
        """测试更新频率控制"""
        print("测试更新频率控制...")
        
        # 配置较低的更新频率
        config = MockMPCDataBusConfig(update_frequency=10.0)  # 10Hz
        integrator = MockMPCDataBusIntegrator(self.data_bus, config)
        
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
        
        # 执行多次集成，使用不同的时间戳避免频率限制
        base_time = time.time()
        for i in range(5):
            mpc_result = self.create_test_mpc_result()
            # 每次更新间隔足够的时间（超过频率限制）
            update_time = base_time + i * 0.1  # 100ms间隔，远超8ms的最小间隔
            self.integrator.update_from_mpc_result(mpc_result, update_time)
        
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
    
    def test_mpc_data_summary(self):
        """测试MPC数据摘要功能"""
        print("测试MPC数据摘要功能...")
        
        # 清除现有数据，确保测试环境干净
        self.data_bus.clear_mpc_data()
        
        # 设置一些MPC数据
        test_force = Vector3D(10.0, 5.0, 80.0)
        self.data_bus.set_desired_contact_force('left_foot', test_force)
        
        test_trajectory = [Vector3D(0.1, 0.0, 0.8), Vector3D(0.15, 0.02, 0.8)]
        self.data_bus.set_mpc_com_trajectory(test_trajectory)
        
        test_zmp = [Vector3D(0.05, 0.0, 0.0)]
        self.data_bus.set_mpc_zmp_trajectory(test_zmp)
        
        self.data_bus.update_mpc_status(0.015, True, 125.6, 'osqp')
        
        test_pos = Vector3D(0.3, 0.1, 0.0)
        self.data_bus.set_target_foot_position_vector3d('left_foot', test_pos)
        
        # 获取数据摘要
        summary = self.data_bus.get_mpc_data_summary()
        
        # 验证摘要内容
        self.assertEqual(summary['desired_forces_count'], 1, "期望力数量应该正确")
        self.assertEqual(summary['com_trajectory_length'], 2, "质心轨迹长度应该正确")
        self.assertEqual(summary['zmp_trajectory_length'], 1, "ZMP轨迹长度应该正确")
        self.assertTrue(summary['mpc_status']['solve_success'], "MPC状态应该正确")
        # 注意：target_foot_positions可能包含预设数据，所以我们检查是否至少有我们设置的数据
        self.assertGreaterEqual(len(summary['target_foot_positions']), 1, "目标足部位置数量应该至少有1个")
        self.assertIn('left_foot', summary['target_foot_positions'], "应该包含我们设置的左脚位置")
        
        print("✓ MPC数据摘要功能测试通过")
    
    def test_clear_mpc_data(self):
        """测试清除MPC数据功能"""
        print("测试清除MPC数据功能...")
        
        # 设置一些MPC数据
        test_force = Vector3D(10.0, 5.0, 80.0)
        self.data_bus.set_desired_contact_force('left_foot', test_force)
        
        test_trajectory = [Vector3D(0.1, 0.0, 0.8)]
        self.data_bus.set_mpc_com_trajectory(test_trajectory)
        
        self.data_bus.update_mpc_status(0.015, True, 125.6, 'osqp')
        
        # 验证数据存在
        self.assertEqual(len(self.data_bus.get_all_desired_contact_forces()), 1)
        self.assertEqual(len(self.data_bus.get_mpc_com_trajectory()), 1)
        self.assertTrue(self.data_bus.get_mpc_status()['solve_success'])
        
        # 清除MPC数据
        self.data_bus.clear_mpc_data()
        
        # 验证数据已清除
        self.assertEqual(len(self.data_bus.get_all_desired_contact_forces()), 0)
        self.assertEqual(len(self.data_bus.get_mpc_com_trajectory()), 0)
        self.assertFalse(self.data_bus.get_mpc_status()['solve_success'])
        
        print("✓ 清除MPC数据功能测试通过")


def run_basic_integration_tests():
    """运行基础集成测试"""
    print("开始MPC数据总线集成基础测试...")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMPCDataBusIntegrationBasic)
    
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
    success = run_basic_integration_tests()
    exit(0 if success else 1) 