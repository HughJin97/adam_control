"""
MPC数据总线集成模块
将MPC求解结果集成到机器人数据总线系统

功能：
1. 支撑力输出：将MPC计算的接触力写入DataBus.desired_force[foot]
2. 质心轨迹输出：提供质心期望位置/速度供WBC跟踪
3. 落脚点输出：智能落脚点规划结果写入DataBus.target_foot_pos
4. 实时数据更新：随控制周期更新数据总线

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .data_bus import DataBus, Vector3D, ContactState, GaitPhase, LegState
from .mpc_solver import MPCSolver, MPCResult, GaitPlan
from .simplified_dynamics_model import LIPMState


class MPCOutputMode(Enum):
    """MPC输出模式"""
    FORCE_ONLY = "force_only"           # 仅输出支撑力
    TRAJECTORY_ONLY = "trajectory_only" # 仅输出质心轨迹
    FOOTSTEP_ONLY = "footstep_only"     # 仅输出落脚点
    COMBINED = "combined"               # 组合输出（推荐）


@dataclass
class MPCDataBusConfig:
    """MPC数据总线配置"""
    # 输出模式
    output_mode: MPCOutputMode = MPCOutputMode.COMBINED
    
    # 力输出配置
    force_output_enabled: bool = True
    force_smoothing_factor: float = 0.8    # 力平滑系数
    force_threshold: float = 1.0           # 最小输出力阈值
    
    # 轨迹输出配置
    trajectory_output_enabled: bool = True
    trajectory_lookahead_steps: int = 3    # 轨迹前瞻步数
    com_velocity_scaling: float = 1.0      # 质心速度缩放
    
    # 落脚点输出配置
    footstep_output_enabled: bool = True
    footstep_update_threshold: float = 0.05 # 落脚点更新阈值[m]
    footstep_safety_margin: float = 0.02   # 安全裕量
    
    # 更新频率控制
    update_frequency: float = 125.0        # 数据更新频率[Hz]
    force_update_rate: float = 250.0       # 力更新频率[Hz]
    trajectory_update_rate: float = 100.0  # 轨迹更新频率[Hz]
    
    # 数据验证
    enable_data_validation: bool = True
    max_force_magnitude: float = 300.0     # 最大力幅值[N]
    max_com_velocity: float = 2.0          # 最大质心速度[m/s]


class MPCDataBusIntegrator:
    """
    MPC数据总线集成器
    
    负责将MPC求解结果正确地写入数据总线，供其他模块使用
    """
    
    def __init__(self, 
                 data_bus: DataBus,
                 config: Optional[MPCDataBusConfig] = None):
        """
        初始化MPC数据总线集成器
        
        Args:
            data_bus: 机器人数据总线实例
            config: MPC数据总线配置
        """
        self.data_bus = data_bus
        self.config = config if config is not None else MPCDataBusConfig()
        
        # 内部状态
        self.last_mpc_result: Optional[MPCResult] = None
        self.last_update_time = 0.0
        self.force_history = {'left_foot': [], 'right_foot': []}
        self.trajectory_buffer = []
        
        # 性能统计
        self.update_count = 0
        self.update_times = []
        self.data_validation_errors = 0
        
        # 足部映射
        self.foot_mapping = {
            'left_foot': 'left_leg',
            'right_foot': 'right_leg',
            'left': 'left_foot',
            'right': 'right_foot'
        }
        
        print(f"MPC数据总线集成器初始化完成")
        print(f"输出模式: {self.config.output_mode.value}")
        print(f"更新频率: {self.config.update_frequency:.1f} Hz")
    
    def update_from_mpc_result(self, mpc_result: MPCResult, current_time: float = None) -> bool:
        """
        从MPC结果更新数据总线
        
        Args:
            mpc_result: MPC求解结果
            current_time: 当前时间戳
            
        Returns:
            bool: 更新是否成功
        """
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
                    print(f"MPC结果验证失败，跳过更新")
                    return False
            
            # 根据配置模式更新数据总线
            success = True
            
            if self.config.output_mode in [MPCOutputMode.FORCE_ONLY, MPCOutputMode.COMBINED]:
                if self.config.force_output_enabled:
                    success &= self._update_contact_forces(mpc_result, current_time)
            
            if self.config.output_mode in [MPCOutputMode.TRAJECTORY_ONLY, MPCOutputMode.COMBINED]:
                if self.config.trajectory_output_enabled:
                    success &= self._update_com_trajectory(mpc_result, current_time)
            
            if self.config.output_mode in [MPCOutputMode.FOOTSTEP_ONLY, MPCOutputMode.COMBINED]:
                if self.config.footstep_output_enabled:
                    success &= self._update_footstep_targets(mpc_result, current_time)
            
            # 更新MPC状态信息
            self._update_mpc_status(mpc_result, current_time)
            
            # 更新轨迹缓存
            self._update_trajectory_cache(mpc_result)
            
            # 更新内部状态
            self.last_mpc_result = mpc_result
            self.last_update_time = current_time
            self.update_count += 1
            
            # 性能统计
            update_time = time.time() - start_time
            self.update_times.append(update_time)
            
            if success:
                print(f"MPC数据总线更新成功: 耗时 {update_time:.3f}ms")
            else:
                print(f"MPC数据总线更新部分失败")
                
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
    
    def _validate_mpc_result(self, mpc_result: MPCResult) -> bool:
        """验证MPC结果的有效性"""
        if not mpc_result.success:
            return False
            
        # 验证接触力
        if len(mpc_result.current_contact_forces) > 0:
            for foot, force in mpc_result.current_contact_forces.items():
                force_magnitude = np.sqrt(force.x**2 + force.y**2 + force.z**2)
                if force_magnitude > self.config.max_force_magnitude:
                    print(f"接触力过大: {foot} = {force_magnitude:.1f}N")
                    return False
        
        # 验证质心速度
        if len(mpc_result.com_velocity_trajectory) > 0:
            current_vel = mpc_result.com_velocity_trajectory[0]
            vel_magnitude = np.sqrt(current_vel.x**2 + current_vel.y**2)
            if vel_magnitude > self.config.max_com_velocity:
                print(f"质心速度过大: {vel_magnitude:.2f}m/s")
                return False
        
        return True
    
    def _update_contact_forces(self, mpc_result: MPCResult, current_time: float) -> bool:
        """更新接触力到数据总线"""
        try:
            # 获取当前时刻的接触力
            current_forces = mpc_result.current_contact_forces
            
            if not current_forces:
                print("警告: MPC结果中无接触力数据")
                return True
            
            # 更新每个足部的期望力
            for foot_name, force_vector in current_forces.items():
                # 标准化足部名称
                standardized_foot = self._standardize_foot_name(foot_name)
                
                # 力平滑处理
                smoothed_force = self._smooth_contact_force(
                    standardized_foot, force_vector, current_time
                )
                
                # 应用力阈值
                if self._get_force_magnitude(smoothed_force) < self.config.force_threshold:
                    smoothed_force = Vector3D(0.0, 0.0, 0.0)
                
                # 写入数据总线
                success = self._write_desired_force_to_databus(standardized_foot, smoothed_force)
                
                if success:
                    print(f"更新 {standardized_foot} 期望力: "
                          f"[{smoothed_force.x:.1f}, {smoothed_force.y:.1f}, {smoothed_force.z:.1f}]N")
                else:
                    print(f"写入 {standardized_foot} 期望力失败")
                    return False
            
            return True
            
        except Exception as e:
            print(f"更新接触力异常: {e}")
            return False
    
    def _update_com_trajectory(self, mpc_result: MPCResult, current_time: float) -> bool:
        """更新质心轨迹到数据总线"""
        try:
            # 获取质心轨迹数据
            if not mpc_result.com_position_trajectory or not mpc_result.com_velocity_trajectory:
                print("警告: MPC结果中无质心轨迹数据")
                return True
            
            # 当前时刻的质心状态
            current_pos = mpc_result.com_position_trajectory[0]
            current_vel = mpc_result.com_velocity_trajectory[0]
            
            # 应用速度缩放
            scaled_vel = Vector3D(
                current_vel.x * self.config.com_velocity_scaling,
                current_vel.y * self.config.com_velocity_scaling,
                current_vel.z * self.config.com_velocity_scaling
            )
            
            # 写入数据总线
            self.data_bus.set_center_of_mass_position(current_pos)
            self.data_bus.set_center_of_mass_velocity(scaled_vel)
            
            # 如果有加速度数据，也更新
            if mpc_result.com_acceleration_trajectory:
                current_acc = mpc_result.com_acceleration_trajectory[0]
                self.data_bus.set_center_of_mass_acceleration(current_acc)
            
            print(f"更新质心状态: 位置[{current_pos.x:.3f}, {current_pos.y:.3f}, {current_pos.z:.3f}], "
                  f"速度[{scaled_vel.x:.3f}, {scaled_vel.y:.3f}]")
            
            return True
            
        except Exception as e:
            print(f"更新质心轨迹异常: {e}")
            return False
    
    def _update_footstep_targets(self, mpc_result: MPCResult, current_time: float) -> bool:
        """更新落脚点目标到数据总线"""
        try:
            # 获取下一步落脚点
            next_footsteps = mpc_result.next_footstep
            
            if not next_footsteps:
                print("警告: MPC结果中无落脚点数据")
                return True
            
            # 更新每个足部的目标位置
            for foot_name, target_pos in next_footsteps.items():
                standardized_foot = self._standardize_foot_name(foot_name)
                
                # 检查是否需要更新（避免频繁小幅调整）
                if self._should_update_footstep(standardized_foot, target_pos):
                    # 应用安全裕量
                    safe_target = self._apply_footstep_safety_margin(target_pos)
                    
                    # 写入数据总线
                    success = self._write_target_foot_position_to_databus(
                        standardized_foot, safe_target
                    )
                    
                    if success:
                        print(f"更新 {standardized_foot} 目标位置: "
                              f"[{safe_target.x:.3f}, {safe_target.y:.3f}, {safe_target.z:.3f}]")
                    else:
                        print(f"写入 {standardized_foot} 目标位置失败")
                        return False
            
            return True
            
        except Exception as e:
            print(f"更新落脚点目标异常: {e}")
            return False
    
    def _standardize_foot_name(self, foot_name: str) -> str:
        """标准化足部名称"""
        foot_name_lower = foot_name.lower()
        return self.foot_mapping.get(foot_name_lower, foot_name_lower)
    
    def _smooth_contact_force(self, foot_name: str, new_force: Vector3D, current_time: float) -> Vector3D:
        """对接触力进行平滑处理"""
        if foot_name not in self.force_history:
            self.force_history[foot_name] = []
        
        history = self.force_history[foot_name]
        
        # 添加新的力数据
        history.append({
            'force': new_force,
            'time': current_time
        })
        
        # 保持历史长度
        max_history = 5
        if len(history) > max_history:
            history.pop(0)
        
        # 如果历史数据不足，直接返回
        if len(history) < 2:
            return new_force
        
        # 指数平滑
        alpha = self.config.force_smoothing_factor
        prev_force = history[-2]['force']
        
        smoothed_force = Vector3D(
            alpha * prev_force.x + (1 - alpha) * new_force.x,
            alpha * prev_force.y + (1 - alpha) * new_force.y,
            alpha * prev_force.z + (1 - alpha) * new_force.z
        )
        
        return smoothed_force
    
    def _get_force_magnitude(self, force: Vector3D) -> float:
        """计算力的幅值"""
        return np.sqrt(force.x**2 + force.y**2 + force.z**2)
    
    def _should_update_footstep(self, foot_name: str, new_target: Vector3D) -> bool:
        """检查是否需要更新落脚点"""
        # 获取当前目标位置
        current_target = self.data_bus.get_target_foot_position(foot_name)
        
        if not current_target:
            return True
        
        # 计算位置变化
        dx = new_target.x - current_target.get('x', 0.0)
        dy = new_target.y - current_target.get('y', 0.0)
        dz = new_target.z - current_target.get('z', 0.0)
        
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return distance > self.config.footstep_update_threshold
    
    def _apply_footstep_safety_margin(self, target_pos: Vector3D) -> Vector3D:
        """应用落脚点安全裕量"""
        # 这里可以实现更复杂的安全检查
        # 目前只是简单返回原位置
        return target_pos
    
    def _write_desired_force_to_databus(self, foot_name: str, force: Vector3D) -> bool:
        """将期望力写入数据总线"""
        try:
            # 使用数据总线的专用MPC接口
            return self.data_bus.set_desired_contact_force(foot_name, force)
                
        except Exception as e:
            print(f"写入期望力到数据总线失败: {e}")
            return False
    
    def _write_target_foot_position_to_databus(self, foot_name: str, position: Vector3D) -> bool:
        """将目标足部位置写入数据总线"""
        try:
            # 使用数据总线的Vector3D接口
            return self.data_bus.set_target_foot_position_vector3d(foot_name, position)
                
        except Exception as e:
            print(f"写入目标足部位置到数据总线失败: {e}")
            return False
    
    def get_integration_statistics(self) -> Dict:
        """获取集成统计信息"""
        avg_update_time = np.mean(self.update_times) if self.update_times else 0.0
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
    
    def print_status(self):
        """打印集成器状态"""
        stats = self.get_integration_statistics()
        
        print("\n=== MPC数据总线集成器状态 ===")
        print(f"更新次数: {stats['update_count']}")
        print(f"平均更新时间: {stats['average_update_time']:.3f}ms")
        print(f"成功率: {stats['success_rate']:.1%}")
        print(f"验证错误: {stats['validation_errors']}")
        print(f"输出模式: {stats['config']['output_mode']}")
        print(f"更新频率: {stats['config']['update_frequency']:.1f}Hz")
        print(f"力输出: {'启用' if stats['config']['force_enabled'] else '禁用'}")
        print(f"轨迹输出: {'启用' if stats['config']['trajectory_enabled'] else '禁用'}")
        print(f"落脚点输出: {'启用' if stats['config']['footstep_enabled'] else '禁用'}")

    def _update_mpc_status(self, mpc_result: MPCResult, current_time: float):
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
    
    def _update_trajectory_cache(self, mpc_result: MPCResult):
        """更新轨迹缓存到数据总线"""
        try:
            # 更新质心轨迹缓存
            if mpc_result.com_position_trajectory:
                self.data_bus.set_mpc_com_trajectory(mpc_result.com_position_trajectory)
            
            # 更新ZMP轨迹缓存
            if mpc_result.zmp_trajectory:
                self.data_bus.set_mpc_zmp_trajectory(mpc_result.zmp_trajectory)
                
        except Exception as e:
            print(f"更新轨迹缓存异常: {e}")


def create_mpc_databus_integrator(data_bus: DataBus, 
                                  output_mode: MPCOutputMode = MPCOutputMode.COMBINED,
                                  update_frequency: float = 125.0) -> MPCDataBusIntegrator:
    """
    创建MPC数据总线集成器
    
    Args:
        data_bus: 数据总线实例
        output_mode: 输出模式
        update_frequency: 更新频率
        
    Returns:
        MPCDataBusIntegrator: 集成器实例
    """
    config = MPCDataBusConfig(
        output_mode=output_mode,
        update_frequency=update_frequency
    )
    
    return MPCDataBusIntegrator(data_bus, config) 