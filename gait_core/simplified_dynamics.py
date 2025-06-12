"""
简化动力学模型模块 - Linear Inverted Pendulum Model (LIPM)
用于建立机器人质心动态的单刚体模型

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import pinocchio as pin
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os

from .data_bus import DataBus, Vector3D, get_data_bus


class ControlMode(Enum):
    """控制模式枚举"""
    FORCE_CONTROL = 0      # 足底水平力控制
    FOOTSTEP_CONTROL = 1   # 落脚位置控制


@dataclass
class LIPMState:
    """LIPM模型状态"""
    # 质心位置 [m]
    com_position: Vector3D = field(default_factory=Vector3D)
    # 质心速度 [m/s]
    com_velocity: Vector3D = field(default_factory=Vector3D)
    # 质心加速度 [m/s²]
    com_acceleration: Vector3D = field(default_factory=Vector3D)
    # 偏航角 [rad]
    yaw: float = 0.0
    # 偏航角速度 [rad/s]
    yaw_rate: float = 0.0
    # 时间戳
    timestamp: float = 0.0


@dataclass
class LIPMInput:
    """LIPM模型输入"""
    # 支撑足位置 [m]
    support_foot_position: Vector3D = field(default_factory=Vector3D)
    # 足底水平力 [N]（力控制模式）
    foot_force: Vector3D = field(default_factory=Vector3D)
    # 下一步落脚位置 [m]（步态控制模式）
    next_footstep: Vector3D = field(default_factory=Vector3D)
    # 控制模式
    control_mode: ControlMode = ControlMode.FORCE_CONTROL


@dataclass
class LIPMParameters:
    """LIPM模型参数"""
    # 机器人总质量 [kg]
    total_mass: float = 50.0
    # 质心高度 [m]
    com_height: float = 0.8
    # 重力加速度 [m/s²]
    gravity: float = 9.81
    # 自然频率 [rad/s]
    natural_frequency: float = 0.0  # 将自动计算
    # 时间常数 [s]
    time_constant: float = 0.0      # 将自动计算
    
    def __post_init__(self):
        """计算派生参数"""
        self.natural_frequency = np.sqrt(self.gravity / self.com_height)
        self.time_constant = 1.0 / self.natural_frequency


class SimplifiedDynamicsModel:
    """
    简化动力学模型类 - 基于Linear Inverted Pendulum Model (LIPM)
    
    功能：
    1. 从URDF模型中提取机器人参数
    2. 建立单刚体LIPM模型
    3. 实现质心动态预测
    4. 支持力控制和步态控制两种模式
    5. 与数据总线集成
    """
    
    def __init__(self, urdf_path: Optional[str] = None, data_bus: Optional[DataBus] = None):
        """
        初始化简化动力学模型
        
        Args:
            urdf_path: URDF文件路径
            data_bus: 数据总线实例
        """
        self.data_bus = data_bus if data_bus is not None else get_data_bus()
        
        # 模型参数
        self.parameters = LIPMParameters()
        
        # 当前状态
        self.current_state = LIPMState()
        
        # 输入命令
        self.input_command = LIPMInput()
        
        # Pinocchio模型（用于参数提取）
        self.robot_model = None
        self.robot_data = None
        
        # 如果提供了URDF路径，加载模型
        if urdf_path and os.path.exists(urdf_path):
            self.load_robot_model(urdf_path)
        
        # 初始化状态
        self._update_state_from_data_bus()
    
    def load_robot_model(self, urdf_path: str) -> bool:
        """
        从URDF文件加载机器人模型
        
        Args:
            urdf_path: URDF文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            # 使用Pinocchio加载URDF模型
            self.robot_model = pin.buildModelFromUrdf(urdf_path)
            self.robot_data = self.robot_model.createData()
            
            # 提取机器人参数
            self._extract_robot_parameters()
            
            print(f"成功加载机器人模型: {urdf_path}")
            print(f"总质量: {self.parameters.total_mass:.2f} kg")
            print(f"质心高度: {self.parameters.com_height:.3f} m")
            print(f"自然频率: {self.parameters.natural_frequency:.3f} rad/s")
            
            return True
            
        except Exception as e:
            print(f"加载URDF模型失败: {e}")
            return False
    
    def _extract_robot_parameters(self):
        """从机器人模型中提取参数"""
        if self.robot_model is None:
            return
        
        try:
            # 计算总质量
            self.parameters.total_mass = pin.computeTotalMass(self.robot_model)
            
            # 计算初始配置下的质心位置来估计质心高度
            q = pin.neutral(self.robot_model)
            pin.centerOfMass(self.robot_model, self.robot_data, q)
            com_position = self.robot_data.com[0]  # 全局质心位置
            
            # 使用质心的Z坐标作为质心高度的估计值
            self.parameters.com_height = max(0.5, abs(com_position[2]))  # 确保合理的高度值
            
            # 重新计算派生参数
            self.parameters.__post_init__()
            
            # 将参数存储到数据总线
            self._store_parameters_to_data_bus()
            
        except Exception as e:
            print(f"提取机器人参数失败: {e}")
            # 使用默认参数
    
    def _store_parameters_to_data_bus(self):
        """将模型参数存储到数据总线"""
        if hasattr(self.data_bus, 'lipm_parameters'):
            self.data_bus.limp_parameters = {
                'total_mass': self.parameters.total_mass,
                'com_height': self.parameters.com_height,
                'gravity': self.parameters.gravity,
                'natural_frequency': self.parameters.natural_frequency,
                'time_constant': self.parameters.time_constant
            }
    
    def _update_state_from_data_bus(self):
        """从数据总线更新当前状态"""
        # 更新质心位置和速度
        com_pos = self.data_bus.get_center_of_mass_position()
        com_vel = self.data_bus.get_center_of_mass_velocity()
        
        if com_pos:
            self.current_state.com_position = com_pos
        if com_vel:
            self.current_state.com_velocity = com_vel
        
        # 更新偏航角信息
        imu_orientation = self.data_bus.get_imu_orientation()
        if imu_orientation:
            # 从四元数提取偏航角
            self.current_state.yaw = self._quaternion_to_yaw(imu_orientation)
        
        angular_vel = self.data_bus.get_imu_angular_velocity()
        if angular_vel:
            self.current_state.yaw_rate = angular_vel.z
        
        # 更新时间戳
        self.current_state.timestamp = self.data_bus.timestamp
    
    def _quaternion_to_yaw(self, quat) -> float:
        """从四元数提取偏航角"""
        # quat = [w, x, y, z]
        w, x, y, z = quat.w, quat.x, quat.y, quat.z
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw
    
    def set_support_foot_position(self, foot_position: Vector3D):
        """设置支撑足位置"""
        self.input_command.support_foot_position = foot_position
    
    def set_foot_force_command(self, force: Vector3D):
        """设置足底水平力命令"""
        self.input_command.foot_force = force
        self.input_command.control_mode = ControlMode.FORCE_CONTROL
    
    def set_next_footstep(self, footstep_position: Vector3D):
        """设置下一步落脚位置"""
        self.input_command.next_footstep = footstep_position
        self.input_command.control_mode = ControlMode.FOOTSTEP_CONTROL
    
    def compute_com_dynamics(self, dt: float) -> LIPMState:
        """
        计算质心动力学
        基于LIMP模型: ẍ = (g/z_c) * (x - p_foot)
        
        Args:
            dt: 时间步长 [s]
            
        Returns:
            LIPMState: 预测的下一状态
        """
        # 获取当前状态
        x = self.current_state.com_position.x
        y = self.current_state.com_position.y
        z = self.current_state.com_position.z
        
        vx = self.current_state.com_velocity.x
        vy = self.current_state.com_velocity.y
        vz = self.current_state.com_velocity.z
        
        # 支撑足位置
        p_foot_x = self.input_command.support_foot_position.x
        p_foot_y = self.input_command.support_foot_position.y
        
        # 使用质心高度或当前Z位置
        z_c = max(self.parameters.com_height, z) if z > 0.1 else self.parameters.com_height
        
        # LIPM动力学方程
        omega_n = np.sqrt(self.parameters.gravity / z_c)  # 自然频率
        
        if self.input_command.control_mode == ControlMode.FORCE_CONTROL:
            # 力控制模式：考虑外部足底力
            fx = self.input_command.foot_force.x
            fy = self.input_command.foot_force.y
            
            # 动力学方程：m*ẍ = mg*(x-p_foot)/z_c + f_x
            ax = omega_n**2 * (x - p_foot_x) + fx / self.parameters.total_mass
            ay = omega_n**2 * (y - p_foot_y) + fy / self.parameters.total_mass
            
        else:
            # 步态控制模式：标准LIPM
            ax = omega_n**2 * (x - p_foot_x)
            ay = omega_n**2 * (y - p_foot_y)
        
        # Z方向保持常数或缓慢变化
        az = 0.0  # 假设质心高度保持不变
        
        # 数值积分（欧拉法）
        next_state = LIPMState()
        
        # 位置积分
        next_state.com_position.x = x + vx * dt
        next_state.com_position.y = y + vy * dt
        next_state.com_position.z = z + vz * dt
        
        # 速度积分
        next_state.com_velocity.x = vx + ax * dt
        next_state.com_velocity.y = vy + ay * dt
        next_state.com_velocity.z = vz + az * dt
        
        # 加速度
        next_state.com_acceleration.x = ax
        next_state.com_acceleration.y = ay
        next_state.com_acceleration.z = az
        
        # 偏航角（假设缓慢变化）
        next_state.yaw = self.current_state.yaw + self.current_state.yaw_rate * dt
        next_state.yaw_rate = self.current_state.yaw_rate  # 假设角速度保持不变
        
        next_state.timestamp = self.current_state.timestamp + dt
        
        return next_state
    
    def predict_com_trajectory(self, time_horizon: float, dt: float) -> List[LIPMState]:
        """
        预测质心轨迹
        
        Args:
            time_horizon: 预测时间范围 [s]
            dt: 时间步长 [s]
            
        Returns:
            List[LIPMState]: 预测轨迹状态序列
        """
        trajectory = []
        steps = int(time_horizon / dt)
        
        # 保存当前状态
        original_state = LIPMState(
            com_position=Vector3D(
                x=self.current_state.com_position.x,
                y=self.current_state.com_position.y,
                z=self.current_state.com_position.z
            ),
            com_velocity=Vector3D(
                x=self.current_state.com_velocity.x,
                y=self.current_state.com_velocity.y,
                z=self.current_state.com_velocity.z
            ),
            com_acceleration=Vector3D(
                x=self.current_state.com_acceleration.x,
                y=self.current_state.com_acceleration.y,
                z=self.current_state.com_acceleration.z
            ),
            yaw=self.current_state.yaw,
            yaw_rate=self.current_state.yaw_rate,
            timestamp=self.current_state.timestamp
        )
        
        # 进行预测
        for i in range(steps):
            next_state = self.compute_com_dynamics(dt)
            trajectory.append(next_state)
            
            # 更新当前状态用于下一步预测
            self.current_state = next_state
        
        # 恢复原始状态
        self.current_state = original_state
        
        return trajectory
    
    def compute_required_cop(self, desired_acceleration: Vector3D) -> Vector3D:
        """
        计算实现期望质心加速度所需的压力中心(COP)位置
        
        Args:
            desired_acceleration: 期望的质心加速度
            
        Returns:
            Vector3D: 所需的COP位置
        """
        # 从LIPM动力学反推：p_foot = x - (z_c/g) * ẍ
        z_c = self.parameters.com_height
        g = self.parameters.gravity
        
        x = self.current_state.com_position.x
        y = self.current_state.com_position.y
        
        cop_x = x - (z_c / g) * desired_acceleration.x
        cop_y = y - (z_c / g) * desired_acceleration.y
        
        return Vector3D(x=cop_x, y=cop_y, z=0.0)
    
    def update(self, dt: float):
        """
        更新模型状态
        
        Args:
            dt: 时间步长 [s]
        """
        # 从数据总线更新状态
        self._update_state_from_data_bus()
        
        # 计算下一状态
        next_state = self.compute_com_dynamics(dt)
        
        # 更新当前状态
        self.current_state = next_state
        
        # 将预测结果写回数据总线
        self._write_state_to_data_bus()
    
    def _write_state_to_data_bus(self):
        """将状态写回数据总线"""
        # 更新质心位置和速度预测
        self.data_bus.set_center_of_mass_position(self.current_state.com_position)
        self.data_bus.set_center_of_mass_velocity(self.current_state.com_velocity)
        
        # 如果数据总线支持，也可以存储加速度预测
        if hasattr(self.data_bus, 'com_acceleration_prediction'):
            self.data_bus.com_acceleration_prediction = self.current_state.com_acceleration
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'parameters': {
                'total_mass': self.parameters.total_mass,
                'com_height': self.parameters.com_height,
                'gravity': self.parameters.gravity,
                'natural_frequency': self.parameters.natural_frequency,
                'time_constant': self.parameters.time_constant
            },
            'current_state': {
                'com_position': {
                    'x': self.current_state.com_position.x,
                    'y': self.current_state.com_position.y,
                    'z': self.current_state.com_position.z
                },
                'com_velocity': {
                    'x': self.current_state.com_velocity.x,
                    'y': self.current_state.com_velocity.y,
                    'z': self.current_state.com_velocity.z
                },
                'yaw': self.current_state.yaw,
                'yaw_rate': self.current_state.yaw_rate
            },
            'control_mode': self.input_command.control_mode.name,
            'support_foot_position': {
                'x': self.input_command.support_foot_position.x,
                'y': self.input_command.support_foot_position.y,
                'z': self.input_command.support_foot_position.z
            }
        }
    
    def print_status(self):
        """打印模型状态"""
        print("\n=== 简化动力学模型状态 ===")
        print(f"总质量: {self.parameters.total_mass:.2f} kg")
        print(f"质心高度: {self.parameters.com_height:.3f} m")
        print(f"自然频率: {self.parameters.natural_frequency:.3f} rad/s")
        print(f"时间常数: {self.parameters.time_constant:.3f} s")
        print(f"质心位置: ({self.current_state.com_position.x:.3f}, {self.current_state.com_position.y:.3f}, {self.current_state.com_position.z:.3f}) m")
        print(f"质心速度: ({self.current_state.com_velocity.x:.3f}, {self.current_state.com_velocity.y:.3f}, {self.current_state.com_velocity.z:.3f}) m/s")
        print(f"偏航角: {np.degrees(self.current_state.yaw):.1f}°")
        print(f"控制模式: {self.input_command.control_mode.name}")
        print("========================\n")


def create_simplified_dynamics_model(urdf_path: Optional[str] = None) -> SimplifiedDynamicsModel:
    """
    创建简化动力学模型实例
    
    Args:
        urdf_path: URDF文件路径
        
    Returns:
        SimplifiedDynamicsModel: 模型实例
    """
    return SimplifiedDynamicsModel(urdf_path=urdf_path) 