"""
Control Sender - 控制指令发送模块
从数据总线读取控制指令并发送到MuJoCo仿真环境

作者: Adam Control Team
版本: 1.0
"""

import mujoco
import numpy as np
from typing import Dict, Optional, List, Tuple
from enum import Enum
from data_bus import get_data_bus, DataBus


class ControlMode(Enum):
    """控制模式枚举"""
    TORQUE = "torque"          # 力矩控制
    POSITION = "position"      # 位置控制
    VELOCITY = "velocity"      # 速度控制
    HYBRID = "hybrid"          # 混合控制


class ControlSender:
    """
    控制指令发送器
    负责从数据总线读取控制指令并发送到MuJoCo仿真环境
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        初始化控制发送器
        
        参数:
            model: MuJoCo模型对象
            data: MuJoCo数据对象
        """
        self.model = model
        self.data = data
        self.data_bus = get_data_bus()
        
        # 缓存执行器映射
        self._actuator_mapping = {}
        self._joint_to_actuator = {}
        self._init_actuator_mappings()
        
        # 控制模式配置
        self.control_mode = ControlMode.TORQUE  # 默认力矩控制
        self.joint_control_modes = {}  # 每个关节的独立控制模式
        
        # 安全限制
        self.enable_safety_checks = True
        self.torque_limits = {}
        self.position_limits = {}
        self.velocity_limits = {}
        self._init_safety_limits()
        
        # 控制增益（用于位置/速度控制）
        self.position_gains = {}
        self.velocity_gains = {}
        self._init_control_gains()
        
        print("ControlSender initialized successfully")
        print(f"Found {len(self._actuator_mapping)} actuators")
    
    def _init_actuator_mappings(self):
        """初始化执行器映射关系"""
        # 遍历所有执行器
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
            )
            if actuator_name:
                self._actuator_mapping[actuator_name] = i
                
                # 获取执行器对应的关节
                trntype = self.model.actuator_trntype[i]
                if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                    joint_id = self.model.actuator_trnid[i, 0]
                    joint_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                    )
                    if joint_name:
                        self._joint_to_actuator[joint_name] = {
                            'actuator_id': i,
                            'actuator_name': actuator_name
                        }
    
    def _init_safety_limits(self):
        """初始化安全限制"""
        for joint_name in self.data_bus.joint_names:
            if joint_name in self._joint_to_actuator:
                actuator_id = self._joint_to_actuator[joint_name]['actuator_id']
                
                # 从模型中获取限制
                if actuator_id < self.model.nu:
                    # 力矩限制
                    if self.model.actuator_ctrllimited[actuator_id]:
                        ctrl_range = self.model.actuator_ctrlrange[actuator_id]
                        self.torque_limits[joint_name] = {
                            'min': ctrl_range[0],
                            'max': ctrl_range[1]
                        }
                    
                    # 位置限制（从关节限制获取）
                    try:
                        joint_id = self.model.joint(joint_name).id
                        if self.model.jnt_limited[joint_id]:
                            joint_range = self.model.jnt_range[joint_id]
                            self.position_limits[joint_name] = {
                                'min': joint_range[0],
                                'max': joint_range[1]
                            }
                    except:
                        pass
                    
                    # 速度限制（使用默认值或从模型获取）
                    self.velocity_limits[joint_name] = {
                        'max': 10.0  # 默认最大速度 rad/s
                    }
    
    def _init_control_gains(self):
        """初始化控制增益"""
        # 默认PD增益
        default_kp = 100.0
        default_kd = 10.0
        
        for joint_name in self.data_bus.joint_names:
            # 根据关节类型设置不同的增益
            if "hip" in joint_name:
                self.position_gains[joint_name] = 150.0
                self.velocity_gains[joint_name] = 15.0
            elif "knee" in joint_name:
                self.position_gains[joint_name] = 120.0
                self.velocity_gains[joint_name] = 12.0
            elif "ankle" in joint_name:
                self.position_gains[joint_name] = 80.0
                self.velocity_gains[joint_name] = 8.0
            elif "arm" in joint_name:
                self.position_gains[joint_name] = 50.0
                self.velocity_gains[joint_name] = 5.0
            else:
                self.position_gains[joint_name] = default_kp
                self.velocity_gains[joint_name] = default_kd
    
    def set_control_mode(self, mode: ControlMode):
        """
        设置全局控制模式
        
        参数:
            mode: 控制模式
        """
        self.control_mode = mode
        print(f"Control mode set to: {mode.value}")
    
    def set_joint_control_mode(self, joint_name: str, mode: ControlMode):
        """
        设置单个关节的控制模式
        
        参数:
            joint_name: 关节名称
            mode: 控制模式
        """
        if joint_name in self.data_bus.joint_names:
            self.joint_control_modes[joint_name] = mode
            print(f"Joint {joint_name} control mode set to: {mode.value}")
    
    def send_commands(self) -> bool:
        """
        主要的控制指令发送函数
        从数据总线读取控制指令并发送到仿真环境
        
        返回:
            bool: 发送是否成功
        """
        try:
            # 根据控制模式发送指令
            if self.control_mode == ControlMode.TORQUE:
                self._send_torque_commands()
            elif self.control_mode == ControlMode.POSITION:
                self._send_position_commands()
            elif self.control_mode == ControlMode.VELOCITY:
                self._send_velocity_commands()
            elif self.control_mode == ControlMode.HYBRID:
                self._send_hybrid_commands()
            
            return True
            
        except Exception as e:
            print(f"Error sending commands: {e}")
            return False
    
    def _send_torque_commands(self):
        """发送力矩控制指令"""
        # 从数据总线获取控制力矩
        control_torques = self.data_bus.get_all_control_torques()
        
        for joint_name, torque in control_torques.items():
            if joint_name not in self._joint_to_actuator:
                continue
            
            actuator_id = self._joint_to_actuator[joint_name]['actuator_id']
            
            # 应用安全检查
            if self.enable_safety_checks:
                torque = self._apply_torque_limits(joint_name, torque)
            
            # 设置控制输入
            self.data.ctrl[actuator_id] = torque
    
    def _send_position_commands(self):
        """发送位置控制指令"""
        # 从数据总线获取期望位置
        control_positions = self.data_bus.get_all_control_positions()
        current_positions = self.data_bus.get_all_joint_positions()
        current_velocities = self.data_bus.get_all_joint_velocities()
        
        for joint_name, desired_position in control_positions.items():
            if joint_name not in self._joint_to_actuator:
                continue
            
            actuator_id = self._joint_to_actuator[joint_name]['actuator_id']
            
            # 应用位置限制
            if self.enable_safety_checks:
                desired_position = self._apply_position_limits(joint_name, desired_position)
            
            # 获取当前状态
            current_pos = current_positions.get(joint_name, 0.0)
            current_vel = current_velocities.get(joint_name, 0.0)
            
            # PD控制计算力矩
            kp = self.position_gains.get(joint_name, 100.0)
            kd = self.velocity_gains.get(joint_name, 10.0)
            
            position_error = desired_position - current_pos
            velocity_error = 0.0 - current_vel  # 期望速度为0
            
            torque = kp * position_error + kd * velocity_error
            
            # 应用力矩限制
            if self.enable_safety_checks:
                torque = self._apply_torque_limits(joint_name, torque)
            
            # 设置控制输入
            self.data.ctrl[actuator_id] = torque
    
    def _send_velocity_commands(self):
        """发送速度控制指令"""
        # 从数据总线获取期望速度
        control_velocities = self.data_bus.get_all_control_velocities()
        current_velocities = self.data_bus.get_all_joint_velocities()
        
        for joint_name, desired_velocity in control_velocities.items():
            if joint_name not in self._joint_to_actuator:
                continue
            
            actuator_id = self._joint_to_actuator[joint_name]['actuator_id']
            
            # 应用速度限制
            if self.enable_safety_checks:
                desired_velocity = self._apply_velocity_limits(joint_name, desired_velocity)
            
            # 获取当前速度
            current_vel = current_velocities.get(joint_name, 0.0)
            
            # P控制计算力矩
            kv = self.velocity_gains.get(joint_name, 10.0)
            velocity_error = desired_velocity - current_vel
            
            torque = kv * velocity_error
            
            # 应用力矩限制
            if self.enable_safety_checks:
                torque = self._apply_torque_limits(joint_name, torque)
            
            # 设置控制输入
            self.data.ctrl[actuator_id] = torque
    
    def _send_hybrid_commands(self):
        """发送混合控制指令（每个关节可以有不同的控制模式）"""
        for joint_name in self.data_bus.joint_names:
            if joint_name not in self._joint_to_actuator:
                continue
            
            # 获取该关节的控制模式
            joint_mode = self.joint_control_modes.get(joint_name, ControlMode.TORQUE)
            
            # 根据关节的控制模式发送相应指令
            if joint_mode == ControlMode.TORQUE:
                torque = self.data_bus.get_control_torque(joint_name)
                if torque is not None:
                    self._send_single_torque_command(joint_name, torque)
            
            elif joint_mode == ControlMode.POSITION:
                position = self.data_bus.get_control_position(joint_name)
                if position is not None:
                    self._send_single_position_command(joint_name, position)
            
            elif joint_mode == ControlMode.VELOCITY:
                velocity = self.data_bus.control_velocities.get(joint_name)
                if velocity is not None:
                    self._send_single_velocity_command(joint_name, velocity)
    
    def _send_single_torque_command(self, joint_name: str, torque: float):
        """发送单个关节的力矩指令"""
        actuator_id = self._joint_to_actuator[joint_name]['actuator_id']
        
        if self.enable_safety_checks:
            torque = self._apply_torque_limits(joint_name, torque)
        
        self.data.ctrl[actuator_id] = torque
    
    def _send_single_position_command(self, joint_name: str, desired_position: float):
        """发送单个关节的位置指令"""
        actuator_id = self._joint_to_actuator[joint_name]['actuator_id']
        
        if self.enable_safety_checks:
            desired_position = self._apply_position_limits(joint_name, desired_position)
        
        # 获取当前状态
        current_pos = self.data_bus.get_joint_position(joint_name) or 0.0
        current_vel = self.data_bus.get_joint_velocity(joint_name) or 0.0
        
        # PD控制
        kp = self.position_gains.get(joint_name, 100.0)
        kd = self.velocity_gains.get(joint_name, 10.0)
        
        position_error = desired_position - current_pos
        torque = kp * position_error - kd * current_vel
        
        if self.enable_safety_checks:
            torque = self._apply_torque_limits(joint_name, torque)
        
        self.data.ctrl[actuator_id] = torque
    
    def _send_single_velocity_command(self, joint_name: str, desired_velocity: float):
        """发送单个关节的速度指令"""
        actuator_id = self._joint_to_actuator[joint_name]['actuator_id']
        
        if self.enable_safety_checks:
            desired_velocity = self._apply_velocity_limits(joint_name, desired_velocity)
        
        # 获取当前速度
        current_vel = self.data_bus.get_joint_velocity(joint_name) or 0.0
        
        # P控制
        kv = self.velocity_gains.get(joint_name, 10.0)
        velocity_error = desired_velocity - current_vel
        torque = kv * velocity_error
        
        if self.enable_safety_checks:
            torque = self._apply_torque_limits(joint_name, torque)
        
        self.data.ctrl[actuator_id] = torque
    
    def _apply_torque_limits(self, joint_name: str, torque: float) -> float:
        """应用力矩限制"""
        if joint_name in self.torque_limits:
            limits = self.torque_limits[joint_name]
            torque = np.clip(torque, limits['min'], limits['max'])
        return torque
    
    def _apply_position_limits(self, joint_name: str, position: float) -> float:
        """应用位置限制"""
        if joint_name in self.position_limits:
            limits = self.position_limits[joint_name]
            position = np.clip(position, limits['min'], limits['max'])
        return position
    
    def _apply_velocity_limits(self, joint_name: str, velocity: float) -> float:
        """应用速度限制"""
        if joint_name in self.velocity_limits:
            max_vel = self.velocity_limits[joint_name]['max']
            velocity = np.clip(velocity, -max_vel, max_vel)
        return velocity
    
    def set_safety_checks(self, enable: bool):
        """启用/禁用安全检查"""
        self.enable_safety_checks = enable
        print(f"Safety checks {'enabled' if enable else 'disabled'}")
    
    def update_control_gains(self, joint_name: str, kp: float = None, kd: float = None):
        """
        更新特定关节的控制增益
        
        参数:
            joint_name: 关节名称
            kp: 位置增益
            kd: 速度增益
        """
        if joint_name in self.data_bus.joint_names:
            if kp is not None:
                self.position_gains[joint_name] = kp
            if kd is not None:
                self.velocity_gains[joint_name] = kd
            print(f"Updated gains for {joint_name}: kp={kp}, kd={kd}")
    
    def get_actuator_info(self) -> Dict[str, Dict]:
        """获取所有执行器的信息"""
        info = {}
        for joint_name, mapping in self._joint_to_actuator.items():
            actuator_id = mapping['actuator_id']
            info[joint_name] = {
                'actuator_name': mapping['actuator_name'],
                'actuator_id': actuator_id,
                'current_control': self.data.ctrl[actuator_id],
                'control_mode': self.joint_control_modes.get(
                    joint_name, self.control_mode
                ).value
            }
        return info
    
    def reset_commands(self):
        """重置所有控制指令为零"""
        self.data.ctrl[:] = 0.0
        print("All control commands reset to zero")


# 便捷函数
def create_control_sender(model: mujoco.MjModel, data: mujoco.MjData) -> ControlSender:
    """
    创建控制发送器实例
    
    参数:
        model: MuJoCo模型
        data: MuJoCo数据
        
    返回:
        ControlSender实例
    """
    return ControlSender(model, data)


def send_commands(control_sender: ControlSender) -> bool:
    """
    便捷函数：发送控制指令
    
    参数:
        control_sender: 控制发送器实例
        
    返回:
        是否成功发送
    """
    return control_sender.send_commands()


# 示例使用
if __name__ == "__main__":
    print("ControlSender module loaded successfully")
    print("To use this module:")
    print("1. Create MuJoCo model and data objects")
    print("2. Create ControlSender: sender = create_control_sender(model, data)")
    print("3. Set control mode: sender.set_control_mode(ControlMode.TORQUE)")
    print("4. Send commands: success = send_commands(sender)")
    print("5. Or use directly: sender.send_commands()") 