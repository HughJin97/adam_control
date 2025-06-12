#!/usr/bin/env python3
"""
足步规划模块 - Foot Placement Module

实现机器人步态过程中的落脚点计算和足步规划功能。
根据当前支撑腿位置、步态参数和运动意图计算摆动腿的目标着地位置。

功能特性：
1. 正向运动学计算足端位置
2. 基于支撑腿的相对落脚点计算
3. 步长和步宽的动态调整
4. 地形适应和稳定性保证
5. 与步态调度器的集成
6. 实时足步目标更新

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import threading
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Union
import math


class FootPlacementStrategy(Enum):
    """足步规划策略"""
    STATIC_WALK = "static_walk"           # 静态行走
    DYNAMIC_WALK = "dynamic_walk"         # 动态行走
    ADAPTIVE = "adaptive"                 # 自适应规划
    TERRAIN_ADAPTIVE = "terrain_adaptive" # 地形自适应
    STABILIZING = "stabilizing"           # 稳定性优先


class TerrainType(Enum):
    """地形类型"""
    FLAT = "flat"                # 平地
    SLOPE = "slope"              # 斜坡
    STAIRS = "stairs"            # 楼梯
    ROUGH = "rough"              # 粗糙地面
    UNKNOWN = "unknown"          # 未知地形


@dataclass
class Vector3D:
    """三维向量类"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y, self.z])
    
    def from_array(self, arr: np.ndarray):
        """从numpy数组设置值"""
        self.x, self.y, self.z = arr[0], arr[1], arr[2]
    
    def distance_to(self, other: 'Vector3D') -> float:
        """计算到另一点的距离"""
        return np.linalg.norm(self.to_array() - other.to_array())
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)


@dataclass
class FootState:
    """足部状态信息"""
    position: Vector3D = field(default_factory=Vector3D)      # 当前位置
    target_position: Vector3D = field(default_factory=Vector3D)  # 目标位置
    velocity: Vector3D = field(default_factory=Vector3D)      # 速度
    in_contact: bool = True                                   # 是否接触地面
    contact_force: float = 0.0                               # 接触力
    last_update_time: float = 0.0                            # 上次更新时间
    
    def update_from_kinematics(self, position: np.ndarray):
        """从运动学计算结果更新位置"""
        self.position.from_array(position)
        self.last_update_time = time.time()


@dataclass
class FootPlacementConfig:
    """足步规划配置"""
    # 基本步态参数
    nominal_step_length: float = 0.15      # 标准步长 [m]
    nominal_step_width: float = 0.12       # 标准步宽 [m]
    step_height: float = 0.05              # 抬脚高度 [m]
    
    # 身体几何参数
    hip_width: float = 0.18                # 髋宽 [m]
    leg_length: float = 0.6                # 腿长 [m]
    foot_length: float = 0.16              # 脚长 [m]
    foot_width: float = 0.08               # 脚宽 [m]
    
    # 动态调整参数
    speed_step_gain: float = 0.8           # 速度-步长增益
    lateral_stability_margin: float = 0.03 # 横向稳定性余量 [m]
    longitudinal_stability_margin: float = 0.02  # 纵向稳定性余量 [m]
    
    # 地形适应参数
    terrain_adaptation_gain: float = 0.5   # 地形适应增益
    max_step_adjustment: float = 0.08      # 最大步态调整 [m]
    ground_clearance: float = 0.02         # 地面间隙 [m]
    
    # 安全限制
    max_step_length: float = 0.25          # 最大步长 [m]
    min_step_length: float = 0.05          # 最小步长 [m]
    max_step_width: float = 0.20           # 最大步宽 [m]
    min_step_width: float = 0.08           # 最小步宽 [m]
    
    # 控制参数
    planning_lookahead: float = 0.2        # 规划前瞻时间 [s]
    update_frequency: float = 50.0         # 更新频率 [Hz]
    smoothing_factor: float = 0.1          # 平滑因子


class ForwardKinematics:
    """正向运动学计算器"""
    
    def __init__(self):
        """初始化运动学参数"""
        # AzureLoong机器人的简化运动学参数
        self.hip_offset = np.array([0.0, 0.09, 0.0])      # 髋关节偏移 [m]
        self.upper_leg_length = 0.30                       # 大腿长度 [m]
        self.lower_leg_length = 0.30                       # 小腿长度 [m]
        self.foot_offset = np.array([0.0, 0.0, -0.05])    # 足部偏移 [m]
        
        # 关节名称映射
        self.left_leg_joints = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
            "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll"
        ]
        self.right_leg_joints = [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", 
            "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll"
        ]
    
    def compute_foot_position(self, joint_angles: Dict[str, float], leg: str) -> np.ndarray:
        """
        计算足端位置
        
        参数:
            joint_angles: 关节角度字典 [rad]
            leg: 腿部标识 ("left" 或 "right")
            
        返回:
            np.ndarray: 足端位置 [x, y, z] [m]
        """
        if leg.lower() in ["left", "left_foot"]:
            joints = self.left_leg_joints
            hip_sign = 1.0  # 左腿
        else:
            joints = self.right_leg_joints
            hip_sign = -1.0  # 右腿
        
        # 获取关节角度
        try:
            hip_yaw = joint_angles.get(joints[0], 0.0)
            hip_roll = joint_angles.get(joints[1], 0.0)
            hip_pitch = joint_angles.get(joints[2], 0.0)
            knee_pitch = joint_angles.get(joints[3], 0.0)
            ankle_pitch = joint_angles.get(joints[4], 0.0)
            ankle_roll = joint_angles.get(joints[5], 0.0)
        except (KeyError, IndexError):
            # 如果关节角度不可用，返回默认位置
            return np.array([0.0, hip_sign * self.hip_offset[1], -self.upper_leg_length - self.lower_leg_length])
        
        # 简化的正向运动学计算
        # 这里使用几何方法近似计算，实际应用中可能需要更精确的DH参数法
        
        # 髋关节位置
        hip_pos = np.array([0.0, hip_sign * self.hip_offset[1], 0.0])
        
        # 考虑髋关节旋转
        # 髋偏摆 (yaw)
        yaw_rotation = np.array([
            [np.cos(hip_yaw), -np.sin(hip_yaw), 0],
            [np.sin(hip_yaw), np.cos(hip_yaw), 0],
            [0, 0, 1]
        ])
        
        # 髋侧摆 (roll)  
        roll_rotation = np.array([
            [1, 0, 0],
            [0, np.cos(hip_roll), -np.sin(hip_roll)],
            [0, np.sin(hip_roll), np.cos(hip_roll)]
        ])
        
        # 髋俯仰 (pitch)
        pitch_rotation = np.array([
            [np.cos(hip_pitch), 0, np.sin(hip_pitch)],
            [0, 1, 0],
            [-np.sin(hip_pitch), 0, np.cos(hip_pitch)]
        ])
        
        # 大腿向量 (髋到膝)
        thigh_vector = np.array([0, 0, -self.upper_leg_length])
        
        # 应用髋关节旋转
        hip_rotation = yaw_rotation @ roll_rotation @ pitch_rotation
        thigh_rotated = hip_rotation @ thigh_vector
        
        # 膝关节位置
        knee_pos = hip_pos + thigh_rotated
        
        # 膝关节旋转 (只有pitch)
        knee_rotation = np.array([
            [np.cos(knee_pitch), 0, np.sin(knee_pitch)],
            [0, 1, 0],
            [-np.sin(knee_pitch), 0, np.cos(knee_pitch)]
        ])
        
        # 小腿向量 (膝到踝)
        shin_vector = np.array([0, 0, -self.lower_leg_length])
        shin_rotated = hip_rotation @ knee_rotation @ shin_vector
        
        # 踝关节位置
        ankle_pos = knee_pos + shin_rotated
        
        # 踝关节旋转
        ankle_pitch_rotation = np.array([
            [np.cos(ankle_pitch), 0, np.sin(ankle_pitch)],
            [0, 1, 0],
            [-np.sin(ankle_pitch), 0, np.cos(ankle_pitch)]
        ])
        
        ankle_roll_rotation = np.array([
            [1, 0, 0],
            [0, np.cos(ankle_roll), -np.sin(ankle_roll)],
            [0, np.sin(ankle_roll), np.cos(ankle_roll)]
        ])
        
        # 足部偏移
        foot_rotated = hip_rotation @ knee_rotation @ ankle_pitch_rotation @ ankle_roll_rotation @ self.foot_offset
        
        # 最终足端位置
        foot_pos = ankle_pos + foot_rotated
        
        return foot_pos
    
    def compute_both_feet_positions(self, joint_angles: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算双足位置
        
        返回:
            Tuple[np.ndarray, np.ndarray]: (左足位置, 右足位置)
        """
        left_pos = self.compute_foot_position(joint_angles, "left")
        right_pos = self.compute_foot_position(joint_angles, "right")
        return left_pos, right_pos


class FootPlacementPlanner:
    """足步规划器"""
    
    def __init__(self, config: Optional[FootPlacementConfig] = None):
        """初始化足步规划器"""
        self.config = config or FootPlacementConfig()
        self.kinematics = ForwardKinematics()
        
        # 足部状态
        self.left_foot = FootState()
        self.right_foot = FootState()
        
        # 规划状态
        self.current_strategy = FootPlacementStrategy.STATIC_WALK
        self.terrain_type = TerrainType.FLAT
        self.body_velocity = Vector3D()
        self.body_heading = 0.0  # 身体朝向角度 [rad]
        
        # 目标位置
        self.target_positions = {
            "left_foot": Vector3D(),
            "right_foot": Vector3D()
        }
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 统计信息
        self.planning_count = 0
        self.last_planning_time = 0.0
        
        print("FootPlacementPlanner initialized")
    
    def update_foot_states_from_kinematics(self, joint_angles: Dict[str, float]):
        """从关节角度更新足部状态"""
        with self._lock:
            left_pos, right_pos = self.kinematics.compute_both_feet_positions(joint_angles)
            
            self.left_foot.update_from_kinematics(left_pos)
            self.right_foot.update_from_kinematics(right_pos)
    
    def set_body_motion_intent(self, velocity: Vector3D, heading: float = 0.0):
        """设置身体运动意图"""
        with self._lock:
            self.body_velocity = velocity
            self.body_heading = heading
    
    def compute_step_parameters(self, swing_leg: str, support_leg: str) -> Dict[str, float]:
        """
        计算步态参数
        
        参数:
            swing_leg: 摆动腿 ("left"/"right")
            support_leg: 支撑腿 ("left"/"right")
            
        返回:
            Dict: 步态参数字典
        """
        # 基础步长和步宽
        base_step_length = self.config.nominal_step_length
        base_step_width = self.config.nominal_step_width
        
        # 根据身体速度调整步长
        forward_speed = self.body_velocity.x
        step_length = base_step_length + forward_speed * self.config.speed_step_gain
        
        # 应用安全限制
        step_length = np.clip(step_length, self.config.min_step_length, self.config.max_step_length)
        
        # 根据横向速度调整步宽
        lateral_speed = self.body_velocity.y
        step_width = base_step_width + abs(lateral_speed) * 0.5
        step_width = np.clip(step_width, self.config.min_step_width, self.config.max_step_width)
        
        # 考虑地形适应
        if self.terrain_type != TerrainType.FLAT:
            terrain_factor = self.config.terrain_adaptation_gain
            step_length *= (1.0 - terrain_factor * 0.2)  # 在复杂地形上步长略减小
            step_width *= (1.0 + terrain_factor * 0.1)   # 步宽略增加以提高稳定性
        
        return {
            'step_length': step_length,
            'step_width': step_width,
            'step_height': self.config.step_height,
            'ground_clearance': self.config.ground_clearance
        }
    
    def compute_target_foot_position(self, swing_leg: str, support_leg: str, 
                                   step_params: Dict[str, float]) -> Vector3D:
        """
        计算摆动腿目标落脚点
        
        参数:
            swing_leg: 摆动腿标识
            support_leg: 支撑腿标识  
            step_params: 步态参数
            
        返回:
            Vector3D: 目标足端位置
        """
        # 获取支撑腿当前位置
        if support_leg.lower() in ["left", "left_foot"]:
            support_pos = self.left_foot.position
        else:
            support_pos = self.right_foot.position
        
        # 计算身体坐标系下的步位移
        step_length = step_params['step_length']
        step_width = step_params['step_width']
        
        # 前进方向位移 (考虑身体朝向)
        forward_displacement = Vector3D(
            step_length * np.cos(self.body_heading),
            step_length * np.sin(self.body_heading),
            0.0
        )
        
        # 横向位移 (相对于身体中线)
        if swing_leg.lower() in ["left", "left_foot"]:
            # 左脚：向左侧偏移（y轴负方向）
            lateral_displacement = Vector3D(
                -step_width * 0.5 * np.sin(self.body_heading),
                -step_width * 0.5 * np.cos(self.body_heading),
                0.0
            )
        else:
            # 右脚：向右侧偏移（y轴正方向）
            lateral_displacement = Vector3D(
                step_width * 0.5 * np.sin(self.body_heading),
                step_width * 0.5 * np.cos(self.body_heading),
                0.0
            )
        
        # 计算目标位置
        target_pos = support_pos + forward_displacement + lateral_displacement
        
        # 设置地面高度 (假设平地为0)
        target_pos.z = step_params['ground_clearance']
        
        # 应用稳定性调整
        target_pos = self._apply_stability_adjustments(target_pos, swing_leg, support_leg)
        
        return target_pos
    
    def _apply_stability_adjustments(self, target_pos: Vector3D, swing_leg: str, 
                                   support_leg: str) -> Vector3D:
        """应用稳定性调整"""
        adjusted_pos = Vector3D(target_pos.x, target_pos.y, target_pos.z)
        
        # 横向稳定性调整
        if swing_leg.lower() in ["left", "left_foot"]:
            adjusted_pos.y -= self.config.lateral_stability_margin
        else:
            adjusted_pos.y += self.config.lateral_stability_margin
        
        # 纵向稳定性调整 (略微前置重心)
        adjusted_pos.x += self.config.longitudinal_stability_margin
        
        return adjusted_pos
    
    def plan_foot_placement(self, swing_leg: str, support_leg: str) -> Vector3D:
        """
        规划足步放置
        
        参数:
            swing_leg: 摆动腿标识
            support_leg: 支撑腿标识
            
        返回:
            Vector3D: 计算得到的目标足端位置
        """
        with self._lock:
            # 计算步态参数
            step_params = self.compute_step_parameters(swing_leg, support_leg)
            
            # 计算目标位置
            target_pos = self.compute_target_foot_position(swing_leg, support_leg, step_params)
            
            # 更新目标位置记录
            foot_key = f"{swing_leg.lower()}_foot"
            if foot_key not in self.target_positions:
                foot_key = swing_leg.lower()
            
            if foot_key in self.target_positions:
                self.target_positions[foot_key] = target_pos
            
            # 更新统计信息
            self.planning_count += 1
            self.last_planning_time = time.time()
            
            return target_pos
    
    def get_current_foot_positions(self) -> Dict[str, Vector3D]:
        """获取当前足部位置"""
        with self._lock:
            return {
                "left_foot": self.left_foot.position,
                "right_foot": self.right_foot.position
            }
    
    def get_target_foot_positions(self) -> Dict[str, Vector3D]:
        """获取目标足部位置"""
        with self._lock:
            return self.target_positions.copy()
    
    def set_terrain_type(self, terrain: TerrainType):
        """设置地形类型"""
        with self._lock:
            self.terrain_type = terrain
    
    def set_planning_strategy(self, strategy: FootPlacementStrategy):
        """设置规划策略"""
        with self._lock:
            self.current_strategy = strategy
    
    def get_planning_statistics(self) -> Dict:
        """获取规划统计信息"""
        with self._lock:
            return {
                "planning_count": self.planning_count,
                "last_planning_time": self.last_planning_time,
                "current_strategy": self.current_strategy.value,
                "terrain_type": self.terrain_type.value,
                "left_foot_position": self.left_foot.position,
                "right_foot_position": self.right_foot.position,
                "target_positions": self.target_positions.copy()
            }
    
    def print_status(self):
        """打印当前状态"""
        stats = self.get_planning_statistics()
        
        print(f"\n=== 足步规划器状态 ===")
        print(f"规划策略: {stats['current_strategy']}")
        print(f"地形类型: {stats['terrain_type']}")
        print(f"规划次数: {stats['planning_count']}")
        print(f"上次规划时间: {stats['last_planning_time']:.3f}")
        
        print(f"\n当前足部位置:")
        left_pos = stats['left_foot_position']
        right_pos = stats['right_foot_position']
        print(f"  左脚: ({left_pos.x:.3f}, {left_pos.y:.3f}, {left_pos.z:.3f})")
        print(f"  右脚: ({right_pos.x:.3f}, {right_pos.y:.3f}, {right_pos.z:.3f})")
        
        print(f"\n目标足部位置:")
        for foot, pos in stats['target_positions'].items():
            print(f"  {foot}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")


# 全局足步规划器实例
_global_foot_planner = None


def get_foot_planner(config: Optional[FootPlacementConfig] = None) -> FootPlacementPlanner:
    """获取全局足步规划器实例"""
    global _global_foot_planner
    if _global_foot_planner is None:
        _global_foot_planner = FootPlacementPlanner(config)
    return _global_foot_planner


if __name__ == "__main__":
    # 测试代码
    planner = get_foot_planner()
    
    print("足步规划器测试")
    planner.print_status()
    
    # 模拟关节角度数据
    joint_angles = {
        "left_hip_yaw": 0.0, "left_hip_roll": 0.05, "left_hip_pitch": -0.1,
        "left_knee_pitch": 0.2, "left_ankle_pitch": -0.1, "left_ankle_roll": -0.05,
        "right_hip_yaw": 0.0, "right_hip_roll": -0.05, "right_hip_pitch": -0.1,
        "right_knee_pitch": 0.2, "right_ankle_pitch": -0.1, "right_ankle_roll": 0.05
    }
    
    # 更新足部状态
    planner.update_foot_states_from_kinematics(joint_angles)
    
    # 设置运动意图
    planner.set_body_motion_intent(Vector3D(0.3, 0.0, 0.0), 0.0)  # 前进0.3m/s
    
    print("\n开始足步规划测试...")
    
    # 模拟左腿摆动时的规划
    target_left = planner.plan_foot_placement("left", "right")
    print(f"左腿目标位置: ({target_left.x:.3f}, {target_left.y:.3f}, {target_left.z:.3f})")
    
    # 模拟右腿摆动时的规划
    target_right = planner.plan_foot_placement("right", "left")
    print(f"右腿目标位置: ({target_right.x:.3f}, {target_right.y:.3f}, {target_right.z:.3f})")
    
    # 打印最终状态
    planner.print_status() 