#!/usr/bin/env python3
"""
步态有限状态机调度器 - Gait Finite State Machine Scheduler

实现步态状态机，用于跟踪和管理机器人的步态阶段转换。
支持基于时间和传感器反馈的状态切换逻辑。

功能特性：
1. 步态状态定义和管理
2. 基于时间的状态转换
3. 基于传感器反馈的状态转换
4. 双触发机制（时间 + 传感器）
5. 状态转换历史记录
6. 调试和监控接口

作者: Adam Control Team
版本: 1.0
"""

import time
import threading
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from collections import deque


class GaitState(Enum):
    """步态状态枚举"""
    # 基础状态
    IDLE = "idle"                           # 空闲状态
    STANDING = "standing"                   # 静止站立
    
    # 单支撑状态
    LEFT_SUPPORT = "left_support"           # 左腿支撑（右腿摆动）
    RIGHT_SUPPORT = "right_support"         # 右腿支撑（左腿摆动）
    
    # 双支撑过渡状态  
    DOUBLE_SUPPORT_LR = "double_support_lr" # 双支撑：左→右过渡
    DOUBLE_SUPPORT_RL = "double_support_rl" # 双支撑：右→左过渡
    
    # 特殊状态
    FLIGHT = "flight"                       # 飞行相（跑步时）
    EMERGENCY_STOP = "emergency_stop"       # 紧急停止


class LegState(Enum):
    """支撑腿状态枚举"""
    LSt = "left_support"        # 左支撑
    RSt = "right_support"       # 右支撑 
    DSt = "double_support"      # 双支撑
    NSt = "no_support"          # 无支撑（飞行相）


class TransitionTrigger(Enum):
    """状态转换触发类型"""
    TIME_BASED = "time_based"           # 基于时间
    SENSOR_BASED = "sensor_based"       # 基于传感器
    HYBRID = "hybrid"                   # 混合触发（时间 + 传感器）
    MANUAL = "manual"                   # 手动触发


@dataclass
class StateTransition:
    """状态转换定义"""
    from_state: GaitState               # 源状态
    to_state: GaitState                 # 目标状态
    trigger_type: TransitionTrigger     # 触发类型
    time_condition: float = 0.0         # 时间条件 [s]
    force_threshold: float = 50.0       # 力阈值 [N]
    velocity_threshold: float = 0.01    # 速度阈值 [m/s]
    condition_func: Optional[Callable] = None  # 自定义条件函数


@dataclass
class StateInfo:
    """状态信息"""
    state: GaitState                    # 当前状态
    entry_time: float                   # 进入时间
    duration: float                     # 持续时间
    leg_state: LegState                 # 支撑腿状态
    expected_duration: float = 0.0      # 期望持续时间
    swing_leg: str = ""                 # 摆动腿 ("left"/"right"/"none")
    support_leg: str = ""               # 支撑腿 ("left"/"right"/"both")


@dataclass
class GaitSchedulerConfig:
    """步态调度器配置"""
    # 基础时间参数
    swing_time: float = 0.4             # 摆动时间 [s]
    stance_time: float = 0.6            # 支撑时间 [s] 
    double_support_time: float = 0.1    # 双支撑时间 [s]
    
    # 传感器阈值
    touchdown_force_threshold: float = 30.0    # 着地力阈值 [N]
    liftoff_force_threshold: float = 10.0      # 离地力阈值 [N]
    contact_velocity_threshold: float = 0.02   # 接触速度阈值 [m/s]
    
    # 触发模式
    use_time_trigger: bool = True              # 使用时间触发
    use_sensor_trigger: bool = True            # 使用传感器触发
    require_both_triggers: bool = False        # 要求两种触发都满足
    
    # 安全参数
    max_swing_time: float = 1.0                # 最大摆动时间 [s]
    min_stance_time: float = 0.2               # 最小支撑时间 [s]
    emergency_force_threshold: float = 200.0   # 紧急停止力阈值 [N]
    
    # 调试选项
    enable_logging: bool = True                # 启用日志
    log_transitions: bool = True               # 记录状态转换
    max_log_entries: int = 1000                # 最大日志条目数


class GaitScheduler:
    """步态有限状态机调度器"""
    
    def __init__(self, config: Optional[GaitSchedulerConfig] = None):
        """初始化步态调度器"""
        self.config = config or GaitSchedulerConfig()
        
        # 状态管理
        self.current_state = GaitState.IDLE
        self.previous_state = GaitState.IDLE
        self.state_start_time = time.time()
        self.total_time = 0.0
        
        # ================== 步态计时器 ==================
        self.swing_start_time = 0.0          # 摆动开始时间
        self.swing_elapsed_time = 0.0        # 摆动经过时间
        self.stance_start_time = 0.0         # 支撑开始时间  
        self.stance_elapsed_time = 0.0       # 支撑经过时间
        self.step_start_time = 0.0           # 当前步开始时间
        self.step_elapsed_time = 0.0         # 当前步经过时间
        self.current_step_phase = "stance"   # 当前步相位 ("swing"/"stance"/"transition")
        
        # 步态切换控制
        self.step_completion_pending = False  # 步完成待处理标志
        self.next_swing_leg = "none"         # 下一个摆动腿
        self.step_transition_threshold = 0.95  # 步切换阈值（占总摆动时间的比例）
        
        # 支撑腿状态
        self.leg_state = LegState.DSt
        self.swing_leg = "none"
        self.support_leg = "both"
        
        # 传感器数据
        self.left_foot_force = 0.0
        self.right_foot_force = 0.0
        self.left_foot_contact = False
        self.right_foot_contact = False
        self.left_foot_velocity = np.array([0.0, 0.0, 0.0])
        self.right_foot_velocity = np.array([0.0, 0.0, 0.0])
        
        # 状态转换定义
        self.transitions: Dict[GaitState, List[StateTransition]] = {}
        self._setup_transitions()
        
        # 历史记录
        self.state_history: deque = deque(maxlen=self.config.max_log_entries)
        self.transition_log: deque = deque(maxlen=self.config.max_log_entries)
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 回调函数
        self.state_change_callbacks: List[Callable] = []
        
        print("GaitScheduler initialized")
    
    def _setup_transitions(self):
        """设置状态转换规则"""
        # IDLE -> STANDING
        self.add_transition(
            GaitState.IDLE, GaitState.STANDING,
            TransitionTrigger.TIME_BASED, time_condition=0.1
        )
        
        # STANDING -> LEFT_SUPPORT (开始行走)
        self.add_transition(
            GaitState.STANDING, GaitState.LEFT_SUPPORT,
            TransitionTrigger.MANUAL  # 手动开始行走
        )
        
        # LEFT_SUPPORT -> DOUBLE_SUPPORT_LR
        self.add_transition(
            GaitState.LEFT_SUPPORT, GaitState.DOUBLE_SUPPORT_LR,
            TransitionTrigger.HYBRID,
            time_condition=self.config.swing_time,
            force_threshold=self.config.touchdown_force_threshold
        )
        
        # DOUBLE_SUPPORT_LR -> RIGHT_SUPPORT
        self.add_transition(
            GaitState.DOUBLE_SUPPORT_LR, GaitState.RIGHT_SUPPORT,
            TransitionTrigger.TIME_BASED,
            time_condition=self.config.double_support_time
        )
        
        # RIGHT_SUPPORT -> DOUBLE_SUPPORT_RL
        self.add_transition(
            GaitState.RIGHT_SUPPORT, GaitState.DOUBLE_SUPPORT_RL,
            TransitionTrigger.HYBRID,
            time_condition=self.config.swing_time,
            force_threshold=self.config.touchdown_force_threshold
        )
        
        # DOUBLE_SUPPORT_RL -> LEFT_SUPPORT
        self.add_transition(
            GaitState.DOUBLE_SUPPORT_RL, GaitState.LEFT_SUPPORT,
            TransitionTrigger.TIME_BASED,
            time_condition=self.config.double_support_time
        )
        
        # 任何状态 -> EMERGENCY_STOP
        for state in GaitState:
            if state != GaitState.EMERGENCY_STOP:
                self.add_transition(
                    state, GaitState.EMERGENCY_STOP,
                    TransitionTrigger.SENSOR_BASED,
                    force_threshold=self.config.emergency_force_threshold
                )
        
        # EMERGENCY_STOP -> STANDING
        self.add_transition(
            GaitState.EMERGENCY_STOP, GaitState.STANDING,
            TransitionTrigger.MANUAL
        )
    
    def add_transition(self, from_state: GaitState, to_state: GaitState, 
                      trigger_type: TransitionTrigger, **kwargs):
        """添加状态转换规则"""
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            trigger_type=trigger_type,
            **kwargs
        )
        
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        self.transitions[from_state].append(transition)
    
    def update_sensor_data(self, left_force: float, right_force: float,
                          left_velocity: np.ndarray, right_velocity: np.ndarray):
        """更新传感器数据"""
        with self._lock:
            self.left_foot_force = left_force
            self.right_foot_force = right_force
            self.left_foot_velocity = left_velocity.copy()
            self.right_foot_velocity = right_velocity.copy()
            
            # 更新接触状态
            self.left_foot_contact = left_force > self.config.touchdown_force_threshold
            self.right_foot_contact = right_force > self.config.touchdown_force_threshold
    
    def update(self, dt: float) -> bool:
        """
        主循环更新接口 - 推进步态有限状态机
        
        此方法应在主控制循环中调用，执行：
        1. 从数据总线读取传感器数据
        2. 更新步态计时器和状态机
        3. 检测步完成条件
        4. 计算并写入下一步目标
        5. 更新数据总线中的步态状态
        
        参数:
            dt: 时间步长 [s]
            
        返回:
            bool: 是否发生状态转换或重要更新
        """
        return self.update_gait_state(dt)
    
    def update_gait_state(self, dt: float) -> bool:
        """
        更新步态状态
        
        参数:
            dt: 时间步长 [s]
            
        返回:
            bool: 是否发生状态转换
        """
        with self._lock:
            self.total_time += dt
            state_changed = False
            
            # ================== 从数据总线读取传感器数据 ==================
            self._read_sensor_data_from_bus()
            
            # ================== 更新步态计时器 ==================
            self._update_gait_timers(dt)
            
            # ================== 检查步完成条件 ==================
            step_completed = self._check_step_completion()
            
            # 检查状态转换条件
            current_duration = time.time() - self.state_start_time
            
            if self.current_state in self.transitions:
                for transition in self.transitions[self.current_state]:
                    # 检查基本转换条件
                    basic_condition = self._check_transition_condition(transition, current_duration)
                    
                    # 检查步完成条件（对于摆动相结束的转换）
                    step_condition = True
                    if (self.current_state in [GaitState.LEFT_SUPPORT, GaitState.RIGHT_SUPPORT] and
                        transition.to_state in [GaitState.DOUBLE_SUPPORT_LR, GaitState.DOUBLE_SUPPORT_RL]):
                        step_condition = step_completed
                    
                    if basic_condition and step_condition:
                        self._transition_to_state(transition.to_state)
                        state_changed = True
                        break
            
            # 更新状态信息
            self._update_leg_states()
            
            # 处理步完成事件
            if step_completed and not self.step_completion_pending:
                self._handle_step_completion()
            
            # ================== 数据总线交互 ==================
            self._update_data_bus_outputs()
            
            return state_changed
    
    def _read_sensor_data_from_bus(self):
        """从数据总线读取传感器数据"""
        try:
            from data_bus import get_data_bus
            data_bus = get_data_bus()
            
            # 读取足底力传感器数据
            left_foot = data_bus.end_effectors.get("left_foot")
            right_foot = data_bus.end_effectors.get("right_foot")
            
            if left_foot:
                self.left_foot_force = left_foot.contact_force_magnitude
                self.left_foot_velocity = left_foot.velocity.to_array()
                self.left_foot_contact = left_foot.contact_state.value == 1
            
            if right_foot:
                self.right_foot_force = right_foot.contact_force_magnitude
                self.right_foot_velocity = right_foot.velocity.to_array()
                self.right_foot_contact = right_foot.contact_state.value == 1
            
            # 读取IMU数据
            self.body_acceleration = data_bus.imu.linear_acceleration.to_array()
            self.body_angular_velocity = data_bus.imu.angular_velocity.to_array()
            
        except Exception as e:
            if self.config.enable_logging:
                print(f"Warning: 传感器数据读取失败: {e}")
    
    def _update_data_bus_outputs(self):
        """更新数据总线中的步态输出数据"""
        try:
            from data_bus import get_data_bus
            data_bus = get_data_bus()
            
            # 更新基本步态状态
            data_bus.current_gait_state = self.current_state.value
            data_bus.legState = self.leg_state.value
            data_bus.swing_leg = self.swing_leg
            data_bus.support_leg = self.support_leg
            
            # 更新计时信息
            data_bus.gait.phase_time = self.swing_elapsed_time
            data_bus.gait.cycle_phase = self._calculate_cycle_phase()
            
            # 更新足部摆动状态
            data_bus.gait.left_in_swing = (self.swing_leg == "left")
            data_bus.gait.right_in_swing = (self.swing_leg == "right")
            data_bus.gait.double_support = (self.swing_leg == "none")
            
            # 触发足步规划更新（如果需要）
            if hasattr(data_bus, 'update_foot_kinematics'):
                data_bus.update_foot_kinematics()
                
        except Exception as e:
            if self.config.enable_logging:
                print(f"Warning: 数据总线更新失败: {e}")
    
    def _calculate_cycle_phase(self) -> float:
        """计算当前步态周期相位 (0.0-1.0)"""
        cycle_time = self.config.swing_time + self.config.double_support_time
        if cycle_time > 0:
            return (self.swing_elapsed_time % cycle_time) / cycle_time
        return 0.0
    
    def _check_transition_condition(self, transition: StateTransition, 
                                  current_duration: float) -> bool:
        """检查状态转换条件"""
        time_satisfied = False
        sensor_satisfied = False
        
        # 检查时间条件
        if transition.trigger_type in [TransitionTrigger.TIME_BASED, TransitionTrigger.HYBRID]:
            time_satisfied = current_duration >= transition.time_condition
        
        # 检查传感器条件
        if transition.trigger_type in [TransitionTrigger.SENSOR_BASED, TransitionTrigger.HYBRID]:
            sensor_satisfied = self._check_sensor_condition(transition)
        
        # 检查自定义条件
        if transition.condition_func:
            custom_satisfied = transition.condition_func(self)
            if transition.trigger_type == TransitionTrigger.HYBRID:
                sensor_satisfied = sensor_satisfied and custom_satisfied
            else:
                sensor_satisfied = custom_satisfied
        
        # 根据触发类型决定是否转换
        if transition.trigger_type == TransitionTrigger.TIME_BASED:
            return time_satisfied
        elif transition.trigger_type == TransitionTrigger.SENSOR_BASED:
            return sensor_satisfied
        elif transition.trigger_type == TransitionTrigger.HYBRID:
            if self.config.require_both_triggers:
                return time_satisfied and sensor_satisfied
            else:
                return time_satisfied or sensor_satisfied
        elif transition.trigger_type == TransitionTrigger.MANUAL:
            return False  # 手动触发需要显式调用
        
        return False
    
    def _check_sensor_condition(self, transition: StateTransition) -> bool:
        """检查传感器条件"""
        # 根据当前状态和目标状态检查相应的传感器条件
        
        # 检查紧急停止
        if transition.to_state == GaitState.EMERGENCY_STOP:
            return (max(self.left_foot_force, self.right_foot_force) > 
                   transition.force_threshold)
        
        # 检查摆动腿着地
        if (self.current_state == GaitState.LEFT_SUPPORT and 
            transition.to_state == GaitState.DOUBLE_SUPPORT_LR):
            # 右腿着地
            return (self.right_foot_force > transition.force_threshold and
                   np.linalg.norm(self.right_foot_velocity) < transition.velocity_threshold)
        
        if (self.current_state == GaitState.RIGHT_SUPPORT and 
            transition.to_state == GaitState.DOUBLE_SUPPORT_RL):
            # 左腿着地
            return (self.left_foot_force > transition.force_threshold and
                   np.linalg.norm(self.left_foot_velocity) < transition.velocity_threshold)
        
        return False
    
    def _transition_to_state(self, new_state: GaitState):
        """转换到新状态"""
        if new_state == self.current_state:
            return
        
        old_state = self.current_state
        old_duration = time.time() - self.state_start_time
        
        # 记录状态历史
        state_info = StateInfo(
            state=old_state,
            entry_time=self.state_start_time,
            duration=old_duration,
            leg_state=self.leg_state,
            swing_leg=self.swing_leg,
            support_leg=self.support_leg
        )
        self.state_history.append(state_info)
        
        # 更新当前状态
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()
        
        # 记录转换日志
        if self.config.log_transitions:
            transition_info = {
                'timestamp': self.state_start_time,
                'from_state': old_state.value,
                'to_state': new_state.value,
                'duration': old_duration,
                'total_time': self.total_time
            }
            self.transition_log.append(transition_info)
        
        # 更新腿部状态
        self._update_leg_states()
        
        # 触发足步规划（在进入单支撑状态时）
        self._trigger_foot_placement_planning(old_state, new_state)
        
        # 调用回调函数
        for callback in self.state_change_callbacks:
            callback(old_state, new_state, old_duration)
        
        if self.config.enable_logging:
            print(f"State transition: {old_state.value} -> {new_state.value} "
                  f"(duration: {old_duration:.3f}s)")
    
    def _update_leg_states(self):
        """更新腿部状态"""
        old_swing_leg = self.swing_leg
        
        if self.current_state == GaitState.LEFT_SUPPORT:
            self.leg_state = LegState.LSt
            self.swing_leg = "right"
            self.support_leg = "left"
            self.current_step_phase = "swing"
        
        elif self.current_state == GaitState.RIGHT_SUPPORT:
            self.leg_state = LegState.RSt
            self.swing_leg = "left"
            self.support_leg = "right"
            self.current_step_phase = "swing"
        
        elif self.current_state in [GaitState.DOUBLE_SUPPORT_LR, 
                                   GaitState.DOUBLE_SUPPORT_RL,
                                   GaitState.STANDING]:
            self.leg_state = LegState.DSt
            self.swing_leg = "none"
            self.support_leg = "both"
            self.current_step_phase = "transition"
        
        elif self.current_state == GaitState.FLIGHT:
            self.leg_state = LegState.NSt
            self.swing_leg = "both"
            self.support_leg = "none"
            self.current_step_phase = "swing"
        
        else:  # IDLE, EMERGENCY_STOP
            self.leg_state = LegState.DSt
            self.swing_leg = "none"
            self.support_leg = "both"
            self.current_step_phase = "stance"
        
        # 检测摆动腿变化，重置摆动计时器
        if old_swing_leg != self.swing_leg and self.swing_leg in ["left", "right"]:
            self._reset_swing_timer()
    
    def _update_gait_timers(self, dt: float):
        """更新步态计时器"""
        current_time = time.time()
        
        # 更新总体步态时间
        self.step_elapsed_time += dt
        
        # 根据当前相位更新相应计时器
        if self.current_step_phase == "swing":
            self.swing_elapsed_time += dt
        elif self.current_step_phase == "stance":
            self.stance_elapsed_time += dt
        
        # 如果处于摆动相，检查是否需要更新摆动计时器
        if self.current_state in [GaitState.LEFT_SUPPORT, GaitState.RIGHT_SUPPORT]:
            if self.swing_start_time == 0.0:
                self.swing_start_time = current_time
                self.swing_elapsed_time = 0.0
    
    def _reset_swing_timer(self):
        """重置摆动计时器"""
        current_time = time.time()
        self.swing_start_time = current_time
        self.swing_elapsed_time = 0.0
        self.step_start_time = current_time
        self.step_elapsed_time = 0.0
        self.step_completion_pending = False
        
        if self.config.enable_logging:
            print(f"[步态计时器] 摆动计时器重置，摆动腿: {self.swing_leg}")
    
    def _reset_stance_timer(self):
        """重置支撑计时器"""
        current_time = time.time()
        self.stance_start_time = current_time
        self.stance_elapsed_time = 0.0
    
    def _check_step_completion(self) -> bool:
        """
        检查步完成条件
        
        返回:
            bool: 是否步完成
        """
        if self.current_step_phase != "swing" or self.swing_leg == "none":
            return False
        
        # 条件1: 时间条件 - 摆动时间达到tSwing
        time_condition = self.swing_elapsed_time >= (self.config.swing_time * self.step_transition_threshold)
        
        # 条件2: 传感器条件 - 摆动脚触地检测
        sensor_condition = False
        
        if self.swing_leg == "left":
            # 左脚摆动，检查左脚触地
            sensor_condition = (self.left_foot_force > self.config.touchdown_force_threshold and
                              np.linalg.norm(self.left_foot_velocity) < self.config.contact_velocity_threshold)
        elif self.swing_leg == "right":
            # 右脚摆动，检查右脚触地
            sensor_condition = (self.right_foot_force > self.config.touchdown_force_threshold and
                              np.linalg.norm(self.right_foot_velocity) < self.config.contact_velocity_threshold)
        
        # 使用混合条件或单独条件
        if self.config.use_time_trigger and self.config.use_sensor_trigger:
            if self.config.require_both_triggers:
                return time_condition and sensor_condition
            else:
                return time_condition or sensor_condition
        elif self.config.use_time_trigger:
            return time_condition
        elif self.config.use_sensor_trigger:
            return sensor_condition
        else:
            return time_condition  # 默认使用时间条件
    
    def _handle_step_completion(self):
        """处理步完成事件"""
        self.step_completion_pending = True
        
        # 准备下一个摆动腿
        if self.swing_leg == "left":
            self.next_swing_leg = "right"
        elif self.swing_leg == "right":
            self.next_swing_leg = "left"
        else:
            self.next_swing_leg = "none"
        
        # 通知数据总线步完成事件
        self._notify_step_completion()
        
        if self.config.enable_logging:
            print(f"[步态完成] 当前步完成，摆动腿: {self.swing_leg}, "
                  f"摆动时间: {self.swing_elapsed_time:.3f}s, "
                  f"下一摆动腿: {self.next_swing_leg}")
    
    def _notify_step_completion(self):
        """通知数据总线步完成事件"""
        try:
            from data_bus import get_data_bus
            data_bus = get_data_bus()
            
            if hasattr(data_bus, 'mark_step_completion'):
                data_bus.mark_step_completion(
                    completed_swing_leg=self.swing_leg,
                    swing_duration=self.swing_elapsed_time,
                    next_swing_leg=self.next_swing_leg
                )
        except ImportError:
            pass  # 数据总线不可用时忽略
        except Exception as e:
            if self.config.enable_logging:
                print(f"Warning: 步完成通知失败: {e}")
    
    def _trigger_foot_placement_planning(self, old_state: GaitState, new_state: GaitState):
        """
        触发足步规划
        
        当进入单支撑状态时（即某一腿刚开始抬起作为摆动腿），
        触发对应摆动腿的足步规划计算
        
        参数:
            old_state: 前一状态
            new_state: 新状态
        """
        # 检查是否进入了需要进行足步规划的状态
        planning_trigger_states = {
            GaitState.LEFT_SUPPORT: "right",   # 左支撑时，右腿摆动
            GaitState.RIGHT_SUPPORT: "left"    # 右支撑时，左腿摆动
        }
        
        if new_state in planning_trigger_states:
            swing_leg = planning_trigger_states[new_state]
            support_leg = "left" if swing_leg == "right" else "right"
            
            try:
                # 尝试从数据总线触发足步规划
                from data_bus import get_data_bus
                data_bus = get_data_bus()
                
                if hasattr(data_bus, 'trigger_foot_placement_planning'):
                    target_pos = data_bus.trigger_foot_placement_planning(swing_leg, support_leg)
                    
                    if target_pos and self.config.enable_logging:
                        print(f"[步态调度器] 触发{swing_leg}腿足步规划: "
                              f"目标({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
                else:
                    # 直接使用足步规划器
                    from foot_placement import get_foot_planner
                    planner = get_foot_planner()
                    
                    # 从数据总线获取关节角度来更新足部状态
                    if hasattr(data_bus, 'get_all_joint_positions'):
                        joint_positions = data_bus.get_all_joint_positions()
                        planner.update_foot_states_from_kinematics(joint_positions)
                    
                    target_pos = planner.plan_foot_placement(swing_leg, support_leg)
                    
                    if self.config.enable_logging:
                        print(f"[步态调度器] 足步规划{swing_leg}腿: "
                              f"目标({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
                        
            except ImportError as e:
                if self.config.enable_logging:
                    print(f"Warning: 无法导入足步规划模块: {e}")
            except Exception as e:
                if self.config.enable_logging:
                    print(f"Warning: 足步规划触发失败: {e}")
        
        # 可选：在双支撑状态结束时也可触发预规划
        elif (old_state in [GaitState.DOUBLE_SUPPORT_LR, GaitState.DOUBLE_SUPPORT_RL] and
              new_state in [GaitState.LEFT_SUPPORT, GaitState.RIGHT_SUPPORT]):
            
            next_swing_leg = "right" if new_state == GaitState.LEFT_SUPPORT else "left"
            
            if self.config.enable_logging:
                print(f"[步态调度器] 准备{next_swing_leg}腿摆动，状态: {old_state.value} -> {new_state.value}")
    
    def start_walking(self):
        """开始行走"""
        with self._lock:
            if self.current_state == GaitState.STANDING:
                self._transition_to_state(GaitState.LEFT_SUPPORT)
            elif self.current_state == GaitState.IDLE:
                self._transition_to_state(GaitState.STANDING)
                # 短暂延迟后开始行走
                time.sleep(0.1)
                self._transition_to_state(GaitState.LEFT_SUPPORT)
    
    def stop_walking(self):
        """停止行走"""
        with self._lock:
            if self.current_state not in [GaitState.IDLE, GaitState.STANDING, 
                                         GaitState.EMERGENCY_STOP]:
                self._transition_to_state(GaitState.STANDING)
    
    def emergency_stop(self):
        """紧急停止"""
        with self._lock:
            self._transition_to_state(GaitState.EMERGENCY_STOP)
    
    def reset(self):
        """重置状态机"""
        with self._lock:
            self.current_state = GaitState.IDLE
            self.previous_state = GaitState.IDLE
            self.state_start_time = time.time()
            self.total_time = 0.0
            self.leg_state = LegState.DSt
            self.swing_leg = "none"
            self.support_leg = "both"
            
            # 清空历史记录
            self.state_history.clear()
            self.transition_log.clear()
    
    def get_current_state_info(self) -> StateInfo:
        """获取当前状态信息"""
        with self._lock:
            current_duration = time.time() - self.state_start_time
            return StateInfo(
                state=self.current_state,
                entry_time=self.state_start_time,
                duration=current_duration,
                leg_state=self.leg_state,
                swing_leg=self.swing_leg,
                support_leg=self.support_leg
            )
    
    def get_state_statistics(self) -> Dict:
        """获取状态统计信息"""
        with self._lock:
            stats = {}
            total_duration = 0.0
            
            # 统计各状态持续时间
            for state_info in self.state_history:
                state_name = state_info.state.value
                if state_name not in stats:
                    stats[state_name] = {
                        'count': 0,
                        'total_duration': 0.0,
                        'avg_duration': 0.0,
                        'min_duration': float('inf'),
                        'max_duration': 0.0
                    }
                
                stats[state_name]['count'] += 1
                stats[state_name]['total_duration'] += state_info.duration
                stats[state_name]['min_duration'] = min(
                    stats[state_name]['min_duration'], state_info.duration)
                stats[state_name]['max_duration'] = max(
                    stats[state_name]['max_duration'], state_info.duration)
                
                total_duration += state_info.duration
            
            # 计算平均值和百分比
            for state_name, data in stats.items():
                data['avg_duration'] = data['total_duration'] / data['count']
                data['percentage'] = (data['total_duration'] / total_duration * 100 
                                    if total_duration > 0 else 0)
            
            return {
                'state_stats': stats,
                'total_duration': total_duration,
                'total_transitions': len(self.state_history),
                'current_state': self.current_state.value,
                'current_duration': time.time() - self.state_start_time
            }
    
    def add_state_change_callback(self, callback: Callable):
        """添加状态变化回调函数"""
        self.state_change_callbacks.append(callback)
    
    def remove_state_change_callback(self, callback: Callable):
        """移除状态变化回调函数"""
        if callback in self.state_change_callbacks:
            self.state_change_callbacks.remove(callback)
    
    def print_status(self):
        """打印当前状态"""
        state_info = self.get_current_state_info()
        print(f"\n=== 步态调度器状态 ===")
        print(f"当前状态: {state_info.state.value}")
        print(f"支撑腿状态: {state_info.leg_state.value}")
        print(f"摆动腿: {state_info.swing_leg}")
        print(f"支撑腿: {state_info.support_leg}")
        print(f"状态持续时间: {state_info.duration:.3f}s")
        print(f"总运行时间: {self.total_time:.3f}s")
        print(f"左脚接触力: {self.left_foot_force:.1f}N")
        print(f"右脚接触力: {self.right_foot_force:.1f}N")
        print(f"左脚接触状态: {self.left_foot_contact}")
        print(f"右脚接触状态: {self.right_foot_contact}")
    
    # ================== 数据接口方法 ==================
    
    def get_gait_state_data(self) -> Dict:
        """
        获取完整的步态状态数据 - 供其他模块使用
        
        返回:
            Dict: 包含所有步态状态信息的字典
        """
        with self._lock:
            return {
                # 基本状态
                "current_state": self.current_state.value,
                "leg_state": self.leg_state.value,
                "swing_leg": self.swing_leg,
                "support_leg": self.support_leg,
                "previous_state": self.previous_state.value,
                
                # 时间信息
                "swing_elapsed_time": self.swing_elapsed_time,
                "stance_elapsed_time": self.stance_elapsed_time,
                "step_elapsed_time": self.step_elapsed_time,
                "total_time": self.total_time,
                "cycle_phase": self._calculate_cycle_phase(),
                "current_step_phase": self.current_step_phase,
                
                # 步态参数
                "swing_time": self.config.swing_time,
                "double_support_time": self.config.double_support_time,
                "stance_time": self.config.stance_time,
                
                # 传感器状态
                "left_foot_contact": self.left_foot_contact,
                "right_foot_contact": self.right_foot_contact,
                "left_foot_force": self.left_foot_force,
                "right_foot_force": self.right_foot_force,
                
                # 步完成信息
                "step_completion_pending": self.step_completion_pending,
                "next_swing_leg": self.next_swing_leg,
                
                # 状态转换信息
                "state_start_time": self.state_start_time,
                "state_transition_count": getattr(self, 'state_transition_count', 0)
            }
    
    def get_target_foot_positions(self) -> Dict[str, Dict[str, float]]:
        """
        获取目标足部位置 - 供足部轨迹规划使用
        
        返回:
            Dict: {"left_foot": {"x": float, "y": float, "z": float}, 
                   "right_foot": {"x": float, "y": float, "z": float}}
        """
        try:
            from data_bus import get_data_bus
            data_bus = get_data_bus()
            return data_bus.target_foot_pos.copy()
        except:
            # 默认位置
            return {
                "left_foot": {"x": 0.0, "y": 0.09, "z": 0.0},
                "right_foot": {"x": 0.0, "y": -0.09, "z": 0.0}
            }
    
    def get_timing_info(self) -> Dict:
        """
        获取步态时间信息 - 供MPC等控制器使用
        
        返回:
            Dict: 包含时间和相位信息
        """
        with self._lock:
            return {
                "current_time": self.total_time,
                "swing_time": self.swing_elapsed_time,
                "swing_duration": self.config.swing_time,
                "swing_progress": min(1.0, self.swing_elapsed_time / self.config.swing_time) if self.config.swing_time > 0 else 0.0,
                "cycle_phase": self._calculate_cycle_phase(),
                "time_to_swing_end": max(0.0, self.config.swing_time - self.swing_elapsed_time),
                "is_swing_phase": self.current_step_phase == "swing",
                "is_stance_phase": self.current_step_phase == "stance",
                "is_transition": self.current_step_phase == "transition",
                "dt": getattr(self, '_last_dt', 0.01)
            }
    
    def get_leg_states(self) -> Dict:
        """
        获取腿部状态信息 - 供运动学和动力学模块使用
        
        返回:
            Dict: 腿部状态信息
        """
        with self._lock:
            return {
                "leg_state": self.leg_state.value,
                "swing_leg": self.swing_leg,
                "support_leg": self.support_leg,
                "left_is_swing": self.swing_leg == "left",
                "right_is_swing": self.swing_leg == "right",
                "left_is_support": self.support_leg in ["left", "both"],
                "right_is_support": self.support_leg in ["right", "both"],
                "in_double_support": self.support_leg == "both",
                "in_single_support": self.support_leg in ["left", "right"]
            }
    
    def set_motion_command(self, forward_velocity: float = 0.0, 
                          lateral_velocity: float = 0.0, 
                          turning_rate: float = 0.0):
        """
        设置运动指令 - 从高层控制模块接收运动意图
        
        参数:
            forward_velocity: 前进速度 [m/s]
            lateral_velocity: 侧向速度 [m/s]  
            turning_rate: 转向速率 [rad/s]
        """
        try:
            from data_bus import get_data_bus
            data_bus = get_data_bus()
            data_bus.set_body_motion_intent(forward_velocity, lateral_velocity, turning_rate)
        except Exception as e:
            if self.config.enable_logging:
                print(f"Warning: 运动指令设置失败: {e}")
    
    def is_ready_for_new_step(self) -> bool:
        """
        检查是否准备开始新步 - 供高层规划器查询
        
        返回:
            bool: 是否可以开始新步
        """
        with self._lock:
            # 在双支撑相或即将完成摆动时可以规划新步
            return (self.current_step_phase in ["transition", "stance"] or 
                    self.swing_elapsed_time > self.config.swing_time * 0.8)
    
    def get_next_swing_prediction(self) -> Dict:
        """
        预测下一步摆动信息 - 供预测性控制使用
        
        返回:
            Dict: 下一步摆动预测信息
        """
        with self._lock:
            time_to_next_swing = 0.0
            
            if self.current_step_phase == "swing":
                # 当前在摆动，预测到下一个摆动的时间
                time_to_swing_end = max(0.0, self.config.swing_time - self.swing_elapsed_time)
                time_to_next_swing = time_to_swing_end + self.config.double_support_time
            elif self.current_step_phase == "transition":
                # 当前在双支撑，预测到下一个摆动的时间
                time_to_next_swing = self.config.double_support_time
            
            return {
                "next_swing_leg": self.next_swing_leg,
                "time_to_next_swing": time_to_next_swing,
                "next_swing_duration": self.config.swing_time,
                "predicted_swing_start_time": self.total_time + time_to_next_swing,
                "predicted_swing_end_time": self.total_time + time_to_next_swing + self.config.swing_time
            }


# 全局步态调度器实例
_global_gait_scheduler = None


def get_gait_scheduler(config: Optional[GaitSchedulerConfig] = None) -> GaitScheduler:
    """获取全局步态调度器实例"""
    global _global_gait_scheduler
    if _global_gait_scheduler is None:
        _global_gait_scheduler = GaitScheduler(config)
    return _global_gait_scheduler


if __name__ == "__main__":
    # 测试代码
    scheduler = get_gait_scheduler()
    
    print("步态调度器测试")
    scheduler.print_status()
    
    # 模拟传感器数据更新
    import numpy as np
    
    print("\n开始行走测试...")
    scheduler.start_walking()
    
    for i in range(50):
        # 模拟传感器数据
        if scheduler.swing_leg == "right":
            left_force = 80.0  # 左脚支撑
            right_force = 5.0 if i < 25 else 35.0  # 右脚后期着地
        elif scheduler.swing_leg == "left":
            right_force = 80.0  # 右脚支撑
            left_force = 5.0 if i % 50 < 25 else 35.0  # 左脚后期着地
        else:
            left_force = right_force = 40.0  # 双支撑
        
        left_vel = np.array([0.0, 0.0, 0.1 if scheduler.swing_leg == "left" else 0.0])
        right_vel = np.array([0.0, 0.0, 0.1 if scheduler.swing_leg == "right" else 0.0])
        
        scheduler.update_sensor_data(left_force, right_force, left_vel, right_vel)
        
        # 更新状态
        state_changed = scheduler.update_gait_state(0.02)  # 50Hz
        
        if state_changed:
            print(f"Step {i}: State changed to {scheduler.current_state.value}")
        
        time.sleep(0.02)  # 50Hz 仿真
        
        if i % 25 == 0:
            scheduler.print_status()
    
    print("\n最终统计:")
    stats = scheduler.get_state_statistics()
    for state, data in stats['state_stats'].items():
        print(f"{state}: {data['count']} 次, 平均 {data['avg_duration']:.3f}s, "
              f"占比 {data['percentage']:.1f}%") 