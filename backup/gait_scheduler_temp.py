import time
from typing import Dict
import numpy as np
from enum import Enum

class GaitState(Enum):
    LEFT_SUPPORT = 1
    RIGHT_SUPPORT = 2
    DOUBLE_SUPPORT_LR = 3
    DOUBLE_SUPPORT_RL = 4
    STANDING = 5

class GaitScheduler:
    def __init__(self, config):
        self.config = config
        self.current_state = GaitState.STANDING
        self.leg_state = GaitState.STANDING
        self.swing_leg = None
        self.support_leg = None
        self.step_count = 0
        self.cycle_count = 0
        self.total_time = 0.0
        self.current_gait_cycle_start = time.time()
        self._reset_gait_timers()

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

    def _update_gait_timers(self, dt: float):
        """
        更新步态计时器
        
        参数:
            dt: 时间步长 [s]
        """
        current_time = time.time()
        
        # 更新摆动相计时器
        if self.current_state in [GaitState.LEFT_SUPPORT, GaitState.RIGHT_SUPPORT]:
            if hasattr(self, '_swing_phase_started') and self._swing_phase_started:
                self.swing_timer += dt
                self.current_swing_duration = current_time - self.swing_start_time
            else:
                # 刚进入摆动相，重置计时器
                self.swing_timer = 0.0
                self.swing_start_time = current_time
                self.current_swing_duration = 0.0
                self._swing_phase_started = True
        else:
            self._swing_phase_started = False
        
        # 更新支撑相计时器
        if self.current_state in [GaitState.DOUBLE_SUPPORT_LR, GaitState.DOUBLE_SUPPORT_RL, 
                                 GaitState.STANDING]:
            if hasattr(self, '_stance_phase_started') and self._stance_phase_started:
                self.stance_timer += dt
                self.current_stance_duration = current_time - self.stance_start_time
            else:
                # 刚进入支撑相，重置计时器
                self.stance_timer = 0.0
                self.stance_start_time = current_time
                self.current_stance_duration = 0.0
                self._stance_phase_started = True
        else:
            self._stance_phase_started = False

    def _should_auto_transition(self, dt: float) -> bool:
        """
        检查是否应该自动进行状态转换
        
        参数:
            dt: 时间步长 [s]
            
        返回:
            bool: 是否应该转换状态
        """
        # 获取步态参数
        try:
            from data_bus import get_data_bus
            data_bus = get_data_bus()
            gait_params = data_bus.get_gait_parameters()
            t_swing = gait_params.timing.t_swing if gait_params else self.config.swing_time
        except:
            t_swing = self.config.swing_time
        
        # 检查摆动相是否完成
        if self.current_state in [GaitState.LEFT_SUPPORT, GaitState.RIGHT_SUPPORT]:
            # 时间条件：摆动时间超过设定值
            time_condition = self.swing_timer >= t_swing
            
            # 传感器条件：摆动脚着地
            sensor_condition = False
            if self.current_state == GaitState.LEFT_SUPPORT:
                # 右腿摆动，检查右脚着地
                sensor_condition = (self.right_foot_force > self.config.touchdown_force_threshold and
                                  np.linalg.norm(self.right_foot_velocity) < self.config.contact_velocity_threshold)
            elif self.current_state == GaitState.RIGHT_SUPPORT:
                # 左腿摆动，检查左脚着地
                sensor_condition = (self.left_foot_force > self.config.touchdown_force_threshold and
                                  np.linalg.norm(self.left_foot_velocity) < self.config.contact_velocity_threshold)
            
            # 混合判断：时间到或传感器检测到着地
            if self.config.use_time_trigger and self.config.use_sensor_trigger:
                if self.config.require_both_triggers:
                    return time_condition and sensor_condition
                else:
                    return time_condition or sensor_condition
            elif self.config.use_time_trigger:
                return time_condition
            elif self.config.use_sensor_trigger:
                return sensor_condition
        
        # 检查双支撑相是否完成
        elif self.current_state in [GaitState.DOUBLE_SUPPORT_LR, GaitState.DOUBLE_SUPPORT_RL]:
            return self.stance_timer >= self.config.double_support_time
        
        return False

    def _get_next_state(self) -> GaitState:
        """
        获取下一个状态
        
        返回:
            GaitState: 下一个应该转换到的状态
        """
        if self.current_state == GaitState.LEFT_SUPPORT:
            # 左支撑 -> 双支撑过渡（左到右）
            return GaitState.DOUBLE_SUPPORT_LR
        
        elif self.current_state == GaitState.RIGHT_SUPPORT:
            # 右支撑 -> 双支撑过渡（右到左）
            return GaitState.DOUBLE_SUPPORT_RL
        
        elif self.current_state == GaitState.DOUBLE_SUPPORT_LR:
            # 双支撑（左到右）-> 右支撑
            return GaitState.RIGHT_SUPPORT
        
        elif self.current_state == GaitState.DOUBLE_SUPPORT_RL:
            # 双支撑（右到左）-> 左支撑
            return GaitState.LEFT_SUPPORT
        
        # 其他状态保持不变
        return self.current_state

    def _mark_step_completed(self):
        """
        标记步态完成事件
        """
        self.step_count += 1
        
        # 检查是否完成了一个完整的步态周期
        # 一个完整周期包含：左支撑-双支撑-右支撑-双支撑
        if self.step_count % 2 == 0:  # 每完成两步算一个完整周期
            self.cycle_count += 1
            
            # 重置周期计时
            self.current_gait_cycle_start = time.time()
        
        # 更新数据总线中的步态事件标记
        try:
            from data_bus import get_data_bus
            data_bus = get_data_bus()
            
            # 标记步态切换事件
            if hasattr(data_bus, 'step_finished'):
                data_bus.step_finished = True
            
            # 更新步数计数器
            if hasattr(data_bus, 'step_count'):
                data_bus.step_count = self.step_count
            
            # 更新周期计数器
            if hasattr(data_bus, 'cycle_count'):
                data_bus.cycle_count = self.cycle_count
            
            # 更新当前腿状态
            if hasattr(data_bus, 'leg_state'):
                data_bus.leg_state = self.leg_state.value
            
            if self.config.enable_logging:
                print(f"[步态调度器] 步态完成: 步数={self.step_count}, 周期={self.cycle_count}, "
                      f"legState={self.leg_state.value}")
                
        except Exception as e:
            if self.config.enable_logging:
                print(f"Warning: 更新数据总线步态事件失败: {e}")

    def _reset_gait_timers(self):
        """重置步态计时器"""
        self.swing_timer = 0.0
        self.stance_timer = 0.0
        self.current_swing_duration = 0.0
        self.current_stance_duration = 0.0
        self.swing_start_time = 0.0
        self.stance_start_time = 0.0
        self._swing_phase_started = False
        self._stance_phase_started = False

    def get_gait_timing_info(self) -> Dict:
        """
        获取步态计时信息
        
        返回:
            Dict: 包含各种计时信息的字典
        """
        return {
            'swing_timer': self.swing_timer,
            'stance_timer': self.stance_timer,
            'current_swing_duration': self.current_swing_duration,
            'current_stance_duration': self.current_stance_duration,
            'step_count': self.step_count,
            'cycle_count': self.cycle_count,
            'total_time': self.total_time,
            'current_state': self.current_state.value,
            'leg_state': self.leg_state.value,
            'swing_leg': self.swing_leg,
            'support_leg': self.support_leg
        }