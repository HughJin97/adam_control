#!/usr/bin/env python3
"""
超稳定版单脚支撑平衡仿真
进一步优化控制参数，确保机器人不跳跃不抽搐
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import deque


class StableSingleFootBalanceSimulation:
    """超稳定版单脚平衡仿真类"""
    
    def __init__(self, model_path: str = "models/scene.xml"):
        """初始化仿真"""
        self.model_path = model_path
        self.model = None
        self.data = None
        self.viewer = None
        
        # 仿真参数
        self.dt = 0.001  # 1ms仿真步长
        self.control_dt = 0.008  # 8ms控制步长 (125Hz)
        self.simulation_time = 0.0
        self.last_control_time = 0.0
        
        # 机器人状态
        self.robot_mass = 45.0  # kg，估计机器人质量
        self.gravity = 9.81
        self.support_foot = "right"  # 支撑脚：'left' 或 'right'
        
        # 关节名称映射
        self.joint_names = [
            'J_waist_pitch', 'J_waist_roll', 'J_waist_yaw',
            'J_hip_l_roll', 'J_hip_l_yaw', 'J_hip_l_pitch', 'J_knee_l_pitch', 'J_ankle_l_pitch', 'J_ankle_l_roll',
            'J_hip_r_roll', 'J_hip_r_yaw', 'J_hip_r_pitch', 'J_knee_r_pitch', 'J_ankle_r_pitch', 'J_ankle_r_roll',
            'J_arm_l_01', 'J_arm_l_02', 'J_arm_l_03', 'J_arm_l_04', 'J_arm_l_05', 'J_arm_l_06', 'J_arm_l_07',
            'J_arm_r_01', 'J_arm_r_02', 'J_arm_r_03', 'J_arm_r_04', 'J_arm_r_05', 'J_arm_r_06', 'J_arm_r_07',
            'J_head_yaw', 'J_head_pitch'
        ]
        
        # 超保守的控制器参数
        self.kp_joint = 10.0   # 关节位置增益 (进一步降低到10)
        self.kd_joint = 2.0    # 关节速度增益 (降低到2)
        self.kp_balance = 20.0 # 平衡控制增益 (降低到20)
        self.kd_balance = 5.0  # 平衡阻尼增益 (降低到5)
        
        # 更严格的力矩限制
        self.max_joint_torque = 30.0  # 最大关节力矩限制降到30
        
        # 数据记录
        self.time_history = deque(maxlen=1000)
        self.com_history = deque(maxlen=1000)
        self.force_history = deque(maxlen=1000)
        self.balance_error_history = deque(maxlen=1000)
        self.joint_torque_history = deque(maxlen=1000)
        
        # 目标姿态（单脚支撑）
        self.target_joint_positions = self._get_stable_single_foot_stance_pose()
        
        # 运行标志
        self.running = False
        
    def _get_stable_single_foot_stance_pose(self) -> Dict[str, float]:
        """获取超稳定的单脚支撑目标关节角度"""
        pose = {}
        
        # 腰部关节 - 保持完全直立
        pose['J_waist_pitch'] = 0.0
        pose['J_waist_roll'] = 0.0
        pose['J_waist_yaw'] = 0.0
        
        if self.support_foot == "right":
            # 右脚支撑，左脚抬起 - 使用极保守的角度
            # 右腿 - 支撑腿，几乎完全直立
            pose['J_hip_r_roll'] = 0.0
            pose['J_hip_r_yaw'] = 0.0
            pose['J_hip_r_pitch'] = -0.02  # 极小的前倾角度
            pose['J_knee_r_pitch'] = 0.05  # 极小的弯曲角度
            pose['J_ankle_r_pitch'] = -0.03
            pose['J_ankle_r_roll'] = 0.0
            
            # 左腿 - 摆动腿，最小抬起幅度
            pose['J_hip_l_roll'] = 0.0
            pose['J_hip_l_yaw'] = 0.0
            pose['J_hip_l_pitch'] = 0.3    # 进一步减小抬起角度
            pose['J_knee_l_pitch'] = -0.5  # 进一步减小弯曲角度
            pose['J_ankle_l_pitch'] = 0.2  # 进一步减小角度
            pose['J_ankle_l_roll'] = 0.0
        else:
            # 左脚支撑，右脚抬起
            # 左腿 - 支撑腿
            pose['J_hip_l_roll'] = 0.0
            pose['J_hip_l_yaw'] = 0.0
            pose['J_hip_l_pitch'] = -0.02
            pose['J_knee_l_pitch'] = 0.05
            pose['J_ankle_l_pitch'] = -0.03
            pose['J_ankle_l_roll'] = 0.0
            
            # 右腿 - 摆动腿
            pose['J_hip_r_roll'] = 0.0
            pose['J_hip_r_yaw'] = 0.0
            pose['J_hip_r_pitch'] = 0.3
            pose['J_knee_r_pitch'] = -0.5
            pose['J_ankle_r_pitch'] = 0.2
            pose['J_ankle_r_roll'] = 0.0
        
        # 手臂 - 使用最保守的平衡姿态
        pose['J_arm_l_01'] = 0.1   # 进一步减小角度
        pose['J_arm_l_02'] = 0.1
        pose['J_arm_l_03'] = -0.1
        pose['J_arm_l_04'] = 0.2   # 进一步减小角度
        pose['J_arm_l_05'] = 0.0
        pose['J_arm_l_06'] = 0.0
        pose['J_arm_l_07'] = 0.0
        
        pose['J_arm_r_01'] = -0.1  # 进一步减小角度
        pose['J_arm_r_02'] = -0.1
        pose['J_arm_r_03'] = 0.1
        pose['J_arm_r_04'] = 0.2   # 进一步减小角度
        pose['J_arm_r_05'] = 0.0
        pose['J_arm_r_06'] = 0.0
        pose['J_arm_r_07'] = 0.0
        
        # 头部 - 保持中性
        pose['J_head_yaw'] = 0.0
        pose['J_head_pitch'] = 0.0
        
        return pose
    
    def load_model(self):
        """加载Mujoco模型"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            print(f"成功加载模型: {self.model_path}")
            print(f"模型有 {self.model.nq} 个广义坐标, {self.model.nv} 个广义速度")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def initialize_robot_pose(self):
        """超稳定的机器人初始化"""
        # 设置基座位置和姿态 - 进一步降低初始高度
        base_pos_idx = self.model.jnt_qposadr[0]  # float_base关节的位置索引
        self.data.qpos[base_pos_idx:base_pos_idx+3] = [0.0, 0.0, 0.85]  # 进一步降低高度到0.85m
        self.data.qpos[base_pos_idx+3:base_pos_idx+7] = [1.0, 0.0, 0.0, 0.0]  # 四元数 (w, x, y, z)
        
        # 设置关节角度
        for joint_name, target_angle in self.target_joint_positions.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                joint_qpos_idx = self.model.jnt_qposadr[joint_id]
                self.data.qpos[joint_qpos_idx] = target_angle
        
        # 清零速度
        self.data.qvel[:] = 0.0
        
        # 前向运动学计算
        mujoco.mj_forward(self.model, self.data)
        
        # 让机器人更长时间稳定
        for _ in range(200):  # 增加到200步
            mujoco.mj_step(self.model, self.data)
        
        print("机器人初始化到超稳定的单脚支撑姿态")
    
    def get_robot_state(self) -> Dict:
        """获取机器人当前状态"""
        # 基座位置和姿态
        base_pos = self.data.qpos[0:3].copy()
        base_quat = self.data.qpos[3:7].copy()
        base_vel = self.data.qvel[0:3].copy()
        base_angvel = self.data.qvel[3:6].copy()
        
        # 质心位置（近似为基座位置）
        com_pos = base_pos.copy()
        com_vel = base_vel.copy()
        
        # 支撑脚位置
        if self.support_foot == "right":
            foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "rf-tc")
        else:
            foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lf-tc")
        
        if foot_site_id >= 0:
            foot_pos = self.data.site_xpos[foot_site_id].copy()
        else:
            foot_pos = np.array([0.0, 0.0, 0.0])
        
        return {
            'base_position': base_pos,
            'base_orientation': base_quat,
            'base_velocity': base_vel,
            'base_angular_velocity': base_angvel,
            'com_position': com_pos,
            'com_velocity': com_vel,
            'support_foot_position': foot_pos,
            'time': self.simulation_time
        }
    
    def compute_balance_control(self, robot_state: Dict) -> Dict:
        """超保守的平衡控制计算"""
        # 计算平衡误差
        com_pos = robot_state['com_position']
        foot_pos = robot_state['support_foot_position']
        
        # 期望质心在支撑脚正上方
        desired_com_x = foot_pos[0]
        desired_com_y = foot_pos[1]
        
        balance_error_x = com_pos[0] - desired_com_x
        balance_error_y = com_pos[1] - desired_com_y
        
        # 进一步限制平衡误差的影响范围
        max_error = 0.05  # 最大误差限制为5cm
        balance_error_x = np.clip(balance_error_x, -max_error, max_error)
        balance_error_y = np.clip(balance_error_y, -max_error, max_error)
        
        # 计算所需的平衡力
        com_vel = robot_state['com_velocity']
        
        # 进一步限制速度的影响
        max_vel = 0.2  # 最大速度限制降到0.2
        com_vel_x = np.clip(com_vel[0], -max_vel, max_vel)
        com_vel_y = np.clip(com_vel[1], -max_vel, max_vel)
        
        # 计算期望的地面反力（超保守的控制）
        desired_force_x = -self.kp_balance * balance_error_x - self.kd_balance * com_vel_x
        desired_force_y = -self.kp_balance * balance_error_y - self.kd_balance * com_vel_y
        desired_force_z = self.robot_mass * self.gravity  # 重力补偿
        
        # 进一步限制水平力的大小
        max_horizontal_force = 10.0  # 进一步减小最大水平力到10N
        desired_force_x = np.clip(desired_force_x, -max_horizontal_force, max_horizontal_force)
        desired_force_y = np.clip(desired_force_y, -max_horizontal_force, max_horizontal_force)
        
        return {
            'balance_error': np.array([balance_error_x, balance_error_y]),
            'desired_force': np.array([desired_force_x, desired_force_y, desired_force_z]),
            'com_position': com_pos,
            'foot_position': foot_pos
        }
    
    def compute_joint_torques(self, robot_state: Dict, balance_control: Dict) -> np.ndarray:
        """超保守的关节力矩计算"""
        torques = np.zeros(self.model.nu)
        
        # 基本的PD控制器维持目标姿态
        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0 and joint_id < len(self.model.jnt_qposadr):
                qpos_idx = self.model.jnt_qposadr[joint_id]
                qvel_idx = self.model.jnt_dofadr[joint_id]
                
                if qpos_idx < len(self.data.qpos) and qvel_idx < len(self.data.qvel):
                    current_pos = self.data.qpos[qpos_idx]
                    current_vel = self.data.qvel[qvel_idx]
                    target_pos = self.target_joint_positions.get(joint_name, 0.0)
                    
                    # 对于支撑腿的踝关节，添加极温和的平衡控制
                    balance_adjustment = 0.0
                    if self.support_foot == "right" and joint_name in ['J_ankle_r_pitch', 'J_ankle_r_roll']:
                        if joint_name == 'J_ankle_r_pitch':
                            balance_adjustment = balance_control['balance_error'][0] * 0.01  # 进一步减小增益
                        elif joint_name == 'J_ankle_r_roll':
                            balance_adjustment = balance_control['balance_error'][1] * 0.01
                    elif self.support_foot == "left" and joint_name in ['J_ankle_l_pitch', 'J_ankle_l_roll']:
                        if joint_name == 'J_ankle_l_pitch':
                            balance_adjustment = balance_control['balance_error'][0] * 0.01
                        elif joint_name == 'J_ankle_l_roll':
                            balance_adjustment = balance_control['balance_error'][1] * 0.01
                    
                    # PD控制 + 平衡调整
                    pos_error = target_pos - current_pos + balance_adjustment
                    vel_error = 0.0 - current_vel
                    
                    # 查找对应的执行器
                    actuator_name = joint_name.replace('J_', 'M_')
                    actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                    
                    if actuator_id >= 0:
                        torque = self.kp_joint * pos_error + self.kd_joint * vel_error
                        
                        # 更严格的力矩限制
                        torque = np.clip(torque, -self.max_joint_torque, self.max_joint_torque)
                        
                        torques[actuator_id] = torque
        
        return torques
    
    def step_simulation(self):
        """执行一步仿真"""
        # 获取机器人状态
        robot_state = self.get_robot_state()
        
        # 控制频率检查
        if self.simulation_time - self.last_control_time >= self.control_dt:
            # 计算平衡控制
            balance_control = self.compute_balance_control(robot_state)
            
            # 计算关节力矩
            torques = self.compute_joint_torques(robot_state, balance_control)
            
            # 应用控制力矩
            self.data.ctrl[:] = torques
            
            # 记录数据
            self.time_history.append(self.simulation_time)
            self.com_history.append(robot_state['com_position'].copy())
            self.force_history.append(balance_control['desired_force'].copy())
            self.balance_error_history.append(balance_control['balance_error'].copy())
            self.joint_torque_history.append(np.linalg.norm(torques))
            
            self.last_control_time = self.simulation_time
        
        # 执行物理仿真步
        mujoco.mj_step(self.model, self.data)
        self.simulation_time += self.dt
    
    def run_simulation(self, duration: float = 10.0, use_viewer: bool = True):
        """运行仿真"""
        if not self.load_model():
            return
        
        self.initialize_robot_pose()
        
        if use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        print(f"开始超稳定版单脚平衡仿真 (支撑脚: {self.support_foot})")
        print(f"仿真时长: {duration}秒")
        print("控制参数已极度优化，确保无跳跃无抽搐")
        print("按 Ctrl+C 停止仿真")
        
        self.running = True
        start_time = time.time()
        step_count = 0
        
        try:
            while self.running and (time.time() - start_time) < duration:
                self.step_simulation()
                step_count += 1
                
                if use_viewer and self.viewer is not None:
                    self.viewer.sync()
                    # 控制仿真速度，避免过快
                    if step_count % 10 == 0:  # 每10步暂停一次
                        time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n仿真被用户中断")
        
        finally:
            self.running = False
            if self.viewer is not None:
                self.viewer.close()
            
            actual_time = time.time() - start_time
            print(f"仿真结束，仿真时长: {self.simulation_time:.2f}秒，实际用时: {actual_time:.2f}秒")
            self.print_simulation_summary()
    
    def print_simulation_summary(self):
        """打印仿真总结"""
        if len(self.balance_error_history) > 0:
            errors = np.array(list(self.balance_error_history))
            avg_error = np.mean(np.linalg.norm(errors, axis=1))
            max_error = np.max(np.linalg.norm(errors, axis=1))
            
            print(f"\n=== 超稳定版仿真总结 ===")
            print(f"支撑脚: {self.support_foot}")
            print(f"平均平衡误差: {avg_error:.4f} m")
            print(f"最大平衡误差: {max_error:.4f} m")
            print(f"控制频率: {1.0/self.control_dt:.1f} Hz")
            print(f"数据记录点数: {len(self.time_history)}")
            
            if len(self.joint_torque_history) > 0:
                torques = np.array(list(self.joint_torque_history))
                avg_torque = np.mean(torques)
                max_torque = np.max(torques)
                print(f"平均关节力矩范数: {avg_torque:.2f} Nm")
                print(f"最大关节力矩范数: {max_torque:.2f} Nm")
                
                # 检查是否有改进
                if max_torque < 100:
                    print("✅ 关节力矩控制优秀")
                elif max_torque < 200:
                    print("✅ 关节力矩控制良好")
                else:
                    print("⚠️  关节力矩仍然较大")
                
                if avg_error < 0.05:
                    print("✅ 平衡控制优秀")
                elif avg_error < 0.1:
                    print("✅ 平衡控制良好")
                else:
                    print("⚠️  平衡误差仍然较大")
    
    def plot_results(self):
        """绘制仿真结果"""
        if len(self.time_history) == 0:
            print("没有数据可绘制")
            return
        
        times = np.array(list(self.time_history))
        com_positions = np.array(list(self.com_history))
        forces = np.array(list(self.force_history))
        errors = np.array(list(self.balance_error_history))
        torques = np.array(list(self.joint_torque_history))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'超稳定版单脚平衡仿真结果 (支撑脚: {self.support_foot})')
        
        # 质心位置
        axes[0, 0].plot(times, com_positions[:, 0], label='X', color='red')
        axes[0, 0].plot(times, com_positions[:, 1], label='Y', color='green')
        axes[0, 0].plot(times, com_positions[:, 2], label='Z', color='blue')
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('质心位置 (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_title('质心位置')
        
        # 平衡误差
        axes[0, 1].plot(times, errors[:, 0], label='X误差', color='red')
        axes[0, 1].plot(times, errors[:, 1], label='Y误差', color='green')
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('平衡误差 (m)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_title('平衡误差')
        
        # 控制力
        axes[0, 2].plot(times, forces[:, 0], label='Fx', color='red')
        axes[0, 2].plot(times, forces[:, 1], label='Fy', color='green')
        axes[0, 2].plot(times, forces[:, 2], label='Fz', color='blue')
        axes[0, 2].set_xlabel('时间 (s)')
        axes[0, 2].set_ylabel('控制力 (N)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        axes[0, 2].set_title('控制力')
        
        # 平衡误差范数
        error_norms = np.linalg.norm(errors, axis=1)
        axes[1, 0].plot(times, error_norms, color='purple')
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('平衡误差范数 (m)')
        axes[1, 0].grid(True)
        axes[1, 0].set_title('平衡误差范数')
        
        # 关节力矩范数
        axes[1, 1].plot(times, torques, color='orange')
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('关节力矩范数 (Nm)')
        axes[1, 1].grid(True)
        axes[1, 1].set_title('关节力矩范数')
        
        # 质心轨迹 (XY平面)
        axes[1, 2].plot(com_positions[:, 0], com_positions[:, 1], color='blue', alpha=0.7)
        axes[1, 2].scatter(com_positions[0, 0], com_positions[0, 1], color='green', s=50, label='起始点')
        axes[1, 2].scatter(com_positions[-1, 0], com_positions[-1, 1], color='red', s=50, label='结束点')
        axes[1, 2].set_xlabel('X位置 (m)')
        axes[1, 2].set_ylabel('Y位置 (m)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        axes[1, 2].set_title('质心轨迹 (XY平面)')
        axes[1, 2].axis('equal')
        
        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='超稳定版单脚平衡仿真')
    parser.add_argument('--support-foot', choices=['left', 'right'], default='right',
                        help='支撑脚选择 (默认: right)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='仿真时长 (秒, 默认: 10.0)')
    parser.add_argument('--no-viewer', action='store_true',
                        help='不使用可视化界面')
    parser.add_argument('--plot', action='store_true',
                        help='仿真结束后绘制结果')
    
    args = parser.parse_args()
    
    # 创建仿真实例
    sim = StableSingleFootBalanceSimulation()
    sim.support_foot = args.support_foot
    sim.target_joint_positions = sim._get_stable_single_foot_stance_pose()
    
    # 运行仿真
    sim.run_simulation(
        duration=args.duration,
        use_viewer=not args.no_viewer
    )
    
    # 绘制结果
    if args.plot:
        sim.plot_results()


if __name__ == "__main__":
    main() 