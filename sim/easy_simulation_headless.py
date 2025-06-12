#!/usr/bin/env python3
"""
Easy Simulation (Headless) - 无GUI仿真调试脚本

用于在没有GUI的情况下调试数据总线和控制系统：
1. 验证数据总线读写正确性
2. 对比MuJoCo内部数据与数据总线数据
3. 检查数据更新频率与仿真步长匹配
4. 测试不同控制模式的响应

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import mujoco
import time
from typing import Dict, List

# 导入控制模块
from data_bus import get_data_bus, Vector3D, ContactState
from sensor_reader import SensorReader
from control_sender import ControlSender, ControlMode


class HeadlessSimulation:
    """无GUI仿真调试类"""
    
    def __init__(self, model_path: str = "models/scene.xml"):
        """初始化仿真"""
        self.model_path = model_path
        self.model = None
        self.data = None
        
        # 控制组件
        self.sensor_reader = None
        self.control_sender = None
        self.data_bus = get_data_bus()
        
        # 仿真参数
        self.sim_time = 0.0
        self.sim_step_time = 0.001  # 1ms 仿真步长
        
        # 调试参数
        self.debug_joints = ["J_hip_l_pitch", "J_hip_r_pitch", "J_knee_l_pitch", "J_knee_r_pitch"]
        
        # 统计信息
        self.stats = {
            "sim_steps": 0,
            "control_cycles": 0,
            "sensor_reads": 0,
            "control_sends": 0,
            "max_joint_error": 0.0,
            "total_joint_error": 0.0,
            "start_time": 0.0
        }
        
        print("HeadlessSimulation initialized")
        self._load_model()
        self._setup_simulation()
    
    def _load_model(self):
        """加载MuJoCo模型"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            print(f"✓ 模型加载成功: {self.model_path}")
            print(f"  - DOF: {self.model.nq}")
            print(f"  - 执行器: {self.model.nu}")
            print(f"  - 传感器: {self.model.nsensor}")
            
            # 设置合理的初始姿态
            self._set_initial_pose()
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise
    
    def _set_initial_pose(self):
        """设置机器人的初始姿态"""
        # 设置一个稳定的站立姿态
        initial_joint_positions = {
            # 左腿
            "J_hip_l_pitch": -0.05,   # 髋关节稍微前倾
            "J_knee_l_pitch": 0.1,    # 膝关节稍微弯曲
            "J_ankle_l_pitch": -0.05, # 踝关节补偿
            
            # 右腿
            "J_hip_r_pitch": -0.05,
            "J_knee_r_pitch": 0.1,
            "J_ankle_r_pitch": -0.05,
            
            # 手臂
            "J_arm_l_02": 0.1,
            "J_arm_r_02": -0.1,
        }
        
        # 应用初始位置
        for joint_name, position in initial_joint_positions.items():
            try:
                joint_id = self.model.joint(joint_name).id
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_idx] = position
            except:
                pass
        
        # 前向动力学计算
        mujoco.mj_forward(self.model, self.data)
        print("✓ 初始姿态设置完成")
    
    def _setup_simulation(self):
        """设置仿真组件"""
        self.sensor_reader = SensorReader(self.model, self.data)
        self.control_sender = ControlSender(self.model, self.data)
        self.control_sender.set_control_mode(ControlMode.TORQUE)
        print("✓ 仿真组件设置完成")
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print(f"\n=== 开始综合测试 ===")
        
        # 1. 数据一致性测试
        self.test_data_consistency()
        
        # 2. 零力矩仿真测试
        self.test_zero_torque_simulation()
        
        # 3. 位置保持控制测试
        self.test_position_hold_control()
        
        # 4. 重力补偿测试
        self.test_gravity_compensation()
        
        # 5. 频率性能测试
        self.test_frequency_performance()
        
        print(f"\n=== 综合测试完成 ===")
    
    def test_data_consistency(self, num_samples: int = 200):
        """测试数据总线一致性"""
        print(f"\n=== 1. 数据一致性测试 ===")
        print(f"测试样本数: {num_samples}")
        
        errors = []
        max_error = 0.0
        
        for i in range(num_samples):
            # 随机设置关节位置
            for joint_name in self.debug_joints:
                try:
                    joint_id = self.model.joint(joint_name).id
                    qpos_idx = self.model.jnt_qposadr[joint_id]
                    self.data.qpos[qpos_idx] = np.random.uniform(-0.8, 0.8)
                except:
                    pass
            
            # 前向动力学
            mujoco.mj_forward(self.model, self.data)
            
            # 读取传感器
            self.sensor_reader.read_sensors()
            
            # 检查一致性
            for joint_name in self.debug_joints:
                mujoco_pos = self._get_mujoco_joint_position(joint_name)
                databus_pos = self.data_bus.get_joint_position(joint_name)
                
                if databus_pos is not None:
                    error = abs(mujoco_pos - databus_pos)
                    errors.append(error)
                    max_error = max(max_error, error)
        
        if errors:
            print(f"平均误差: {np.mean(errors):.10f}")
            print(f"最大误差: {np.max(errors):.10f}")
            print(f"标准差: {np.std(errors):.10f}")
            
            if max_error < 1e-10:
                print("✓ 数据一致性测试：完美")
            elif max_error < 1e-6:
                print("✓ 数据一致性测试：通过")
            else:
                print("✗ 数据一致性测试：失败")
        
        return max_error < 1e-6
    
    def test_zero_torque_simulation(self, duration: float = 3.0):
        """测试零力矩仿真"""
        print(f"\n=== 2. 零力矩仿真测试 ===")
        print(f"仿真时长: {duration}s (机器人应该在重力作用下倒塌)")
        
        # 重置到初始状态
        self._set_initial_pose()
        
        # 记录初始状态
        initial_com_height = self.data.subtree_com[0][2]  # 质心高度
        
        self.sim_time = 0.0
        self.stats["sim_steps"] = 0
        start_time = time.time()
        
        while self.sim_time < duration:
            # 零力矩控制
            for joint_name in self.data_bus.joint_names:
                self.data_bus.set_control_torque(joint_name, 0.0)
            
            # 读取传感器
            self.sensor_reader.read_sensors()
            
            # 发送控制指令
            self.control_sender.send_commands()
            
            # 仿真步进
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.sim_step_time
            self.stats["sim_steps"] += 1
        
        # 检查结果
        final_com_height = self.data.subtree_com[0][2]
        height_drop = initial_com_height - final_com_height
        
        elapsed_time = time.time() - start_time
        avg_fps = self.stats["sim_steps"] / elapsed_time
        
        print(f"初始质心高度: {initial_com_height:.3f}m")
        print(f"最终质心高度: {final_com_height:.3f}m")
        print(f"质心下降: {height_drop:.3f}m")
        print(f"平均仿真频率: {avg_fps:.1f} Hz")
        
        if height_drop > 0.05:  # 质心下降超过5cm
            print("✓ 零力矩测试：机器人正常倒塌")
        else:
            print("? 零力矩测试：机器人意外稳定")
        
        return height_drop > 0.05
    
    def test_position_hold_control(self, duration: float = 2.0):
        """测试位置保持控制"""
        print(f"\n=== 3. 位置保持控制测试 ===")
        print(f"仿真时长: {duration}s (机器人应该保持初始姿态)")
        
        # 重置到初始状态
        self._set_initial_pose()
        
        # 记录目标位置
        target_positions = {}
        for joint_name in self.debug_joints:
            target_positions[joint_name] = self._get_mujoco_joint_position(joint_name)
        
        self.sim_time = 0.0
        position_errors = []
        max_position_error = 0.0
        
        while self.sim_time < duration:
            # 位置保持控制
            kp = 100.0
            kd = 10.0
            
            for joint_name in self.data_bus.joint_names:
                current_pos = self.data_bus.get_joint_position(joint_name) or 0.0
                current_vel = self.data_bus.get_joint_velocity(joint_name) or 0.0
                
                # 对于关键关节使用记录的目标位置，其他关节保持当前位置
                if joint_name in target_positions:
                    target_pos = target_positions[joint_name]
                else:
                    target_pos = current_pos
                
                # PD控制
                pos_error = target_pos - current_pos
                vel_error = 0.0 - current_vel
                control_torque = kp * pos_error + kd * vel_error
                
                # 限制力矩
                control_torque = np.clip(control_torque, -100.0, 100.0)
                self.data_bus.set_control_torque(joint_name, control_torque)
                
                # 记录误差
                if joint_name in self.debug_joints:
                    error = abs(pos_error)
                    position_errors.append(error)
                    max_position_error = max(max_position_error, error)
            
            # 读取传感器和发送控制
            self.sensor_reader.read_sensors()
            self.control_sender.send_commands()
            
            # 仿真步进
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.sim_step_time
        
        # 分析结果
        if position_errors:
            avg_error = np.mean(position_errors)
            print(f"平均位置误差: {avg_error:.6f} rad")
            print(f"最大位置误差: {max_position_error:.6f} rad")
            
            if max_position_error < 0.01:  # 1度以内
                print("✓ 位置保持测试：优秀")
            elif max_position_error < 0.05:  # 3度以内
                print("✓ 位置保持测试：良好")
            else:
                print("△ 位置保持测试：需要调整参数")
        
        return max_position_error < 0.05
    
    def test_gravity_compensation(self, duration: float = 2.0):
        """测试重力补偿"""
        print(f"\n=== 4. 重力补偿测试 ===")
        print(f"仿真时长: {duration}s (机器人应该悬浮在空中)")
        
        # 重置到初始状态
        self._set_initial_pose()
        
        initial_com_height = self.data.subtree_com[0][2]
        initial_positions = {}
        for joint_name in self.debug_joints:
            initial_positions[joint_name] = self._get_mujoco_joint_position(joint_name)
        
        self.sim_time = 0.0
        com_height_history = []
        
        while self.sim_time < duration:
            # 重力补偿控制
            # 设置零加速度并计算所需力矩
            self.data.qacc[:] = 0.0
            mujoco.mj_inverse(self.model, self.data)
            
            # 将计算的力矩设置到数据总线
            for joint_name in self.data_bus.joint_names:
                if joint_name in self.control_sender._joint_to_actuator:
                    actuator_id = self.control_sender._joint_to_actuator[joint_name]['actuator_id']
                    if actuator_id < len(self.data.qfrc_inverse):
                        gravity_torque = self.data.qfrc_inverse[actuator_id]
                        self.data_bus.set_control_torque(joint_name, gravity_torque)
            
            # 读取传感器和发送控制
            self.sensor_reader.read_sensors()
            self.control_sender.send_commands()
            
            # 仿真步进
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.sim_step_time
            
            # 记录质心高度
            com_height_history.append(self.data.subtree_com[0][2])
        
        # 分析结果
        final_com_height = self.data.subtree_com[0][2]
        height_change = abs(final_com_height - initial_com_height)
        height_std = np.std(com_height_history) if com_height_history else 0.0
        
        print(f"初始质心高度: {initial_com_height:.4f}m")
        print(f"最终质心高度: {final_com_height:.4f}m")
        print(f"高度变化: {height_change:.4f}m")
        print(f"高度标准差: {height_std:.4f}m")
        
        # 检查关节位置漂移
        max_joint_drift = 0.0
        for joint_name in self.debug_joints:
            current_pos = self._get_mujoco_joint_position(joint_name)
            initial_pos = initial_positions[joint_name]
            drift = abs(current_pos - initial_pos)
            max_joint_drift = max(max_joint_drift, drift)
        
        print(f"最大关节漂移: {max_joint_drift:.4f} rad")
        
        if height_change < 0.01 and max_joint_drift < 0.02:
            print("✓ 重力补偿测试：优秀")
            return True
        elif height_change < 0.05 and max_joint_drift < 0.1:
            print("✓ 重力补偿测试：良好")
            return True
        else:
            print("△ 重力补偿测试：需要改进")
            return False
    
    def test_frequency_performance(self, duration: float = 1.0):
        """测试频率性能"""
        print(f"\n=== 5. 频率性能测试 ===")
        print(f"测试时长: {duration}s")
        
        self.sim_time = 0.0
        step_times = []
        sensor_times = []
        control_times = []
        
        start_time = time.time()
        
        while self.sim_time < duration:
            step_start = time.time()
            
            # 传感器读取
            sensor_start = time.time()
            self.sensor_reader.read_sensors()
            sensor_time = time.time() - sensor_start
            sensor_times.append(sensor_time)
            
            # 控制计算 (简单PD控制)
            control_start = time.time()
            for joint_name in self.data_bus.joint_names:
                current_pos = self.data_bus.get_joint_position(joint_name) or 0.0
                current_vel = self.data_bus.get_joint_velocity(joint_name) or 0.0
                control_torque = -10.0 * current_pos - 1.0 * current_vel
                control_torque = np.clip(control_torque, -50.0, 50.0)
                self.data_bus.set_control_torque(joint_name, control_torque)
            control_time = time.time() - control_start
            control_times.append(control_time)
            
            # 发送控制指令
            self.control_sender.send_commands()
            
            # 仿真步进
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.sim_step_time
            
            step_time = time.time() - step_start
            step_times.append(step_time)
        
        # 性能分析
        total_time = time.time() - start_time
        num_steps = len(step_times)
        
        print(f"总步数: {num_steps}")
        print(f"实际频率: {num_steps/total_time:.1f} Hz")
        print(f"平均步长时间: {np.mean(step_times)*1000:.2f} ms")
        print(f"最大步长时间: {np.max(step_times)*1000:.2f} ms")
        print(f"平均传感器时间: {np.mean(sensor_times)*1000:.2f} ms")
        print(f"平均控制时间: {np.mean(control_times)*1000:.2f} ms")
        
        target_freq = 1000.0  # 目标1000Hz
        achieved_freq = num_steps / total_time
        
        if achieved_freq >= target_freq * 0.8:
            print("✓ 性能测试：良好")
            return True
        else:
            print("△ 性能测试：需要优化")
            return False
    
    def _get_mujoco_joint_position(self, joint_name: str) -> float:
        """直接从MuJoCo获取关节位置"""
        try:
            joint_id = self.model.joint(joint_name).id
            qpos_idx = self.model.jnt_qposadr[joint_id]
            return self.data.qpos[qpos_idx]
        except:
            return 0.0
    
    def print_system_info(self):
        """打印系统信息"""
        print(f"\n=== 系统信息 ===")
        
        # 读取一次传感器数据
        self.sensor_reader.read_sensors()
        
        # 关节信息
        print("关节状态:")
        for joint_name in self.debug_joints:
            pos = self.data_bus.get_joint_position(joint_name)
            vel = self.data_bus.get_joint_velocity(joint_name)
            torque = self.data_bus.get_joint_torque(joint_name)
            print(f"  {joint_name:15s}: pos={pos:7.3f}, vel={vel:7.3f}, torque={torque:7.3f}")
        
        # IMU信息
        imu_ori = self.data_bus.get_imu_orientation()
        imu_vel = self.data_bus.get_imu_angular_velocity()
        print(f"IMU姿态: quat({imu_ori.w:.3f}, {imu_ori.x:.3f}, {imu_ori.y:.3f}, {imu_ori.z:.3f})")
        print(f"IMU角速度: ({imu_vel.x:.3f}, {imu_vel.y:.3f}, {imu_vel.z:.3f})")
        
        # 质心信息
        com_pos = self.data_bus.get_center_of_mass_position()
        com_vel = self.data_bus.get_center_of_mass_velocity()
        print(f"质心位置: ({com_pos.x:.3f}, {com_pos.y:.3f}, {com_pos.z:.3f})")
        print(f"质心速度: ({com_vel.x:.3f}, {com_vel.y:.3f}, {com_vel.z:.3f})")
        
        # 接触信息
        left_contact = self.data_bus.get_end_effector_contact_state("left_foot")
        right_contact = self.data_bus.get_end_effector_contact_state("right_foot")
        print(f"足部接触: 左={left_contact.name if left_contact else 'None'}, 右={right_contact.name if right_contact else 'None'}")


def main():
    """主函数"""
    print("=== Easy Simulation (Headless) 调试工具 ===\n")
    
    try:
        # 创建仿真实例
        sim = HeadlessSimulation()
        
        # 打印系统信息
        sim.print_system_info()
        
        # 运行综合测试
        sim.run_comprehensive_test()
        
    except Exception as e:
        print(f"仿真过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 调试完成 ===")


if __name__ == "__main__":
    main() 