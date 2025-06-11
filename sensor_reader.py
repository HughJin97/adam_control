"""
Sensor Reader - 传感器数据读取模块
从MuJoCo仿真环境读取传感器数据并更新数据总线

作者: Adam Control Team
版本: 1.0
"""

import mujoco
import numpy as np
from typing import Dict, Optional, Tuple
from data_bus import (
    get_data_bus, 
    Vector3D, 
    Quaternion, 
    ContactState,
    JointData
)


class SensorReader:
    """
    传感器数据读取器
    负责从MuJoCo仿真环境读取各种传感器数据并更新到数据总线
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        初始化传感器读取器
        
        参数:
            model: MuJoCo模型对象
            data: MuJoCo数据对象
        """
        self.model = model
        self.data = data
        self.data_bus = get_data_bus()
        
        # 缓存传感器ID以提高性能
        self._sensor_ids = {}
        self._joint_ids = {}
        self._init_sensor_mappings()
        
        print("SensorReader initialized successfully")
    
    def _init_sensor_mappings(self):
        """初始化传感器映射，缓存传感器和关节的ID"""
        # 缓存传感器ID
        sensor_names = [
            "baselink-quat",       # IMU四元数
            "baselink-velocity",   # 基座速度
            "baselink-gyro",       # 陀螺仪
            "baselink-baseAcc",    # 加速度计
            "lf-touch",           # 左脚接触
            "rf-touch"            # 右脚接触
        ]
        
        for name in sensor_names:
            try:
                self._sensor_ids[name] = self.model.sensor(name).id
            except KeyError:
                print(f"Warning: Sensor '{name}' not found in model")
        
        # 缓存关节ID
        for joint_name in self.data_bus.joint_names:
            try:
                self._joint_ids[joint_name] = self.model.joint(joint_name).id
            except KeyError:
                print(f"Warning: Joint '{joint_name}' not found in model")
    
    def read_sensors(self) -> bool:
        """
        主要的传感器读取函数
        从仿真环境读取所有传感器数据并更新数据总线
        
        返回:
            bool: 读取是否成功
        """
        try:
            # 1. 读取关节传感器数据
            self._read_joint_sensors()
            
            # 2. 读取IMU数据
            self._read_imu_sensors()
            
            # 3. 读取足部接触传感器
            self._read_contact_sensors()
            
            # 4. 计算质心状态
            self._compute_center_of_mass()
            
            # 5. 计算末端执行器状态
            self._compute_end_effector_states()
            
            # 更新时间戳
            self.data_bus.update_timestamp()
            
            return True
            
        except Exception as e:
            print(f"Error reading sensors: {e}")
            return False
    
    def _read_joint_sensors(self):
        """读取关节传感器数据（位置、速度、力矩）"""
        for joint_name, joint_id in self._joint_ids.items():
            if joint_id is None:
                continue
            
            # 获取关节的qpos和qvel索引
            qpos_idx = self.model.jnt_qposadr[joint_id]
            qvel_idx = self.model.jnt_dofadr[joint_id]
            
            # 读取关节位置和速度
            position = self.data.qpos[qpos_idx]
            velocity = self.data.qvel[qvel_idx]
            
            # 读取关节力矩 (如果有执行器)
            torque = 0.0
            actuator_name = f"M_{joint_name[2:]}"  # 执行器名称格式: M_xxx
            try:
                actuator_id = self.model.actuator(actuator_name).id
                torque = self.data.actuator_force[actuator_id]
            except:
                pass
            
            # 更新数据总线
            self.data_bus.set_joint_position(joint_name, position)
            self.data_bus.set_joint_velocity(joint_name, velocity)
            self.data_bus.set_joint_torque(joint_name, torque)
            
            # 更新关节数据的其他字段
            joint_data = self.data_bus.joints[joint_name]
            joint_data.acceleration = 0.0  # MuJoCo可以通过qacc获取
            
            # 获取关节限位
            joint_range = self.model.jnt_range[joint_id]
            joint_data.position_limit_min = joint_range[0]
            joint_data.position_limit_max = joint_range[1]
    
    def _read_imu_sensors(self):
        """读取IMU传感器数据"""
        # 读取姿态四元数
        if "baselink-quat" in self._sensor_ids:
            sensor_id = self._sensor_ids["baselink-quat"]
            quat_data = self.data.sensordata[sensor_id:sensor_id+4]
            quaternion = Quaternion(
                w=quat_data[0],
                x=quat_data[1],
                y=quat_data[2],
                z=quat_data[3]
            )
            self.data_bus.set_imu_orientation(quaternion)
            
            # 计算欧拉角
            euler_angles = self._quaternion_to_euler(quaternion)
            self.data_bus.imu.roll = euler_angles[0]
            self.data_bus.imu.pitch = euler_angles[1]
            self.data_bus.imu.yaw = euler_angles[2]
        
        # 读取角速度
        if "baselink-gyro" in self._sensor_ids:
            sensor_id = self._sensor_ids["baselink-gyro"]
            gyro_data = self.data.sensordata[sensor_id:sensor_id+3]
            angular_velocity = Vector3D(
                x=gyro_data[0],
                y=gyro_data[1],
                z=gyro_data[2]
            )
            self.data_bus.set_imu_angular_velocity(angular_velocity)
        
        # 读取线加速度
        if "baselink-baseAcc" in self._sensor_ids:
            sensor_id = self._sensor_ids["baselink-baseAcc"]
            acc_data = self.data.sensordata[sensor_id:sensor_id+3]
            linear_acceleration = Vector3D(
                x=acc_data[0],
                y=acc_data[1],
                z=acc_data[2]
            )
            self.data_bus.set_imu_linear_acceleration(linear_acceleration)
    
    def _read_contact_sensors(self):
        """读取足部接触传感器"""
        # 读取左脚接触
        if "lf-touch" in self._sensor_ids:
            sensor_id = self._sensor_ids["lf-touch"]
            touch_value = self.data.sensordata[sensor_id]
            contact_state = ContactState.CONTACT if touch_value > 0.1 else ContactState.NO_CONTACT
            self.data_bus.set_end_effector_contact_state("left_foot", contact_state)
            
            # 更新接触力大小
            self.data_bus.end_effectors["left_foot"].contact_force_magnitude = abs(touch_value)
        
        # 读取右脚接触
        if "rf-touch" in self._sensor_ids:
            sensor_id = self._sensor_ids["rf-touch"]
            touch_value = self.data.sensordata[sensor_id]
            contact_state = ContactState.CONTACT if touch_value > 0.1 else ContactState.NO_CONTACT
            self.data_bus.set_end_effector_contact_state("right_foot", contact_state)
            
            # 更新接触力大小
            self.data_bus.end_effectors["right_foot"].contact_force_magnitude = abs(touch_value)
    
    def _compute_center_of_mass(self):
        """计算质心位置和速度"""
        # 获取质心位置
        com_pos = self.data.subtree_com[0]  # 根body的质心
        com_position = Vector3D(
            x=com_pos[0],
            y=com_pos[1],
            z=com_pos[2]
        )
        self.data_bus.set_center_of_mass_position(com_position)
        
        # 计算质心速度 (通过数值微分或从基座速度估计)
        if "baselink-velocity" in self._sensor_ids:
            sensor_id = self._sensor_ids["baselink-velocity"]
            vel_data = self.data.sensordata[sensor_id:sensor_id+3]
            com_velocity = Vector3D(
                x=vel_data[0],
                y=vel_data[1],
                z=vel_data[2]
            )
            self.data_bus.set_center_of_mass_velocity(com_velocity)
    
    def _compute_end_effector_states(self):
        """计算末端执行器（手脚）的位置和速度"""
        # 定义末端执行器对应的body名称
        end_effector_bodies = {
            "left_foot": "Link_ankle_l_roll",
            "right_foot": "Link_ankle_r_roll",
            "left_hand": "Link_arm_l_07",
            "right_hand": "Link_arm_r_07"
        }
        
        for ee_name, body_name in end_effector_bodies.items():
            try:
                body_id = self.model.body(body_name).id
                
                # 获取body的全局位置
                body_pos = self.data.xpos[body_id]
                position = Vector3D(
                    x=body_pos[0],
                    y=body_pos[1],
                    z=body_pos[2]
                )
                self.data_bus.set_end_effector_position(ee_name, position)
                
                # 获取body的全局速度
                body_vel = self.data.cvel[body_id][:3]  # 线速度部分
                velocity = Vector3D(
                    x=body_vel[0],
                    y=body_vel[1],
                    z=body_vel[2]
                )
                self.data_bus.end_effectors[ee_name].velocity = velocity
                
                # 获取body的姿态（四元数）
                body_quat = self.data.xquat[body_id]
                orientation = Quaternion(
                    w=body_quat[0],
                    x=body_quat[1],
                    y=body_quat[2],
                    z=body_quat[3]
                )
                self.data_bus.end_effectors[ee_name].orientation = orientation
                
                # 获取角速度
                body_angular_vel = self.data.cvel[body_id][3:]  # 角速度部分
                angular_velocity = Vector3D(
                    x=body_angular_vel[0],
                    y=body_angular_vel[1],
                    z=body_angular_vel[2]
                )
                self.data_bus.end_effectors[ee_name].angular_velocity = angular_velocity
                
            except KeyError:
                print(f"Warning: Body '{body_name}' not found for end effector '{ee_name}'")
    
    def _quaternion_to_euler(self, q: Quaternion) -> Tuple[float, float, float]:
        """
        将四元数转换为欧拉角（roll, pitch, yaw）
        
        参数:
            q: 四元数
            
        返回:
            (roll, pitch, yaw) 弧度制
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # 使用90度
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def get_contact_forces(self) -> Dict[str, np.ndarray]:
        """
        获取接触力详细信息
        
        返回:
            包含每个接触点力信息的字典
        """
        contact_forces = {}
        
        # 遍历所有接触
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # 获取接触体的名称
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # 获取接触力
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            
            # 判断是否是足部接触
            try:
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
                
                # 检查是否涉及足部
                if "ankle_l" in str(geom1_name) or "ankle_l" in str(geom2_name):
                    contact_forces["left_foot"] = force[:3]  # 只取线性力
                elif "ankle_r" in str(geom1_name) or "ankle_r" in str(geom2_name):
                    contact_forces["right_foot"] = force[:3]
                    
            except:
                pass
        
        return contact_forces
    
    def get_joint_limits_status(self) -> Dict[str, Dict[str, bool]]:
        """
        检查所有关节是否接近限位
        
        返回:
            每个关节的限位状态字典
        """
        limits_status = {}
        
        for joint_name in self.data_bus.joint_names:
            if joint_name not in self._joint_ids:
                continue
                
            joint_data = self.data_bus.joints[joint_name]
            position = joint_data.position
            
            # 检查是否接近限位（在限位的10%范围内）
            range_size = joint_data.position_limit_max - joint_data.position_limit_min
            threshold = 0.1 * range_size
            
            near_lower_limit = (position - joint_data.position_limit_min) < threshold
            near_upper_limit = (joint_data.position_limit_max - position) < threshold
            
            limits_status[joint_name] = {
                "near_lower_limit": near_lower_limit,
                "near_upper_limit": near_upper_limit,
                "within_limits": self.data_bus.validate_joint_limits(joint_name)
            }
        
        return limits_status


# 便捷函数
def create_sensor_reader(model: mujoco.MjModel, data: mujoco.MjData) -> SensorReader:
    """
    创建传感器读取器实例
    
    参数:
        model: MuJoCo模型
        data: MuJoCo数据
        
    返回:
        SensorReader实例
    """
    return SensorReader(model, data)


def read_sensors(sensor_reader: SensorReader) -> bool:
    """
    便捷函数：读取所有传感器数据
    
    参数:
        sensor_reader: 传感器读取器实例
        
    返回:
        是否成功读取
    """
    return sensor_reader.read_sensors()


# 示例使用
if __name__ == "__main__":
    import time
    
    # 这里是示例代码，实际使用时需要提供真实的model和data
    print("SensorReader module loaded successfully")
    print("To use this module:")
    print("1. Create MuJoCo model and data objects")
    print("2. Create SensorReader: reader = create_sensor_reader(model, data)")
    print("3. Read sensors: success = read_sensors(reader)")
    print("4. Access data via DataBus: bus = get_data_bus()") 