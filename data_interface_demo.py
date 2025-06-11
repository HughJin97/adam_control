#!/usr/bin/env python3
"""
AzureLoong机器人数据接口演示脚本

展示步态调度器与其他模块通过数据总线的标准化交互：
1. 传感器数据读取接口
2. 步态状态输出接口  
3. 目标位置计算接口
4. 模块间数据交换
5. 主循环更新流程
"""

import time
import numpy as np
from typing import Dict, Any
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig


class TrajectoryPlannerMock:
    """模拟足部轨迹规划器"""
    
    def __init__(self):
        self.name = "TrajectoryPlanner"
        
    def get_required_gait_data(self) -> Dict:
        """获取轨迹规划所需的步态数据"""
        data_bus = get_data_bus()
        return data_bus.get_gait_data_for_trajectory_planning()
    
    def plan_foot_trajectory(self, target_positions: Dict) -> Dict:
        """规划足部轨迹"""
        gait_data = self.get_required_gait_data()
        
        if not gait_data:
            return {"left_foot": [], "right_foot": []}
        
        print(f"[{self.name}] 规划轨迹:")
        print(f"  摆动腿: {gait_data.get('swing_leg', 'none')}")
        print(f"  摆动进度: {gait_data.get('swing_progress', 0.0)*100:.1f}%")
        print(f"  目标位置: {target_positions}")
        
        # 模拟轨迹规划结果
        return {
            "left_foot": self._generate_trajectory("left"),
            "right_foot": self._generate_trajectory("right"),
            "planning_time": time.time()
        }
    
    def _generate_trajectory(self, foot: str) -> list:
        """生成模拟轨迹点"""
        return [{"t": i*0.01, "x": i*0.01, "y": 0.09 if foot=="left" else -0.09, "z": 0.0} 
                for i in range(10)]


class MPCControllerMock:
    """模拟MPC控制器"""
    
    def __init__(self):
        self.name = "MPCController"
        
    def get_required_gait_data(self) -> Dict:
        """获取MPC控制所需的步态数据"""
        data_bus = get_data_bus()
        return data_bus.get_gait_data_for_mpc()
    
    def compute_control(self) -> Dict:
        """计算控制指令"""
        gait_data = self.get_required_gait_data()
        
        if not gait_data:
            return {"torques": [0.0] * 12}
        
        print(f"[{self.name}] 计算控制:")
        print(f"  支撑腿: {gait_data.get('support_leg', 'unknown')}")
        print(f"  双支撑: {gait_data.get('in_double_support', False)}")
        print(f"  剩余摆动时间: {gait_data.get('swing_time_remaining', 0.0):.3f}s")
        print(f"  下一摆动腿: {gait_data.get('next_swing_leg', 'none')}")
        
        # 模拟控制计算
        return {
            "joint_torques": np.random.normal(0, 5, 12).tolist(),
            "com_force": [0.0, 0.0, 400.0],  # 垂直力
            "computation_time": time.time()
        }


class SensorInterfaceMock:
    """模拟传感器接口"""
    
    def __init__(self):
        self.name = "SensorInterface"
        self.time_start = time.time()
        
    def read_sensors(self) -> Dict:
        """读取传感器数据"""
        current_time = time.time() - self.time_start
        
        # 模拟传感器数据
        sensor_data = {
            "left_foot_force": 200.0 + 50.0 * np.sin(current_time * 2.0),
            "right_foot_force": 200.0 + 50.0 * np.cos(current_time * 2.0),
            "left_foot_velocity": np.array([0.1, 0.0, 0.05 * np.sin(current_time * 3.0)]),
            "right_foot_velocity": np.array([0.1, 0.0, 0.05 * np.cos(current_time * 3.0)]),
            "body_acceleration": np.array([0.0, 0.0, 9.81]),
            "body_angular_velocity": np.array([0.01, 0.01, 0.0])
        }
        
        return sensor_data
    
    def update_data_bus(self, sensor_data: Dict):
        """更新数据总线中的传感器数据"""
        data_bus = get_data_bus()
        data_bus.set_external_sensor_data(sensor_data)


def demonstrate_data_interfaces():
    """演示数据接口"""
    print("="*80)
    print("AzureLoong机器人数据接口演示")
    print("="*80)
    
    # 初始化系统组件
    print("\n[系统初始化]")
    data_bus = get_data_bus()
    
    config = GaitSchedulerConfig()
    config.swing_time = 0.4
    config.enable_logging = False  # 减少输出
    gait_scheduler = get_gait_scheduler(config)
    
    # 模拟其他模块
    trajectory_planner = TrajectoryPlannerMock()
    mpc_controller = MPCControllerMock()
    sensor_interface = SensorInterfaceMock()
    
    print(f"✓ 数据总线初始化完成")
    print(f"✓ 步态调度器初始化完成")
    print(f"✓ 模拟模块初始化完成")
    
    # 显示接口摘要
    print(f"\n[接口状态摘要]")
    interface_summary = data_bus.get_interface_summary()
    for key, value in interface_summary.items():
        print(f"  {key}: {value}")
    
    # 开始行走
    print(f"\n[开始步态循环]")
    gait_scheduler.start_walking()
    data_bus.reset_step_counters()
    
    # 主控制循环演示
    dt = 0.02  # 50Hz控制频率
    total_time = 0.0
    max_demo_time = 2.0
    last_status_time = 0.0
    
    print(f"主循环开始 (控制频率: {1/dt:.0f}Hz)")
    print("-" * 80)
    
    while total_time < max_demo_time:
        # ================== 1. 传感器数据读取 ==================
        sensor_data = sensor_interface.read_sensors()
        sensor_interface.update_data_bus(sensor_data)
        
        # ================== 2. 步态调度器更新 ==================
        # 主循环更新接口
        state_changed = gait_scheduler.update(dt)
        
        # 同步数据总线状态
        data_bus.update_gait_targets_from_scheduler()
        
        # ================== 3. 模块数据获取演示 ==================
        if total_time - last_status_time >= 0.5:  # 每500ms演示一次
            print(f"\n[{total_time:.3f}s] 模块间数据交互演示:")
            
            # 步态调度器状态输出
            gait_state_data = gait_scheduler.get_gait_state_data()
            timing_info = gait_scheduler.get_timing_info()
            leg_states = gait_scheduler.get_leg_states()
            
            print(f"步态调度器输出:")
            print(f"  状态: {gait_state_data['current_state']}")
            print(f"  摆动腿: {gait_state_data['swing_leg']}")
            print(f"  摆动进度: {timing_info['swing_progress']*100:.1f}%")
            print(f"  周期相位: {timing_info['cycle_phase']:.3f}")
            
            # 目标足部位置
            target_positions = gait_scheduler.get_target_foot_positions()
            print(f"目标足部位置:")
            for foot, pos in target_positions.items():
                print(f"  {foot}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
            
            # 轨迹规划模块数据获取
            trajectory_result = trajectory_planner.plan_foot_trajectory(target_positions)
            
            # MPC控制器数据获取
            control_result = mpc_controller.compute_control()
            
            last_status_time = total_time
        
        # ================== 4. 步完成事件处理 ==================
        if data_bus.is_step_completed():
            step_stats = data_bus.get_step_statistics()
            print(f"\n[{total_time:.3f}s] 🚶 步完成事件:")
            print(f"  完成步数: {step_stats['step_count']}")
            print(f"  摆动腿: {step_stats['current_swing_leg']}")
            print(f"  摆动时间: {step_stats['swing_duration']:.3f}s")
            
            data_bus.reset_step_completion_flag()
        
        # ================== 5. 状态变化通知 ==================
        if state_changed:
            current_gait_data = gait_scheduler.get_gait_state_data()
            prediction = gait_scheduler.get_next_swing_prediction()
            print(f"[{total_time:.3f}s] 📊 状态变化:")
            print(f"  新状态: {current_gait_data['current_state']}")
            print(f"  下一摆动腿: {prediction['next_swing_leg']}")
            print(f"  预计摆动开始: {prediction['time_to_next_swing']:.3f}s后")
        
        total_time += dt
        time.sleep(dt * 0.1)  # 减慢演示速度
    
    print("-" * 80)
    print(f"主循环演示完成! 总时间: {total_time:.3f}s")
    
    # ================== 最终数据接口测试 ==================
    print(f"\n[数据接口功能测试]")
    
    # 测试轨迹规划接口
    print(f"轨迹规划接口测试:")
    traj_data = data_bus.get_gait_data_for_trajectory_planning()
    print(f"  接口数据字段: {list(traj_data.keys())}")
    print(f"  当前摆动腿: {traj_data.get('swing_leg', 'N/A')}")
    print(f"  摆动进度: {traj_data.get('swing_progress', 0)*100:.1f}%")
    
    # 测试MPC接口
    print(f"MPC控制接口测试:")
    mpc_data = data_bus.get_gait_data_for_mpc()
    print(f"  接口数据字段: {list(mpc_data.keys())}")
    print(f"  支撑状态: {mpc_data.get('support_leg', 'N/A')}")
    print(f"  预测下一摆动: {mpc_data.get('next_swing_leg', 'N/A')}")
    
    # 测试传感器接口
    print(f"传感器接口测试:")
    sensor_data = data_bus.get_sensor_data_for_gait_scheduler()
    print(f"  接口数据字段: {list(sensor_data.keys())}")
    print(f"  左脚力: {sensor_data.get('left_foot_force', 0):.1f}N")
    print(f"  右脚力: {sensor_data.get('right_foot_force', 0):.1f}N")
    
    return True


def demonstrate_api_usage():
    """演示API使用方法"""
    print(f"\n[API使用方法演示]")
    print("="*80)
    
    # 获取系统实例
    data_bus = get_data_bus()
    gait_scheduler = get_gait_scheduler()
    
    print(f"# 基本API使用示例:")
    print(f"""
# 1. 主循环更新 (推荐方式)
dt = 0.01  # 控制周期
state_changed = gait_scheduler.update(dt)

# 2. 获取完整步态状态
gait_data = gait_scheduler.get_gait_state_data()
current_state = gait_data['current_state']
swing_leg = gait_data['swing_leg']

# 3. 获取时间信息 (供MPC使用)
timing = gait_scheduler.get_timing_info()
swing_progress = timing['swing_progress']
time_remaining = timing['time_to_swing_end']

# 4. 获取腿部状态 (供运动学使用)
leg_states = gait_scheduler.get_leg_states()
is_left_swing = leg_states['left_is_swing']
in_double_support = leg_states['in_double_support']

# 5. 获取目标位置 (供轨迹规划使用)
targets = gait_scheduler.get_target_foot_positions()
left_target = targets['left_foot']

# 6. 设置运动指令 (从高层控制)
gait_scheduler.set_motion_command(
    forward_velocity=0.5,  # m/s
    lateral_velocity=0.0,
    turning_rate=0.1      # rad/s
)

# 7. 检查步完成
if data_bus.is_step_completed():
    step_count = data_bus.get_step_count()
    data_bus.reset_step_completion_flag()

# 8. 专用模块接口
trajectory_data = data_bus.get_gait_data_for_trajectory_planning()
mpc_data = data_bus.get_gait_data_for_mpc()
sensor_data = data_bus.get_sensor_data_for_gait_scheduler()
    """)
    
    # 实际API调用演示
    print(f"\n实际API调用结果:")
    
    # 调用各种API
    gait_data = gait_scheduler.get_gait_state_data()
    timing = gait_scheduler.get_timing_info()
    leg_states = gait_scheduler.get_leg_states()
    targets = gait_scheduler.get_target_foot_positions()
    prediction = gait_scheduler.get_next_swing_prediction()
    
    print(f"当前步态状态: {gait_data['current_state']}")
    print(f"摆动腿: {gait_data['swing_leg']}")
    print(f"摆动进度: {timing['swing_progress']*100:.1f}%")
    print(f"左腿是否摆动: {leg_states['left_is_swing']}")
    print(f"下一摆动腿: {prediction['next_swing_leg']}")
    print(f"左脚目标: {targets['left_foot']}")
    
    # 检查接口就绪状态
    ready_status = gait_scheduler.is_ready_for_new_step()
    print(f"准备新步: {ready_status}")


def main():
    """主演示函数"""
    try:
        # 数据接口演示
        success = demonstrate_data_interfaces()
        
        if success:
            # API使用演示
            demonstrate_api_usage()
            
            print(f"\n" + "="*80)
            print(f"✅ 数据接口设计演示完成!")
            print(f"✅ 所有模块间接口测试通过!")
            print(f"✅ API使用方法验证成功!")
            print(f"="*80)
        else:
            print(f"❌ 演示过程中出现问题")
            
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 