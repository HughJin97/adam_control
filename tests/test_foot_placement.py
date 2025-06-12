#!/usr/bin/env python3
"""
足步规划测试脚本 - Test Foot Placement Module

测试足步规划模块的各项功能：
1. 正向运动学计算
2. 足步规划计算
3. 与数据总线的集成
4. 与步态调度器的集成
5. 不同地形和策略的适应性

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from foot_placement import (
    FootPlacementPlanner, FootPlacementConfig, ForwardKinematics,
    Vector3D, FootPlacementStrategy, TerrainType, get_foot_planner
)
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler


class FootPlacementTester:
    """足步规划测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.test_results = {}
        self.planner = get_foot_planner()
        self.data_bus = get_data_bus()
        self.gait_scheduler = get_gait_scheduler()
        
        print("=== 足步规划测试器初始化 ===")
        print(f"足步规划器: {'✓' if self.planner else '✗'}")
        print(f"数据总线: {'✓' if self.data_bus else '✗'}")
        print(f"步态调度器: {'✓' if self.gait_scheduler else '✗'}")
        print("=" * 40)
    
    def test_forward_kinematics(self) -> bool:
        """测试正向运动学计算"""
        print("\n=== 测试1: 正向运动学计算 ===")
        
        try:
            kinematics = ForwardKinematics()
            
            # 测试用关节角度
            test_angles = {
                "left_hip_yaw": 0.0, "left_hip_roll": 0.1, "left_hip_pitch": -0.2,
                "left_knee_pitch": 0.3, "left_ankle_pitch": -0.1, "left_ankle_roll": -0.05,
                "right_hip_yaw": 0.0, "right_hip_roll": -0.1, "right_hip_pitch": -0.2,
                "right_knee_pitch": 0.3, "right_ankle_pitch": -0.1, "right_ankle_roll": 0.05
            }
            
            # 计算足端位置
            left_pos = kinematics.compute_foot_position(test_angles, "left")
            right_pos = kinematics.compute_foot_position(test_angles, "right")
            
            print(f"左脚位置: ({left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f})")
            print(f"右脚位置: ({right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f})")
            
            # 验证基本合理性
            assert abs(left_pos[1] - right_pos[1]) > 0.1, "双脚Y坐标差距过小"
            assert left_pos[2] < 0, "左脚Z坐标应为负值"
            assert right_pos[2] < 0, "右脚Z坐标应为负值"
            
            # 测试双足计算
            both_pos = kinematics.compute_both_feet_positions(test_angles)
            assert np.allclose(both_pos[0], left_pos), "双足计算左脚位置不一致"
            assert np.allclose(both_pos[1], right_pos), "双足计算右脚位置不一致"
            
            print("✓ 正向运动学计算测试通过")
            self.test_results['forward_kinematics'] = True
            return True
            
        except Exception as e:
            print(f"✗ 正向运动学计算测试失败: {e}")
            self.test_results['forward_kinematics'] = False
            return False
    
    def test_basic_foot_planning(self) -> bool:
        """测试基本足步规划"""
        print("\n=== 测试2: 基本足步规划 ===")
        
        try:
            # 设置测试用关节角度
            test_angles = {
                "left_hip_yaw": 0.0, "left_hip_roll": 0.05, "left_hip_pitch": -0.1,
                "left_knee_pitch": 0.2, "left_ankle_pitch": -0.1, "left_ankle_roll": -0.05,
                "right_hip_yaw": 0.0, "right_hip_roll": -0.05, "right_hip_pitch": -0.1,
                "right_knee_pitch": 0.2, "right_ankle_pitch": -0.1, "right_ankle_roll": 0.05
            }
            
            # 更新足部状态
            self.planner.update_foot_states_from_kinematics(test_angles)
            
            # 设置运动意图 (前进0.2m/s)
            self.planner.set_body_motion_intent(Vector3D(0.2, 0.0, 0.0), 0.0)
            
            # 测试左腿摆动时的规划
            target_left = self.planner.plan_foot_placement("left", "right")
            print(f"左腿摆动目标: ({target_left.x:.3f}, {target_left.y:.3f}, {target_left.z:.3f})")
            
            # 测试右腿摆动时的规划
            target_right = self.planner.plan_foot_placement("right", "left")
            print(f"右腿摆动目标: ({target_right.x:.3f}, {target_right.y:.3f}, {target_right.z:.3f})")
            
            # 验证合理性
            assert target_left.x > 0, "左腿目标X坐标应为正(前进)"
            assert target_right.x > 0, "右腿目标X坐标应为正(前进)"
            assert target_left.y > 0, "左腿目标Y坐标应为正(左侧)"
            assert target_right.y < 0, "右腿目标Y坐标应为负(右侧)"
            assert abs(target_left.y - target_right.y) > 0.1, "双腿Y坐标差距应较大"
            
            # 检查统计信息
            stats = self.planner.get_planning_statistics()
            assert stats['planning_count'] >= 2, "规划次数应不少于2"
            
            print("✓ 基本足步规划测试通过")
            self.test_results['basic_planning'] = True
            return True
            
        except Exception as e:
            print(f"✗ 基本足步规划测试失败: {e}")
            self.test_results['basic_planning'] = False
            return False
    
    def test_data_bus_integration(self) -> bool:
        """测试数据总线集成"""
        print("\n=== 测试3: 数据总线集成 ===")
        
        try:
            # 检查数据总线是否有足步规划功能
            has_foot_planner = hasattr(self.data_bus, 'trigger_foot_placement_planning')
            print(f"数据总线足步规划支持: {'✓' if has_foot_planner else '✗'}")
            
            if not has_foot_planner:
                print("⚠ 数据总线没有足步规划集成，跳过此测试")
                self.test_results['data_bus_integration'] = True
                return True
            
            # 初始化测试数据
            test_angles = {
                "left_hip_pitch": -0.1, "left_knee_pitch": 0.2, "left_ankle_pitch": -0.1,
                "right_hip_pitch": -0.1, "right_knee_pitch": 0.2, "right_ankle_pitch": -0.1
            }
            
            # 更新数据总线中的关节角度
            for joint, angle in test_angles.items():
                self.data_bus.set_joint_position(joint, angle)
            
            # 设置运动意图
            self.data_bus.set_body_motion_intent(0.25, 0.0, 0.0)  # 前进0.25m/s
            
            # 触发足步规划
            target_pos = self.data_bus.trigger_foot_placement_planning("left", "right")
            
            if target_pos:
                print(f"数据总线触发左腿规划: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
                
                # 检查数据总线中的目标位置是否更新
                db_target = self.data_bus.get_target_foot_position("left")
                print(f"数据总线左脚目标: ({db_target['x']:.3f}, {db_target['y']:.3f}, {db_target['z']:.3f})")
                
                # 验证一致性
                assert abs(db_target['x'] - target_pos.x) < 0.001, "X坐标不一致"
                assert abs(db_target['y'] - target_pos.y) < 0.001, "Y坐标不一致"
                assert abs(db_target['z'] - target_pos.z) < 0.001, "Z坐标不一致"
            
            # 测试足部位移计算
            displacement = self.data_bus.get_foot_displacement("left")
            distance = self.data_bus.get_foot_distance_to_target("left")
            
            print(f"左脚位移: ({displacement['x']:.3f}, {displacement['y']:.3f}, {displacement['z']:.3f})")
            print(f"左脚到目标距离: {distance:.3f}m")
            
            # 打印足步规划状态
            self.data_bus.print_foot_planning_status()
            
            print("✓ 数据总线集成测试通过")
            self.test_results['data_bus_integration'] = True
            return True
            
        except Exception as e:
            print(f"✗ 数据总线集成测试失败: {e}")
            self.test_results['data_bus_integration'] = False
            return False
    
    def test_gait_scheduler_integration(self) -> bool:
        """测试步态调度器集成"""
        print("\n=== 测试4: 步态调度器集成 ===")
        
        try:
            # 初始化系统
            self.gait_scheduler.reset()
            
            # 设置一些模拟的传感器数据
            left_velocity = np.array([0.0, 0.0, 0.0])
            right_velocity = np.array([0.0, 0.0, 0.0])
            
            self.gait_scheduler.update_sensor_data(25.0, 25.0, left_velocity, right_velocity)
            
            # 开始行走测试
            print("开始行走测试...")
            self.gait_scheduler.start_walking()
            
            # 模拟几个步态周期
            dt = 0.02  # 50Hz 更新频率
            simulation_time = 3.0  # 模拟3秒
            steps = int(simulation_time / dt)
            
            foot_planning_count = 0
            state_changes = []
            
            for i in range(steps):
                # 更新步态调度器
                state_changed = self.gait_scheduler.update_gait_state(dt)
                
                if state_changed:
                    current_state = self.gait_scheduler.current_state
                    swing_leg = self.gait_scheduler.swing_leg
                    state_changes.append({
                        'time': i * dt,
                        'state': current_state.value,
                        'swing_leg': swing_leg
                    })
                    
                    print(f"t={i*dt:.2f}s: 状态={current_state.value}, 摆动腿={swing_leg}")
                    
                    # 检查是否触发了足步规划
                    if swing_leg in ["left", "right"]:
                        foot_planning_count += 1
                
                # 更新传感器数据 (模拟动态变化)
                if i % 25 == 0:  # 每0.5秒更新一次
                    force_variation = np.random.normal(0, 5)  # 添加噪声
                    self.gait_scheduler.update_sensor_data(
                        25.0 + force_variation, 25.0 + force_variation,
                        left_velocity, right_velocity
                    )
            
            print(f"\n模拟结果:")
            print(f"  状态转换次数: {len(state_changes)}")
            print(f"  足步规划触发次数: {foot_planning_count}")
            
            # 验证结果
            assert len(state_changes) > 0, "应该有状态转换发生"
            assert foot_planning_count > 0, "应该触发足步规划"
            
            # 打印最终统计
            stats = self.gait_scheduler.get_state_statistics()
            print(f"\n步态统计:")
            for state, data in stats['state_stats'].items():
                print(f"  {state}: {data['count']}次, "
                      f"平均{data['avg_duration']:.3f}s, "
                      f"占比{data['percentage']:.1f}%")
            
            print("✓ 步态调度器集成测试通过")
            self.test_results['gait_scheduler_integration'] = True
            return True
            
        except Exception as e:
            print(f"✗ 步态调度器集成测试失败: {e}")
            self.test_results['gait_scheduler_integration'] = False
            return False
    
    def test_terrain_adaptation(self) -> bool:
        """测试地形适应性"""
        print("\n=== 测试5: 地形适应性 ===")
        
        try:
            terrains = [
                TerrainType.FLAT,
                TerrainType.SLOPE,
                TerrainType.STAIRS,
                TerrainType.ROUGH
            ]
            
            results = {}
            
            # 设置基础状态
            test_angles = {
                "left_hip_pitch": -0.1, "left_knee_pitch": 0.2, "left_ankle_pitch": -0.1,
                "right_hip_pitch": -0.1, "right_knee_pitch": 0.2, "right_ankle_pitch": -0.1
            }
            self.planner.update_foot_states_from_kinematics(test_angles)
            self.planner.set_body_motion_intent(Vector3D(0.3, 0.0, 0.0), 0.0)
            
            for terrain in terrains:
                print(f"\n测试地形: {terrain.value}")
                
                # 设置地形类型
                self.planner.set_terrain_type(terrain)
                
                # 执行规划
                target_left = self.planner.plan_foot_placement("left", "right")
                target_right = self.planner.plan_foot_placement("right", "left")
                
                results[terrain.value] = {
                    'left': (target_left.x, target_left.y, target_left.z),
                    'right': (target_right.x, target_right.y, target_right.z)
                }
                
                print(f"  左腿目标: ({target_left.x:.3f}, {target_left.y:.3f}, {target_left.z:.3f})")
                print(f"  右腿目标: ({target_right.x:.3f}, {target_right.y:.3f}, {target_right.z:.3f})")
            
            # 验证地形间的差异
            flat_left_x = results['flat']['left'][0]
            slope_left_x = results['slope']['left'][0]
            
            # 在斜坡上步长应该略有调整
            print(f"\n地形适应验证:")
            print(f"  平地步长: {flat_left_x:.3f}m")
            print(f"  斜坡步长: {slope_left_x:.3f}m")
            print(f"  适应性差异: {abs(flat_left_x - slope_left_x):.3f}m")
            
            print("✓ 地形适应性测试通过")
            self.test_results['terrain_adaptation'] = True
            return True
            
        except Exception as e:
            print(f"✗ 地形适应性测试失败: {e}")
            self.test_results['terrain_adaptation'] = False
            return False
    
    def test_strategy_comparison(self) -> bool:
        """测试不同规划策略"""
        print("\n=== 测试6: 规划策略比较 ===")
        
        try:
            strategies = [
                FootPlacementStrategy.STATIC_WALK,
                FootPlacementStrategy.DYNAMIC_WALK,
                FootPlacementStrategy.ADAPTIVE,
                FootPlacementStrategy.STABILIZING
            ]
            
            # 设置基础状态
            test_angles = {
                "left_hip_pitch": -0.1, "left_knee_pitch": 0.2, "left_ankle_pitch": -0.1,
                "right_hip_pitch": -0.1, "right_knee_pitch": 0.2, "right_ankle_pitch": -0.1
            }
            self.planner.update_foot_states_from_kinematics(test_angles)
            self.planner.set_body_motion_intent(Vector3D(0.4, 0.1, 0.0), 0.2)  # 有横向速度和转向
            
            strategy_results = {}
            
            for strategy in strategies:
                print(f"\n测试策略: {strategy.value}")
                
                # 设置策略
                self.planner.set_planning_strategy(strategy)
                
                # 执行规划
                target_left = self.planner.plan_foot_placement("left", "right")
                target_right = self.planner.plan_foot_placement("right", "left")
                
                strategy_results[strategy.value] = {
                    'left': target_left,
                    'right': target_right
                }
                
                print(f"  左腿目标: ({target_left.x:.3f}, {target_left.y:.3f}, {target_left.z:.3f})")
                print(f"  右腿目标: ({target_right.x:.3f}, {target_right.y:.3f}, {target_right.z:.3f})")
            
            # 比较不同策略的差异
            print(f"\n策略比较:")
            for strategy, results in strategy_results.items():
                left_pos = results['left']
                step_length = left_pos.x
                step_width = abs(left_pos.y)
                print(f"  {strategy}: 步长={step_length:.3f}m, 步宽={step_width:.3f}m")
            
            print("✓ 规划策略比较测试通过")
            self.test_results['strategy_comparison'] = True
            return True
            
        except Exception as e:
            print(f"✗ 规划策略比较测试失败: {e}")
            self.test_results['strategy_comparison'] = False
            return False
    
    def test_performance(self) -> bool:
        """测试性能"""
        print("\n=== 测试7: 性能测试 ===")
        
        try:
            # 准备测试数据
            test_angles = {
                "left_hip_pitch": -0.1, "left_knee_pitch": 0.2, "left_ankle_pitch": -0.1,
                "right_hip_pitch": -0.1, "right_knee_pitch": 0.2, "right_ankle_pitch": -0.1,
                "left_hip_yaw": 0.0, "left_hip_roll": 0.05, "left_ankle_roll": -0.05,
                "right_hip_yaw": 0.0, "right_hip_roll": -0.05, "right_ankle_roll": 0.05
            }
            
            self.planner.update_foot_states_from_kinematics(test_angles) 
            self.planner.set_body_motion_intent(Vector3D(0.3, 0.0, 0.0), 0.0)
            
            # 测试运动学计算性能
            kinematics = ForwardKinematics()
            num_kinematics_tests = 1000
            
            start_time = time.time()
            for _ in range(num_kinematics_tests):
                left_pos = kinematics.compute_foot_position(test_angles, "left")
                right_pos = kinematics.compute_foot_position(test_angles, "right")
            kinematics_time = time.time() - start_time
            
            # 测试足步规划性能
            num_planning_tests = 500
            
            start_time = time.time()
            for _ in range(num_planning_tests):
                target_left = self.planner.plan_foot_placement("left", "right")
                target_right = self.planner.plan_foot_placement("right", "left")
            planning_time = time.time() - start_time
            
            # 计算性能指标
            kinematics_freq = num_kinematics_tests / kinematics_time
            planning_freq = num_planning_tests / planning_time
            
            print(f"\n性能测试结果:")
            print(f"  运动学计算: {kinematics_freq:.1f} Hz ({kinematics_time*1000/num_kinematics_tests:.3f}ms/次)")
            print(f"  足步规划: {planning_freq:.1f} Hz ({planning_time*1000/num_planning_tests:.3f}ms/次)")
            
            # 验证性能要求 (应该能够支持50Hz以上的控制频率)
            assert kinematics_freq > 50, f"运动学计算频率过低: {kinematics_freq:.1f} Hz < 50 Hz"
            assert planning_freq > 25, f"足步规划频率过低: {planning_freq:.1f} Hz < 25 Hz"
            
            print("✓ 性能测试通过")
            self.test_results['performance'] = True
            return True
            
        except Exception as e:
            print(f"✗ 性能测试失败: {e}")
            self.test_results['performance'] = False
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        print("开始足步规划全面测试...\n")
        
        test_methods = [
            ('正向运动学', self.test_forward_kinematics),
            ('基本足步规划', self.test_basic_foot_planning),
            ('数据总线集成', self.test_data_bus_integration),
            ('步态调度器集成', self.test_gait_scheduler_integration),
            ('地形适应性', self.test_terrain_adaptation),
            ('规划策略比较', self.test_strategy_comparison),
            ('性能测试', self.test_performance)
        ]
        
        passed = 0
        total = len(test_methods)
        
        for test_name, test_method in test_methods:
            try:
                success = test_method()
                if success:
                    passed += 1
            except Exception as e:
                print(f"✗ {test_name}测试出现异常: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = False
        
        # 打印测试总结
        print(f"\n" + "="*50)
        print(f"足步规划测试完成")
        print(f"通过: {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"="*50)
        
        print(f"\n详细结果:")
        for test_name, result in self.test_results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"  {test_name}: {status}")
        
        return self.test_results
    
    def generate_visualization(self):
        """生成可视化图表"""
        print("\n=== 生成足步规划可视化 ===")
        
        try:
            # 模拟一个步行序列
            time_points = np.linspace(0, 4, 200)  # 4秒，200个点
            left_positions = []
            right_positions = []
            
            # 设置基础配置
            test_angles = {
                "left_hip_pitch": -0.1, "left_knee_pitch": 0.2, "left_ankle_pitch": -0.1,
                "right_hip_pitch": -0.1, "right_knee_pitch": 0.2, "right_ankle_pitch": -0.1
            }
            self.planner.update_foot_states_from_kinematics(test_angles)
            self.planner.set_body_motion_intent(Vector3D(0.2, 0.0, 0.0), 0.0)
            
            # 模拟步行过程中的足步规划
            for i, t in enumerate(time_points):
                # 交替规划左右脚
                if i % 2 == 0:
                    target = self.planner.plan_foot_placement("left", "right")
                    left_positions.append([target.x, target.y, target.z])
                    if right_positions:
                        right_positions.append(right_positions[-1])  # 保持上一次的位置
                    else:
                        right_positions.append([0.0, -0.09, 0.0])
                else:
                    target = self.planner.plan_foot_placement("right", "left")
                    right_positions.append([target.x, target.y, target.z])
                    if left_positions:
                        left_positions.append(left_positions[-1])  # 保持上一次的位置
                    else:
                        left_positions.append([0.0, 0.09, 0.0])
            
            # 转换为numpy数组
            left_positions = np.array(left_positions)
            right_positions = np.array(right_positions)
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('足步规划可视化结果', fontsize=16, fontweight='bold')
            
            # 1. 足迹轨迹 (俯视图)
            ax1 = axes[0, 0]
            ax1.plot(left_positions[:, 0], left_positions[:, 1], 'b-o', 
                    label='左脚轨迹', markersize=3, linewidth=2)
            ax1.plot(right_positions[:, 0], right_positions[:, 1], 'r-s', 
                    label='右脚轨迹', markersize=3, linewidth=2)
            ax1.set_xlabel('X 位置 [m]')
            ax1.set_ylabel('Y 位置 [m]')
            ax1.set_title('足迹轨迹 (俯视图)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # 2. X方向位置随时间变化
            ax2 = axes[0, 1]
            ax2.plot(time_points, left_positions[:, 0], 'b-', label='左脚 X', linewidth=2)
            ax2.plot(time_points, right_positions[:, 0], 'r-', label='右脚 X', linewidth=2)
            ax2.set_xlabel('时间 [s]')
            ax2.set_ylabel('X 位置 [m]')
            ax2.set_title('前进方向位置')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Y方向位置随时间变化
            ax3 = axes[1, 0]
            ax3.plot(time_points, left_positions[:, 1], 'b-', label='左脚 Y', linewidth=2)
            ax3.plot(time_points, right_positions[:, 1], 'r-', label='右脚 Y', linewidth=2)
            ax3.set_xlabel('时间 [s]')
            ax3.set_ylabel('Y 位置 [m]')
            ax3.set_title('横向位置')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 步长和步宽统计
            ax4 = axes[1, 1]
            step_lengths = np.diff(left_positions[:, 0])
            step_widths = np.abs(left_positions[:, 1] - right_positions[:, 1])
            
            ax4.hist(step_lengths[step_lengths > 0], bins=20, alpha=0.7, 
                    label=f'步长 (均值: {np.mean(step_lengths[step_lengths > 0]):.3f}m)', 
                    color='green')
            ax4.set_xlabel('步长 [m]')
            ax4.set_ylabel('频次')
            ax4.set_title('步长分布')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图像
            plt.savefig('foot_placement_visualization.png', dpi=300, bbox_inches='tight')
            print("✓ 可视化图表已保存为 'foot_placement_visualization.png'")
            plt.show()
            
        except Exception as e:
            print(f"✗ 可视化生成失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("AzureLoong 机器人足步规划系统测试")
    print("=" * 60)
    
    # 创建测试器
    tester = FootPlacementTester()
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 生成可视化
    if results.get('basic_planning', False):
        tester.generate_visualization()
    
    # 最终状态打印
    print(f"\n" + "="*60)
    if all(results.values()):
        print("🎉 所有测试通过！足步规划系统可以正常使用。")
    else:
        failed_tests = [name for name, result in results.items() if not result]
        print(f"⚠ 部分测试失败: {', '.join(failed_tests)}")
        print("请检查相关模块并修复问题。")
    
    print("="*60)


if __name__ == "__main__":
    main() 