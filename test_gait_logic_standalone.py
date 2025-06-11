#!/usr/bin/env python3
"""
AzureLoong机器人步态逻辑独立测试

专门测试步态调度器的核心逻辑，不涉及复杂控制：
1. 固定机器人，仅推进状态机
2. 手动触发摆动完成条件
3. 验证legState左右支撑交替
4. 检查target_foot_pos合理性
5. 确保状态切换无跳变卡顿
6. 验证数据总线字段逻辑更新
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig, GaitState, LegState


class GaitLogicTester:
    """步态逻辑测试器"""
    
    def __init__(self):
        self.test_results = []
        self.state_history = []
        self.leg_state_history = []
        self.target_position_history = []
        self.timing_history = []
        
        # 测试参数
        self.dt = 0.01  # 10ms 控制周期
        self.manual_trigger_mode = True
        
        # 初始化系统
        self.data_bus = get_data_bus()
        
        # 配置步态调度器（简化参数便于测试）
        config = GaitSchedulerConfig()
        config.swing_time = 0.4          # 400ms 摆动时间
        config.stance_time = 0.6         # 600ms 支撑时间  
        config.double_support_time = 0.1 # 100ms 双支撑时间
        config.use_sensor_trigger = False # 禁用传感器触发，仅用时间
        config.enable_logging = True
        
        self.gait_scheduler = get_gait_scheduler(config)
        
        print("✓ 步态逻辑测试器初始化完成")
        print(f"  摆动时间: {config.swing_time}s")
        print(f"  支撑时间: {config.stance_time}s") 
        print(f"  双支撑时间: {config.double_support_time}s")
        print(f"  控制周期: {self.dt}s ({1/self.dt:.0f}Hz)")
    
    def setup_fixed_robot_mode(self):
        """设置固定机器人模式（不实际行走）"""
        # 设置传感器数据为静态值
        self.gait_scheduler.left_foot_force = 200.0  # N
        self.gait_scheduler.right_foot_force = 200.0  # N
        self.gait_scheduler.left_foot_contact = True
        self.gait_scheduler.right_foot_contact = True
        self.gait_scheduler.left_foot_velocity = np.array([0.0, 0.0, 0.0])
        self.gait_scheduler.right_foot_velocity = np.array([0.0, 0.0, 0.0])
        
        print("✓ 固定机器人模式设置完成（双脚着地，无运动）")
    
    def record_state(self):
        """记录当前状态信息"""
        # 获取步态调度器状态
        gait_data = self.gait_scheduler.get_gait_state_data()
        timing_info = self.gait_scheduler.get_timing_info()
        leg_states = self.gait_scheduler.get_leg_states()
        targets = self.gait_scheduler.get_target_foot_positions()
        
        # 记录状态历史
        state_record = {
            'timestamp': time.time(),
            'total_time': gait_data['total_time'],
            'current_state': gait_data['current_state'],
            'leg_state': gait_data['leg_state'],
            'swing_leg': gait_data['swing_leg'],
            'support_leg': gait_data['support_leg'],
            'swing_elapsed_time': gait_data['swing_elapsed_time'],
            'swing_progress': timing_info['swing_progress'],
            'cycle_phase': timing_info['cycle_phase'],
            'target_positions': targets.copy(),
            'step_phase': gait_data['current_step_phase']
        }
        
        self.state_history.append(state_record)
        
        # 分别记录关键状态
        self.leg_state_history.append({
            'time': gait_data['total_time'],
            'leg_state': gait_data['leg_state'],
            'swing_leg': gait_data['swing_leg'],
            'support_leg': gait_data['support_leg']
        })
        
        self.target_position_history.append({
            'time': gait_data['total_time'],
            'left_foot': targets['left_foot'].copy(),
            'right_foot': targets['right_foot'].copy()
        })
        
        self.timing_history.append({
            'time': gait_data['total_time'],
            'swing_elapsed': gait_data['swing_elapsed_time'],
            'swing_progress': timing_info['swing_progress'],
            'cycle_phase': timing_info['cycle_phase']
        })
    
    def manually_trigger_swing_completion(self, leg: str):
        """手动触发摆动完成条件"""
        print(f"[手动触发] {leg}腿摆动完成")
        
        # 模拟着地条件
        if leg == "left":
            self.gait_scheduler.left_foot_force = 250.0  # 增加接触力
            self.gait_scheduler.left_foot_velocity = np.array([0.0, 0.0, 0.0])  # 速度归零
            self.gait_scheduler.left_foot_contact = True
        elif leg == "right":
            self.gait_scheduler.right_foot_force = 250.0
            self.gait_scheduler.right_foot_velocity = np.array([0.0, 0.0, 0.0])
            self.gait_scheduler.right_foot_contact = True
        
        # 也可以直接设置时间条件满足
        if hasattr(self.gait_scheduler, 'swing_elapsed_time'):
            self.gait_scheduler.swing_elapsed_time = self.gait_scheduler.config.swing_time
    
    def validate_leg_state_alternation(self) -> bool:
        """验证腿部状态交替是否正确"""
        print("\n[验证] 腿部状态交替逻辑")
        
        if len(self.leg_state_history) < 3:
            print("❌ 状态历史记录不足，无法验证交替")
            return False
        
        transitions = []
        for i in range(1, len(self.leg_state_history)):
            prev = self.leg_state_history[i-1]
            curr = self.leg_state_history[i]
            
            if prev['leg_state'] != curr['leg_state']:
                transitions.append({
                    'time': curr['time'],
                    'from': prev['leg_state'],
                    'to': curr['leg_state'],
                    'swing_from': prev['swing_leg'],
                    'swing_to': curr['swing_leg']
                })
        
        print(f"检测到 {len(transitions)} 次状态转换:")
        for i, trans in enumerate(transitions):
            print(f"  {i+1}. {trans['time']:.3f}s: {trans['from']} → {trans['to']}")
            print(f"     摆动腿: {trans['swing_from']} → {trans['swing_to']}")
        
        # 验证交替逻辑
        expected_sequence = ["double_support", "left_support", "double_support", "right_support"]
        alternation_correct = True
        
        for i, trans in enumerate(transitions):
            expected_idx = i % len(expected_sequence)
            if trans['to'] != expected_sequence[expected_idx]:
                print(f"❌ 状态转换异常: 期望 {expected_sequence[expected_idx]}, 实际 {trans['to']}")
                alternation_correct = False
        
        if alternation_correct:
            print("✅ 腿部状态交替正确")
        
        return alternation_correct
    
    def validate_target_positions(self) -> bool:
        """验证目标足部位置的合理性"""
        print("\n[验证] 目标足部位置合理性")
        
        position_valid = True
        
        for i, record in enumerate(self.target_position_history):
            left_pos = record['left_foot']
            right_pos = record['right_foot']
            
            # 检查位置范围合理性
            for foot, pos in [("left", left_pos), ("right", right_pos)]:
                x, y, z = pos['x'], pos['y'], pos['z']
                
                # 合理性检查
                if abs(x) > 1.0:  # 前后位置不应超过1米
                    print(f"❌ {foot}脚X坐标异常: {x:.3f}m (时间: {record['time']:.3f}s)")
                    position_valid = False
                
                if abs(y) > 0.5:  # 左右位置不应超过0.5米
                    print(f"❌ {foot}脚Y坐标异常: {y:.3f}m (时间: {record['time']:.3f}s)")
                    position_valid = False
                
                if abs(z) > 0.2:  # 高度不应超过0.2米
                    print(f"❌ {foot}脚Z坐标异常: {z:.3f}m (时间: {record['time']:.3f}s)")
                    position_valid = False
        
        # 检查左右脚间距
        for record in self.target_position_history[::10]:  # 每10个记录检查一次
            left_pos = record['left_foot']
            right_pos = record['right_foot']
            
            y_distance = abs(left_pos['y'] - right_pos['y'])
            if y_distance < 0.08 or y_distance > 0.4:  # 间距应在8-40cm之间
                print(f"❌ 左右脚间距异常: {y_distance:.3f}m (时间: {record['time']:.3f}s)")
                position_valid = False
        
        if position_valid:
            print("✅ 目标位置在合理范围内")
            
            # 显示位置统计
            all_left_x = [r['left_foot']['x'] for r in self.target_position_history]
            all_right_x = [r['right_foot']['x'] for r in self.target_position_history]
            all_left_y = [r['left_foot']['y'] for r in self.target_position_history]
            all_right_y = [r['right_foot']['y'] for r in self.target_position_history]
            
            print(f"  左脚X范围: [{min(all_left_x):.3f}, {max(all_left_x):.3f}]m")
            print(f"  右脚X范围: [{min(all_right_x):.3f}, {max(all_right_x):.3f}]m")
            print(f"  左脚Y范围: [{min(all_left_y):.3f}, {max(all_left_y):.3f}]m")
            print(f"  右脚Y范围: [{min(all_right_y):.3f}, {max(all_right_y):.3f}]m")
        
        return position_valid
    
    def validate_state_transitions(self) -> bool:
        """验证状态转换的连续性（无跳变卡顿）"""
        print("\n[验证] 状态转换连续性")
        
        transitions_smooth = True
        
        # 检查状态转换时间的合理性
        state_durations = {}
        current_state_start = 0
        
        for i, record in enumerate(self.state_history):
            if i == 0:
                current_state = record['current_state']
                current_state_start = record['total_time']
                continue
            
            if record['current_state'] != current_state:
                # 状态发生了转换
                duration = record['total_time'] - current_state_start
                
                if current_state not in state_durations:
                    state_durations[current_state] = []
                state_durations[current_state].append(duration)
                
                print(f"  {current_state} 持续 {duration:.3f}s")
                
                # 检查持续时间是否合理
                if current_state == "left_support" or current_state == "right_support":
                    if duration < 0.1 or duration > 2.0:  # 支撑相应在0.1-2.0s范围内
                        print(f"❌ {current_state} 持续时间异常: {duration:.3f}s")
                        transitions_smooth = False
                
                elif "double_support" in current_state:
                    if duration < 0.05 or duration > 0.5:  # 双支撑应在0.05-0.5s范围内
                        print(f"❌ {current_state} 持续时间异常: {duration:.3f}s")
                        transitions_smooth = False
                
                current_state = record['current_state']
                current_state_start = record['total_time']
        
        # 检查时间连续性
        for i in range(1, len(self.timing_history)):
            prev_time = self.timing_history[i-1]['time']
            curr_time = self.timing_history[i]['time']
            time_gap = curr_time - prev_time
            
            if time_gap > self.dt * 2:  # 时间间隔不应超过2个控制周期
                print(f"❌ 时间跳跃: {prev_time:.3f}s → {curr_time:.3f}s (间隔 {time_gap:.3f}s)")
                transitions_smooth = False
        
        if transitions_smooth:
            print("✅ 状态转换连续平滑")
        
        return transitions_smooth
    
    def validate_data_bus_consistency(self) -> bool:
        """验证数据总线字段的逻辑一致性"""
        print("\n[验证] 数据总线字段一致性")
        
        consistency_ok = True
        
        # 检查数据总线状态
        for i, record in enumerate(self.state_history[::5]):  # 每5个记录检查一次
            # 获取数据总线当前状态
            current_leg_state = self.data_bus.legState
            current_swing_leg = self.data_bus.swing_leg
            current_support_leg = self.data_bus.support_leg
            
            # 与步态调度器状态对比
            if current_leg_state != record['leg_state']:
                print(f"❌ legState 不一致: DataBus={current_leg_state}, Scheduler={record['leg_state']}")
                consistency_ok = False
            
            if current_swing_leg != record['swing_leg']:
                print(f"❌ swing_leg 不一致: DataBus={current_swing_leg}, Scheduler={record['swing_leg']}")
                consistency_ok = False
            
            if current_support_leg != record['support_leg']:
                print(f"❌ support_leg 不一致: DataBus={current_support_leg}, Scheduler={record['support_leg']}")
                consistency_ok = False
        
        # 检查目标位置同步
        data_bus_targets = self.data_bus.target_foot_pos
        scheduler_targets = self.gait_scheduler.get_target_foot_positions()
        
        for foot in ['left_foot', 'right_foot']:
            if foot in data_bus_targets and foot in scheduler_targets:
                db_pos = data_bus_targets[foot]
                sc_pos = scheduler_targets[foot]
                
                for coord in ['x', 'y', 'z']:
                    if abs(db_pos.get(coord, 0) - sc_pos.get(coord, 0)) > 1e-6:
                        print(f"❌ {foot} {coord}坐标不一致: DataBus={db_pos.get(coord, 0):.6f}, Scheduler={sc_pos.get(coord, 0):.6f}")
                        consistency_ok = False
        
        if consistency_ok:
            print("✅ 数据总线字段与调度器状态一致")
        
        return consistency_ok
    
    def print_state_summary(self):
        """打印状态摘要"""
        print("\n" + "="*80)
        print("步态逻辑测试状态摘要")
        print("="*80)
        
        if not self.state_history:
            print("❌ 无状态历史记录")
            return
        
        total_time = self.state_history[-1]['total_time']
        total_records = len(self.state_history)
        
        print(f"测试总时间: {total_time:.3f}s")
        print(f"记录总数: {total_records}")
        print(f"平均记录间隔: {total_time/total_records:.4f}s")
        
        # 状态统计
        state_counts = {}
        for record in self.state_history:
            state = record['current_state']
            state_counts[state] = state_counts.get(state, 0) + 1
        
        print(f"\n状态分布:")
        for state, count in state_counts.items():
            percentage = count / total_records * 100
            print(f"  {state}: {count} ({percentage:.1f}%)")
        
        # 腿部状态统计
        leg_state_counts = {}
        for record in self.leg_state_history:
            leg_state = record['leg_state']
            leg_state_counts[leg_state] = leg_state_counts.get(leg_state, 0) + 1
        
        print(f"\n腿部状态分布:")
        for leg_state, count in leg_state_counts.items():
            percentage = count / len(self.leg_state_history) * 100
            print(f"  {leg_state}: {count} ({percentage:.1f}%)")
        
        # 显示最后几个状态
        print(f"\n最近状态变化:")
        for i in range(max(0, len(self.state_history)-5), len(self.state_history)):
            record = self.state_history[i]
            print(f"  {record['total_time']:.3f}s: {record['current_state']} | "
                  f"摆动腿={record['swing_leg']} | 进度={record['swing_progress']*100:.1f}%")


def run_standalone_gait_logic_test():
    """运行独立步态逻辑测试"""
    print("="*80)
    print("AzureLoong机器人步态逻辑独立测试")
    print("="*80)
    
    # 初始化测试器
    tester = GaitLogicTester()
    tester.setup_fixed_robot_mode()
    
    # 开始步态但不实际移动
    print(f"\n[测试开始] 启动步态状态机")
    tester.gait_scheduler.start_walking()
    
    # 测试参数
    test_duration = 3.0  # 测试3秒
    test_cycles = int(test_duration / tester.dt)
    
    manual_trigger_times = [
        (1.0, "right"),  # 1秒时手动完成右腿摆动
        (2.0, "left"),   # 2秒时手动完成左腿摆动
        (2.8, "right"),  # 2.8秒时再次完成右腿摆动
    ]
    
    trigger_idx = 0
    
    print(f"运行 {test_duration}s 测试 ({test_cycles} 个周期)")
    print("-" * 80)
    
    # 主测试循环
    for cycle in range(test_cycles):
        current_time = cycle * tester.dt
        
        # 检查是否需要手动触发
        if (trigger_idx < len(manual_trigger_times) and 
            current_time >= manual_trigger_times[trigger_idx][0]):
            
            trigger_time, trigger_leg = manual_trigger_times[trigger_idx]
            tester.manually_trigger_swing_completion(trigger_leg)
            trigger_idx += 1
        
        # 推进状态机
        state_changed = tester.gait_scheduler.update_gait_state(tester.dt)
        
        # 记录状态
        tester.record_state()
        
        # 显示重要状态变化
        if state_changed:
            current_record = tester.state_history[-1]
            print(f"[{current_time:.3f}s] 状态变化: {current_record['current_state']} | "
                  f"摆动腿: {current_record['swing_leg']} | "
                  f"支撑腿: {current_record['support_leg']}")
    
    print("-" * 80)
    print(f"测试完成! 共 {len(tester.state_history)} 个状态记录")
    
    # 验证测试结果
    print(f"\n开始验证测试结果...")
    
    all_tests_passed = True
    
    # 1. 验证腿部状态交替
    if not tester.validate_leg_state_alternation():
        all_tests_passed = False
    
    # 2. 验证目标位置合理性
    if not tester.validate_target_positions():
        all_tests_passed = False
    
    # 3. 验证状态转换连续性
    if not tester.validate_state_transitions():
        all_tests_passed = False
    
    # 4. 验证数据总线一致性
    if not tester.validate_data_bus_consistency():
        all_tests_passed = False
    
    # 打印摘要
    tester.print_state_summary()
    
    # 最终结果
    print(f"\n" + "="*80)
    if all_tests_passed:
        print("🎉 步态逻辑测试全部通过!")
        print("✅ legState 左右支撑交替正确")
        print("✅ target_foot_pos 位置合理")
        print("✅ 状态转换连续平滑")
        print("✅ 数据总线字段一致")
    else:
        print("❌ 步态逻辑测试发现问题")
        print("请检查上述验证结果中的错误信息")
    
    print("="*80)
    
    return all_tests_passed


if __name__ == "__main__":
    try:
        success = run_standalone_gait_logic_test()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 