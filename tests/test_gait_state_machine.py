#!/usr/bin/env python3
"""
步态有限状态机功能正确性测试

验证要求：
1. 左腿支撑 → 右腿支撑（或含双支撑过渡）按设定节拍循环
2. 连续 ≥ 20 个步态周期无跳变/卡死
3. legState 与仿真足底接触状态一致：
   - 支撑腿始终检测到接触
   - 摆动腿在 tSwing 内不接触
   - 落地后立即转为支撑
"""

import time
import numpy as np
from typing import List, Dict, Tuple
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig, GaitState, LegState


class GaitStateMachineTester:
    """步态状态机测试器"""
    
    def __init__(self):
        # 测试配置
        self.target_cycles = 20  # 目标测试周期数
        self.dt = 0.005  # 5ms 控制周期，更精确
        
        # 状态记录
        self.cycle_history = []
        self.contact_history = []
        self.state_transitions = []
        self.errors = []
        
        # 初始化系统
        config = GaitSchedulerConfig()
        config.swing_time = 0.4         # 400ms 摆动时间
        config.stance_time = 0.6        # 600ms 支撑时间
        config.double_support_time = 0.1 # 100ms 双支撑时间
        config.use_time_trigger = True   # 使用时间触发
        config.use_sensor_trigger = True # 同时使用传感器触发
        config.require_both_triggers = False # 不要求两个触发同时满足
        config.enable_logging = False
        
        self.scheduler = get_gait_scheduler(config)
        self.data_bus = get_data_bus()
        
        # 仿真足底接触状态
        self.sim_left_contact = True   # 仿真左脚接触状态
        self.sim_right_contact = True  # 仿真右脚接触状态
        
        print("✓ 步态状态机测试器初始化完成")
        print(f"  目标测试周期: {self.target_cycles}")
        print(f"  控制频率: {1/self.dt:.0f}Hz")
        print(f"  摆动时间: {config.swing_time}s")
        
    def simulate_foot_contact(self, current_time: float):
        """模拟足底接触状态"""
        # 根据当前步态状态模拟接触
        current_state = self.scheduler.current_state
        swing_leg = self.scheduler.swing_leg
        swing_progress = self.scheduler.swing_elapsed_time / self.scheduler.config.swing_time
        
        # 支撑腿始终接触
        if swing_leg == "right":
            self.sim_left_contact = True   # 左腿支撑，始终接触
            # 右腿摆动期间不接触，但在摆动末期（>95%）重新接触
            if swing_progress < 0.95:
                self.sim_right_contact = False
            else:
                self.sim_right_contact = True  # 摆动末期着地
        elif swing_leg == "left":
            self.sim_right_contact = True  # 右腿支撑，始终接触
            # 左腿摆动期间不接触，但在摆动末期重新接触
            if swing_progress < 0.95:
                self.sim_left_contact = False
            else:
                self.sim_left_contact = True
        else:
            # 双支撑期间都接触
            self.sim_left_contact = True
            self.sim_right_contact = True
        
        # 更新调度器的传感器数据
        self.scheduler.left_foot_contact = self.sim_left_contact
        self.scheduler.right_foot_contact = self.sim_right_contact
        
        # 设置相应的力传感器值
        self.scheduler.left_foot_force = 200.0 if self.sim_left_contact else 0.0
        self.scheduler.right_foot_force = 200.0 if self.sim_right_contact else 0.0
        
        # 设置速度（接触时速度为0）
        self.scheduler.left_foot_velocity = np.array([0.0, 0.0, 0.0]) if self.sim_left_contact else np.array([0.1, 0.0, 0.0])
        self.scheduler.right_foot_velocity = np.array([0.0, 0.0, 0.0]) if self.sim_right_contact else np.array([0.1, 0.0, 0.0])
    
    def record_state(self, current_time: float):
        """记录当前状态"""
        # 记录接触状态历史
        contact_record = {
            'time': current_time,
            'sim_left_contact': self.sim_left_contact,
            'sim_right_contact': self.sim_right_contact,
            'scheduler_left_contact': self.scheduler.left_foot_contact,
            'scheduler_right_contact': self.scheduler.right_foot_contact,
            'leg_state': self.scheduler.leg_state.value,
            'swing_leg': self.scheduler.swing_leg,
            'support_leg': self.scheduler.support_leg,
            'swing_progress': self.scheduler.swing_elapsed_time / self.scheduler.config.swing_time if self.scheduler.swing_leg != "none" else 0.0
        }
        self.contact_history.append(contact_record)
    
    def detect_state_transitions(self):
        """检测状态转换"""
        if len(self.contact_history) < 2:
            return
        
        prev_record = self.contact_history[-2]
        curr_record = self.contact_history[-1]
        
        # 检测腿部状态转换
        if prev_record['leg_state'] != curr_record['leg_state']:
            transition = {
                'time': curr_record['time'],
                'from_state': prev_record['leg_state'],
                'to_state': curr_record['leg_state'],
                'from_swing': prev_record['swing_leg'],
                'to_swing': curr_record['swing_leg']
            }
            self.state_transitions.append(transition)
            
            print(f"[{curr_record['time']:.3f}s] 状态转换: {transition['from_state']} → {transition['to_state']}")
            print(f"  摆动腿: {transition['from_swing']} → {transition['to_swing']}")
    
    def validate_contact_consistency(self) -> Tuple[bool, List[str]]:
        """验证接触状态一致性"""
        print("\n" + "="*60)
        print("接触状态一致性验证")
        print("="*60)
        
        errors = []
        
        for i, record in enumerate(self.contact_history):
            time_str = f"{record['time']:.3f}s"
            
            # 1. 验证支撑腿始终接触
            if record['support_leg'] == "left":
                if not record['sim_left_contact']:
                    errors.append(f"{time_str}: 左腿支撑但仿真显示不接触")
                if not record['scheduler_left_contact']:
                    errors.append(f"{time_str}: 左腿支撑但调度器显示不接触")
            elif record['support_leg'] == "right":
                if not record['sim_right_contact']:
                    errors.append(f"{time_str}: 右腿支撑但仿真显示不接触")
                if not record['scheduler_right_contact']:
                    errors.append(f"{time_str}: 右腿支撑但调度器显示不接触")
            elif record['support_leg'] == "both":
                if not (record['sim_left_contact'] and record['sim_right_contact']):
                    errors.append(f"{time_str}: 双支撑但仿真显示有腿不接触")
            
            # 2. 验证摆动腿在摆动期间不接触（除了着地瞬间）
            if record['swing_leg'] == "left" and record['swing_progress'] < 0.9:
                if record['sim_left_contact']:
                    errors.append(f"{time_str}: 左腿摆动期间({record['swing_progress']:.1%})仿真显示接触")
            elif record['swing_leg'] == "right" and record['swing_progress'] < 0.9:
                if record['sim_right_contact']:
                    errors.append(f"{time_str}: 右腿摆动期间({record['swing_progress']:.1%})仿真显示接触")
            
            # 3. 验证落地后立即转为支撑
            if record['swing_leg'] == "left" and record['swing_progress'] >= 0.95:
                if record['sim_left_contact'] and record['leg_state'] not in ["double_support", "left_support"]:
                    errors.append(f"{time_str}: 左腿已着地但未转为支撑状态")
            elif record['swing_leg'] == "right" and record['swing_progress'] >= 0.95:
                if record['sim_right_contact'] and record['leg_state'] not in ["double_support", "right_support"]:
                    errors.append(f"{time_str}: 右腿已着地但未转为支撑状态")
        
        # 显示验证结果
        if not errors:
            print("✅ 接触状态一致性验证通过")
            print(f"  验证记录数: {len(self.contact_history)}")
            print(f"  无一致性错误")
        else:
            print(f"❌ 发现 {len(errors)} 个一致性错误:")
            for error in errors[:10]:  # 只显示前10个错误
                print(f"    {error}")
            if len(errors) > 10:
                print(f"    ... 还有 {len(errors) - 10} 个错误")
        
        return len(errors) == 0, errors
    
    def analyze_gait_cycles(self) -> Tuple[bool, int]:
        """分析步态周期"""
        print("\n" + "="*60)
        print("步态周期分析")
        print("="*60)
        
        # 分析状态转换序列
        if len(self.state_transitions) < 4:
            print(f"❌ 状态转换不足: {len(self.state_transitions)} < 4")
            return False, 0
        
        # 计算完整周期数
        # 一个完整周期：left_support → double_support → right_support → double_support → left_support
        cycle_count = 0
        i = 0
        
        while i < len(self.state_transitions) - 3:
            # 查找周期模式
            if (self.state_transitions[i]['to_state'] == 'left_support' and
                i + 3 < len(self.state_transitions)):
                
                # 检查后续状态是否符合周期模式
                pattern_match = True
                expected_states = ['double_support', 'right_support', 'double_support']
                
                for j, expected_state in enumerate(expected_states):
                    if i + j + 1 >= len(self.state_transitions):
                        pattern_match = False
                        break
                    if self.state_transitions[i + j + 1]['to_state'] != expected_state:
                        pattern_match = False
                        break
                
                if pattern_match:
                    cycle_count += 1
                    cycle_start_time = self.state_transitions[i]['time']
                    cycle_end_time = self.state_transitions[i + 3]['time']
                    cycle_duration = cycle_end_time - cycle_start_time
                    
                    cycle_info = {
                        'cycle_number': cycle_count,
                        'start_time': cycle_start_time,
                        'end_time': cycle_end_time,
                        'duration': cycle_duration
                    }
                    self.cycle_history.append(cycle_info)
                    
                    print(f"周期 {cycle_count}: {cycle_start_time:.3f}s - {cycle_end_time:.3f}s (时长: {cycle_duration:.3f}s)")
                    i += 4  # 跳过这个完整周期
                else:
                    i += 1
            else:
                i += 1
        
        print(f"\n检测到完整步态周期数: {cycle_count}")
        
        if cycle_count >= self.target_cycles:
            print(f"✅ 达到目标周期数 ({cycle_count} >= {self.target_cycles})")
            return True, cycle_count
        else:
            print(f"❌ 未达到目标周期数 ({cycle_count} < {self.target_cycles})")
            return False, cycle_count
    
    def check_no_stuck_or_jump(self) -> bool:
        """检查无卡死或跳变"""
        print("\n" + "="*60)
        print("卡死/跳变检查")
        print("="*60)
        
        no_issues = True
        
        # 检查状态持续时间
        for i, trans in enumerate(self.state_transitions[1:], 1):
            prev_trans = self.state_transitions[i-1]
            duration = trans['time'] - prev_trans['time']
            
            # 检查状态持续时间是否合理
            if duration < 0.05:  # 少于50ms可能是跳变
                print(f"⚠️ 可能的状态跳变: {prev_trans['to_state']} 持续仅 {duration:.3f}s")
                no_issues = False
            elif duration > 2.0:  # 超过2s可能是卡死
                print(f"⚠️ 可能的状态卡死: {prev_trans['to_state']} 持续 {duration:.3f}s")
                no_issues = False
        
        # 检查状态转换的合理性
        valid_transitions = [
            ('left_support', 'double_support'),
            ('double_support', 'right_support'),
            ('right_support', 'double_support'),
            ('double_support', 'left_support')
        ]
        
        for trans in self.state_transitions:
            transition_pair = (trans['from_state'], trans['to_state'])
            if transition_pair not in valid_transitions:
                print(f"❌ 无效状态转换: {trans['from_state']} → {trans['to_state']} 在 {trans['time']:.3f}s")
                no_issues = False
        
        if no_issues:
            print("✅ 未发现卡死或跳变问题")
        
        return no_issues
    
    def run_test(self) -> bool:
        """运行主测试"""
        print("="*70)
        print("步态有限状态机功能正确性测试")
        print("="*70)
        
        # 启动步态
        self.scheduler.start_walking()
        print(f"步态启动: {self.scheduler.current_state.value}")
        
        # 测试参数
        max_test_time = 30.0  # 最大测试时间30s
        test_steps = int(max_test_time / self.dt)
        
        print(f"开始测试: 最大时长{max_test_time}s, 目标周期{self.target_cycles}")
        print("-" * 70)
        
        start_time = time.time()
        
        # 主测试循环
        for step in range(test_steps):
            current_time = step * self.dt
            
            # 模拟足底接触
            self.simulate_foot_contact(current_time)
            
            # 推进状态机
            state_changed = self.scheduler.update_gait_state(self.dt)
            
            # 记录状态
            self.record_state(current_time)
            
            # 检测状态转换
            if state_changed:
                self.detect_state_transitions()
            
            # 检查是否达到目标周期数
            if len(self.cycle_history) >= self.target_cycles:
                print(f"✅ 达到目标周期数，测试在 {current_time:.3f}s 完成")
                break
            
            # 每1秒显示进度
            if step % int(1.0 / self.dt) == 0 and step > 0:
                cycles_detected = len(self.cycle_history)
                print(f"[{current_time:.1f}s] 进度: {cycles_detected}/{self.target_cycles} 周期")
        
        test_duration = time.time() - start_time
        print(f"测试完成，用时 {test_duration:.2f}s")
        print("-" * 70)
        
        # 验证结果
        print("\n开始结果验证...")
        
        # 1. 验证步态周期
        cycles_ok, cycle_count = self.analyze_gait_cycles()
        
        # 2. 验证接触一致性
        contact_ok, contact_errors = self.validate_contact_consistency()
        
        # 3. 检查卡死/跳变
        no_issues_ok = self.check_no_stuck_or_jump()
        
        # 最终结果
        print("\n" + "="*70)
        print("测试结果总结")
        print("="*70)
        
        all_tests_passed = cycles_ok and contact_ok and no_issues_ok
        
        print(f"1. 步态周期测试: {'✅ 通过' if cycles_ok else '❌ 失败'} ({cycle_count}/{self.target_cycles})")
        print(f"2. 接触一致性测试: {'✅ 通过' if contact_ok else '❌ 失败'} ({len(contact_errors)} 错误)")
        print(f"3. 卡死/跳变检查: {'✅ 通过' if no_issues_ok else '❌ 失败'}")
        
        if all_tests_passed:
            print("\n🎉 步态有限状态机功能正确性测试全部通过!")
            print("✅ 状态按设定节拍循环")
            print("✅ 连续多周期无跳变/卡死")
            print("✅ legState与接触状态一致")
        else:
            print("\n❌ 测试发现问题，需要进一步调试")
        
        print("="*70)
        
        return all_tests_passed


def main():
    """主函数"""
    try:
        tester = GaitStateMachineTester()
        success = tester.run_test()
        return success
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 