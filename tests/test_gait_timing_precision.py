#!/usr/bin/env python3
"""
步态节拍精度测试

验证要求：
1. 每步摆动时长 tSwing 达到配置值
2. 整体步态周期 Tcycle 达到配置值  
3. 实测周期误差 ≤ ±5%
4. 通过日志和自动脚本统计 Δt
"""

import time
import numpy as np
import statistics
from typing import List, Dict, Tuple
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig, GaitState, LegState


class GaitTimingTester:
    """步态节拍精度测试器"""
    
    def __init__(self):
        # 测试配置
        self.target_test_cycles = 10  # 目标测试周期数
        self.dt = 0.001  # 1ms 高精度控制周期
        
        # 配置步态参数
        config = GaitSchedulerConfig()
        config.swing_time = 0.4         # 400ms 摆动时间
        config.stance_time = 0.6        # 600ms 支撑时间
        config.double_support_time = 0.1 # 100ms 双支撑时间
        config.use_time_trigger = True
        config.use_sensor_trigger = True
        config.require_both_triggers = False
        config.enable_logging = True
        
        self.scheduler = get_gait_scheduler(config)
        self.data_bus = get_data_bus()
        
        # 期望值
        self.expected_swing_time = config.swing_time
        self.expected_stance_time = config.stance_time
        self.expected_double_support_time = config.double_support_time
        # 一个完整周期 = 左摆动 + 双支撑 + 右摆动 + 双支撑
        self.expected_cycle_time = 2 * (config.swing_time + config.double_support_time)
        
        # 测量数据
        self.swing_measurements = []  # 摆动时长测量
        self.stance_measurements = []  # 支撑时长测量
        self.double_support_measurements = []  # 双支撑时长测量
        self.cycle_measurements = []  # 完整周期测量
        
        # 状态记录
        self.state_history = []
        self.transition_times = []
        
        print("="*80)
        print("步态节拍精度测试")
        print("="*80)
        print(f"期望摆动时间: {self.expected_swing_time*1000:.1f}ms")
        print(f"期望双支撑时间: {self.expected_double_support_time*1000:.1f}ms")
        print(f"期望完整周期: {self.expected_cycle_time*1000:.1f}ms")
        print(f"测量精度: {self.dt*1000:.1f}ms")
        print(f"目标误差: ≤ ±5%")
    
    def setup_precise_timing(self):
        """设置精确计时"""
        # 设置初始传感器状态
        self.scheduler.left_foot_force = 200.0
        self.scheduler.right_foot_force = 200.0
        self.scheduler.left_foot_contact = True
        self.scheduler.right_foot_contact = True
        
        # 启动步态
        self.scheduler.start_walking()
        
        print(f"\n步态启动完成:")
        print(f"  初始状态: {self.scheduler.current_state.value}")
        print(f"  初始摆动腿: {self.scheduler.swing_leg}")
    
    def simulate_realistic_contact(self, current_time: float):
        """模拟真实的足底接触"""
        swing_leg = self.scheduler.swing_leg
        swing_progress = 0.0
        
        if swing_leg != "none" and self.scheduler.config.swing_time > 0:
            swing_progress = self.scheduler.swing_elapsed_time / self.scheduler.config.swing_time
        
        # 模拟真实的接触时序
        if swing_leg == "right":
            # 左腿支撑，始终接触
            self.scheduler.left_foot_contact = True
            self.scheduler.left_foot_force = 200.0
            # 右腿在摆动95%时着地
            if swing_progress >= 0.95:
                self.scheduler.right_foot_contact = True
                self.scheduler.right_foot_force = 200.0
            else:
                self.scheduler.right_foot_contact = False
                self.scheduler.right_foot_force = 0.0
                
        elif swing_leg == "left":
            # 右腿支撑，始终接触
            self.scheduler.right_foot_contact = True
            self.scheduler.right_foot_force = 200.0
            # 左腿在摆动95%时着地
            if swing_progress >= 0.95:
                self.scheduler.left_foot_contact = True
                self.scheduler.left_foot_force = 200.0
            else:
                self.scheduler.left_foot_contact = False
                self.scheduler.left_foot_force = 0.0
        else:
            # 双支撑期间都接触
            self.scheduler.left_foot_contact = True
            self.scheduler.right_foot_contact = True
            self.scheduler.left_foot_force = 200.0
            self.scheduler.right_foot_force = 200.0
    
    def record_state_transition(self, current_time: float, state_changed: bool):
        """记录状态转换"""
        current_state = {
            'time': current_time,
            'state': self.scheduler.current_state.value,
            'leg_state': self.scheduler.leg_state.value,
            'swing_leg': self.scheduler.swing_leg,
            'swing_elapsed': self.scheduler.swing_elapsed_time,
            'total_time': self.scheduler.total_time
        }
        self.state_history.append(current_state)
        
        if state_changed and len(self.state_history) > 1:
            prev_state = self.state_history[-2]
            
            transition = {
                'time': current_time,
                'from_state': prev_state['state'],
                'to_state': current_state['state'],
                'duration': current_time - prev_state['time'],
                'from_swing': prev_state['swing_leg'],
                'to_swing': current_state['swing_leg']
            }
            self.transition_times.append(transition)
            
            print(f"[{current_time:.3f}s] {transition['from_state']} → {transition['to_state']} "
                  f"(时长: {transition['duration']*1000:.1f}ms)")
    
    def analyze_swing_times(self):
        """分析摆动时间"""
        print(f"\n{'='*60}")
        print("摆动时间分析")
        print('='*60)
        
        swing_durations = []
        
        # 分析状态转换，寻找摆动阶段
        for i, trans in enumerate(self.transition_times):
            # 检测摆动结束的转换（单支撑 → 双支撑）
            if (trans['from_state'] in ['left_support', 'right_support'] and 
                'double_support' in trans['to_state']):
                
                swing_duration = trans['duration']
                swing_durations.append(swing_duration)
                
                error_ms = (swing_duration - self.expected_swing_time) * 1000
                error_percent = (swing_duration - self.expected_swing_time) / self.expected_swing_time * 100
                
                print(f"摆动 {len(swing_durations)}: {swing_duration*1000:.1f}ms "
                      f"(误差: {error_ms:+.1f}ms, {error_percent:+.1f}%)")
        
        self.swing_measurements = swing_durations
        
        if swing_durations:
            avg_swing = statistics.mean(swing_durations)
            std_swing = statistics.stdev(swing_durations) if len(swing_durations) > 1 else 0.0
            min_swing = min(swing_durations)
            max_swing = max(swing_durations)
            
            avg_error_percent = (avg_swing - self.expected_swing_time) / self.expected_swing_time * 100
            
            print(f"\n摆动时间统计:")
            print(f"  期望值: {self.expected_swing_time*1000:.1f}ms")
            print(f"  测量数: {len(swing_durations)}")
            print(f"  平均值: {avg_swing*1000:.1f}ms (误差: {avg_error_percent:+.1f}%)")
            print(f"  标准差: {std_swing*1000:.1f}ms")
            print(f"  最小值: {min_swing*1000:.1f}ms")
            print(f"  最大值: {max_swing*1000:.1f}ms")
            
            # 检查精度要求
            if abs(avg_error_percent) <= 5.0:
                print(f"  ✅ 摆动时间精度合格 (|{avg_error_percent:.1f}%| ≤ 5%)")
                return True
            else:
                print(f"  ❌ 摆动时间精度不合格 (|{avg_error_percent:.1f}%| > 5%)")
                return False
        else:
            print("❌ 未检测到摆动时间数据")
            return False
    
    def analyze_double_support_times(self):
        """分析双支撑时间"""
        print(f"\n{'='*60}")
        print("双支撑时间分析")
        print('='*60)
        
        double_support_durations = []
        
        # 分析双支撑阶段的持续时间
        for i, trans in enumerate(self.transition_times):
            # 检测双支撑结束的转换（双支撑 → 单支撑）
            if ('double_support' in trans['from_state'] and 
                trans['to_state'] in ['left_support', 'right_support']):
                
                double_support_duration = trans['duration']
                double_support_durations.append(double_support_duration)
                
                error_ms = (double_support_duration - self.expected_double_support_time) * 1000
                error_percent = (double_support_duration - self.expected_double_support_time) / self.expected_double_support_time * 100
                
                print(f"双支撑 {len(double_support_durations)}: {double_support_duration*1000:.1f}ms "
                      f"(误差: {error_ms:+.1f}ms, {error_percent:+.1f}%)")
        
        self.double_support_measurements = double_support_durations
        
        if double_support_durations:
            avg_ds = statistics.mean(double_support_durations)
            std_ds = statistics.stdev(double_support_durations) if len(double_support_durations) > 1 else 0.0
            avg_error_percent = (avg_ds - self.expected_double_support_time) / self.expected_double_support_time * 100
            
            print(f"\n双支撑时间统计:")
            print(f"  期望值: {self.expected_double_support_time*1000:.1f}ms")
            print(f"  测量数: {len(double_support_durations)}")
            print(f"  平均值: {avg_ds*1000:.1f}ms (误差: {avg_error_percent:+.1f}%)")
            print(f"  标准差: {std_ds*1000:.1f}ms")
            
            if abs(avg_error_percent) <= 5.0:
                print(f"  ✅ 双支撑时间精度合格 (|{avg_error_percent:.1f}%| ≤ 5%)")
                return True
            else:
                print(f"  ❌ 双支撑时间精度不合格 (|{avg_error_percent:.1f}%| > 5%)")
                return False
        else:
            print("❌ 未检测到双支撑时间数据")
            return False
    
    def analyze_cycle_times(self):
        """分析完整步态周期"""
        print(f"\n{'='*60}")
        print("完整步态周期分析")
        print('='*60)
        
        cycle_durations = []
        
        # 查找完整周期：left_support → double_support_lr → right_support → double_support_rl → left_support
        cycle_states = ['left_support', 'double_support_lr', 'right_support', 'double_support_rl']
        
        i = 0
        while i < len(self.transition_times) - 3:
            # 查找周期起点
            if self.transition_times[i]['to_state'] == 'left_support':
                cycle_start_time = self.transition_times[i]['time']
                cycle_start_idx = i
                
                # 检查是否有完整的周期序列
                cycle_complete = True
                for j, expected_state in enumerate(cycle_states[1:], 1):  # 跳过起始状态
                    if (cycle_start_idx + j >= len(self.transition_times) or
                        self.transition_times[cycle_start_idx + j]['to_state'] != expected_state):
                        cycle_complete = False
                        break
                
                if cycle_complete and cycle_start_idx + 4 < len(self.transition_times):
                    # 找到下一个left_support作为周期结束
                    if self.transition_times[cycle_start_idx + 4]['to_state'] == 'left_support':
                        cycle_end_time = self.transition_times[cycle_start_idx + 4]['time']
                        cycle_duration = cycle_end_time - cycle_start_time
                        cycle_durations.append(cycle_duration)
                        
                        error_ms = (cycle_duration - self.expected_cycle_time) * 1000
                        error_percent = (cycle_duration - self.expected_cycle_time) / self.expected_cycle_time * 100
                        
                        print(f"周期 {len(cycle_durations)}: {cycle_duration*1000:.1f}ms "
                              f"(误差: {error_ms:+.1f}ms, {error_percent:+.1f}%)")
                        
                        i += 4  # 跳到下一个周期
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        self.cycle_measurements = cycle_durations
        
        if cycle_durations:
            avg_cycle = statistics.mean(cycle_durations)
            std_cycle = statistics.stdev(cycle_durations) if len(cycle_durations) > 1 else 0.0
            avg_error_percent = (avg_cycle - self.expected_cycle_time) / self.expected_cycle_time * 100
            
            print(f"\n完整周期统计:")
            print(f"  期望值: {self.expected_cycle_time*1000:.1f}ms")
            print(f"  测量数: {len(cycle_durations)}")
            print(f"  平均值: {avg_cycle*1000:.1f}ms (误差: {avg_error_percent:+.1f}%)")
            print(f"  标准差: {std_cycle*1000:.1f}ms")
            
            if abs(avg_error_percent) <= 5.0:
                print(f"  ✅ 周期时间精度合格 (|{avg_error_percent:.1f}%| ≤ 5%)")
                return True
            else:
                print(f"  ❌ 周期时间精度不合格 (|{avg_error_percent:.1f}%| > 5%)")
                return False
        else:
            print("❌ 未检测到完整周期数据")
            return False
    
    def generate_timing_log(self):
        """生成详细的时序日志"""
        print(f"\n{'='*60}")
        print("详细时序日志")
        print('='*60)
        
        print(f"时间戳\t\t状态\t\t\t持续时间(ms)\t摆动进度")
        print("-" * 70)
        
        for i, record in enumerate(self.state_history[::10]):  # 每10个记录显示一次
            if record['swing_leg'] != 'none':
                swing_progress = record['swing_elapsed'] / self.scheduler.config.swing_time * 100
                progress_str = f"{swing_progress:.1f}%"
            else:
                progress_str = "N/A"
            
            if i > 0:
                duration_ms = (record['time'] - self.state_history[(i-1)*10]['time']) * 1000
            else:
                duration_ms = 0.0
            
            print(f"{record['time']:.3f}s\t\t{record['state']:<15}\t{duration_ms:.1f}ms\t\t{progress_str}")
    
    def run_precision_test(self) -> bool:
        """运行精度测试"""
        print(f"\n开始精度测试...")
        
        self.setup_precise_timing()
        
        # 测试参数
        max_test_time = 15.0  # 最大测试时间15秒
        test_steps = int(max_test_time / self.dt)
        
        print(f"测试参数:")
        print(f"  最大测试时间: {max_test_time}s")
        print(f"  控制周期: {self.dt*1000:.1f}ms")
        print(f"  总步数: {test_steps}")
        print("-" * 80)
        
        start_time = time.time()
        
        # 主测试循环
        for step in range(test_steps):
            current_time = step * self.dt
            
            # 模拟真实传感器
            self.simulate_realistic_contact(current_time)
            
            # 推进状态机
            state_changed = self.scheduler.update_gait_state(self.dt)
            
            # 记录状态
            self.record_state_transition(current_time, state_changed)
            
            # 检查是否有足够的测量数据
            if len(self.cycle_measurements) >= self.target_test_cycles:
                print(f"✅ 达到目标周期数 ({len(self.cycle_measurements)})，测试完成")
                break
            
            # 每秒显示进度
            if step % int(1.0 / self.dt) == 0 and step > 0:
                cycles_detected = len(self.cycle_measurements)
                swings_detected = len(self.swing_measurements)
                print(f"[{current_time:.1f}s] 进度: {cycles_detected} 周期, {swings_detected} 摆动")
        
        test_duration = time.time() - start_time
        print(f"\n测试完成，用时 {test_duration:.2f}s")
        print(f"总状态转换: {len(self.transition_times)}")
        
        # 分析结果
        swing_ok = self.analyze_swing_times()
        double_support_ok = self.analyze_double_support_times()
        cycle_ok = self.analyze_cycle_times()
        
        # 生成日志
        self.generate_timing_log()
        
        # 最终结果
        print(f"\n{'='*80}")
        print("节拍精度测试结果")
        print('='*80)
        
        all_tests_passed = swing_ok and double_support_ok and cycle_ok
        
        print(f"1. 摆动时间精度: {'✅ 通过' if swing_ok else '❌ 失败'}")
        print(f"2. 双支撑时间精度: {'✅ 通过' if double_support_ok else '❌ 失败'}")
        print(f"3. 完整周期精度: {'✅ 通过' if cycle_ok else '❌ 失败'}")
        
        if all_tests_passed:
            print(f"\n🎉 所有节拍精度测试通过!")
            print(f"✅ tSwing 误差 ≤ ±5%")
            print(f"✅ Tcycle 误差 ≤ ±5%")
            print(f"✅ 时序控制精确")
        else:
            print(f"\n❌ 部分精度测试未通过")
            print(f"需要调整控制参数或算法")
        
        print('='*80)
        
        return all_tests_passed


def main():
    """主函数"""
    try:
        tester = GaitTimingTester()
        success = tester.run_precision_test()
        return success
    except Exception as e:
        print(f"❌ 精度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 