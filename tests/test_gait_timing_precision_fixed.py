#!/usr/bin/env python3
"""
步态节拍精度测试 - 修复版

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


class SimplifiedGaitTester:
    """简化的步态精度测试器"""
    
    def __init__(self):
        # 测试配置
        self.target_test_cycles = 5  # 目标测试周期数
        self.dt = 0.005  # 5ms 控制周期
        
        # 配置步态参数  
        config = GaitSchedulerConfig()
        config.swing_time = 0.4         # 400ms 摆动时间
        config.stance_time = 0.6        # 600ms 支撑时间
        config.double_support_time = 0.1 # 100ms 双支撑时间
        config.use_time_trigger = True
        config.use_sensor_trigger = True
        config.require_both_triggers = False  # 只要时间或传感器条件满足即可
        config.enable_logging = False  # 关闭详细日志
        config.touchdown_force_threshold = 30.0
        
        self.scheduler = get_gait_scheduler(config)
        self.data_bus = get_data_bus()
        
        # 期望值
        self.expected_swing_time = config.swing_time
        self.expected_double_support_time = config.double_support_time
        # 一个完整周期 = 摆动 + 双支撑 + 摆动 + 双支撑
        self.expected_cycle_time = 2 * (config.swing_time + config.double_support_time)
        
        # 测量数据
        self.swing_measurements = []
        self.double_support_measurements = []
        self.cycle_measurements = []
        
        # 状态记录
        self.state_changes = []
        self.current_time = 0.0
        
        print("="*80)
        print("简化步态节拍精度测试")
        print("="*80)
        print(f"期望摆动时间: {self.expected_swing_time*1000:.1f}ms")
        print(f"期望双支撑时间: {self.expected_double_support_time*1000:.1f}ms") 
        print(f"期望完整周期: {self.expected_cycle_time*1000:.1f}ms")
        print(f"控制周期: {self.dt*1000:.1f}ms")
        print(f"目标误差: ≤ ±5%")
    
    def setup_initial_state(self):
        """设置初始状态"""
        # 确保传感器状态正确
        self.scheduler.left_foot_force = 100.0
        self.scheduler.right_foot_force = 100.0
        self.scheduler.left_foot_contact = True
        self.scheduler.right_foot_contact = True
        
        # 启动步态
        self.scheduler.start_walking()
        print(f"\n初始状态: {self.scheduler.current_state.value}")
        print(f"初始摆动腿: {self.scheduler.swing_leg}")
    
    def simulate_perfect_sensors(self):
        """模拟完美的传感器响应"""
        current_state = self.scheduler.current_state
        swing_progress = 0.0
        
        if self.scheduler.swing_leg != "none" and self.scheduler.config.swing_time > 0:
            swing_progress = self.scheduler.swing_elapsed_time / self.scheduler.config.swing_time
        
        # 根据状态和进度设置传感器
        if current_state == GaitState.LEFT_SUPPORT:
            # 左腿支撑，右腿摆动
            self.scheduler.left_foot_contact = True
            self.scheduler.left_foot_force = 100.0
            
            # 在摆动95%时右腿着地
            if swing_progress >= 0.95:
                self.scheduler.right_foot_contact = True
                self.scheduler.right_foot_force = 100.0
            else:
                self.scheduler.right_foot_contact = False
                self.scheduler.right_foot_force = 0.0
                
        elif current_state == GaitState.RIGHT_SUPPORT:
            # 右腿支撑，左腿摆动
            self.scheduler.right_foot_contact = True
            self.scheduler.right_foot_force = 100.0
            
            # 在摆动95%时左腿着地
            if swing_progress >= 0.95:
                self.scheduler.left_foot_contact = True
                self.scheduler.left_foot_force = 100.0
            else:
                self.scheduler.left_foot_contact = False
                self.scheduler.left_foot_force = 0.0
        else:
            # 双支撑或其他状态
            self.scheduler.left_foot_contact = True
            self.scheduler.right_foot_contact = True
            self.scheduler.left_foot_force = 100.0
            self.scheduler.right_foot_force = 100.0
    
    def record_state_change(self, old_state: str, new_state: str, duration: float):
        """记录状态变化"""
        change_record = {
            'time': self.current_time,
            'from_state': old_state,
            'to_state': new_state,
            'duration': duration
        }
        self.state_changes.append(change_record)
        
        print(f"[{self.current_time:.3f}s] {old_state} → {new_state} "
              f"(时长: {duration*1000:.1f}ms)")
    
    def run_timing_test(self) -> bool:
        """运行时序测试"""
        print(f"\n开始时序测试...")
        
        self.setup_initial_state()
        
        # 测试参数
        max_test_time = 8.0  # 最大测试时间
        test_steps = int(max_test_time / self.dt)
        
        print(f"测试参数:")
        print(f"  最大测试时间: {max_test_time}s")
        print(f"  控制周期: {self.dt*1000:.1f}ms")
        print(f"  总步数: {test_steps}")
        print("-" * 80)
        
        last_state = self.scheduler.current_state.value
        state_start_time = 0.0
        
        # 主测试循环
        for step in range(test_steps):
            self.current_time = step * self.dt
            
            # 模拟传感器
            self.simulate_perfect_sensors()
            
            # 推进状态机
            state_changed = self.scheduler.update_gait_state(self.dt)
            
            # 检查状态变化
            current_state = self.scheduler.current_state.value
            if current_state != last_state:
                state_duration = self.current_time - state_start_time
                self.record_state_change(last_state, current_state, state_duration)
                
                last_state = current_state
                state_start_time = self.current_time
            
            # 检查是否有足够的测量数据
            if len(self.state_changes) >= 20:  # 足够的状态变化来分析周期
                break
            
            # 每秒显示进度
            if step % int(1.0 / self.dt) == 0 and step > 0:
                print(f"[{self.current_time:.1f}s] 状态: {current_state}, 变化数: {len(self.state_changes)}")
        
        print(f"\n测试完成，总状态变化: {len(self.state_changes)}")
        
        # 分析结果
        swing_ok = self.analyze_swing_timing()
        ds_ok = self.analyze_double_support_timing()
        cycle_ok = self.analyze_cycle_timing()
        
        # 最终结果
        print(f"\n{'='*80}")
        print("节拍精度测试结果")
        print('='*80)
        
        all_tests_passed = swing_ok and ds_ok and cycle_ok
        
        print(f"1. 摆动时间精度: {'✅ 通过' if swing_ok else '❌ 失败'}")
        print(f"2. 双支撑时间精度: {'✅ 通过' if ds_ok else '❌ 失败'}")
        print(f"3. 完整周期精度: {'✅ 通过' if cycle_ok else '❌ 失败'}")
        
        if all_tests_passed:
            print(f"\n🎉 所有节拍精度测试通过!")
            print(f"✅ tSwing 误差 ≤ ±5%")
            print(f"✅ Tcycle 误差 ≤ ±5%")
            print(f"✅ 时序控制精确")
        else:
            print(f"\n❌ 部分精度测试未通过")
        
        print('='*80)
        return all_tests_passed
    
    def analyze_swing_timing(self) -> bool:
        """分析摆动时间精度"""
        print(f"\n{'='*60}")
        print("摆动时间分析")
        print('='*60)
        
        swing_durations = []
        
        # 查找摆动阶段（单支撑 → 双支撑的转换）
        for change in self.state_changes:
            if (change['from_state'] in ['left_support', 'right_support'] and
                'double_support' in change['to_state']):
                
                swing_duration = change['duration']
                swing_durations.append(swing_duration)
                
                error_ms = (swing_duration - self.expected_swing_time) * 1000
                error_percent = (swing_duration - self.expected_swing_time) / self.expected_swing_time * 100
                
                print(f"摆动 {len(swing_durations)}: {swing_duration*1000:.1f}ms "
                      f"(误差: {error_ms:+.1f}ms, {error_percent:+.1f}%)")
        
        if swing_durations:
            avg_swing = statistics.mean(swing_durations)
            avg_error_percent = (avg_swing - self.expected_swing_time) / self.expected_swing_time * 100
            
            print(f"\n摆动时间统计:")
            print(f"  期望值: {self.expected_swing_time*1000:.1f}ms")
            print(f"  测量数: {len(swing_durations)}")
            print(f"  平均值: {avg_swing*1000:.1f}ms (误差: {avg_error_percent:+.1f}%)")
            
            self.swing_measurements = swing_durations
            
            if abs(avg_error_percent) <= 5.0:
                print(f"  ✅ 摆动时间精度合格 (|{avg_error_percent:.1f}%| ≤ 5%)")
                return True
            else:
                print(f"  ❌ 摆动时间精度不合格 (|{avg_error_percent:.1f}%| > 5%)")
                return False
        else:
            print("❌ 未检测到摆动时间数据")
            return False
    
    def analyze_double_support_timing(self) -> bool:
        """分析双支撑时间精度"""
        print(f"\n{'='*60}")
        print("双支撑时间分析")  
        print('='*60)
        
        ds_durations = []
        
        # 查找双支撑阶段（双支撑 → 单支撑的转换）
        for change in self.state_changes:
            if ('double_support' in change['from_state'] and
                change['to_state'] in ['left_support', 'right_support']):
                
                ds_duration = change['duration']
                ds_durations.append(ds_duration)
                
                error_ms = (ds_duration - self.expected_double_support_time) * 1000
                error_percent = (ds_duration - self.expected_double_support_time) / self.expected_double_support_time * 100
                
                print(f"双支撑 {len(ds_durations)}: {ds_duration*1000:.1f}ms "
                      f"(误差: {error_ms:+.1f}ms, {error_percent:+.1f}%)")
        
        if ds_durations:
            avg_ds = statistics.mean(ds_durations)
            avg_error_percent = (avg_ds - self.expected_double_support_time) / self.expected_double_support_time * 100
            
            print(f"\n双支撑时间统计:")
            print(f"  期望值: {self.expected_double_support_time*1000:.1f}ms")
            print(f"  测量数: {len(ds_durations)}")
            print(f"  平均值: {avg_ds*1000:.1f}ms (误差: {avg_error_percent:+.1f}%)")
            
            self.double_support_measurements = ds_durations
            
            if abs(avg_error_percent) <= 5.0:
                print(f"  ✅ 双支撑时间精度合格 (|{avg_error_percent:.1f}%| ≤ 5%)")
                return True
            else:
                print(f"  ❌ 双支撑时间精度不合格 (|{avg_error_percent:.1f}%| > 5%)")
                return False
        else:
            print("❌ 未检测到双支撑时间数据")
            return False
    
    def analyze_cycle_timing(self) -> bool:
        """分析完整周期精度"""
        print(f"\n{'='*60}")
        print("完整步态周期分析")
        print('='*60)
        
        cycle_durations = []
        
        # 查找完整周期模式
        cycle_pattern = ['left_support', 'double_support_lr', 'right_support', 'double_support_rl']
        
        i = 0
        while i <= len(self.state_changes) - 4:
            # 检查是否匹配周期模式
            pattern_match = True
            for j, expected_to_state in enumerate(cycle_pattern):
                if (i + j >= len(self.state_changes) or 
                    self.state_changes[i + j]['to_state'] != expected_to_state):
                    pattern_match = False
                    break
            
            if pattern_match:
                # 计算周期时间
                cycle_start_time = self.state_changes[i]['time'] - self.state_changes[i]['duration']
                cycle_end_time = self.state_changes[i + 3]['time']
                cycle_duration = cycle_end_time - cycle_start_time
                
                cycle_durations.append(cycle_duration)
                
                error_ms = (cycle_duration - self.expected_cycle_time) * 1000
                error_percent = (cycle_duration - self.expected_cycle_time) / self.expected_cycle_time * 100
                
                print(f"周期 {len(cycle_durations)}: {cycle_duration*1000:.1f}ms "
                      f"(误差: {error_ms:+.1f}ms, {error_percent:+.1f}%)")
                
                i += 4  # 跳到下一个周期
            else:
                i += 1
        
        if cycle_durations:
            avg_cycle = statistics.mean(cycle_durations)
            avg_error_percent = (avg_cycle - self.expected_cycle_time) / self.expected_cycle_time * 100
            
            print(f"\n完整周期统计:")
            print(f"  期望值: {self.expected_cycle_time*1000:.1f}ms")
            print(f"  测量数: {len(cycle_durations)}")
            print(f"  平均值: {avg_cycle*1000:.1f}ms (误差: {avg_error_percent:+.1f}%)")
            
            self.cycle_measurements = cycle_durations
            
            if abs(avg_error_percent) <= 5.0:
                print(f"  ✅ 周期时间精度合格 (|{avg_error_percent:.1f}%| ≤ 5%)")
                return True
            else:
                print(f"  ❌ 周期时间精度不合格 (|{avg_error_percent:.1f}%| > 5%)")
                return False
        else:
            print("❌ 未检测到完整周期数据")
            return False


def main():
    """主函数"""
    try:
        tester = SimplifiedGaitTester()
        success = tester.run_timing_test()
        return success
    except Exception as e:
        print(f"❌ 精度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 