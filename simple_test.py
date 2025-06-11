import sys
sys.path.append('.')

from foot_placement import get_foot_planner, Vector3D
from data_bus import get_data_bus

print("=== 足步规划测试 ===")

planner = get_foot_planner()

joint_angles = {
    "left_hip_pitch": -0.1, "left_knee_pitch": 0.2, "left_ankle_pitch": -0.1,
    "right_hip_pitch": -0.1, "right_knee_pitch": 0.2, "right_ankle_pitch": -0.1
}

planner.update_foot_states_from_kinematics(joint_angles)
planner.set_body_motion_intent(Vector3D(0.3, 0.0, 0.0), 0.0)

target_left = planner.plan_foot_placement("left", "right")
target_right = planner.plan_foot_placement("right", "left")

print(f"左腿目标: ({target_left.x:.3f}, {target_left.y:.3f}, {target_left.z:.3f})")
print(f"右腿目标: ({target_right.x:.3f}, {target_right.y:.3f}, {target_right.z:.3f})")

print("\n=== 数据总线测试 ===")

data_bus = get_data_bus()

if hasattr(data_bus, 'trigger_foot_placement_planning'):
    data_bus.set_body_motion_intent(0.25, 0.0, 0.0)
    target_pos = data_bus.trigger_foot_placement_planning("left", "right")
    
    if target_pos:
        print(f"规划成功: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
    
    data_bus.print_foot_planning_status()
else:
    print("数据总线未集成足步规划")

print("\n✓ 测试完成") 