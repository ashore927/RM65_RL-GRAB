"""可视化末端和目标位置"""
import numpy as np
import yaml
import time
from robot_env import RobotGraspEnv

with open('config/simplified_object_picking.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config['simulation'] = config.get('simulation', {})
config['simulation']['visualize'] = True
config['simulation']['real_time'] = True

env = RobotGraspEnv(config, test=True)
pc = env.base_env.physics_client

# 添加可视化标记
def add_debug_sphere(pc, pos, color, radius=0.01):
    """添加调试球体"""
    vis_id = pc.createVisualShape(pc.GEOM_SPHERE, radius=radius, rgbaColor=color)
    body_id = pc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_id, basePosition=pos)
    return body_id

obs = env.reset()

# 获取位置
tip_pos, _ = env._get_tip_position()
obj_pos, _ = pc.getBasePositionAndOrientation(env.object_id)

# 获取gripper base位置
gripper_link = env._get_gripper_base_link_index()
gripper_pos, _ = env.base_env.models[0].links[gripper_link].get_pose()

print("=== 位置信息 ===")
print(f"Gripper base: {np.array(gripper_pos)}")
print(f"Tip (指腹中心): {tip_pos}")
print(f"Object: {np.array(obj_pos)}")
print(f"Tip - Object XY: [{tip_pos[0]-obj_pos[0]:.4f}, {tip_pos[1]-obj_pos[1]:.4f}]")
print(f"XY距离: {np.linalg.norm(tip_pos[:2] - np.array(obj_pos)[:2]):.4f}")

# 添加标记
# 红色 = 物体
add_debug_sphere(pc, obj_pos, [1, 0, 0, 1], 0.02)
# 绿色 = 指腹中心
add_debug_sphere(pc, tip_pos, [0, 1, 0, 1], 0.015)
# 蓝色 = gripper base
add_debug_sphere(pc, gripper_pos, [0, 0, 1, 1], 0.015)

print("\n可视化标记：红色=物体, 绿色=指腹中心, 蓝色=gripper base")
print("按Ctrl+C退出...")

# 持续更新位置显示
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass

env.close()
