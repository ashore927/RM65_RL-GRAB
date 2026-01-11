"""快速验证修复效果 - 带 PyBullet GUI 实时可视化"""

import numpy as np
import yaml
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_env import RobotGraspEnv


# ============ 全局配置 ============
USE_GUI = True           # 是否启用 PyBullet GUI 可视化
SLOW_MOTION = True       # 是否慢动作播放（便于观察）
STEP_DELAY = 0.02        # 每步延迟时间(秒)，仅在 SLOW_MOTION=True 时生效


def load_config(config_path='config/simplified_object_picking.yaml'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 启用 GUI 可视化
    if USE_GUI:
        if 'simulation' not in config:
            config['simulation'] = {}
        config['simulation']['visualize'] = True
        config['simulation']['real_time'] = False  # 手动控制速度
    
    return config


def step_with_viz(env, action):
    """执行一步并添加可视化延迟"""
    obs, reward, done, info = env.step(action)
    if USE_GUI and SLOW_MOTION:
        time.sleep(STEP_DELAY)
    return obs, reward, done, info


def test_action_tracking():
    """测试动作跟踪率"""
    print("=" * 60)
    print("测试1: 动作跟踪率")
    print("=" * 60)
    
    config = load_config()
    env = RobotGraspEnv(config, evaluate=True, test=True)
    
    print(f"\n当前配置:")
    print(f"  ee_action_scale = {env.ee_action_scale} m")
    print(f"  action_repeat = {env.action_repeat}")
    
    results = {}
    for direction, name in [([1,0,0,0,0], "X"), ([0,1,0,0,0], "Y"), ([0,0,1,0,0], "Z"), ([0,0,-1,0,0], "-Z")]:
        env.reset()
        pre_tip, _ = env._get_tip_position()
        
        action = np.array(direction, dtype=np.float32)
        step_with_viz(env, action)
        
        post_tip, _ = env._get_tip_position()
        displacement = np.linalg.norm(np.array(post_tip) - np.array(pre_tip))
        expected = env.ee_action_scale
        ratio = displacement / expected * 100
        results[name] = ratio
        
        print(f"  {name}方向: 期望={expected:.4f}m, 实际={displacement:.4f}m, 跟踪率={ratio:.1f}%")
    
    avg_ratio = np.mean(list(results.values()))
    print(f"\n  平均跟踪率: {avg_ratio:.1f}%")
    
    env.close()
    return avg_ratio > 70  # 跟踪率>70%视为通过


def test_tip_position():
    """测试指尖位置计算"""
    print("\n" + "=" * 60)
    print("测试2: 指尖位置计算")
    print("=" * 60)
    
    config = load_config()
    env = RobotGraspEnv(config, evaluate=True, test=True)
    env.reset()
    
    pc = env.base_env.physics_client
    gripper_link = env._get_gripper_base_link_index()
    gripper_pos, gripper_orn = env.base_env.models[0].links[gripper_link].get_pose()
    gripper_euler = pc.getEulerFromQuaternion(gripper_orn)
    
    tip_pos, _ = env._get_tip_position()
    
    print(f"\n  夹爪基座位置: [{gripper_pos[0]:.4f}, {gripper_pos[1]:.4f}, {gripper_pos[2]:.4f}]")
    print(f"  夹爪朝向 (RPY): [{np.rad2deg(gripper_euler[0]):.1f}°, {np.rad2deg(gripper_euler[1]):.1f}°, {np.rad2deg(gripper_euler[2]):.1f}°]")
    print(f"  指尖位置: [{tip_pos[0]:.4f}, {tip_pos[1]:.4f}, {tip_pos[2]:.4f}]")
    
    # 检查指尖是否在合理位置（应该在基座下方，Z > 0）
    z_offset = gripper_pos[2] - tip_pos[2]
    print(f"  指尖相对基座Z偏移: {z_offset:.4f}m (应该约为+0.10m)")
    
    valid = tip_pos[2] > 0 and z_offset > 0.05
    print(f"  结果: {'✓ 通过' if valid else '✗ 失败'}")
    
    env.close()
    return valid


def test_scripted_grasp():
    """测试脚本化抓取"""
    print("\n" + "=" * 60)
    print("测试3: 脚本化抓取流程")
    print("=" * 60)
    
    config = load_config()
    env = RobotGraspEnv(config, evaluate=True, test=True)
    env.reset()
    
    pc = env.base_env.physics_client
    robot_id = env.base_env.robot_id
    obj_pos_init, _ = pc.getBasePositionAndOrientation(env.object_id)
    print(f"\n  物体初始位置: [{obj_pos_init[0]:.4f}, {obj_pos_init[1]:.4f}, {obj_pos_init[2]:.4f}]")
    
    # 获取物体尺寸
    aabb_min, aabb_max = pc.getAABB(env.object_id)
    obj_size = np.array(aabb_max) - np.array(aabb_min)
    obj_center_z = obj_pos_init[2] + obj_size[2] / 2
    print(f"  物体尺寸: [{obj_size[0]:.4f}, {obj_size[1]:.4f}, {obj_size[2]:.4f}]")
    print(f"  物体中心Z: {obj_center_z:.4f}")
    
    # 可视化标记：在物体位置添加调试线
    if USE_GUI:
        # 画一个十字标记物体位置
        pc.addUserDebugLine([obj_pos_init[0]-0.05, obj_pos_init[1], obj_pos_init[2]],
                           [obj_pos_init[0]+0.05, obj_pos_init[1], obj_pos_init[2]],
                           [1, 0, 0], 2, 0)  # 红色X轴
        pc.addUserDebugLine([obj_pos_init[0], obj_pos_init[1]-0.05, obj_pos_init[2]],
                           [obj_pos_init[0], obj_pos_init[1]+0.05, obj_pos_init[2]],
                           [0, 1, 0], 2, 0)  # 绿色Y轴
        pc.addUserDebugLine([obj_pos_init[0], obj_pos_init[1], obj_pos_init[2]],
                           [obj_pos_init[0], obj_pos_init[1], obj_pos_init[2]+0.1],
                           [0, 0, 1], 2, 0)  # 蓝色Z轴
        print("  [可视化] 已在物体位置添加坐标轴标记")
        time.sleep(1)  # 暂停让用户观察初始状态
    
    # 阶段1: 移动到物体上方
    print("\n  阶段1: 移动到物体上方...")
    # X 方向映射：robot_env 现在默认不反转 X；若配置 action_mapping.invert_x=True，则需要在脚本侧取反
    x_sign = -1.0 if getattr(env, 'invert_x_action', False) else 1.0
    for step in range(50):
        obj_pos, _ = pc.getBasePositionAndOrientation(env.object_id)
        tip_pos, _ = env._get_tip_position()
        
        # 目标：物体正上方5cm
        target_xy = np.array([obj_pos[0], obj_pos[1]])
        current_xy = np.array([tip_pos[0], tip_pos[1]])
        xy_error = target_xy - current_xy
        xy_dist = np.linalg.norm(xy_error)
        
        if xy_dist < 0.02:
            print(f"    Step {step}: XY对齐完成 (误差={xy_dist:.4f}m)")
            break
        
        # 动作：水平移动
        action = np.zeros(5, dtype=np.float32)
        direction = xy_error / (xy_dist + 1e-6)
        action[0] = x_sign * direction[0]
        action[1] = direction[1]
        action[2] = 0.0  # 保持高度
        action[3] = 1.0  # 保持张开
        
        step_with_viz(env, action)
        
        if step % 10 == 0:
            print(f"    Step {step}: XY距离={xy_dist:.4f}m")
    
    if USE_GUI:
        print("  [可视化] XY对齐完成，暂停1秒...")
        time.sleep(1)
    
    # 阶段2: 下降到抓取高度
    print("\n  阶段2: 下降到抓取高度...")
    # 关键修复：目标高度设为物体中心，让夹爪手指能包围物体
    # 物体底部Z≈0.017，物体中心Z≈0.039，物体高度≈0.044m
    # 夹爪手指需要在物体中心高度才能夹住
    target_z = obj_center_z  # 物体中心高度
    print(f"    目标指尖高度: {target_z:.4f}m (物体中心高度)")
    print(f"    物体底部Z: {obj_pos_init[2]:.4f}m, 物体中心Z: {obj_center_z:.4f}m")
    
    for step in range(100):  # 增加步数确保到位
        tip_pos, _ = env._get_tip_position()
        z_error = target_z - tip_pos[2]
        
        # 检查是否会撞到桌面（指尖不能低于0.06m）
        if tip_pos[2] < 0.06:
            print(f"    Step {step}: 接近桌面，停止下降 (Z={tip_pos[2]:.4f}m)")
            break
        
        if abs(z_error) < 0.01:
            print(f"    Step {step}: 高度到位 (Z={tip_pos[2]:.4f}m)")
            break
        
        action = np.zeros(5, dtype=np.float32)
        action[2] = np.clip(z_error * 3, -1, 1)  # Z方向移动，降低增益更平稳
        action[3] = 1.0  # 保持张开
        
        step_with_viz(env, action)
        
        if step % 10 == 0:
            print(f"    Step {step}: 指尖Z={tip_pos[2]:.4f}, 目标Z={target_z:.4f}")
    
    if USE_GUI:
        print("  [可视化] 高度到位，准备闭合夹爪，暂停1秒...")
        time.sleep(1)
    
    # 阶段3: 闭合夹爪
    print("\n  阶段3: 闭合夹爪...")
    contact_count = 0
    robot_id = env.base_env.robot_id
    
    for step in range(120):  # 增加闭合步数
        action = np.zeros(5, dtype=np.float32)
        action[3] = -1.0  # 闭合
        
        step_with_viz(env, action)
        
        if env._check_finger_contact():
            contact_count += 1
        
        if step % 30 == 0:
            obj_pos, _ = pc.getBasePositionAndOrientation(env.object_id)
            contact = env._check_finger_contact()
            tip_pos, _ = env._get_tip_position()
            # 获取 WSG50 两指位置（米）
            left_pos = float('nan')
            right_pos = float('nan')
            if env.gripper_joints_cache and 'wsg50_finger_left_joint' in env.gripper_joints_cache:
                left_pos = pc.getJointState(robot_id, env.gripper_joints_cache['wsg50_finger_left_joint'])[0]
            if env.gripper_joints_cache and 'wsg50_finger_right_joint' in env.gripper_joints_cache:
                right_pos = pc.getJointState(robot_id, env.gripper_joints_cache['wsg50_finger_right_joint'])[0]
            print(
                f"    Step {step}: left={left_pos:.5f}m right={right_pos:.5f}m, 接触={contact}, 物体Z={obj_pos[2]:.4f}"
            )
    
    if USE_GUI:
        print("  [可视化] 夹爪闭合完成，准备抬升，暂停1秒...")
        time.sleep(1)
    
    # 阶段4: 抬升
    print("\n  阶段4: 抬升...")
    for step in range(100):
        action = np.zeros(5, dtype=np.float32)
        action[2] = 1.0   # 向上
        action[3] = -1.0  # 保持闭合
        
        step_with_viz(env, action)
        
        if step % 25 == 0:
            obj_pos, _ = pc.getBasePositionAndOrientation(env.object_id)
            tip_pos, _ = env._get_tip_position()
            contact = env._check_finger_contact()
            print(f"    Step {step}: 指尖Z={tip_pos[2]:.4f}, 物体Z={obj_pos[2]:.4f}, 接触={contact}")
    
    # 检查结果
    obj_pos_final, _ = pc.getBasePositionAndOrientation(env.object_id)
    lift_height = obj_pos_final[2] - obj_pos_init[2]
    
    print(f"\n  最终结果:")
    print(f"    物体抬升高度: {lift_height:.4f}m")
    print(f"    接触次数: {contact_count}")
    
    success = lift_height > 0.05
    print(f"    结果: {'✓ 抓取成功!' if success else '✗ 抓取失败'}")
    
    if USE_GUI:
        print("\n  [可视化] 测试完成，窗口将在3秒后关闭...")
        time.sleep(3)
    
    env.close()
    return success


def main():
    print("\n" + "=" * 60)
    print("robot_env.py 修复验证")
    print("=" * 60)
    print(f"  GUI可视化: {'开启' if USE_GUI else '关闭'}")
    print(f"  慢动作: {'开启' if SLOW_MOTION else '关闭'}")
    if SLOW_MOTION:
        print(f"  每步延迟: {STEP_DELAY}秒")
    
    results = []
    
    results.append(("动作跟踪率", test_action_tracking()))
    results.append(("指尖位置", test_tip_position()))
    results.append(("脚本化抓取", test_scripted_grasp()))
    
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n所有测试通过! 可以开始训练。")
    else:
        print("\n存在失败项，需要进一步调试。")


if __name__ == '__main__':
    main()
