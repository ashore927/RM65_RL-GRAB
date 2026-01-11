"""可视化抓取调试 - 使用PyBullet GUI模式"""

import numpy as np
import yaml
import os
import sys
import time
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path='config/simplified_object_picking.yaml'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def visualize_grasp():
    """可视化抓取过程"""
    print("=" * 60)
    print("可视化抓取调试")
    print("=" * 60)
    print("\n按 Enter 继续每个阶段，按 Q 退出")
    
    # 修改配置以启用GUI模式
    config = load_config()
    
    # 强制使用GUI模式
    if 'simulation' not in config:
        config['simulation'] = {}
    config['simulation']['visualize'] = True  # 关键：设置 visualize 为 True
    config['simulation']['real_time'] = False  # 不用实时，便于观察
    
    from robot_env import RobotGraspEnv
    env = RobotGraspEnv(config, evaluate=True, test=True)
    env.reset()
    
    pc = env.base_env.physics_client
    robot_id = env.base_env.robot_id
    
    # 设置相机视角（俯视角度）
    pc.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[-0.3, 0, 0.1]
    )
    
    # 获取物体信息
    obj_pos_init, _ = pc.getBasePositionAndOrientation(env.object_id)
    aabb_min, aabb_max = pc.getAABB(env.object_id)
    obj_size = np.array(aabb_max) - np.array(aabb_min)
    obj_center_z = obj_pos_init[2] + obj_size[2] / 2
    
    print(f"\n物体初始位置: [{obj_pos_init[0]:.4f}, {obj_pos_init[1]:.4f}, {obj_pos_init[2]:.4f}]")
    print(f"物体尺寸: [{obj_size[0]:.4f}, {obj_size[1]:.4f}, {obj_size[2]:.4f}]")
    print(f"物体底部Z: {obj_pos_init[2]:.4f}, 物体中心Z: {obj_center_z:.4f}, 物体顶部Z: {aabb_max[2]:.4f}")
    
    # 添加可视化标记
    # 物体位置标记（红色球）
    obj_marker = pc.addUserDebugPoints(
        [obj_pos_init], [[1, 0, 0]], pointSize=10
    )
    
    # 添加目标抓取高度的水平线（绿色）- 现在是物体底部高度
    target_grasp_z = obj_pos_init[2]
    pc.addUserDebugLine(
        [-0.5, -0.2, target_grasp_z],
        [-0.1, 0.2, target_grasp_z],
        [0, 1, 0], lineWidth=2
    )
    
    # 添加物体中心高度线（黄色）
    pc.addUserDebugLine(
        [-0.5, -0.2, obj_center_z],
        [-0.1, 0.2, obj_center_z],
        [1, 1, 0], lineWidth=2
    )
    
    # 添加文字说明
    pc.addUserDebugText(
        f"Target (obj bottom): {target_grasp_z:.3f}m",
        [-0.1, 0.2, target_grasp_z + 0.01],
        [0, 1, 0], textSize=1.0
    )
    pc.addUserDebugText(
        f"Object center: {obj_center_z:.3f}m",
        [-0.1, 0.2, obj_center_z + 0.01],
        [1, 1, 0], textSize=1.0
    )
    
    def wait_for_input():
        """等待用户输入"""
        try:
            user_input = input("按 Enter 继续，输入 'q' 退出: ")
            if user_input.lower() == 'q':
                return False
        except:
            pass
        return True
    
    def add_tip_marker(tip_pos, color=[0, 0, 1]):
        """添加指尖位置标记"""
        return pc.addUserDebugPoints([tip_pos], [color], pointSize=8)
    
    def show_gripper_state():
        """显示夹爪状态"""
        tip_pos, _ = env._get_tip_position()
        obj_pos, _ = pc.getBasePositionAndOrientation(env.object_id)
        contact = env._check_finger_contact()
        
        gripper_angle = 0
        if env.gripper_joints_cache and 'joint_b1' in env.gripper_joints_cache:
            gripper_angle = pc.getJointState(robot_id, env.gripper_joints_cache['joint_b1'])[0]
        
        print(f"  指尖(计算): [{tip_pos[0]:.4f}, {tip_pos[1]:.4f}, {tip_pos[2]:.4f}]")
        print(f"  物体位置: [{obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f}]")
        
        # 显示实际夹爪手指位置
        finger_z_values = []
        for i in range(pc.getNumJoints(robot_id)):
            info = pc.getJointInfo(robot_id, i)
            link_name = info[12].decode('utf-8')
            if link_name in ['gripper_link3', 'gripper_link4']:  # 实际指尖
                link_state = pc.getLinkState(robot_id, i)
                finger_z_values.append(link_state[0][2])
        if finger_z_values:
            avg_finger_z = np.mean(finger_z_values)
            print(f"  实际手指Z: {avg_finger_z:.4f} (link3/4平均)")
            print(f"  手指与物体中心差: {avg_finger_z - obj_center_z:.4f}m")
        
        print(f"  夹爪角度: {gripper_angle:.3f} (-0.75=张开, 0=闭合)")
        print(f"  接触: {contact}")
        
        return tip_pos, obj_pos, contact
    
    # ============ 阶段0: 初始状态 ============
    print("\n" + "=" * 40)
    print("阶段0: 初始状态")
    print("=" * 40)
    show_gripper_state()
    
    # 显示夹爪各个link的位置
    print("\n夹爪Link位置:")
    gripper_link_names = ['gripper_link1', 'gripper_link2', 'gripper_link3', 'gripper_link4']
    num_joints = pc.getNumJoints(robot_id)
    for i in range(num_joints):
        info = pc.getJointInfo(robot_id, i)
        link_name = info[12].decode('utf-8')
        if link_name in gripper_link_names:
            link_state = pc.getLinkState(robot_id, i)
            link_pos = link_state[0]
            print(f"  {link_name}: [{link_pos[0]:.4f}, {link_pos[1]:.4f}, {link_pos[2]:.4f}]")
    
    if not wait_for_input():
        env.close()
        return
    
    # ============ 阶段1: 移动到物体上方 ============
    print("\n" + "=" * 40)
    print("阶段1: 移动到物体上方")
    print("=" * 40)
    
    for step in range(50):
        obj_pos, _ = pc.getBasePositionAndOrientation(env.object_id)
        tip_pos, _ = env._get_tip_position()
        
        target_xy = np.array([obj_pos[0], obj_pos[1]])
        current_xy = np.array([tip_pos[0], tip_pos[1]])
        xy_error = target_xy - current_xy
        xy_dist = np.linalg.norm(xy_error)
        
        if xy_dist < 0.02:
            print(f"  Step {step}: XY对齐完成")
            break
        
        action = np.zeros(5, dtype=np.float32)
        direction = xy_error / (xy_dist + 1e-6)
        action[0] = -direction[0]
        action[1] = direction[1]
        action[3] = 1.0
        
        env.step(action)
        time.sleep(0.01)  # 放慢以便观察
    
    show_gripper_state()
    if not wait_for_input():
        env.close()
        return
    
    # ============ 阶段2: 下降到抓取高度 ============
    print("\n" + "=" * 40)
    print("阶段2: 下降到抓取高度")
    print("=" * 40)
    
    # 分析实际手指位置与计算指尖的关系
    # 初始时: gripper_base_link Z ≈ 0.247, gripper_link3/4 Z ≈ 0.161
    # tip_offset = 0.085 计算的是 gripper_base_link 到 "计算指尖" 的偏移
    # 但实际夹爪手指(link3/4)位置更高约 0.085m - (0.247-0.161) = 0m？
    # 实际偏差 = 0.247 - 0.161 = 0.086 ≈ tip_offset，所以计算应该正确
    
    # 关键问题：目标高度应该让实际手指到达物体中心高度
    # 物体中心Z ≈ 0.039m，需要实际手指下降到这个高度
    # 由于计算指尖 = gripper_base - 0.085，而实际手指也约在这个位置
    # 所以目标指尖高度应该 = 物体中心高度
    
    target_z = obj_center_z  # 物体中心高度，让手指能包围物体
    print(f"目标指尖高度: {target_z:.4f}m (物体中心)")
    print(f"物体底部Z: {obj_pos_init[2]:.4f}m, 物体中心Z: {obj_center_z:.4f}m")
    
    for step in range(100):
        tip_pos, _ = env._get_tip_position()
        z_error = target_z - tip_pos[2]
        
        # 防止撞桌面
        if tip_pos[2] < 0.01:
            print(f"  Step {step}: 接近桌面，停止")
            break
        
        if abs(z_error) < 0.01:
            print(f"  Step {step}: 高度到位")
            break
        
        action = np.zeros(5, dtype=np.float32)
        action[2] = np.clip(z_error * 3, -1, 1)
        action[3] = 1.0
        
        env.step(action)
        time.sleep(0.01)
        
        if step % 20 == 0:
            print(f"  Step {step}: Z={tip_pos[2]:.4f}, 目标={target_z:.4f}")
    
    show_gripper_state()
    
    # 显示当前夹爪位置与物体的关系
    print("\n当前夹爪Link位置:")
    for i in range(num_joints):
        info = pc.getJointInfo(robot_id, i)
        link_name = info[12].decode('utf-8')
        if link_name in gripper_link_names:
            link_state = pc.getLinkState(robot_id, i)
            link_pos = link_state[0]
            obj_pos, _ = pc.getBasePositionAndOrientation(env.object_id)
            dist_to_obj = np.linalg.norm(np.array(link_pos) - np.array(obj_pos))
            print(f"  {link_name}: Z={link_pos[2]:.4f}, 到物体={dist_to_obj:.4f}m")
    
    if not wait_for_input():
        env.close()
        return
    
    # ============ 阶段3: 闭合夹爪 ============
    print("\n" + "=" * 40)
    print("阶段3: 闭合夹爪")
    print("=" * 40)
    
    contact_count = 0
    
    for step in range(150):
        action = np.zeros(5, dtype=np.float32)
        action[3] = -1.0
        
        env.step(action)
        time.sleep(0.005)  # 稍微放慢
        
        if env._check_finger_contact():
            contact_count += 1
        
        if step % 30 == 0:
            tip_pos, obj_pos, contact = show_gripper_state()
            print()
    
    print(f"\n总接触次数: {contact_count}")
    
    if not wait_for_input():
        env.close()
        return
    
    # ============ 阶段4: 抬升 ============
    print("\n" + "=" * 40)
    print("阶段4: 抬升")
    print("=" * 40)
    
    for step in range(100):
        action = np.zeros(5, dtype=np.float32)
        action[2] = 1.0   # 向上
        action[3] = -1.0  # 保持闭合
        
        env.step(action)
        time.sleep(0.01)
        
        if step % 25 == 0:
            tip_pos, obj_pos, contact = show_gripper_state()
            print()
    
    # ============ 最终结果 ============
    print("\n" + "=" * 40)
    print("最终结果")
    print("=" * 40)
    
    obj_pos_final, _ = pc.getBasePositionAndOrientation(env.object_id)
    lift_height = obj_pos_final[2] - obj_pos_init[2]
    
    print(f"物体抬升高度: {lift_height:.4f}m")
    print(f"接触次数: {contact_count}")
    print(f"结果: {'✓ 抓取成功!' if lift_height > 0.05 else '✗ 抓取失败'}")
    
    print("\n按 Enter 关闭...")
    try:
        input()
    except:
        pass
    
    env.close()


def visualize_gripper_only():
    """仅可视化夹爪开合"""
    print("=" * 60)
    print("夹爪开合可视化")
    print("=" * 60)
    
    config = load_config()
    if 'simulation' not in config:
        config['simulation'] = {}
    config['simulation']['visualize'] = True
    config['simulation']['real_time'] = False
    
    from robot_env import RobotGraspEnv
    env = RobotGraspEnv(config, evaluate=True, test=True)
    env.reset()
    
    pc = env.base_env.physics_client
    robot_id = env.base_env.robot_id
    
    # 设置相机视角（侧视）
    pc.resetDebugVisualizerCamera(
        cameraDistance=0.3,
        cameraYaw=90,
        cameraPitch=-10,
        cameraTargetPosition=[-0.25, 0, 0.15]
    )
    
    print("\n观察夹爪开合动作...")
    print("张开夹爪...")
    
    for _ in range(50):
        env._control_gripper(1.0)
        pc.stepSimulation()
        time.sleep(0.02)
    
    print("闭合夹爪...")
    for i in range(100):
        env._control_gripper(-1.0)
        pc.stepSimulation()
        time.sleep(0.02)
        
        if i % 20 == 0:
            gripper_angle = 0
            if env.gripper_joints_cache and 'joint_b1' in env.gripper_joints_cache:
                gripper_angle = pc.getJointState(robot_id, env.gripper_joints_cache['joint_b1'])[0]
            print(f"  Step {i}: 夹爪角度={gripper_angle:.3f}")
    
    print("\n按 Enter 关闭...")
    try:
        input()
    except:
        pass
    
    env.close()


def visualize_manual_grasp():
    """手动控制抓取 - 使用滑块"""
    print("=" * 60)
    print("手动控制抓取")
    print("=" * 60)
    print("""
控制说明:
  使用 PyBullet GUI 中的滑块控制:
  - X/Y/Z 滑块: 控制末端移动方向
  - Gripper 滑块: 正值=张开, 负值=闭合
""")
    
    config = load_config()
    if 'simulation' not in config:
        config['simulation'] = {}
    config['simulation']['visualize'] = True
    config['simulation']['real_time'] = False
    
    from robot_env import RobotGraspEnv
    env = RobotGraspEnv(config, evaluate=True, test=True)
    env.reset()
    
    pc = env.base_env.physics_client
    
    # 设置相机
    pc.resetDebugVisualizerCamera(
        cameraDistance=0.6,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[-0.3, 0, 0.1]
    )
    
    # 添加控制滑块
    gripper_slider = pc.addUserDebugParameter("Gripper", -1, 1, 1)
    x_slider = pc.addUserDebugParameter("X", -1, 1, 0)
    y_slider = pc.addUserDebugParameter("Y", -1, 1, 0)
    z_slider = pc.addUserDebugParameter("Z", -1, 1, 0)
    
    print("使用GUI中的滑块控制机械臂和夹爪")
    print("按 Enter 退出...")
    
    try:
        while True:
            # 读取滑块值
            gripper_val = pc.readUserDebugParameter(gripper_slider)
            x_val = pc.readUserDebugParameter(x_slider)
            y_val = pc.readUserDebugParameter(y_slider)
            z_val = pc.readUserDebugParameter(z_slider)
            
            # 构建动作
            action = np.array([x_val, y_val, z_val, gripper_val, 0], dtype=np.float32)
            
            # 执行动作
            env.step(action)
            
            # 显示状态
            tip_pos, _ = env._get_tip_position()
            obj_pos, _ = pc.getBasePositionAndOrientation(env.object_id)
            contact = env._check_finger_contact()
            
            # 更新调试文字
            pc.addUserDebugText(
                f"Tip: [{tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f}]",
                [-0.1, 0.3, 0.3], [1, 1, 1], textSize=1.0,
                replaceItemUniqueId=100
            )
            pc.addUserDebugText(
                f"Contact: {contact}",
                [-0.1, 0.3, 0.25], [1, 0, 0] if not contact else [0, 1, 0], textSize=1.0,
                replaceItemUniqueId=101
            )
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        pass
    
    env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='grasp',
                       choices=['grasp', 'gripper', 'manual'],
                       help='grasp: 自动抓取流程, gripper: 夹爪测试, manual: 手动控制')
    args = parser.parse_args()
    
    if args.mode == 'grasp':
        visualize_grasp()
    elif args.mode == 'gripper':
        visualize_gripper_only()
    elif args.mode == 'manual':
        visualize_manual_grasp()
