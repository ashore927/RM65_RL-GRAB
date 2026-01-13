"""测量夹爪距离诊断脚本 - 简化版

直接显示各 Link 的坐标，在仿真窗口中用圆球标明，旁边写上坐标值。

标记的点：
1. wsg50_base_link - 末端执行器中心（黄色）
2. wsg50_finger_left Link 中心（红色）
3. wsg50_finger_right Link 中心（绿色）
4. wsg50_finger_left 末端点 - 碰撞盒顶端，真正的指尖（深红色）
5. wsg50_finger_right 末端点 - 碰撞盒顶端，真正的指尖（深绿色）

使用方法：
    python measure_gripper_distances.py
"""

import numpy as np
import pybullet as p
import pybullet_data
import time


def main():
    """主函数：显示各 Link 坐标"""
    
    print("=" * 60)
    print("夹爪坐标显示脚本")
    print("=" * 60)
    
    # 连接仿真引擎
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # 设置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.3]
    )
    
    # 加载地面
    plane_id = p.loadURDF("plane.urdf")
    
    # 加载机器人 URDF
    urdf_path = "D:/Code/RM65_RL-Grab/URDF/rm65_wsg50.urdf"
    
    try:
        robot_id = p.loadURDF(
            urdf_path,
            [0, 0, 0],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
    except Exception as e:
        print(f"Error loading robot URDF: {e}")
        p.disconnect()
        return
    
    # 收集关键 link 的索引
    link_indices = {}
    joint_indices = {}
    
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8') if isinstance(info[1], bytes) else str(info[1])
        link_name = info[12].decode('utf-8') if isinstance(info[12], bytes) else str(info[12])
        
        if link_name in ['wsg50_base_link', 'wsg50_finger_left', 'wsg50_finger_right']:
            link_indices[link_name] = i
            
        if joint_name in ['wsg50_finger_left_joint', 'wsg50_finger_right_joint']:
            joint_indices[joint_name] = i
    
    # 设置机械臂为用户指定的初始姿态
    neutral_rad = [
        0.0,        # Joint1:  0° - 居中位置，双向均有活动空间
        -0.034601,  # Joint2:  -1.9825°
        1.578469,   # Joint3:  90.4396°
        0.000004,   # Joint4:   0.0002°
        1.597716,   # Joint5:  91.5424°
        0.0         # Joint6:  0° - 居中
    ]
    
    for i in range(min(6, num_joints)):
        p.resetJointState(robot_id, i, neutral_rad[i])
    
    # 夹爪完全张开
    if 'wsg50_finger_left_joint' in joint_indices:
        p.resetJointState(robot_id, joint_indices['wsg50_finger_left_joint'], -0.005)
    if 'wsg50_finger_right_joint' in joint_indices:
        p.resetJointState(robot_id, joint_indices['wsg50_finger_right_joint'], -0.005)
    
    # 运行仿真让状态稳定
    for _ in range(100):
        p.stepSimulation()
    
    # ==================== 获取坐标 ====================
    
    print("\n" + "=" * 60)
    print("各 Link 中心坐标 (世界坐标系)")
    print("=" * 60)
    
    # 1. wsg50_base_link
    base_pos = None
    if 'wsg50_base_link' in link_indices:
        state = p.getLinkState(robot_id, link_indices['wsg50_base_link'], computeForwardKinematics=True)
        base_pos = np.array(state[4])
        base_orn = state[5]
        print(f"\nwsg50_base_link (末端执行器中心):")
        print(f"  X = {base_pos[0]:.6f} m = {base_pos[0]*1000:.2f} mm")
        print(f"  Y = {base_pos[1]:.6f} m = {base_pos[1]*1000:.2f} mm")
        print(f"  Z = {base_pos[2]:.6f} m = {base_pos[2]*1000:.2f} mm")
    else:
        print("警告: 未找到 wsg50_base_link")
    
    # 2. wsg50_finger_left
    left_pos = None
    left_orn = None
    if 'wsg50_finger_left' in link_indices:
        state = p.getLinkState(robot_id, link_indices['wsg50_finger_left'], computeForwardKinematics=True)
        left_pos = np.array(state[4])
        left_orn = state[5]
        print(f"\nwsg50_finger_left (左指 Link 中心):")
        print(f"  X = {left_pos[0]:.6f} m = {left_pos[0]*1000:.2f} mm")
        print(f"  Y = {left_pos[1]:.6f} m = {left_pos[1]*1000:.2f} mm")
        print(f"  Z = {left_pos[2]:.6f} m = {left_pos[2]*1000:.2f} mm")
    else:
        print("警告: 未找到 wsg50_finger_left")
    
    # 3. wsg50_finger_right
    right_pos = None
    right_orn = None
    if 'wsg50_finger_right' in link_indices:
        state = p.getLinkState(robot_id, link_indices['wsg50_finger_right'], computeForwardKinematics=True)
        right_pos = np.array(state[4])
        right_orn = state[5]
        print(f"\nwsg50_finger_right (右指 Link 中心):")
        print(f"  X = {right_pos[0]:.6f} m = {right_pos[0]*1000:.2f} mm")
        print(f"  Y = {right_pos[1]:.6f} m = {right_pos[1]*1000:.2f} mm")
        print(f"  Z = {right_pos[2]:.6f} m = {right_pos[2]*1000:.2f} mm")
    else:
        print("警告: 未找到 wsg50_finger_right")
    
    # ==================== 计算指尖末端点 ====================
    # 根据 URDF: 碰撞盒 origin xyz="0 0 0.021", box size="0.01 0.01 0.075"
    # 碰撞盒中心在局部 Z=0.021 处，高度 0.075 m
    # 真正的指尖（没有 joint 的那一端）= 碰撞盒顶端
    # 碰撞盒顶端在 Link 局部坐标系中: z = 0.021 + 0.075/2 = 0.0585 m
    
    COLLISION_CENTER_Z = 0.021
    COLLISION_HALF_HEIGHT = 0.075 / 2  # = 0.0375
    TIP_LOCAL_Z = COLLISION_CENTER_Z + COLLISION_HALF_HEIGHT  # = 0.0585 (顶端 = 真正指尖)
    
    print("\n" + "=" * 60)
    print("指尖末端点坐标 (碰撞盒顶端 = 真正的指尖)")
    print("=" * 60)
    print(f"\n根据 URDF 碰撞盒定义:")
    print(f"  碰撞盒中心 (局部 Z): {COLLISION_CENTER_Z} m = {COLLISION_CENTER_Z*1000} mm")
    print(f"  碰撞盒高度: {COLLISION_HALF_HEIGHT*2} m = {COLLISION_HALF_HEIGHT*2*1000} mm")
    print(f"  指尖末端 (局部 Z): {TIP_LOCAL_Z} m = {TIP_LOCAL_Z*1000} mm")
    
    left_tip = None
    right_tip = None
    
    if left_pos is not None and left_orn is not None:
        # 将局部坐标转换到世界坐标
        rot_matrix = np.array(p.getMatrixFromQuaternion(left_orn)).reshape(3, 3)
        local_offset = np.array([0, 0, TIP_LOCAL_Z])
        left_tip = left_pos + rot_matrix @ local_offset
        print(f"\nwsg50_finger_left 末端点 (真正的指尖):")
        print(f"  X = {left_tip[0]:.6f} m = {left_tip[0]*1000:.2f} mm")
        print(f"  Y = {left_tip[1]:.6f} m = {left_tip[1]*1000:.2f} mm")
        print(f"  Z = {left_tip[2]:.6f} m = {left_tip[2]*1000:.2f} mm")
    
    if right_pos is not None and right_orn is not None:
        rot_matrix = np.array(p.getMatrixFromQuaternion(right_orn)).reshape(3, 3)
        local_offset = np.array([0, 0, TIP_LOCAL_Z])
        right_tip = right_pos + rot_matrix @ local_offset
        print(f"\nwsg50_finger_right 末端点 (真正的指尖):")
        print(f"  X = {right_tip[0]:.6f} m = {right_tip[0]*1000:.2f} mm")
        print(f"  Y = {right_tip[1]:.6f} m = {right_tip[1]*1000:.2f} mm")
        print(f"  Z = {right_tip[2]:.6f} m = {right_tip[2]*1000:.2f} mm")
    
    # ==================== 可视化 ====================
    print("\n" + "=" * 60)
    print("仿真窗口中的标记")
    print("=" * 60)
    print("  黄色球: wsg50_base_link (末端执行器中心)")
    print("  红色球: wsg50_finger_left Link 中心")
    print("  绿色球: wsg50_finger_right Link 中心")
    print("  深红色球: wsg50_finger_left 末端点 (真正的指尖)")
    print("  深绿色球: wsg50_finger_right 末端点 (真正的指尖)")
    print("\n按 Ctrl+C 退出...")
    
    # 添加可视化标记
    
    # 1. wsg50_base_link - 黄色
    if base_pos is not None:
        p.addUserDebugPoints([base_pos.tolist()], [[1, 1, 0]], pointSize=20)
        coord_text = f"wsg50_base_link\n({base_pos[0]*1000:.1f}, {base_pos[1]*1000:.1f}, {base_pos[2]*1000:.1f}) mm"
        p.addUserDebugText(coord_text, (base_pos + [0.02, 0, 0.02]).tolist(), [1, 1, 0], textSize=1.0)
    
    # 2. wsg50_finger_left Link 中心 - 红色
    if left_pos is not None:
        p.addUserDebugPoints([left_pos.tolist()], [[1, 0, 0]], pointSize=15)
        coord_text = f"finger_left center\n({left_pos[0]*1000:.1f}, {left_pos[1]*1000:.1f}, {left_pos[2]*1000:.1f}) mm"
        p.addUserDebugText(coord_text, (left_pos + [-0.06, 0, 0.01]).tolist(), [1, 0, 0], textSize=1.0)
    
    # 3. wsg50_finger_right Link 中心 - 绿色
    if right_pos is not None:
        p.addUserDebugPoints([right_pos.tolist()], [[0, 1, 0]], pointSize=15)
        coord_text = f"finger_right center\n({right_pos[0]*1000:.1f}, {right_pos[1]*1000:.1f}, {right_pos[2]*1000:.1f}) mm"
        p.addUserDebugText(coord_text, (right_pos + [0.03, 0, 0.01]).tolist(), [0, 1, 0], textSize=1.0)
    
    # 4. 左指末端点（真正的指尖）- 深红色
    if left_tip is not None:
        p.addUserDebugPoints([left_tip.tolist()], [[0.8, 0, 0]], pointSize=15)
        coord_text = f"left_tip (fingertip)\n({left_tip[0]*1000:.1f}, {left_tip[1]*1000:.1f}, {left_tip[2]*1000:.1f}) mm"
        p.addUserDebugText(coord_text, (left_tip + [-0.06, 0, -0.02]).tolist(), [0.8, 0, 0], textSize=1.0)
        # 画线连接 Link 中心和末端
        if left_pos is not None:
            p.addUserDebugLine(left_pos.tolist(), left_tip.tolist(), [1, 0.5, 0.5], lineWidth=2)
    
    # 5. 右指末端点（真正的指尖）- 深绿色
    if right_tip is not None:
        p.addUserDebugPoints([right_tip.tolist()], [[0, 0.8, 0]], pointSize=15)
        coord_text = f"right_tip (fingertip)\n({right_tip[0]*1000:.1f}, {right_tip[1]*1000:.1f}, {right_tip[2]*1000:.1f}) mm"
        p.addUserDebugText(coord_text, (right_tip + [0.03, 0, -0.02]).tolist(), [0, 0.8, 0], textSize=1.0)
        # 画线连接 Link 中心和末端
        if right_pos is not None:
            p.addUserDebugLine(right_pos.tolist(), right_tip.tolist(), [0.5, 1, 0.5], lineWidth=2)
    
    # 6. 两个指尖末端之间的连线和距离
    if left_tip is not None and right_tip is not None:
        # 计算两点间的距离
        fingertip_distance = np.linalg.norm(left_tip - right_tip)
        print(f"\n两个指尖末端点之间的距离:")
        print(f"  距离 = {fingertip_distance:.6f} m = {fingertip_distance*1000:.2f} mm")
        
        # 画青色连线
        p.addUserDebugLine(left_tip.tolist(), right_tip.tolist(), [0, 1, 1], lineWidth=3)
        
        # 在连线中点标注距离
        midpoint = (left_tip + right_tip) / 2
        distance_text = f"距离: {fingertip_distance*1000:.2f} mm"
        p.addUserDebugText(distance_text, (midpoint + [0, 0, 0.02]).tolist(), [0, 1, 1], textSize=1.5)
    
    # 画线连接 base_link 到各个点
    if base_pos is not None:
        if left_pos is not None:
            p.addUserDebugLine(base_pos.tolist(), left_pos.tolist(), [1, 0.5, 0], lineWidth=1)
        if right_pos is not None:
            p.addUserDebugLine(base_pos.tolist(), right_pos.tolist(), [0.5, 1, 0], lineWidth=1)
    
    # 保持仿真运行
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\n\n已退出仿真")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
