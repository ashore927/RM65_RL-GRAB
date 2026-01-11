"""
机械臂可抓取范围分析工具

在指定高度平面上扫描 XY 网格，测试每个点的 IK 可达性，
并可视化机械臂的有效工作空间范围。
"""

import time
import numpy as np
from robot_env import RobotGraspEnv

# 配置
config = {
    'scene': {'scene_type': 'OnTable'},
    'simulation': {
        'visualize': True,
        'real_time': False,  # 关闭实时，加速扫描
        'auto_init_test_env': True,
    }
}

# 扫描参数
SCAN_X_MIN = -0.4      # X 扫描范围（米）
SCAN_X_MAX = 0.6
SCAN_Y_MIN = -0.5
SCAN_Y_MAX = 0.5
SCAN_Z_HEIGHT = 0.15   # 固定扫描高度（桌面上方 15cm，抓取高度）
GRID_RESOLUTION = 0.02 # 网格分辨率（2cm）

# 关节限位安全边界
JOINT_LIMIT_MARGIN = 0.1  # 弧度


def check_ik_reachable(env, target_pos, target_orn=None):
    """检测目标位置是否可达（IK 有效解且关节不超限）
    
    Args:
        env: RobotGraspEnv 实例
        target_pos: [x, y, z] 目标末端位置
        target_orn: 目标末端姿态（四元数），默认垂直向下
        
    Returns:
        dict: {
            'reachable': bool,  # 是否可达
            'joint_angles': list or None,  # IK 解
            'reason': str,  # 不可达原因
        }
    """
    pc = env.base_env.physics_client
    robot_id = env.base_env.robot_id
    gripper_link = env._get_gripper_base_link_index()
    
    # 默认末端姿态：垂直向下
    if target_orn is None:
        # 与 robot_env 统一的末端“正常”姿态
        target_orn = pc.getQuaternionFromEuler([np.pi, 0, -np.pi/2])
    
    # 求解 IK
    try:
        joint_poses = pc.calculateInverseKinematics(
            robot_id,
            gripper_link,
            target_pos,
            target_orn,
            maxNumIterations=200,
            residualThreshold=1e-4
        )
    except Exception as e:
        return {'reachable': False, 'joint_angles': None, 'reason': f'IK exception: {e}'}
    
    arm_angles = joint_poses[:6]
    
    # 检查关节是否在限位内
    for i, angle in enumerate(arm_angles):
        low, high = env.arm_joint_limits[i]
        if angle < low + JOINT_LIMIT_MARGIN or angle > high - JOINT_LIMIT_MARGIN:
            return {
                'reachable': False,
                'joint_angles': list(arm_angles),
                'reason': f'Joint {i+1} out of limit: {np.rad2deg(angle):.1f}° (limit: {np.rad2deg(low):.1f}° ~ {np.rad2deg(high):.1f}°)'
            }
    
    # 验证 IK 解精度：临时设置关节角，检查实际末端位置
    # 保存当前关节状态
    original_angles = []
    for j in range(6):
        state = pc.getJointState(robot_id, j)
        original_angles.append(state[0])
    
    # 设置 IK 解
    for j, angle in enumerate(arm_angles):
        pc.resetJointState(robot_id, j, angle)
    
    # 获取实际末端位置
    actual_pos, _ = env.base_env.models[0].links[gripper_link].get_pose()
    
    # 恢复原始关节状态
    for j, angle in enumerate(original_angles):
        pc.resetJointState(robot_id, j, angle)
    
    # 检查位置误差
    pos_error = np.linalg.norm(np.array(actual_pos) - np.array(target_pos))
    if pos_error > 0.01:  # 误差超过 1cm 视为不可达
        return {
            'reachable': False,
            'joint_angles': list(arm_angles),
            'reason': f'IK position error too large: {pos_error*100:.1f}cm'
        }
    
    return {'reachable': True, 'joint_angles': list(arm_angles), 'reason': 'OK'}


def scan_workspace(env, z_height=SCAN_Z_HEIGHT, show_markers=True):
    """扫描工作空间并可视化
    
    Args:
        env: RobotGraspEnv 实例
        z_height: 扫描平面高度
        show_markers: 是否显示可视化标记
        
    Returns:
        dict: 扫描结果统计
    """
    pc = env.base_env.physics_client
    
    # 生成网格点
    x_range = np.arange(SCAN_X_MIN, SCAN_X_MAX + GRID_RESOLUTION, GRID_RESOLUTION)
    y_range = np.arange(SCAN_Y_MIN, SCAN_Y_MAX + GRID_RESOLUTION, GRID_RESOLUTION)
    
    total_points = len(x_range) * len(y_range)
    reachable_points = []
    unreachable_points = []
    
    print(f"开始扫描工作空间...")
    print(f"  X 范围: [{SCAN_X_MIN}, {SCAN_X_MAX}] m")
    print(f"  Y 范围: [{SCAN_Y_MIN}, {SCAN_Y_MAX}] m")
    print(f"  Z 高度: {z_height} m")
    print(f"  网格分辨率: {GRID_RESOLUTION*100:.0f} cm")
    print(f"  总测试点数: {total_points}")
    print()
    
    marker_ids = []
    scan_count = 0
    
    for x in x_range:
        for y in y_range:
            target_pos = [x, y, z_height]
            result = check_ik_reachable(env, target_pos)
            
            scan_count += 1
            if scan_count % 100 == 0:
                print(f"  进度: {scan_count}/{total_points} ({100*scan_count/total_points:.1f}%)")
            
            if result['reachable']:
                reachable_points.append((x, y))
                color = [0, 1, 0, 0.6]  # 绿色 = 可达
            else:
                unreachable_points.append((x, y, result['reason']))
                color = [1, 0, 0, 0.3]  # 红色 = 不可达
            
            # 可视化标记
            if show_markers:
                marker_id = pc.createVisualShape(
                    shapeType=pc.GEOM_SPHERE,
                    radius=GRID_RESOLUTION * 0.4,
                    rgbaColor=color
                )
                body_id = pc.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=marker_id,
                    basePosition=[x, y, z_height]
                )
                marker_ids.append(body_id)
    
    # 统计结果
    reachable_count = len(reachable_points)
    unreachable_count = len(unreachable_points)
    reachable_ratio = reachable_count / total_points * 100
    
    print()
    print("=" * 50)
    print("扫描完成！")
    print("=" * 50)
    print(f"  可达点数: {reachable_count} ({reachable_ratio:.1f}%)")
    print(f"  不可达点数: {unreachable_count} ({100-reachable_ratio:.1f}%)")
    
    # 计算可达范围边界
    if reachable_points:
        xs = [p[0] for p in reachable_points]
        ys = [p[1] for p in reachable_points]
        print()
        print("可达范围边界:")
        print(f"  X: [{min(xs):.3f}, {max(xs):.3f}] m")
        print(f"  Y: [{min(ys):.3f}, {max(ys):.3f}] m")
        
        # 计算近似工作空间面积
        area = reachable_count * (GRID_RESOLUTION ** 2)
        print(f"  近似面积: {area*10000:.1f} cm²")
    
    return {
        'reachable_points': reachable_points,
        'unreachable_points': unreachable_points,
        'marker_ids': marker_ids,
        'z_height': z_height,
    }


def scan_multiple_heights(env, heights=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]):
    """在多个高度扫描工作空间
    
    Args:
        env: RobotGraspEnv 实例
        heights: 要扫描的高度列表
        
    Returns:
        dict: 各高度的扫描结果
    """
    results = {}
    
    for z in heights:
        print(f"\n{'='*60}")
        print(f"扫描高度: Z = {z} m")
        print('='*60)
        result = scan_workspace(env, z_height=z, show_markers=False)
        results[z] = result
        
        # 显示该高度的简要统计
        total = len(result['reachable_points']) + len(result['unreachable_points'])
        reachable = len(result['reachable_points'])
        print(f"  → 可达率: {100*reachable/total:.1f}%")
    
    return results


def draw_workspace_boundary(env, z_height=SCAN_Z_HEIGHT):
    """绘制工作空间边界轮廓
    
    使用极坐标扫描找到各方向的最远可达点
    """
    pc = env.base_env.physics_client
    
    # 以机械臂基座为圆心，极坐标扫描
    angles = np.linspace(0, 2*np.pi, 72)  # 每 5 度一个采样
    radii = np.arange(0.05, 0.8, 0.02)    # 从 5cm 到 80cm
    
    boundary_points = []
    
    print("正在计算工作空间边界...")
    
    for angle in angles:
        max_r = 0
        for r in radii:
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            target_pos = [x, y, z_height]
            
            result = check_ik_reachable(env, target_pos)
            if result['reachable']:
                max_r = r
        
        if max_r > 0:
            bx = max_r * np.cos(angle)
            by = max_r * np.sin(angle)
            boundary_points.append([bx, by, z_height])
    
    # 绘制边界线
    if len(boundary_points) > 1:
        for i in range(len(boundary_points)):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % len(boundary_points)]
            pc.addUserDebugLine(
                p1, p2,
                lineColorRGB=[0, 1, 1],
                lineWidth=3,
                lifeTime=0
            )
    
    print(f"边界绘制完成，共 {len(boundary_points)} 个边界点")
    
    # ============ 计算并输出半圆可达范围 ============
    if boundary_points:
        # 计算各方向可达半径
        radii_by_angle = []
        for p in boundary_points:
            r = np.sqrt(p[0]**2 + p[1]**2)
            angle = np.arctan2(p[1], p[0])  # [-π, π]
            radii_by_angle.append((angle, r, p[0], p[1]))
        
        # 按角度排序
        radii_by_angle.sort(key=lambda x: x[0])
        
        # 分区统计
        # 前方 (Y > 0): 角度在 [0, π]
        # 后方 (Y < 0): 角度在 [-π, 0]
        # 左侧 (X < 0): 角度在 [π/2, π] 或 [-π, -π/2]
        # 右侧 (X > 0): 角度在 [-π/2, π/2]
        
        front_radii = [r for ang, r, x, y in radii_by_angle if y > 0.01]
        back_radii = [r for ang, r, x, y in radii_by_angle if y < -0.01]
        left_radii = [r for ang, r, x, y in radii_by_angle if x < -0.01]
        right_radii = [r for ang, r, x, y in radii_by_angle if x > 0.01]
        all_radii = [r for ang, r, x, y in radii_by_angle]
        
        # 计算统计值
        min_r = min(all_radii) if all_radii else 0
        max_r = max(all_radii) if all_radii else 0
        avg_r = np.mean(all_radii) if all_radii else 0
        
        front_max = max(front_radii) if front_radii else 0
        front_min = min(front_radii) if front_radii else 0
        back_max = max(back_radii) if back_radii else 0
        left_max = max(left_radii) if left_radii else 0
        right_max = max(right_radii) if right_radii else 0
        
        # 计算安全可达半圆（所有方向都能到达的最小半径）
        safe_r = min_r
        
        # 找出最远点
        max_point = max(radii_by_angle, key=lambda x: x[1])
        max_angle_deg = np.rad2deg(max_point[0])
        
        print()
        print("=" * 60)
        print(f"高度 Z = {z_height} m 的可达范围分析")
        print("=" * 60)
        print()
        print("【半圆可达范围】")
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │  最小可达半径 (安全圆):  {min_r*100:6.1f} cm     │")
        print(f"  │  最大可达半径:           {max_r*100:6.1f} cm     │")
        print(f"  │  平均可达半径:           {avg_r*100:6.1f} cm     │")
        print(f"  └─────────────────────────────────────────┘")
        print()
        print("【各方向最大可达距离】")
        print(f"  前方 (+Y): {front_max*100:5.1f} cm    后方 (-Y): {back_max*100:5.1f} cm")
        print(f"  左侧 (-X): {left_max*100:5.1f} cm    右侧 (+X): {right_max*100:5.1f} cm")
        print()
        print(f"  最远可达点: ({max_point[2]*100:.1f}, {max_point[3]*100:.1f}) cm")
        print(f"             方向角: {max_angle_deg:.1f}°, 距离: {max_point[1]*100:.1f} cm")
        print()
        print("【推荐抓取范围】")
        print(f"  安全半圆范围: 以基座为圆心，半径 {safe_r*100:.1f} cm 的圆内区域")
        print(f"  有效工作半圆: 前方 180° 范围，半径 {min(front_min, safe_r)*100:.1f} ~ {front_max*100:.1f} cm")
        print()
        
        # ASCII 可视化
        print("【俯视图示意】(单位: 10cm)")
        print()
        _draw_ascii_workspace(radii_by_angle, z_height)
    
    return boundary_points


def _draw_ascii_workspace(radii_by_angle, z_height):
    """绘制 ASCII 工作空间俯视图"""
    # 创建字符网格
    size = 21  # 21x21 的网格
    grid = [[' ' for _ in range(size)] for _ in range(size)]
    center = size // 2
    scale = 0.1  # 每格 10cm
    
    # 绘制坐标轴
    for i in range(size):
        grid[center][i] = '─'
        grid[i][center] = '│'
    grid[center][center] = '┼'
    grid[center][size-1] = '→'  # +X
    grid[0][center] = '↑'       # +Y
    
    # 绘制边界点
    for ang, r, x, y in radii_by_angle:
        gx = int(round(x / scale)) + center
        gy = center - int(round(y / scale))  # Y 轴向上
        if 0 <= gx < size and 0 <= gy < size:
            grid[gy][gx] = '●'
    
    # 标记基座位置
    grid[center][center] = '◎'
    
    # 打印网格
    print(f"       +Y (前方)")
    for row in grid:
        print("    " + ''.join(row))
    print(f"       -Y (后方)        → +X (右侧)")
    print()
    print(f"  ◎ = 机械臂基座    ● = 可达边界点")
    print(f"  每格 = 10cm")


def interactive_test(env):
    """交互式测试：点击位置测试可达性"""
    pc = env.base_env.physics_client
    
    print()
    print("=" * 50)
    print("交互式可达性测试")
    print("=" * 50)
    print("在 PyBullet 窗口中按住 Ctrl 并点击鼠标左键选择目标位置")
    print("按 Q 键退出")
    print()
    
    test_marker = None
    
    while True:
        # 检查键盘事件
        keys = pc.getKeyboardEvents()
        if ord('q') in keys or ord('Q') in keys:
            print("退出交互测试")
            break
        
        # 检查鼠标事件
        mouse_events = pc.getMouseEvents()
        for event in mouse_events:
            event_type, mouse_x, mouse_y, button_idx, button_state = event
            
            # Ctrl + 左键点击
            if event_type == 2 and button_idx == 0 and button_state & 1:
                # 获取点击位置的射线
                # 注意：这里简化处理，假设点击在桌面平面上
                # 实际应该使用 rayTest 获取精确位置
                
                # 获取相机参数
                view_matrix = pc.getDebugVisualizerCamera()[2]
                proj_matrix = pc.getDebugVisualizerCamera()[3]
                
                # 这里使用简化方法：从当前视角投射
                width, height = 1280, 720
                # 归一化屏幕坐标
                x_norm = (2.0 * mouse_x / width) - 1.0
                y_norm = 1.0 - (2.0 * mouse_y / height)
                
                # 简化：在固定高度测试
                test_z = SCAN_Z_HEIGHT
                
                # 使用 rayTest 获取点击位置
                cam_info = pc.getDebugVisualizerCamera()
                cam_pos = cam_info[11]  # camera position
                cam_target = cam_info[12]  # camera target
                
                # 构造射线
                ray_from = list(cam_pos)
                ray_dir = np.array(cam_target) - np.array(cam_pos)
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                ray_to = np.array(ray_from) + ray_dir * 10
                
                hit = pc.rayTest(ray_from, ray_to.tolist())
                if hit[0][0] >= 0:  # 有碰撞
                    hit_pos = hit[0][3]
                    target_pos = [hit_pos[0], hit_pos[1], test_z]
                    
                    # 测试可达性
                    result = check_ik_reachable(env, target_pos)
                    
                    # 更新标记
                    if test_marker is not None:
                        pc.removeBody(test_marker)
                    
                    color = [0, 1, 0, 1] if result['reachable'] else [1, 0, 0, 1]
                    marker_visual = pc.createVisualShape(
                        shapeType=pc.GEOM_SPHERE,
                        radius=0.03,
                        rgbaColor=color
                    )
                    test_marker = pc.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=marker_visual,
                        basePosition=target_pos
                    )
                    
                    # 打印结果
                    status = "✓ 可达" if result['reachable'] else "✗ 不可达"
                    print(f"测试位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] - {status}")
                    if not result['reachable']:
                        print(f"  原因: {result['reason']}")
        
        time.sleep(0.01)


def main():
    print("=" * 60)
    print("机械臂工作空间分析工具")
    print("=" * 60)
    print()
    
    # 初始化环境
    env = RobotGraspEnv(config, evaluate=False)
    pc = env.base_env.physics_client
    env.reset()
    
    print("环境初始化完成")
    print()
    
    # 显示菜单
    while True:
        print()
        print("请选择操作:")
        print("  1. 扫描单一高度工作空间（可视化标记）")
        print("  2. 扫描多个高度工作空间（统计对比）")
        print("  3. 绘制工作空间边界轮廓")
        print("  4. 交互式可达性测试")
        print("  5. 退出")
        print()
        
        try:
            choice = input("请输入选项 (1-5): ").strip()
        except EOFError:
            choice = '1'
        
        if choice == '1':
            try:
                z = float(input(f"请输入扫描高度（默认 {SCAN_Z_HEIGHT} m）: ").strip() or SCAN_Z_HEIGHT)
            except (ValueError, EOFError):
                z = SCAN_Z_HEIGHT
            scan_workspace(env, z_height=z, show_markers=True)
            print("\n可视化标记已显示，绿色=可达，红色=不可达")
            print("在 PyBullet 窗口中查看结果")
            input("按 Enter 继续...")
            
        elif choice == '2':
            results = scan_multiple_heights(env)
            print("\n各高度可达率汇总:")
            for z, result in sorted(results.items()):
                total = len(result['reachable_points']) + len(result['unreachable_points'])
                reachable = len(result['reachable_points'])
                print(f"  Z={z:.2f}m: {100*reachable/total:.1f}%")
            input("按 Enter 继续...")
            
        elif choice == '3':
            try:
                z = float(input(f"请输入边界高度（默认 {SCAN_Z_HEIGHT} m）: ").strip() or SCAN_Z_HEIGHT)
            except (ValueError, EOFError):
                z = SCAN_Z_HEIGHT
            draw_workspace_boundary(env, z_height=z)
            print("\n边界轮廓已绘制（青色线条）")
            input("按 Enter 继续...")
            
        elif choice == '4':
            interactive_test(env)
            
        elif choice == '5':
            print("退出程序")
            break
        else:
            print("无效选项，请重新输入")
    
    env.close()


if __name__ == '__main__':
    main()
