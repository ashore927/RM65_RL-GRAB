"""仿真世界环境模块

此模块定义 `World` 类，它封装了基于 PyBullet 的物理仿真环境，
并提供一组管理仿真（初始化、重置、步进、模型管理）和快速搭建测试场景
（桌面、机器人、物体、RGBD 相机调试标记）的实用方法。

主要用途：作为强化学习环境或机器人控制测试平台的基础组件。
"""

import pybullet as p
import time
import gym
import os
import math
import pybullet_data
import xml.etree.ElementTree as ET

from gym.utils import seeding
from numpy.random import RandomState
from robot import Robot
from pybullet_utils import bullet_client
from sensor import RGBDSensor

class Env(gym.Env):
    """仿真世界类，封装 PyBullet 物理场景并兼容 Gym API。

    该类负责：
    - 管理与 PyBullet 的连接（BulletClient）
    - 控制仿真步长、重置与实时同步
    - 加载并管理模型（URDF）
    - 提供一些常用的场景搭建工具（用于测试和调试）
    - 与上层 Scene 对象协作以构建具体任务场景
    """

    def __init__(self, config, evaluate, test, validate):
        """初始化 Env 实例。

        Args:
            config: 一个字典，包含 'scene' 和 'simulation' 等配置项。
                其中 'scene' 用于选择场景类型（OnTable/OnFloor），
                'simulation' 包含可视化、实时等仿真参数。
            evaluate: 是否处于评估模式（决定随机种子是否固定）。
            test, validate: 透传给 Scene 构造函数的标志（由上层任务决定）。
        """
        # 初始化随机数生成器（可在 evaluate 模式下固定种子）
        self._rng = self.seed(evaluate=evaluate)

        # 仿真时间管理
        self.sim_time = 0.
        # PyBullet 常用时间步（240Hz），可按需修改
        self._time_step = 1. / 240.
        # 求解器迭代次数，影响接触求解精度
        self._solver_iterations = 150

        # 仿真相关的配置节
        simulation_cfg = config['simulation']
        visualize = simulation_cfg.get('visualize', True)
        # 是否以实时速度运行仿真（True 会在 step 后 sleep 以同步真实时间）
        self._real_time = simulation_cfg.get('real_time', True)
        # 创建 BulletClient（GUI 或 DIRECT 模式）
        self.physics_client = bullet_client.BulletClient(
            p.GUI if visualize else p.DIRECT)

        # 存放通过 add_model/load 模型得到的 Model 实例列表
        self.models = []


        self.table_id = None
        self.robot_id = None
        self.objects = []  # 记录通过 loadURDF 加载的 object ids
        self.sensor = None  # RGBD 传感器对象（若初始化成功）
        
        # 使用独立方法构建/初始化测试环境（加载机器人、桌子、物体、传感器等）
        # 以便在 reset_sim 中也能方便地重新构建场景
        self.init_env()

    def add_robot(self, path, start_pos, start_orn, scaling=1.):
        robot = Robot(self.physics_client)
        # 加载机器人时将 base 设置为固定（useFixedBase=True），
        # 以后在 init_test_env 中统一用电机锁定各关节。
        robot.load_model(path, start_pos, start_orn, scaling, static=True)
        self.models.append(robot)
        try:
            return robot.model_id
        except AttributeError:
            return robot

    def lock_all_joints(self, body_id, force_scale=50.0, min_force=10.0):
        """锁定关节并启用限位约束"""
        num_j = self.physics_client.getNumJoints(body_id)
        
        # 识别夹爪关节名称（WSG50 平行夹爪）
        gripper_joint_names = {'wsg50_finger_left_joint', 'wsg50_finger_right_joint'}
        
        for joint_idx in range(num_j):
            info = self.physics_client.getJointInfo(body_id, joint_idx)
            joint_type = info[2]
            joint_name = info[1].decode('utf-8') if isinstance(info[1], bytes) else str(info[1])
            
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                lower_limit = info[8]
                upper_limit = info[9]
                
                # 对夹爪关节使用更大的力矩，防止外力推动和穿模
                if joint_name in gripper_joint_names:
                    control_force = 500.0  # 夹爪关节使用超强力矩
                    limit_force = 5000.0   # 限位约束力提高5倍
                else:
                    control_force = max(info[10] * force_scale, min_force)
                    limit_force = 1000.0
                

                """解决夹爪半开半合/慢速问题：

                    将 force 增加到 500.0（极大值）。
                    关键点：显式设置 maxVelocity=10.0。URDF 中默认限制是 1.0 rad/s，这导致夹爪开合至少需要 0.75 秒，看起来很慢且像卡住。现在允许它以 10 倍速度运动。
                    调整了 Gain 参数以匹配高速运动。
                """
                # 启用关节限位（默认是禁用的）
                self.physics_client.setJointMotorControl2(
                    body_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=self.physics_client.getJointState(body_id, joint_idx)[0],
                    force=control_force,
                    maxVelocity=0.5  # 限制速度防止突变
                )
                
                # 关键：显式启用关节限位约束
                self.physics_client.changeDynamics(
                    body_id, joint_idx,
                    jointLowerLimit=lower_limit,
                    jointUpperLimit=upper_limit,
                    jointLimitForce=limit_force,
                    jointDamping=1.0  # 增加阻尼稳定关节
                )
    
    def run(self, duration):
        for _ in range(int(duration / self._time_step)):
            self.step_sim()

    def step_sim(self):
        """将仿真推进一步，并处理实时同步（如果启用）。"""
        self.physics_client.stepSimulation()
        self.sim_time += self._time_step
        if self._real_time:
            time.sleep(max(0., self.sim_time -
                       time.time() + self._real_start_time))

    def reset_sim(self):
        # [优化] 不再调用 resetSimulation()，避免销毁所有对象
        # 仅重置物理状态和时间
        self.sim_time = 0.
        self._real_start_time = time.time()

        # 1. 重置机器人关节到初始姿态
        self.set_initial_joint_angles()
        
        # 2. 重置桌上物体的位置和姿态
        # 强制将方块以平面朝上放置（四元数 [0,0,0,1] 表示无旋转）
        # 小扰动角度（±1 度以内）
        eps = 0.02
        rx = self._rng.uniform(-eps, eps)
        ry = self._rng.uniform(-eps, eps)
        rz = self._rng.uniform(-eps, eps)
        orn = self.physics_client.getQuaternionFromEuler([rx, ry, rz])
        
        # 假设第一个物体是立方体，重置到初始位置
        if len(self.objects) > 0:
            cube_id = self.objects[0]
            # 重置位置和速度（线速度/角速度清零）
            self.physics_client.resetBasePositionAndOrientation(
                cube_id, [-0.3, 0.0, 0.1], orn
            )
            self.physics_client.resetBaseVelocity(cube_id, [0, 0, 0], [0, 0, 0])

        # 3. 让仿真跑一小段时间以稳定物体（消除重置带来的瞬时穿模力）
        for _ in range(50):
            self.physics_client.stepSimulation()
            
        # 4. 重新锁定关节（防止外力干扰）
        if self.robot_id is not None:
            self.lock_all_joints(self.robot_id)
        

    def set_initial_joint_angles(self):
        """Set the robot to a neutral pose by resetting joint states.

        使用弧度列表调用 `resetJointState`，末端姿态为垂直向下，指向X正半轴。
        """
        # 末端姿态居中的关节角（弧度）
        # Joint1 设为 0（居中），提供双向±2π的活动空间
        neutral_rad = [
            0.0,        # Joint1:  0° - 居中位置，双向均有活动空间
            -0.034601,  # Joint2:  -1.9825°
            1.578469,   # Joint3:  90.4396°
            0.000004,   # Joint4:   0.0002°
            1.597716,   # Joint5:  91.5424°
            0.0         # Joint6:  0° - 居中
        ]

        for joint_index, joint_angle in enumerate(neutral_rad):
            self.physics_client.resetJointState(self.robot_id, joint_index, joint_angle)

    def add_objects_on_table(self):
        """放置几个示例物体，使用可控质量的 createMultiBody（避免 URDF 质量过小）。"""
        # 立方体（缩小尺寸和质量，使其容易被小夹爪夹取）
        cube_scale = 0.2 ** (1 / 3)  # 体积缩小到40%
        cube_half = 0.03 * cube_scale  # 夹爪更容易夹住
        cube_mass = 0.15  # 质量极轻，便于夹爪夹取和举起
        box_col = self.physics_client.createCollisionShape(
            self.physics_client.GEOM_BOX, halfExtents=[cube_half, cube_half, cube_half])
        box_vis = self.physics_client.createVisualShape(
            self.physics_client.GEOM_BOX, halfExtents=[cube_half, cube_half, cube_half], rgbaColor=[1, 0, 0, 1])
        cube_id = self.physics_client.createMultiBody(
            baseMass=cube_mass,
            baseCollisionShapeIndex=box_col,
            baseVisualShapeIndex=box_vis,
            basePosition=[-0.3, 0.0, 0.1],  # 前方（正Y）30cm处
            baseOrientation=[0, 0, 0, 1]
        )
        self.objects.append(cube_id)
        
        # 为立方体设置高摩擦力，便于夹爪抓取
        self.physics_client.changeDynamics(
            cube_id, -1,
            lateralFriction=1.5,  # 极高摩擦，防止夹爪滑动
            spinningFriction=0.5,
            rollingFriction=0.1,
            restitution=0.0,      # 完全非弹性
            linearDamping=0.05,
            angularDamping=0.05
        )

        # 强制将方块以平面朝上放置（四元数 [0,0,0,1] 表示无旋转）
        # 小扰动角度（±1 度以内）
        eps = 0.02
        rx = self._rng.uniform(-eps, eps)
        ry = self._rng.uniform(-eps, eps)
        rz = self._rng.uniform(-eps, eps)

        orn = self.physics_client.getQuaternionFromEuler([rx, ry, rz])
        self.physics_client.resetBasePositionAndOrientation(
            cube_id, [-0.3, 0.0, 0.1], orn
        )


    def init_env(self, camera_config=None):
        """初始化或重建测试环境：机器人、桌子、物体和相机。
        在调用 resetSimulation() 之后可以安全调用此方法（它会将模型加载到新的物理世界中）。
        """
        # 设置 pybullet 内置资源搜索路径
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 确保重力与物理参数在加载物体前生效，避免物体“悬浮”
        self.physics_client.setGravity(0, 0, -9.81)
        # 使用高精度参数，确保仿真稳定性
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self._time_step,
            numSolverIterations=500,              # 提高到500次迭代
            numSubSteps=10,                       # 增加子步数
            contactBreakingThreshold=0.0005,      # 更小的接触断裂阈值
            allowedCcdPenetration=0.0001,         # 更严格的CCD穿透限制
            enableConeFriction=1,
            erp=0.8,                              # 误差修正参数
            contactERP=0.9,                       # 接触误差修正
            frictionERP=0.9,                      # 摩擦误差修正
            constraintSolverType=p.CONSTRAINT_SOLVER_LCP_DANTZIG,
            globalCFM=0.00001
        )

        # 加载机器人并记录 body id
        # 使用集成了 WSG50 夹爪的 URDF
        self.robot_id = self.add_robot("./URDF/rm65_wsg50.urdf", [0, 0, 0], [0, 0, 0, 1], scaling=1.)

        # 解析 URDF 中的 mimic 关系（如果有），用于在 UI 中自动联动从动关节
        try:
            urdf_path = os.path.join("./URDF/rm65_wsg50.urdf")
            self._parse_urdf_mimics(urdf_path)
        except Exception:
            self.mimic_slave_to_master = {}
            self.mimic_master_to_slaves = {}

        # 加载桌子（X正半轴）
        self.table_id = self.physics_client.loadURDF(
            os.path.join("./URDF/objects/table/table.urdf"),
            basePosition=[-0.65, 0.0, -0.63],
            baseOrientation=[0, 0, 0, 1]
        )
        
        # 给桌子表面加高friction，防止物体滑动
        self.physics_client.changeDynamics(
            self.table_id, -1,
            lateralFriction=3.0,
            spinningFriction=2.5,
            contactStiffness=50000,
            contactDamping=5000
        )

        # 固定机器人基座到世界
        self.physics_client.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )

        # 关节初始位姿
        self.set_initial_joint_angles()

        # 在桌面放置示例物体
        self.objects = []
        self.add_objects_on_table()

        # 确保物体具有合适的摩擦/弹性，减少被反弹的情况
        # 仅对非球体、非桌子的物体应用通用设置
        # 球体和桌子已在创建时单独设置了高摩擦
        for oid in self.objects:
            # 简单判断：如果是球体（mass=0.5且有特殊设置），跳过覆盖
            # 这里通过 getDynamicsInfo 检查 friction 是否已经是高值来决定是否覆盖
            dyn = self.physics_client.getDynamicsInfo(oid, -1)
            current_lat_fric = dyn[1]
            
            # 如果摩擦力已经很高（>1.0），说明是特殊设置的物体（如球体），跳过
            if current_lat_fric > 1.0:
                continue
                
            self.physics_client.changeDynamics(
                oid, -1,
                restitution=0.05,
                lateralFriction=0.7,
                spinningFriction=0.02,
                rollingFriction=0.02,
                linearDamping=0.0,
                angularDamping=0.01,
            )
        
        # 为夹爪链接设置特殊的碰撞和动力学参数，防止穿模
        if self.robot_id is not None:
            # WSG50 平行夹爪的 link 名称
            gripper_link_names = ['wsg50_base_link', 'wsg50_gripper_left', 'wsg50_gripper_right', 
                                    'wsg50_finger_left', 'wsg50_finger_right']
            num_j = self.physics_client.getNumJoints(self.robot_id)
            for j in range(num_j):
                info = self.physics_client.getJointInfo(self.robot_id, j)
                link_name = info[12].decode('utf-8') if isinstance(info[12], bytes) else str(info[12])
                if link_name in gripper_link_names:
                    # 夹爪指尖需要极高摩擦
                    if 'finger' in link_name:
                        lateral_friction = 2.0  # 指尖极高摩擦
                        contact_stiffness = 50000.0
                        contact_damping = 5000.0
                    else:
                        lateral_friction = 1.0
                        contact_stiffness = 10000.0
                        contact_damping = 1000.0
                    
                    self.physics_client.changeDynamics(
                        self.robot_id, j,
                        lateralFriction=lateral_friction,  # 指尖或链接的摩擦
                        spinningFriction=0.1,
                        rollingFriction=0.05,
                        restitution=0.0,            # 无弹性
                        linearDamping=0.1,          # 线性阻尼
                        angularDamping=0.2,         # 角阻尼
                        contactStiffness=contact_stiffness,   # 接触刚度
                        contactDamping=contact_damping,       # 接触阻尼
                        collisionMargin=0.0001      # 极小碰撞边距
                    )
        
        # 加载并设置完初始角度后，用电机锁住所有关节，防止外力直接改变角度
        if self.robot_id is not None:
            self.lock_all_joints(self.robot_id)
        # 初始化相机配置
        if camera_config is None:
            # [优化] 移除不再使用的 yaml 路径，sensor.py 已改为硬编码
            camera_config = {
                'randomize': {
                    'focal_length': 5.0,
                    'optical_center': 2.0,
                    'translation': 0.1,
                    'rotation': 0.1
                },
                'full_observation': True
            }

        try:
            self.sensor = RGBDSensor(camera_config, self.physics_client, self.robot_id, randomize=True)
        except Exception:
            self.sensor = None

        # 在相机固定的 link 上绘制调试坐标
        # cam_link = self.find_link_index("camera_link")
        # if cam_link == -1:
        cam_link = 7
        self.mark_camera_position(cam_link)

        # 记录 realtime 同步起始时间
        self._real_start_time = time.time()
        # 创建实时关节控制面板（滑块），用于在 GUI 中实时调节每个关节目标角
        # try:
        #     self.create_joint_control_panel()
        # except Exception:
        #     pass

        # 设置初始相机为桌面侧方视角，便于观察机械臂与桌面
        self.set_side_camera_view()
        # 可视化相机视锥（调试用）
        # self.sensor.draw_camera_frustum(depth=1.0)

    def get_gripper_force(self):
        """获取夹爪与立方体之间的接触力，用于调试为什么夹不起来"""
        if not self.objects:
            return 0.0
        
        cube_id = self.objects[0]  # 假设第一个物体是立方体
        # [优化] 移除 try-except
        contacts = self.physics_client.getContactPoints(bodyA=cube_id, bodyB=self.robot_id)
        if not contacts:
            return 0.0
        total_force = sum([contact[9] for contact in contacts])  # contact[9] 是法向力
        return total_force

    def get_sensor_data(self):
        """If RGBDSensor 已初始化，则返回 (rgb, depth, mask)，否则返回 (None, None, None)。"""
        if self.sensor is None:
            return None, None, None
        # [优化] 移除 try-except
        return self.sensor.get_state()

    def mark_camera_position(self, link_index):
        """在给定的机器人关节索引处绘制 RGB 轴和文本标签.

        该标记使用 `addUserDebugLine` 和 `addUserDebugText` 并绑定到 `parentLinkIndex`，
        因此会随机器人 Link 一同移动。
        """
        axis_length = 0.15
        line_width = 4

        # X (red)
        self.physics_client.addUserDebugLine(
            lineFromXYZ=[0, 0, 0], lineToXYZ=[axis_length, 0, 0],
            lineColorRGB=[1, 0, 0], lineWidth=line_width,
            parentObjectUniqueId=self.robot_id, parentLinkIndex=link_index
        )
        # Y (green)
        self.physics_client.addUserDebugLine(
            lineFromXYZ=[0, 0, 0], lineToXYZ=[0, axis_length, 0],
            lineColorRGB=[0, 1, 0], lineWidth=line_width,
            parentObjectUniqueId=self.robot_id, parentLinkIndex=link_index
        )
        # Z (blue)
        self.physics_client.addUserDebugLine(
            lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0, axis_length],
            lineColorRGB=[0, 0, 1], lineWidth=line_width,
            parentObjectUniqueId=self.robot_id, parentLinkIndex=link_index
        )

        # label
        self.physics_client.addUserDebugText(
            text="CAMERA_EYE",
            textPosition=[0, 0, axis_length + 0.02],
            textColorRGB=[1, 1, 0],
            textSize=1.5,
            parentObjectUniqueId=self.robot_id,
            parentLinkIndex=link_index
        )

    def find_link_index(self, link_name):
        """根据 link 名称查找机器人中的 link 索引，找不到返回 -1。"""
        num_joints = self.physics_client.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = self.physics_client.getJointInfo(self.robot_id, i)
            # info[12] 为 bytes 类型的 link name
            try:
                name = info[12].decode('utf-8')
            except Exception:
                name = str(info[12])
            if name == link_name:
                return i
        return -1

    def set_side_camera_view(self, distance=1.2, yaw=0, pitch=-30, target_pos=None):
        """把 PyBullet 视角设置为桌面侧方观察（默认距离/角度可调）。

        distance: 摄像机与目标点的距离（米）
        yaw: 水平绕 Z 轴角度（度），90 为正侧面
        pitch: 垂直角度（度），负值向下看
        target_pos: 目标位置 [x,y,z]，若为 None 则自动以 table/robot 附近为目标
        """
        if target_pos is None:
            # 优先用桌子位置作为观察目标，否则用机器人基座或原点
            if self.table_id is not None:
                # table 在 init_env 中以 basePosition=[0.65, 0.0, -0.63] 加载，
                # 将观察目标提升到桌面大致高度（z≈0）
                target_pos = [-0.65, 0.0, 0.0]
            elif self.robot_id is not None:
                try:
                    bp, _ = self.physics_client.getBasePositionAndOrientation(self.robot_id)
                    target_pos = [bp[0], bp[1], bp[2] + 0.2]
                except Exception:
                    target_pos = [0, 0, 0]
            else:
                target_pos = [0, 0, 0]

        # 调用 BulletClient 的 resetDebugVisualizerCamera 设定视角
        try:
            self.physics_client.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=yaw,
                cameraPitch=pitch,
                cameraTargetPosition=target_pos
            )
        except Exception:
            # 在非 GUI（DIRECT）模式下会抛异常，忽略
            pass

    def create_joint_control_panel(self):
        """在 PyBullet GUI 中为每个可动关节创建滑块（User Debug Parameters）。

        创建后会把参数 id 存入 `self.joint_ui_params`，键为 joint index。
        """
        self.joint_ui_params = {}
        if self.robot_id is None:
            return
        num_j = self.physics_client.getNumJoints(self.robot_id)
        for j in range(num_j):
            info = self.physics_client.getJointInfo(self.robot_id, j)
            jtype = info[2]
            name = info[1].decode('utf-8') if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            # 仅为可旋转/线性关节创建滑块
            # 如果该关节是 mimic 的从动（slave），则不为其创建独立滑块，由主关节驱动
            if hasattr(self, 'mimic_slave_to_master') and j in self.mimic_slave_to_master:
                continue
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                lower, upper = info[8], info[9]
                # 如果无有效限位，则给出默认范围
                if not (isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and lower < upper):
                    lower, upper = -3.1416, 3.1416
                # 当前关节角作为初始值（并 clamp 到范围内）
                try:
                    cur = self.physics_client.getJointState(self.robot_id, j)[0]
                except Exception:
                    cur = 0.0
                cur = max(min(cur, upper), lower)
                pid = self.physics_client.addUserDebugParameter(name, lower, upper, cur)
                self.joint_ui_params[j] = pid

        # 为主关节创建完 UI 后，确保从动关节的数据结构存在（即便没有滑块）
        if not hasattr(self, 'mimic_master_to_slaves'):
            self.mimic_master_to_slaves = {}

    def update_joints_from_ui(self):
        """读取滑块的值并把目标位置下发为 POSITION_CONTROL，从而以电机驱动关节。

        应在主循环每步调用以实时采样滑块值。
        """
        if not hasattr(self, 'joint_ui_params') or self.robot_id is None:
            return
        
        # 识别夹爪主关节（WSG50 平行夹爪）
        gripper_master_joints = {'wsg50_finger_left_joint', 'wsg50_finger_right_joint'}
        
        for j, pid in self.joint_ui_params.items():
            try:
                target = float(self.physics_client.readUserDebugParameter(pid))
            except Exception:
                continue
            info = self.physics_client.getJointInfo(self.robot_id, j)
            joint_name = info[1].decode('utf-8') if isinstance(info[1], bytes) else str(info[1])
            
            # 夹爪主关节使用超强力矩
            if joint_name in gripper_master_joints:
                max_force = 500.0
                max_vel = 0.5
                pos_gain = 0.8
            else:
                effort = info[10] if info and len(info) > 10 else 50.0
                max_force = max(20.0, float(effort) * 50.0)
                max_vel = 1.0
                pos_gain = 0.3
            
            try:
                self.physics_client.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=max_force,
                    maxVelocity=max_vel,
                    positionGain=pos_gain
                )
            except Exception:
                pass

        # 处理 mimic 关系：为所有从动关节设置与主关节联动的目标位置
        # self.mimic_master_to_slaves: master_index -> [(slave_index, multiplier, offset), ...]
        if hasattr(self, 'mimic_master_to_slaves'):
            for master_idx, slaves in self.mimic_master_to_slaves.items():
                # 获取主关节当前目标：优先从 UI 参数读取，否则读取实际关节角
                if master_idx in self.joint_ui_params:
                    try:
                        master_target = float(self.physics_client.readUserDebugParameter(self.joint_ui_params[master_idx]))
                    except Exception:
                        master_target = self.physics_client.getJointState(self.robot_id, master_idx)[0]
                else:
                    master_target = self.physics_client.getJointState(self.robot_id, master_idx)[0]

                for (slave_idx, mult, offs) in slaves:
                    slave_target = float(master_target) * float(mult) + float(offs)
                    try:
                        info = self.physics_client.getJointInfo(self.robot_id, slave_idx)
                        # mimic从动关节使用超强力矩，确保严格跟随不失步
                        max_force = 500.0  # 固定使用大力矩
                        self.physics_client.setJointMotorControl2(
                            bodyIndex=self.robot_id,
                            jointIndex=slave_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=slave_target,
                            force=max_force,
                            maxVelocity=0.5,  # 限制速度保持同步
                            positionGain=0.8  # 增加位置增益
                        )
                    except Exception:
                        continue

    def _parse_urdf_mimics(self, urdf_path):
        """解析 URDF 文件中的 <mimic> 标签，构建主/从关节映射。

        结果保存在两个 dict 中：
            - self.mimic_slave_to_master: slave_index -> (master_index, multiplier, offset)
            - self.mimic_master_to_slaves: master_index -> [(slave_index, multiplier, offset), ...]
        """
        self.mimic_slave_to_master = {}
        self.mimic_master_to_slaves = {}
        if not os.path.exists(urdf_path):
            return
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except Exception:
            return

        # 先创建 joint name -> index 的映射
        joint_name_to_index = {}
        num_j = self.physics_client.getNumJoints(self.robot_id)
        for j in range(num_j):
            info = self.physics_client.getJointInfo(self.robot_id, j)
            jname = info[1].decode('utf-8') if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            joint_name_to_index[jname] = j

        # 遍历 URDF 中的 joint 元素，查找包含 mimic 子标签的关节（这些是从动）
        for joint_elem in root.findall('joint'):
            j_name = joint_elem.get('name')
            if j_name is None:
                continue
            mimic = joint_elem.find('mimic')
            if mimic is None:
                continue
            master_name = mimic.get('joint')
            mult = float(mimic.get('multiplier', '1.0'))
            offs = float(mimic.get('offset', '0.0'))
            # 将名称映射到索引
            if j_name in joint_name_to_index and master_name in joint_name_to_index:
                slave_idx = joint_name_to_index[j_name]
                master_idx = joint_name_to_index[master_name]
                self.mimic_slave_to_master[slave_idx] = (master_idx, mult, offs)
                self.mimic_master_to_slaves.setdefault(master_idx, []).append((slave_idx, mult, offs))
        return

    def close(self):
        """断开与 PyBullet 的连接并释放资源。"""
        self.physics_client.disconnect()
    
    def seed(self, seed=None, evaluate=False, validate=False):
        """设置随机数种子并返回一个 RandomState 实例。

        当 `evaluate` 为 True 时，使用固定种子以保证可重复性（用于评估场景）。
        否则使用传入的可选随机种子或默认随机生成器。
        返回值：用于采样的 numpy RandomState 对象。
        """
        if evaluate:
            self._validate = validate
            # 为评估创建固定 RNG，保证每次实验对象序列一致
            self._rng = RandomState(1)
        else:
            self._validate = False
            # 若 seed 为 None，则 np_random 会随机化种子
            self._rng, seed = seeding.np_random(seed)
        return self._rng



