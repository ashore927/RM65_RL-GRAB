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
        """使用 POSITION_CONTROL 将 `body_id` 的所有旋转/平移关节锁定在当前的位置，
        使关节能够抵抗外部力（如鼠标拖动）。

        参数：
            body_id: int，pybullet 物体的唯一 ID
            force_scale: URDF 力量的乘数，用于计算电机力
            min_force: 最小电机力
        """
        num_j = self.physics_client.getNumJoints(body_id)
        # 提高求解器迭代次数以增加约束求解稳定性
        try:
            self.physics_client.setPhysicsEngineParameter(numSolverIterations=200)
        except Exception:
            pass

        for j in range(num_j):
            try:
                info = self.physics_client.getJointInfo(body_id, j)
                jtype = info[2]
                lower, upper, effort = info[8], info[9], info[10]
                if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    cur = self.physics_client.getJointState(body_id, j)[0]
                    # clamp 到上下限（若有限位）
                    if lower < upper:
                        cur = min(max(cur, lower), upper)
                    max_force = max(min_force, float(effort) * force_scale if effort is not None else min_force)
                    # 使用 POSITION_CONTROL 将电机锁住到当前角度
                    self.physics_client.setJointMotorControl2(
                        bodyIndex=body_id,
                        jointIndex=j,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=cur,
                        force=max_force
                    )
                    # 增加关节阻尼以减少抖动
                    try:
                        self.physics_client.changeDynamics(body_id, j, linearDamping=0.04, angularDamping=0.04)
                    except Exception:
                        pass
            except Exception:
                # 跳过无法处理的关节
                continue
    
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
        self.physics_client.resetSimulation()
        # 提高求解精度并启用 CCD 限制穿模
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self._time_step,
            numSolverIterations=max(self._solver_iterations, 300),
            contactBreakingThreshold=0.001,
            allowedCcdPenetration=0.001,
            enableConeFriction=1)
        self.physics_client.setGravity(0., 0., -9.81)    
        self.models = []
        self.sim_time = 0.

        self._real_start_time = time.time()

        # reset 后重建测试场景（init_test_env 会加载桌子、机器人等）
        try:
            self.init_test_env()
        except Exception:
            # 忽略重建错误，保留空场景以便上层处理
            pass
        

    def set_initial_joint_angles(self):
        """Set the robot to a neutral pose by resetting joint states.

        使用角度（度）列表并将其转换为弧度后调用 `resetJointState`。
        """
        neutral_deg = [0.0, 10, 80, 0, 70, 0]
        neutral_rad = [math.radians(angle) for angle in neutral_deg]

        for joint_index, joint_angle in enumerate(neutral_rad):
            # 使用 physics_client 的 resetJointState，适用于 BulletClient
            try:
                self.physics_client.resetJointState(self.robot_id, joint_index, joint_angle)
            except Exception:
                # 如果某个 joint_index 超出范围，忽略并继续
                pass

    def add_objects_on_table(self):
        """放置几个示例物体，使用可控质量的 createMultiBody（避免 URDF 质量过小）。"""
        # 立方体（50g -> 1.0kg）
        box_col = self.physics_client.createCollisionShape(self.physics_client.GEOM_BOX, halfExtents=[0.03,0.03,0.03])
        box_vis = self.physics_client.createVisualShape(self.physics_client.GEOM_BOX, halfExtents=[0.03,0.03,0.03], rgbaColor=[1,0,0,1])
        cube_id = self.physics_client.createMultiBody(baseMass=2.0, baseCollisionShapeIndex=box_col, baseVisualShapeIndex=box_vis,
                                                     basePosition=[-0.5, 0.0, 0.1], baseOrientation=[0,0,0,1])
        self.objects.append(cube_id)

        # 强制将方块以平面朝上放置（四元数 [0,0,0,1] 表示无旋转）
        self.physics_client.resetBasePositionAndOrientation(cube_id, [-0.5, 0.0, 0.1], [0, 0, 0, 1])

        # 球体（指定半径和质量）
        sph_col = self.physics_client.createCollisionShape(self.physics_client.GEOM_SPHERE, radius=0.03)
        sph_vis = self.physics_client.createVisualShape(self.physics_client.GEOM_SPHERE, radius=0.03, rgbaColor=[0,1,0,1])
        sphere_id = self.physics_client.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=sph_col, baseVisualShapeIndex=sph_vis,
                                                       basePosition=[-0.4, 0.1, 0.1], baseOrientation=[0,0,0,1])
        self.objects.append(sphere_id)

        # 对于复杂 URDF（如 duck），建议直接编辑 URDF 的 inertial mass；
        # 临时替换为简单可视对象或手动修改 URDF 并 reload
        try:
            duck_id = self.physics_client.loadURDF(os.path.join("./URDF/objects/duck/duck.urdf"),
                                                  basePosition=[-0.4, 0.15, 0.1],
                                                  baseOrientation=[0, 0, 0, 1], globalScaling=1)
            # 打印质量，若太小请修改 urdf 中的 <inertial>
            dyn = self.physics_client.getDynamicsInfo(duck_id, -1)
            print("duck mass:", dyn[0])
            self.objects.append(duck_id)
        except Exception:
            pass

    def init_env(self, camera_config=None):
        """初始化或重建测试环境：机器人、桌子、物体和相机。
        在调用 resetSimulation() 之后可以安全调用此方法（它会将模型加载到新的物理世界中）。
        """
        # 设置 pybullet 内置资源搜索路径
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 确保重力与物理参数在加载物体前生效，避免物体“悬浮”
        try:
            self.physics_client.setGravity(0, 0, -9.81)
            self.physics_client.setPhysicsEngineParameter(fixedTimeStep=self._time_step,
                                                          numSolverIterations=max(self._solver_iterations, 300),
                                                          contactBreakingThreshold=0.001,
                                                          allowedCcdPenetration=0.001)
        except Exception:
            pass

        # 加载机器人并记录 body id
        self.robot_id = self.add_robot("./URDF/rm65.urdf", [0, 0, 0], [0, 0, 0, 1], scaling=1.)

        # 解析 URDF 中的 mimic 关系（如果有），用于在 UI 中自动联动从动关节
        try:
            urdf_path = os.path.join("./URDF/rm65.urdf")
            self._parse_urdf_mimics(urdf_path)
        except Exception:
            self.mimic_slave_to_master = {}
            self.mimic_master_to_slaves = {}

        # 加载桌子
        self.table_id = self.physics_client.loadURDF(
            os.path.join("./URDF/objects/table/table.urdf"),
            basePosition=[-0.65, 0.0, -0.63],
            baseOrientation=[0, 0, 0, 1]
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
        try:
            body_list = list(self.objects)
            if self.table_id is not None:
                body_list.append(self.table_id)
            for bid in body_list:
                try:
                    # 只设置 base (-1)，根据需要可设置每个 link
                    self.physics_client.changeDynamics(bid, -1,
                                                       restitution=0.0,
                                                       lateralFriction=1.0,
                                                       spinningFriction=0.1,
                                                       rollingFriction=0.1)
                except Exception:
                    pass
        except Exception:
            pass
        
        # 让仿真跑一段时间以让物体在重力和碰撞下稳定落到桌面上
        try:
            self.run(1.0)  # 1s，视情况调整
        except Exception:
            pass

        # 调试输出：打印质量与接触信息，便于排查仍然悬浮/弹开的原因
        try:
            for oid in self.objects:
                dyn = self.physics_client.getDynamicsInfo(oid, -1)
                print(f"obj {oid} mass={dyn[0]}, friction={dyn[1] if len(dyn)>1 else 'N/A'}")
                contacts = self.physics_client.getContactPoints(bodyA=oid, bodyB=self.table_id) if self.table_id is not None else []
                print(f"obj {oid} contact points with table: {len(contacts)}")
        except Exception:
            pass

        # 加载并设置完初始角度后，用电机锁住所有关节，防止外力直接改变角度
        try:
            if self.robot_id is not None:
                self.lock_all_joints(self.robot_id)
        except Exception:
            pass

        # 初始化相机配置
        if camera_config is None:
            camera_config = {
                'camera_info': './config/camera_info.yaml',
                'transform': './config/camera_transform.yaml',
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
        cam_link = self.find_link_index("camera_link")
        if cam_link == -1:
            cam_link = 7
        self.mark_camera_position(cam_link)

        # 记录 realtime 同步起始时间
        self._real_start_time = time.time()
        # 创建实时关节控制面板（滑块），用于在 GUI 中实时调节每个关节目标角
        try:
            self.create_joint_control_panel()
        except Exception:
            pass

        # 设置初始相机为桌面侧方视角，便于观察机械臂与桌面
        try:
            # 若没有指定目标点，则以桌面附近为目标
            self.set_side_camera_view()
        except Exception:
            pass
        # 可视化相机视锥（调试用）
        self.sensor.draw_camera_frustum(depth=1.0)

    def get_sensor_data(self):
        """If RGBDSensor 已初始化，则返回 (rgb, depth, mask)，否则返回 (None, None, None)。"""
        if self.sensor is None:
            return None, None, None
        try:
            return self.sensor.get_state()
        except Exception:
            return None, None, None

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
                # table 在 init_env 中以 basePosition=[-0.65, 0.0, -0.63] 加载，
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
        for j, pid in self.joint_ui_params.items():
            try:
                target = float(self.physics_client.readUserDebugParameter(pid))
            except Exception:
                continue
            info = self.physics_client.getJointInfo(self.robot_id, j)
            effort = info[10] if info and len(info) > 10 else 50.0
            # 设定一个合理的力矩上限，确保电机能抵抗外力
            max_force = max(20.0, float(effort) * 50.0)
            try:
                self.physics_client.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=max_force
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
                        effort = info[10] if info and len(info) > 10 else 50.0
                        max_force = max(20.0, float(effort) * 50.0)
                        self.physics_client.setJointMotorControl2(
                            bodyIndex=self.robot_id,
                            jointIndex=slave_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=slave_target,
                            force=max_force
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



