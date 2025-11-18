import pybullet as p
import pybullet_data
import time
import cv2
import math
import numpy as np

class PyBulletEnvironment:
    def __init__(self):
        # 连接到 PyBullet 模拟器
        self.client_id = p.connect(p.GUI)  # 保存连接 ID
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 直接调用全局方法
        
        # 启用重力
        p.setGravity(0, 0, -9.8)  # 设置重力方向为负 Z 轴，值为 -9.8 m/s²
        # 设置时间步长
        self.time_step = 1.0 / 1000.0  # 默认时间步长为 1/240 秒
        p.setTimeStep(self.time_step)

        # 加载桌子
        self.table_id = p.loadURDF(
            "./URDF/objects/table/table.urdf",
            basePosition=[-0.65, 0.0, -0.63],
            baseOrientation=[0, 0, 0, 1]
        )

        # 加载机械臂
        self.robot_id = p.loadURDF(
            "./URDF/rm65.urdf",  # 替换为你的 URDF 文件路径
            basePosition=[0, 0, 0],  # 设置机械臂的基座位置为 [0, 0, 0]
            baseOrientation=[0, 0, 0, 1]  # 设置机械臂的基座方向为默认方向
        )

        # 固定机械臂的基座
        p.createConstraint(
            parentBodyUniqueId=self.robot_id,  # 机械臂的 ID
            parentLinkIndex=-1,  # 基座的链接索引，-1 表示基座
            childBodyUniqueId=-1,  # -1 表示固定到世界坐标系
            childLinkIndex=-1,  # 无需指定子链接
            jointType=p.JOINT_FIXED,  # 使用固定约束
            jointAxis=[0, 0, 0],  # 约束轴（固定约束不需要轴）
            parentFramePosition=[0, 0, 0],  # 机械臂基座的位置
            childFramePosition=[0, 0, 0]  # 世界坐标系的原点
        )

        # 设置机械臂的初始关节角
        self.set_initial_joint_angles()

                # 在桌子上放置物体
        self.objects = []
        self.add_objects_on_table()

    def set_initial_joint_angles(self):
        # 初始关节角（单位：度）
        neutral_deg = [0.0, 10, 80, 0, 70, 0]
        # 将角度转换为弧度
        neutral_rad = [math.radians(angle) for angle in neutral_deg]

        # 设置每个关节的初始角度
        for joint_index, joint_angle in enumerate(neutral_rad):
            p.resetJointState(self.robot_id, joint_index, joint_angle)

    def add_objects_on_table(self):
        # 放置一个立方体
        cube_id = p.loadURDF(
            "./URDF/objects/cube_small/cube_small.urdf",
            basePosition=[-0.5, 0.0, 0.1],  # 在桌子上的位置
            baseOrientation=[0, 0, 0, 1]
        )
        self.objects.append(cube_id)

        # 放置一个缩小的球体
        sphere_id = p.loadURDF(
            "./URDF/objects/sphere2/sphere2.urdf",
            basePosition=[-0.4, 0.1, 0.1],  # 在桌子上的位置
            baseOrientation=[0, 0, 0, 1],
            globalScaling=0.05  # 缩小球体的体积（缩放比例为 0.5）
        )
        self.objects.append(sphere_id)

        duck_id = p.loadURDF(
            "./URDF/objects/duck/duck.urdf",
            basePosition=[-0.4, 0.1, 0.1],  # 在桌子上的位置
            baseOrientation=[0, 0, 0, 1],
            globalScaling=1  # 缩小球体的体积（缩放比例为 0.5）
        )
        self.objects.append(duck_id)

    def get_camera_image(self):
        # 设置相机参数
        # resolution = (640, 480)
        resolution = (320, 240)
        fov = 87  # 水平视场角
        near_plane = 0.05
        far_plane = 2.0

        # 相机内参
        intrinsics = {
            "fx": 617.368,
            "fy": 617.368,
            "cx": 327.846,
            "cy": 242.789
        }

        # 获取机械臂末端的位姿
        link_state = p.getLinkState(self.robot_id, 8, computeForwardKinematics=1)  # 假设第8个链接是末端
        world_pos, world_orn = p.multiplyTransforms(
            link_state[4],
            link_state[5],
            [0.01, 0.015, 0.1],  # 安装偏移
            p.getQuaternionFromEuler([1.5708, 3.1416, 1.5708])  # 偏移方向
        )

        # 构建视图矩阵
        rot_matrix = np.array(p.getMatrixFromQuaternion(world_orn)).reshape(3, 3)
        target_pos = world_pos + rot_matrix @ np.array([0, 0, 0.1])  # 观察方向
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=world_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=rot_matrix @ np.array([0, -1, 0])  # Y轴向下
        )

        # 构建投影矩阵
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=resolution[0] / resolution[1],
            nearVal=near_plane,
            farVal=far_plane
        )

        # 获取原始图像
        _, _, rgb, depth_raw, _ = p.getCameraImage(
            width=resolution[0],
            height=resolution[1],
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # 处理深度图像
        depth_buffer = np.array(depth_raw).reshape(resolution[1], resolution[0])
        depth = far_plane * near_plane / (far_plane - (far_plane - near_plane) * depth_buffer)  # 深度转换

        # 添加深度噪声
        depth_noise = 0.02  # 深度噪声标准差
        noise = np.random.normal(0, depth_noise, depth.shape)
        depth += noise * depth

        # 转换为点云
        u = np.arange(resolution[0])
        v = np.arange(resolution[1])
        u, v = np.meshgrid(u, v)
        points = np.zeros((resolution[1], resolution[0], 3))
        points[..., 2] = depth  # Z = depth
        points[..., 0] = (u - intrinsics["cx"]) * points[..., 2] / intrinsics["fx"]
        points[..., 1] = (v - intrinsics["cy"]) * points[..., 2] / intrinsics["fy"]
        points = points.reshape(-1, 3)  # 转换为点云格式

        # 转换 RGB 图像为 OpenCV 格式
        rgb_image = np.array(rgb).reshape(resolution[1], resolution[0], 4)[..., :3]  # 去掉 alpha 通道
        rgb_image = rgb_image.astype(np.uint8)  # 确保数据类型为 uint8
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # 转换为 OpenCV 的 BGR 格式

        return rgb_image, depth, points


    def close(self):
        # 断开连接
        p.disconnect(self.client_id)

if __name__ == "__main__":
    env = PyBulletEnvironment()
    
    # 模拟主循环
    try:
        while True:
            p.stepSimulation()  # 更新模拟
            time.sleep(env.time_step)  # 按时间步长暂停

            # 获取相机画面
            camera_image = env.get_camera_image()


    except KeyboardInterrupt:
        print("模拟已终止。")
    env.close()
    cv2.destroyAllWindows()