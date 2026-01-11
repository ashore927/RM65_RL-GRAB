import copy
import os

import cv2
import gym
import numpy as np
import pybullet as p
# import tensorflow as tf

from utils import io_utils, transform_utils, camera_utils
# from manipulation_main.gripperEnv import encoders

class RGBDSensor:
    """收集场景的合成 RGBD 图像。该类还提供随机化相机参数的功能。
    
        属性：
            camera：对模拟 RGBD 相机的引用。
    """

    def __init__(self, config, physics_client, robot_id, randomize=True):  # 传感器配置字典：内参、外参、随机化参数等；机器人对象
        self._physics_client = physics_client   # 获取物理客户端
        self._robot = robot_id  # 获取机器人对象
        full_obs = config.get('full_observation', False)  # 是否使用完整观测（RGB + 深度）
        # [优化] 直接硬编码相机参数，避免频繁读取 YAML 文件
        # 硬编码 camera_info.yaml 内容 (64x64)
        self._camera_info = {
            'height': 64,
            'width': 64,
            'K': [69.76, 0.0, 32.19, 0.0, 77.25, 32.0, 0.0, 0.0, 1.0],
            'near': 0.02,
            'far': 2.0
        }
        
        # 硬编码 camera_transform.yaml 内容
        extrinsics_dict = {
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.5, -0.5, 0.5, -0.5]
        }
        
        self._transform = transform_utils.from_dict(extrinsics_dict)

        self._randomize = config.get('randomize', None) if randomize else None  # 获取随机化参数


        self._construct_camera(self._camera_info, self._transform)  # 构建相机对象
        

        self.state_space = gym.spaces.Box(low=0, high=1,  # 定义状态空间
                shape=(self.camera.info.height, self.camera.info.width, 1))  # 默认depth sensor state space
        if full_obs:  # 如果使用完整观测 RGB + 深度 + 遮罩
            #RGB + Depth
            self.state_space = gym.spaces.Box(low=0, high=255,
                shape=(self.camera.info.height, self.camera.info.width, 5))  # 定义状态空间为RGBD图像
        #TODO: Check for the config parameter to decide if depth or rgb is used  # TODO: 检查配置参数以决定使用深度还是RGB
        # RGB output  # 仅RGB输出
        # self.state_space = gym.spaces.Box(low=0, high=255,
        #   shape=(self.camera.info.height, self.camera.info.width, 3))  

        # Depth sensor state space 仅深度
        # self.state_space = gym.spaces.Box(low=0, high=1,
        #   shape=(self.camera.info.height, self.camera.info.width, 1))


    def reset(self):  # 重置相机参数
        if self._randomize is None:  
            return

        camera_info = copy.deepcopy(self._camera_info)  # 复制相机内参
        transform = np.copy(self._transform)  # 复制相机外参

        f = self._randomize['focal_length']  # 获取随机化参数
        c = self._randomize['optical_center']  # 获取随机化参数
        t = self._randomize['translation']  # 获取随机化参数
        r = self._randomize['rotation']  # 获取随机化参数

        # 随机化焦距 fx 和 fy
        camera_info['K'][0] += np.random.uniform(-f, f)
        camera_info['K'][4] += np.random.uniform(-f, f)
        # 随机化光学中心 cx 和 cy
        camera_info['K'][2] += np.random.uniform(-c, c)
        camera_info['K'][5] += np.random.uniform(-c, c)
        # 随机化平移
        magnitue = np.random.uniform(0., t)
        direction = transform_utils.random_unit_vector() # 生成一个随机单位向量， Eg: [0.26726124 0.53452248 0.80178373]
        transform[:3, 3] += magnitue * direction
        # 随机化旋转
        angle = np.random.uniform(0., r)
        axis = transform_utils.random_unit_vector() # 生成一个随机单位向量作为旋转轴
        q = transform_utils.quaternion_about_axis(angle, axis) # 根据旋转轴和角度生成四元数
        transform = np.dot(transform_utils.quaternion_matrix(q), transform)  # 应用旋转
        # 重建相机
        self._construct_camera(camera_info, transform)

    def get_state(self):  # 获取当前状态
        """Render an RGBD image and mask from the current viewpoint."""
        h_world_robot = transform_utils.from_pose(*self.get_pose())  # 获取机器人相对于世界的变换矩阵
        h_camera_world = np.linalg.inv(
            np.dot(h_world_robot, self._h_robot_camera))  # 计算相机相对于世界的变换矩阵
        rgb, depth, mask = self.camera.render_images(h_camera_world)  # 渲染图像
        return rgb, depth, mask  # 返回RGB图像、深度图像和遮罩

    def get_pose(self):
        """Return the pose of the model base."""
        pos, orn, _, _, _, _ = self._physics_client.getLinkState(self._robot, 7)
        return (pos, orn)

    def draw_camera_frustum(self, depth=1.0, color=[0, 1, 0]):
        """在世界坐标系中绘制相机视锥（以像素平面 depth 米处为截面）。
        depth: 在相机坐标系下的截面深度（米）
        color: 线条颜色
        """
        try:
            # 相机在世界中的变换：h_world_camera = h_world_robot * h_robot_camera
            h_world_robot = transform_utils.from_pose(*self.get_pose())
            h_world_camera = np.dot(h_world_robot, self._h_robot_camera)
            origin = h_world_camera[:3, 3]
            pc = self._physics_client
            # 相机内参
            K = self.camera.info.K
            fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
            w = self.camera.info.width; h = self.camera.info.height

            def pixel_to_world(u, v, d):
                # 像素 -> 相机坐标
                x = (u - cx) / fx * d
                y = (v - cy) / fy * d
                p_cam = np.array([x, y, d, 1.0])
                p_world = h_world_camera.dot(p_cam)
                return p_world[:3]

            corners = [
                pixel_to_world(0, 0, depth),
                pixel_to_world(w, 0, depth),
                pixel_to_world(w, h, depth),
                pixel_to_world(0, h, depth)
            ]

            def draw_dashed(a, b, color):
                vec = b - a
                L = np.linalg.norm(vec)
                if L <= 1e-6:
                    return
                dir = vec / L
                dash_len = 0.02
                gap_len = 0.01
                step = dash_len + gap_len
                t = 0.0
                while t < L:
                    s0 = t
                    s1 = min(t + dash_len, L)
                    p0 = (a + dir * s0).tolist()
                    p1 = (a + dir * s1).tolist()
                    pc.addUserDebugLine(p0, p1, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
                    t += step

            # 从相机原点到四角：虚线
            for c in corners:
                draw_dashed(origin, c, color)

            # 连接四角形成截面（虚线）
            for i in range(4):
                a = corners[i]
                b = corners[(i + 1) % 4]
                draw_dashed(a, b, color)

            # # 绘制相机坐标轴（短线）
            # axis_len = depth * 0.2
            # x_axis = h_world_camera[:3, 0] * axis_len + origin
            # y_axis = h_world_camera[:3, 1] * axis_len + origin
            # z_axis = h_world_camera[:3, 2] * axis_len + origin
            # self._physics_client.addUserDebugLine(origin.tolist(), x_axis.tolist(), [1, 0, 0], 3.0, 0)
            # self._physics_client.addUserDebugLine(origin.tolist(), y_axis.tolist(), [0, 1, 0], 3.0, 0)
            # self._physics_client.addUserDebugLine(origin.tolist(), z_axis.tolist(), [0, 0, 1], 3.0, 0)
        except Exception:
            pass

    def align_view_to_camera(self, distance=1.0):
        """把 PyBullet GUI 视角尽量对齐到相机位姿（近似 yaw/pitch）。
        distance: GUI 相机与目标点的距离（米）
        """
        try:
            h_world_robot = transform_utils.from_pose(*self.get_pose())
            h_world_camera = np.dot(h_world_robot, self._h_robot_camera)
            cam_pos = h_world_camera[:3, 3]
            # 相机前向（z 轴），注意方向约定：取 -z 作为朝向到 target
            forward = h_world_camera[:3, 2]
            # 计算 yaw/pitch（近似）
            import math
            yaw = math.degrees(math.atan2(forward[1], forward[0]))
            pitch = -math.degrees(math.asin(forward[2]))
            target = (cam_pos + forward * 0.1).tolist()
            # 使用 BulletClient 的 resetDebugVisualizerCamera 对齐 GUI 视角
            try:
                self._physics_client.resetDebugVisualizerCamera(cameraDistance=distance,
                                                                cameraYaw=yaw,
                                                                cameraPitch=pitch,
                                                                cameraTargetPosition=target)
            except Exception:
                pass
        except Exception:
            pass

    def _construct_camera(self, camera_info, transform):  # 构建相机对象
        self.camera = RGBDCamera(self._physics_client, camera_info)  # 创建RGBD相机对象
        self._h_robot_camera = transform  # 保存机器人到相机的变换矩阵
 
class RGBDCamera(object):  # RGBD相机类
    """OpenCV compliant camera model using PyBullet's built-in renderer.

    Attributes:
        info (CameraInfo): The intrinsics of this camera.

    使用 PyBullet 内置渲染器的 OpenCV 兼容相机模型。

    属性：
    info (CameraInfo)：该相机的内参。
    """

    def __init__(self, physics_client, config):  # 初始化相机
        self._physics_client = physics_client  # 保存物理客户端
        self.info = camera_utils.CameraInfo.from_dict(config)      # YAML 加载相机内参
        self._near = config['near']  # 近裁剪面
        self._far = config['far']  # 远裁剪面

        self.projection_matrix = _build_projection_matrix(
            self.info.height, self.info.width, self.info.K, self._near, self._far)  # 构建投影矩阵

    def render_images(self, view_matrix):  # 渲染图像
        """
        渲染合成的 RGB 和深度图像。

        参数：
        view_matrix: 从世界坐标系到相机坐标系的变换矩阵。

        返回：
        一个包含 RGB（高度 x 宽度 x 3 的 uint8）和深度（高度 x 宽度 的 float32）图像以及分割掩码的元组。
        """
        gl_view_matrix = view_matrix.copy()  # 复制视图矩阵
        gl_view_matrix[2, :] *= -1  # flip the Z axis to comply to OpenGL  # 翻转Z轴以符合OpenGL规范
        gl_view_matrix = gl_view_matrix.flatten(order='F')  # 转换为列优先顺序的扁平数组

        gl_projection_matrix = self.projection_matrix.flatten(order='F')  # 转换为列优先顺序的扁平数组

        result = self._physics_client.getCameraImage(  
            width=self.info.width,
            height=self.info.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_projection_matrix,
            renderer=p.ER_TINY_RENDERER)  # 获取相机图像

        # Extract RGB image  # 提取RGB图像
        rgb = np.asarray(result[2], dtype=np.uint8)  # 获取RGB数据
        rgb = np.reshape(rgb, (self.info.height, self.info.width, 4))[:, :, :3]  # 去掉alpha通道
        # Extract depth image  # 提取深度图像
        near, far = self._near, self._far  # 获取近远裁剪面
        depth_buffer = np.asarray(result[3], np.float32).reshape(
            (self.info.height, self.info.width))  # 获取深度缓冲区
        depth = 1. * far * near / (far - (far - near) * depth_buffer)  # 深度转换公式

        # Extract segmentation mask
        mask = result[4]  # 获取分割掩码

        return rgb, depth, mask


def _gl_ortho(left, right, bottom, top, near, far):  #  构建正交投影矩阵
    """Implementation of OpenGL's glOrtho subroutine."""    # OpenGL 的 glOrtho 子程序的实现。
    ortho = np.diag([2./(right-left), 2./(top-bottom), - 2./(far-near), 1.])  # 构建对角矩阵
    ortho[0, 3] = - (right + left) / (right - left)  # 设置平移分量
    ortho[1, 3] = - (top + bottom) / (top - bottom)  # 设置平移分量
    ortho[2, 3] = - (far + near) / (far - near)   # 设置平移分量
    return ortho  # 返回正交投影矩阵


def _build_projection_matrix(height, width, K, near, far):  # 构建投影矩阵
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    perspective = np.array([[fx, 0., -cx, 0.],
                            [0., fy, -cy, 0.],
                            [0., 0., near + far, near * far],
                            [0., 0., -1., 0.]])   # 构建透视投影矩阵
    ortho = _gl_ortho(0., width, height, 0., near, far)  # 构建正交投影矩阵
    return np.matmul(ortho, perspective)  # 返回投影矩阵