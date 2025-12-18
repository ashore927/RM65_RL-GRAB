import os
import time
import pybullet as p
import numpy as np
import functools
import gym
import collections
from gym import spaces 
from enum import Enum

from manipulation_main.common import io_utils
from manipulation_main.common import transformations
from manipulation_main.common import transform_utils
from manipulation_main.gripperEnv import sensor, actuator
from manipulation_main.simulation.simulation import World 
from manipulation_main.gripperEnv.rewards import Reward, SimplifiedReward, ShapedCustomReward
from manipulation_main.gripperEnv.curriculum import WorkspaceCurriculum

def _reset(robot, actuator, depth_sensor, skip_empty_states=False):  # 定义重置函数
    """ 重置机器人环境，确保摄像头的视野中有物体. """
    ok = False
    while not ok:
        robot.reset_sim() # 重置仿真世界和场景
        robot.reset_model() # 重置机器人模型和状态
        actuator.reset() # 重置执行器状态
        _, _, mask = depth_sensor.get_state() # 获取深度传感器的状态（包括掩码）
        ok = len(np.unique(mask)) > 2  # 检查掩码中是否有足够的唯一值（表示有物体在视野内）

        if not skip_empty_states:
            ok = True

class RobotEnv(World):  # 定义机器人环境类，继承自World类

    class Events(Enum):        # 定义机器人环境中的事件枚举
        START_OF_EPISODE = 0   # 每个回合开始时触发的事件
        END_OF_EPISODE = 1     # 每个回合结束时触发的事件
        CLOSE = 2              # 环境关闭时触发的事件
        CHECKPOINT = 3         # 达到检查点时触发的事件

    class Status(Enum):        # 定义机器人环境中的状态枚举
        RUNNING = 0            # 机器人环境正在运行的状态
        SUCCESS = 1            # 机器人成功完成任务的状态
        FAIL = 2               # 机器人未能完成任务的状态
        TIME_LIMIT = 3         # 达到时间限制的状态

    def __init__(self, config, evaluate=False, test=False, validate=False):
        if not isinstance(config, dict):     # 检查配置是否为字典类型
            config = io_utils.load_yaml(config)   # 如果不是，则加载YAML配置文件
        
        super().__init__(config, evaluate=evaluate, test=test, validate=validate)  # 初始化父类World
        self._step_time = collections.deque(maxlen=10000)  # 用于记录每个步骤的时间
        self.time_horizon = config['time_horizon']         # 设置时间范围
        self._workspace = {'lower': np.array([-1., -1., -1]),  # 定义工作空间的边界
                           'upper': np.array([1., 1., 1.])}    # 上界
        self.model_path = config['robot']['model_path']   # 机器人模型路径
        self._simplified = config['simplified']   # 简化环境标志
        self.depth_obs = config.get('depth_observation', False)   # 深度观测标志 
        self.full_obs = config.get('full_observation', False)     # 完整观测标志
        self._initial_height = 0.3   # 初始高度
        self._init_ori = transformations.quaternion_from_euler(np.pi, 0., 0.)  # 初始方向
        self.main_joints = [0, 1, 2, 3] #FIXME make it better  # 主要关节索引
        self._left_finger_id = 7   # 左手指关节ID
        self._right_finger_id = 9  # 右手指关节ID
        self._fingers = [self._left_finger_id, self._right_finger_id]   # 手指关节列表

        self._model = None  # 机器人模型初始化为None
        self._joints = None  # 机器人关节初始化为None
        self._left_finger, self._right_finger = None, None   # 左右手指初始化为None
        self._actuator = actuator.Actuator(self, config, self._simplified)  # 初始化执行器

        self._camera = sensor.RGBDSensor(config['sensor'], self)  # 初始化RGB-D传感器

        # Assign the reward fn
        if self._simplified:  # 如果是简化环境
            self._reward_fn = SimplifiedReward(config['reward'], self)
        elif config['reward']['custom']:  # 如果使用自定义奖励函数
            self._reward_fn = ShapedCustomReward(config['reward'], self)
        else:      # 否则使用标准奖励函数
            self._reward_fn = Reward(config['reward'], self)

        # Setup sensors
        if self.depth_obs or self.full_obs:  # 如果使用深度观测或完整观测
            self._sensors = [self._camera]
        else:                                # 否则使用编码深度图传感器
            self._encoder = sensor.EncodedDepthImgSensor(
                                    config, self._camera, self)
            self._sensors = [self._encoder]
        if not self._simplified:            # 如果不是简化环境
            self._sensors.append(self._actuator)  # 添加执行器传感器

        self.curriculum = WorkspaceCurriculum(config['curriculum'], self, evaluate)  # 初始化课程学习

        self.history = self.curriculum._history  # 课程学习历史记录
        self._callbacks = {RobotEnv.Events.START_OF_EPISODE: [],  # 初始化事件回调字典
                        RobotEnv.Events.END_OF_EPISODE: [],  # 每个事件对应一个空列表
                        RobotEnv.Events.CLOSE: [],           
                        RobotEnv.Events.CHECKPOINT: []}
        self.register_events(evaluate, config)   # 注册事件回调函数
        self.sr_mean = 0.   # 成功率均值初始化为0
        self.setup_spaces()  # 设置动作空间和观测空间

    def register_events(self, evaluate, config):   # 注册事件回调函数
        # Setup the reset function    
        skip_empty_states = True if evaluate else config['skip_empty_initial_state']  # 确定是否跳过空状态
        reset = functools.partial(_reset, self, self._actuator, self._camera,
                                skip_empty_states)  # 创建部分函数用于重置环境

        # Register callbacks
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, reset)  # 注册重置回调函数
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._camera.reset)  # 注册摄像头重置回调函数
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._reward_fn.reset)  # 注册奖励函数重置回调函数
        self.register_callback(RobotEnv.Events.END_OF_EPISODE, self.curriculum.update)  # 注册课程学习更新回调函数
        self.register_callback(RobotEnv.Events.CLOSE, super().close) # 注册关闭回调函数

    def reset(self):  # 重置机器人环境
        self._trigger_event(RobotEnv.Events.START_OF_EPISODE)  # 触发每个回合开始事件
        self.episode_step = 0  # 重置回合步骤计数器
        self.episode_rewards = np.zeros(self.time_horizon)  # 初始化回合奖励数组
        self.status = RobotEnv.Status.RUNNING  # 设置初始状态为运行中
        self.obs = self._observe()  # 获取初始观测值

        return self.obs  # 返回初始观测值

    def reset_model(self): # 重置机器人模型
        """Reset the task.

        Returns:
            Observation of the initial state.
        """
        self.endEffectorAngle = 0.  # 初始化末端执行器角度
        start_pos = [0., 0., self._initial_height]  # 设置机器人起始位置
        self._model = self.add_model(self.model_path, start_pos, self._init_ori)  # 添加机器人模型到环境中
        self._joints = self._model.joints  # 获取机器人关节列表
        self.robot_id = self._model.model_id  # 获取机器人模型ID
        self._left_finger = self._model.joints[self._left_finger_id]  # 获取左手指关节
        self._right_finger = self._model.joints[self._right_finger_id]  # 获取右手指关节

    def _trigger_event(self, event, *event_args):  # 触发事件回调函数
        for fn, args, kwargs in self._callbacks[event]:  # 遍历注册的回调函数
            fn(*(event_args + args), **kwargs)  # 调用回调函数

    def register_callback(self, event, fn, *args, **kwargs):  # 注册事件回调函数
        """Register a callback associated with the given event."""
        self._callbacks[event].append((fn, args, kwargs))  # 将回调函数添加到事件对应的列表中

    def step(self, action): # 执行一步动作
        """Advance the Task by one step.

        Args:
            action (np.ndarray): The action to be executed.

        Returns:
            A tuple (obs, reward, done, info), where done is a boolean flag
            indicating whether the current episode finished.
        """
        if self._model is None:
            self.reset()

        self._actuator.step(action)  # 执行动作

        new_obs = self._observe()  # 获取新的观测值

        reward, self.status = self._reward_fn(self.obs, action, new_obs)  # 计算奖励和状态
        self.episode_rewards[self.episode_step] = reward  # 记录当前步骤的奖励

        if self.status != RobotEnv.Status.RUNNING:  # 如果状态不是运行中
            done = True
        elif self.episode_step == self.time_horizon - 1:  # 如果达到时间限制
            done, self.status = True, RobotEnv.Status.TIME_LIMIT
        else:
            done = False

        if done:  # 如果回合结束
            self._trigger_event(RobotEnv.Events.END_OF_EPISODE, self)

        self.episode_step += 1  # 增加回合步骤计数器
        self.obs = new_obs  # 更新观测值
        if len(self.curriculum._history) != 0:  # 如果课程学习历史记录不为空
            self.sr_mean = np.mean(self.curriculum._history)  # 计算成功率均值
        super().step_sim()  # 执行仿真步骤
        return self.obs, reward, done, {"is_success":self.status==RobotEnv.Status.SUCCESS, "episode_step": self.episode_step, "episode_rewards": self.episode_rewards, "status": self.status}

    def _observe(self):  # 获取观测值
        if not self.depth_obs and not self.full_obs:  # 如果不使用深度观测和完整观测
            obs = np.array([])  # 初始化观测值为空数组
            for sensor in self._sensors:  # 遍历传感器列表
                obs = np.append(obs, sensor.get_state())
            return obs
        else:
            rgb, depth, _ = self._camera.get_state()  #  获取RGB-D传感器状态
            sensor_pad = np.zeros(self._camera.state_space.shape[:2])  # 初始化传感器填充数组
            if self._simplified:  # 如果是简化环境
                #FIXME one dimensional depth observation is not working properly  # FIXME 一维深度观测无法正常工作
                # depth = depth[:, :, np.newaxis]  # 将深度图像扩展为三维
                obs_stacked = np.dstack((depth, sensor_pad))  # 堆叠深度图像和传感器填充
                return obs_stacked  # 返回观测值
                # return depth

            sensor_pad = np.zeros(self._camera.state_space.shape[:2])  # 初始化传感器填充数组
            sensor_pad[0][0] = self._actuator.get_state()  # 获取执行器状态并填充到传感器填充数组
            if self.full_obs:  # 如果使用完整观测
                obs_stacked = np.dstack((rgb, depth, sensor_pad))  # 堆叠RGB图像、深度图像和传感器填充
            else:
                obs_stacked = np.dstack((depth, sensor_pad))  # 堆叠深度图像和传感器填充
            return obs_stacked

    def setup_spaces(self):  # 设置动作空间和观测空间
        self.action_space = self._actuator.setup_action_space()  # 设置动作空间
        if not self.depth_obs and not self.full_obs:  # 如果不使用深度观测和完整观测
            low, high = np.array([]), np.array([])  # 初始化观测空间的低高值为空数组
            for sensor in self._sensors:  # 遍历传感器列表
                low = np.append(low, sensor.state_space.low)  # 追加传感器的低值
                high = np.append(high, sensor.state_space.high)  # 追加传感器的高值
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32) # 设置观测空间为盒子空间
        else: # 如果使用深度观测或完整观测
            shape = self._camera.state_space.shape  # 获取摄像头状态空间的形状
            if self._simplified:  # 如果是简化环境
                # Depth
                # self.observation_space = self._camera.state_space
                self.observation_space = gym.spaces.Box(low=0, high=255,
                                    shape=(shape[0], shape[1], 2))  # 设置观测空间为盒子空间（深度图像和填充）
            else:  # 如果不是简化环境
                if self.full_obs: # RGB + Depth + Actuator  
                    self.observation_space = gym.spaces.Box(low=0, high=255,
                                                        shape=(shape[0], shape[1], 5))  # 设置观测空间为盒子空间（RGB图像、深度图像和填充）
                else: # Depth + Actuator obs
                    self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                        shape=(shape[0], shape[1], 2))  # 设置观测空间为盒子空间（深度图像和填充）

    def reset_robot_pose(self, target_pos, target_orn):  # 重置机器人位置
        """ Reset the world coordination of the robot base. Useful for test purposes """
        self.reset_base(self._model.model_id, target_pos, target_orn)  # 重置机器人基座位置
        self.run(0.1)  # 运行仿真一段时间

    def absolute_pose(self, target_pos, target_orn):  # 设置机器人绝对位置
        # target_pos = self._enforce_constraints(target_pos)

        target_pos[1] *= -1  # Invert Y axis
        target_pos[2] = -1 * (target_pos[2] - self._initial_height)  # Adjust Z axis

        # _, _, yaw = transform_utils.euler_from_quaternion(target_orn)
        # yaw *= -1
        yaw = target_orn  # Assume target_orn is yaw angle directly
        comp_pos = np.r_[target_pos, yaw]  # 组合位置和偏航角

        for i, joint in enumerate(self.main_joints):  # 遍历主要关节
            self._joints[joint].set_position(comp_pos[i])
        
        self.run(0.1)

    def relative_pose(self, translation, yaw_rotation):  # 设置机器人相对位置
        pos, orn = self._model.get_pose()  # 获取机器人当前位置和方向
        _, _, yaw = transform_utils.euler_from_quaternion(orn)  # 获取偏航角
        #Calculate transformation matrices
        T_world_old = transformations.compose_matrix(
            angles=[np.pi, 0., yaw], translate=pos)  # 计算旧的世界变换矩阵
        T_old_to_new = transformations.compose_matrix(
            angles=[0., 0., yaw_rotation], translate=translation)  # 计算从旧到新的变换矩阵
        T_world_new = np.dot(T_world_old, T_old_to_new)  # 计算新的世界变换矩阵
        self.endEffectorAngle += yaw_rotation  # 更新末端执行器角度
        target_pos, target_orn = transform_utils.to_pose(T_world_new)  # 获取新的位置和方向
        self.absolute_pose(target_pos, self.endEffectorAngle)  # 设置绝对位置

    def close_gripper(self):  # 关闭夹爪
        self.gripper_close = True
        self._target_joint_pos = 0.05
        self._left_finger.set_position(self._target_joint_pos)
        self._right_finger.set_position(self._target_joint_pos)

        self.run(0.2)

    def open_gripper(self):  # 打开夹爪
        self.gripper_close = False
        self._target_joint_pos = 0.0
        self._left_finger.set_position(self._target_joint_pos)
        self._right_finger.set_position(self._target_joint_pos)

        self.run(0.2)

    def _enforce_constraints(self, position): # 强制执行位置约束
        """Enforce constraints on the next robot movement."""
        if self._workspace:
            position = np.clip(position,
                               self._workspace['lower'],
                               self._workspace['upper'])
        return position
    
    def get_gripper_width(self):  # 获取夹爪开合宽度
        """Query the current opening width of the gripper."""
        left_finger_pos = 0.05 - self._left_finger.get_position()
        right_finger_pos = 0.05 - self._right_finger.get_position()

        return left_finger_pos + right_finger_pos

    def object_detected(self, tol=0.005):  # 夹爪检测物体
        """Grasp detection by checking whether the fingers stalled while closing."""
        return self._target_joint_pos == 0.05 and self.get_gripper_width() > tol

    def get_pose(self):  # 获取机器人位置
        return self._model.get_pose()

    def is_simplified(self):  # 检查环境是否简化
        return self._simplified

    def is_discrete(self):  # 检查动作空间是否为离散
        return self._actuator.is_discrete()