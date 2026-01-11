import numpy as np
import gym
from gym import spaces
from env import Env
import math

"""强化学习环境包装模块

基于 Env 类构建的 Gym 兼容强化学习环境，专注于机械臂夹取任务。
提供标准的 reset/step 接口、观测空间、动作空间和奖励函数。
"""


def quaternion_to_rotation_matrix(q):
    """将四元数转换为3x3旋转矩阵。
    
    参数:
        q: 四元数 [x, y, z, w] (PyBullet 格式)
    
    返回:
        3x3 旋转矩阵
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    # 归一化四元数
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-8:
        return np.eye(3)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # 计算旋转矩阵
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)
    
    return R


class RobotGraspEnv(gym.Env):
    """机械臂夹取强化学习环境
    
    任务：控制机械臂的6个关节和夹爪，将桌面上的物体夹取并举起。
    
    观测：状态向量（末端位姿、夹爪开合、物体相对末端位姿）
    动作：末端位置增量 + 夹爪开合
    奖励：距离/接触/抬升/成功
    """
    
    def __init__(self, config, evaluate=False, test=False, validate=False):
        """初始化强化学习环境。

        参数：
            config: 包含 scene/simulation 配置的字典
            evaluate: 评估模式开关
            test, validate: 额外标志位
        """
        print(f"[DEBUG] RobotEnv initialized (Modified Version)", flush=True)
        self.base_env = Env(config, evaluate, test, validate)
        self.config = config
        self.evaluate = evaluate
        
        # 环境参数
        self.num_joints = 6  # 机械臂关节数（关节1~6）
        # self.arm_joint_limits = [
        #     (-6.28, 6.28),
        #     (-2.268, 2.268),
        #     (-2.355, 2.355),
        #     (-3.1, 3.1),
        #     (-2.233, 2.233),
        #     (-6.28, 6.28),
        # ]
        self.arm_joint_limits = [
            (-2*np.pi, 2*np.pi),  # Joint1: ±2π（360°）
            (-3.1, 3.1),
            (-3.1, 3.1),
            (-3.1, 3.1),
            (-3.1, 3.1),
            (-2*np.pi, 2*np.pi),  # Joint6: ±2π（360°）
        ]
        
        # PD 控制参数（极端降速，应对超大幅度运动）
        self.kp = 2.0    # 位置增益最小化
        self.kd = 60.0   # 速度阻尼最大化
        self.gripper_kp = 1.0
        self.gripper_kd = 5.0
        
        # 减少单次探索长度，加快 reset 频率
        self.max_episode_steps = 80
        self.current_step = 0
        
        # 动作空间：末端位置增量 (dx, dy, dz) + 末端偏航增量 (dyaw) + 夹爪控制，共5维，范围 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        # 关键配置：ee_action_scale 必须与 action_repeat 配合
        # 配置原则：ee_action_scale * 跟踪率 ≈ 实际位移
        self.ee_action_scale = 0.02  # 降低到 0.02，让移动更精细
        self.ee_yaw_scale = 0.1      # 每步最大偏航角增量（约 5.7度）
        self.approach_bias_scale = 0.01  # 接触/近接时沿 z 轴微调的尺度（米）
        self.approach_gate_dist = 0.05   # 触发近接调节的距离阈值（米）

        # 动作映射配置：默认不反转 X（使 action[0]=+1 对应世界 +X 方向移动）
        action_map_cfg = (config or {}).get('action_mapping', {}) if isinstance(config, dict) else {}
        self.invert_x_action = bool(action_map_cfg.get('invert_x', False))
        
        # 末端参考点：使用指腹平面中心（由 finger 几何/关节决定）
        
        # 观测空间：归一化的 15 维向量
        # 0-2: 末端位置 (x, y, z)
        # 3:   末端偏航 (yaw/π)
        # 4:   夹爪开合幅度
        # 5-7: 物体相对末端位置 (dx, dy, dz)
        # 8:   物体相对末端偏航 (Δyaw / π)
        # 9-14: 机械臂6个关节角 (归一化)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )

        # 观测归一化尺度（确保数值落在可学习的范围内）
        self.ee_pos_norm_scale = 1.0   # 米；工作空间在约 1m 量级，直接缩放到 [-1, 1]
        self.rel_pos_norm_scale = 0.5  # 米；物体相对末端的位置误差一般在 0.5m 以内
        
        # 任务相关参数
        self.target_height = 0.3  # 物体需要达到的高度

        # 末端工作空间限制（用于避免 IK 目标越界导致“末端不动 -> stalled”）
        # 可通过 config['ee_workspace'] 配置：
        #   ee_workspace: { enabled: true, min: [x,y,z], max: [x,y,z] }
        ws_cfg = (config or {}).get('ee_workspace', {}) if isinstance(config, dict) else {}
        self.ee_workspace_enabled = bool(ws_cfg.get('enabled', True))
        self.ee_pos_min = np.array(ws_cfg.get('min', [-0.3, -0.6, 0.115]), dtype=np.float32)  # Z下限=115mm，考虑夹爪指尖偏移约100mm
        self.ee_pos_max = np.array(ws_cfg.get('max', [0.8, 0.6, 0.55]), dtype=np.float32)
        self._last_target_was_clipped = False
        
        # 课程学习阶段控制（支持外部按步数/成功率切换）
        reward_cfg = (config or {}).get('reward', {}) if isinstance(config, dict) else {}
        self.training_phase = int(reward_cfg.get('training_phase', 0))

        # 根据课程学习阶段调整动作范围
        if self.training_phase == 0:
            self.ee_action_scale = 0.05  # 早期阶段更精细的移动
        else:
            self.ee_action_scale = 0.08  # 后期阶段正常范围

        # 奖励系数（只保留靠近奖励，无需额外参数）

        # 分阶段权重：删除，因为只保留靠近奖励

        # 观测噪声配置：简化模式下禁用
        obs_cfg = (config or {}).get('observation', {}) if isinstance(config, dict) else {}
        self.obs_noise_pos_std = 0.0   # 禁用位置噪声
        self.obs_noise_yaw_deg = 0.0   # 禁用偏航噪声
        self.cube_obs_noise = {'pos': np.zeros(3, dtype=np.float32), 'yaw': 0.0}

        # 可选：将偏航相关观测维度置零（保持观测维度不变，便于兼容旧模型/脚本）。
        # - obs[3] = 末端偏航（通常为固定值，因为末端姿态被锁定）
        # - obs[8] = 物体相对末端偏航
        self.zero_ee_yaw_obs = bool(obs_cfg.get('zero_ee_yaw', False))
        self.zero_rel_yaw_obs = bool(obs_cfg.get('zero_rel_yaw', False))
        self.zero_yaw_obs = bool(obs_cfg.get('zero_yaw', False))

        # 领域随机化配置：简化模式下禁用
        dr_cfg = (config or {}).get('domain_randomization', {}) if isinstance(config, dict) else {}
        self.dr_base_mass = 0.1  # 固定质量
        self.dr_mass_range = [1.0, 1.0]  # 无质量变化
        self.dr_cube_friction_range = [1.0, 1.0]  # 固定摩擦
        self.dr_table_friction_range = [1.0, 1.0]  # 固定摩擦
        self.dr_xy_noise = 0.0   # 无位置噪声
        self.dr_yaw_noise = 0.0  # 无偏航噪声
        # 控制噪声：禁用
        self.control_noise_std = 0.0   # 无末端指令噪声
        self.gripper_noise_std = 0.0   # 无夹爪指令噪声
        
        # 控制参数
        self.control_frequency = 240  # Hz
        self.sim_substeps = 1   # 降低子步数，减少物理引擎计算量
        self.action_repeat = 50  # 增加到 50，减少每步位移
        self.gripper_force_threshold = 5.0
        # 固定末端姿态指向桌面（便于训练垂直抓取）
        try:
            self.ee_down_orn = self.base_env.physics_client.getQuaternionFromEuler([np.pi, 0, -np.pi/2])
        except Exception:
            self.ee_down_orn = [0, 0, 0, 1]

        # 课程学习阶段控制（阶段 0/1/2，对应靠近/接触/抬升悬停）
        self.curriculum_enabled = True
        self.curriculum_stage = 0
        self.curriculum_complete = False
        self.curriculum_hover_counter = 0
        self.curriculum_xy_thresh = 0.03          # 3 cm 平面对准阈值（更容易学习）
        self.curriculum_approach_height = 0.10     # 阶段 1：末端需在物体上方 10 cm
        self.curriculum_lift_height = 0.20         # 阶段 3：物体抬升 20 cm
        self.curriculum_hover_time = 3.0           # 阶段 3：悬停 3 s
        step_dt = (self.action_repeat * self.sim_substeps) / float(self.control_frequency)
        self.curriculum_hover_steps = max(1, int(np.ceil(self.curriculum_hover_time / max(step_dt, 1e-8))))

        # 缓存变量初始化
        self.desired_angles = None
        self.gripper_joints_cache = {}
        self.gripper_link_indices = []
        self.pc_cache = None
        self.gripper_base_link_idx = None
        self.initial_object_height = None
        self.object_id = None
        self.current_yaw = np.pi / 2  # 初始偏航角 (对应 [np.pi, 0, np.pi/2])
        
        # 初始化缓存（避免 AttributeError）
        if self.base_env.robot_id is not None:
             self._init_gripper_joints_cache()
             self._init_gripper_link_indices()

    def set_reward_phase(self, phase):
        """设置奖励函数的课程阶段"""
        if phase not in [0, 1, 2]:
            print(f"Warning: Unknown reward phase {phase}, defaulting to 0")
            phase = 0
        # 为保持兼容，映射到 training_phase
        self.training_phase = phase
        print(f"Reward Phase switched to: {phase}")
        
        # 记录初始物体位置用于计算成功标准
        self.initial_object_height = None
        self.object_id = None
        
        # 缓存变量
        self.gripper_joints_cache = None
        self.gripper_link_indices = None  # 缓存夹爪 link 索引
        self.pc_cache = None
        self.gripper_base_link_idx = None  # 夹爪基座 link 索引
        
    def object_detected(self):
        """判断是否成功夹住物体 (根据用户要求: 夹爪闭合状态 + 手指间距/接触)
        
        条件：
        1. 双指都接触到物体 (both_contact=True)
        2. 夹爪并未完全闭合 (说明中间有物体撑住)
        """
        _, _, both_contact = self._check_dual_finger_contact()
        
        if not both_contact:
            return False
            
        # 检查夹爪宽度，确保不是“空抓”到闭合限位
        # WSG50 闭合限位是 0.025 (每指)，允许一定误差
        # 如果关节位置接近 0.025，说明夹空了
        is_not_empty = True
        if self.gripper_joints_cache and 'wsg50_finger_left_joint' in self.gripper_joints_cache:
            try:
                joint_idx = self.gripper_joints_cache['wsg50_finger_left_joint']
                pos = self.base_env.physics_client.getJointState(self.base_env.robot_id, joint_idx)[0]
                # 0.025 是闭合，0.024 以上视为闭合（留 1mm 裕量）
                if pos > 0.024:
                    is_not_empty = False
            except Exception:
                pass
        
        return both_contact and is_not_empty

    def _compute_reward(self):
        """计算奖励函数 - 三阶段设计（修复版）
        
        阶段 1：XY 对齐
            - 末端 base_link 中心与物体中心在 XY 平面对齐
            - 对齐阈值：XY 距离 < 5mm
            - 高度保持：惩罚高度偏离，防止上升逃避
            
        阶段 2：下降 + 抓取（仅当 XY 已对齐）
            - 下降到物体高度，闭合夹爪包裹物体
            - 抓取判定：双指接触物体且夹爪未完全闭合
            
        阶段 3：抬升（仅当已抓住物体）
            - 将物体抬升至初始高度 + 0.3m
            - 达成后获得终止奖励
        """
        reward = 0.0
        reward_info = {}
        
        # ========== 获取状态信息 ==========
        tip_pos, _ = self._get_tip_position()
        
        if self.object_id is None:
            return -0.01, {'total': -0.01}
        
        obj_pos, _ = self.base_env.physics_client.getBasePositionAndOrientation(self.object_id)
        
        # 计算距离
        rel_pos = np.array(obj_pos) - np.array(tip_pos)
        xy_dist = float(np.linalg.norm(rel_pos[:2]))
        z_dist = float(tip_pos[2] - obj_pos[2])  # 末端比物体高多少
        
        # 记录物体初始高度（用于抬升目标计算）
        if self.initial_object_height is None:
            self.initial_object_height = obj_pos[2]
        
        # 记录末端初始高度（用于高度保持）
        if not hasattr(self, '_initial_tip_height') or self._initial_tip_height is None:
            self._initial_tip_height = tip_pos[2]
        
        # 阶段判断
        XY_ALIGN_THRESHOLD = 0.005  # 5mm
        is_xy_aligned = xy_dist < XY_ALIGN_THRESHOLD
        is_grasped = self.object_detected()
        target_lift_height = self.initial_object_height + 0.30  # 初始高度 + 30cm
        is_lifted = obj_pos[2] > target_lift_height
        
        # ========== 基础时间惩罚 ==========
        TIME_PENALTY = -0.1
        reward += TIME_PENALTY
        
        # ========== 阶段 1：XY 对齐 ==========
        # XY 进展奖励（始终有效，大幅增强）
        if not hasattr(self, '_last_xy_dist') or self._last_xy_dist is None:
            self._last_xy_dist = xy_dist
        
        xy_progress = self._last_xy_dist - xy_dist  # 正值表示靠近
        XY_PROGRESS_SCALE = 10.0  # 大幅增强：从5.0提高到10.0
        xy_progress_reward = XY_PROGRESS_SCALE * xy_progress
        reward += xy_progress_reward
        self._last_xy_dist = xy_dist
        reward_info['xy_progress'] = xy_progress_reward
        reward_info['xy_dist'] = xy_dist
        
        # XY 对齐达成奖励（增强）
        XY_ALIGN_BONUS = 5.0  # 大幅增强：从2.0提高到5.0
        if is_xy_aligned:
            reward += XY_ALIGN_BONUS
            reward_info['xy_aligned'] = XY_ALIGN_BONUS
        
        # [新增] 接近奖励：当XY距离小于特定阈值时给予额外奖励
        # 解决策略在35mm处停滞的问题，提供更强的终点吸引力
        CLOSE_THRESHOLD_1 = 0.03  # 30mm
        CLOSE_THRESHOLD_2 = 0.02  # 20mm  
        CLOSE_THRESHOLD_3 = 0.01  # 10mm
        
        # if xy_dist < CLOSE_THRESHOLD_1:
        #     CLOSE_BONUS_1 = 1.0
        #     reward += CLOSE_BONUS_1
        #     reward_info['close_30mm'] = CLOSE_BONUS_1
        # if xy_dist < CLOSE_THRESHOLD_2:
        #     CLOSE_BONUS_2 = 2.0
        #     reward += CLOSE_BONUS_2
        #     reward_info['close_20mm'] = CLOSE_BONUS_2
        if xy_dist < CLOSE_THRESHOLD_3:
            CLOSE_BONUS_3 = 5.0
            reward += CLOSE_BONUS_3
            reward_info['close_10mm'] = CLOSE_BONUS_3
        
        # [新增] Z轴逃逸惩罚：防止策略在接近目标时向上逃逸
        # 当XY距离小于30mm时，惩罚正向dz动作（向上移动）
        if xy_dist < 0.03 and hasattr(self, '_current_action'):
            dz_action = self._current_action[2] if len(self._current_action) > 2 else 0
            if dz_action > 0:  # 正dz表示向上移动
                Z_ESCAPE_PENALTY_SCALE = 3.0  # 惩罚系数
                z_escape_penalty = Z_ESCAPE_PENALTY_SCALE * dz_action
                reward -= z_escape_penalty
                reward_info['z_escape_penalty'] = -z_escape_penalty
        
        # ========== 阶段 2：下降 + 抓取（仅当 XY 已对齐）==========
        # 【简化版】：使用统一的距离势能差奖励，避免复杂分级导致局部最优
        # if is_xy_aligned and not is_grasped:
        if is_xy_aligned:
            # 计算3D距离（XY已对齐，主要是Z方向）
            dist_3d = np.sqrt(xy_dist**2 + z_dist**2)
            
            # 初始化距离追踪
            if not hasattr(self, '_last_dist_3d') or self._last_dist_3d is None:
                self._last_dist_3d = dist_3d
            
            # 【核心奖励】距离减少 = 正奖励（势能差方式）
            dist_progress = self._last_dist_3d - dist_3d
            DIST_PROGRESS_SCALE = 2000.0  # 大幅增强：每减少1cm获得2.0奖励
            dist_reward = DIST_PROGRESS_SCALE * dist_progress
            reward += dist_reward
            self._last_dist_3d = dist_3d
            reward_info['dist_progress'] = dist_reward
            
            # 【简化】距离越小，每步额外奖励越大（势能形式）
            # 这鼓励策略快速接近，而不是慢慢移动
            POTENTIAL_SCALE = 5.0
            potential_reward = POTENTIAL_SCALE / (dist_3d + 0.01)  # 距离越小奖励越大
            potential_reward = min(potential_reward, 50.0)  # 限制最大值
            reward += potential_reward
            reward_info['potential'] = potential_reward
            
            # 【修复】夹爪控制：考虑 wsg50_base_link 到指尖的偏移量（约100mm）
            # 当 z_dist < 180mm 时，指尖距物体约 80mm，可以开始闭合
            if z_dist < 0.18:  # 物体在夹爪抓取范围内
                if hasattr(self, '_current_action') and len(self._current_action) > 4:
                    gripper_action = self._current_action[4]
                    if gripper_action < -0.2:  # 闭合夹爪
                        GRIPPER_REWARD = 1000.0
                        reward += GRIPPER_REWARD
                        reward_info['gripper_close'] = GRIPPER_REWARD
            
        else:
            self._last_dist_3d = None
        
        # 抓住物体的一次性奖励
        if is_grasped and not getattr(self, '_grasp_rewarded', False):
            GRASP_SUCCESS_BONUS = 500.0  # 增强：从5.0提高到10.0
            reward += GRASP_SUCCESS_BONUS
            reward_info['grasp_success'] = GRASP_SUCCESS_BONUS
            self._grasp_rewarded = True
        elif not is_grasped:
            self._grasp_rewarded = False
        
        # ========== 阶段 3：抬升（仅当已抓住物体）==========
        if is_grasped:
            # 保持抓取奖励
            HOLD_BONUS = 100.0  # 增强：从0.5提高到1.0
            reward += HOLD_BONUS
            reward_info['hold'] = HOLD_BONUS
            
            # 抬升进展奖励
            if not hasattr(self, '_last_obj_z') or self._last_obj_z is None:
                self._last_obj_z = obj_pos[2]
            
            lift_progress = obj_pos[2] - self._last_obj_z  # 正值表示抬升
            LIFT_PROGRESS_SCALE = 100.0  # 增强：从5.0提高到10.0
            lift_progress_reward = LIFT_PROGRESS_SCALE * lift_progress
            reward += lift_progress_reward
            self._last_obj_z = obj_pos[2]
            reward_info['lift_progress'] = lift_progress_reward
            
            # 任务完成奖励
            if is_lifted:
                TASK_COMPLETE_BONUS = 1000.0  # 增强：从20.0提高到50.0
                reward += TASK_COMPLETE_BONUS
                reward_info['task_complete'] = TASK_COMPLETE_BONUS
        else:
            self._last_obj_z = None
        
        # [移除] 过早下压惩罚 - 这导致策略通过上升来逃避
        # 现在用高度保持惩罚替代
        
        reward_info['total'] = reward
        reward_info['is_grasped'] = float(is_grasped)
        reward_info['is_xy_aligned'] = float(is_xy_aligned)
        
        return reward, reward_info

    def reset(self):
        """重置环境状态。"""
        # [关键修复] 重置步数计数器，否则后续 Episode 会因为达到 max_step 而立即结束
        self.current_step = 0
        
        # 调用底层环境复位（包括机器人关节复位、物体随机放置等）
        self.base_env.reset_sim()
        
        # [Fix] 重置 desired_angles 为当前关节状态
        # 避免 IK 继承上一回合的怪异姿态作为 rest pose，导致向怪异姿态（如原点/卷曲）收敛
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        self.desired_angles = np.zeros(self.num_joints, dtype=np.float32)
        for i in range(self.num_joints):
             # 假设关节 0 ~ num_joints-1 是手臂关节
             state = pc.getJointState(robot_id, i)
             self.desired_angles[i] = state[0]
        
        # 获取目标物体 ID（假设列表中的第一个物体为目标）
        if hasattr(self.base_env, 'objects') and len(self.base_env.objects) > 0:
            self.object_id = self.base_env.objects[0]
        else:
            self.object_id = None
            
        # 在 reset 中主动初始化缓存，避免后续懒加载问题
        self._init_gripper_joints_cache()
        self._init_gripper_link_indices()
        self._init_gripper_joint_limits()  # 启用夹爪关节硬限位
        
        self.is_stalled = False  # 卡住检测状态
        self.success_hold_counter = 0  # 成功保持计数器
        self._last_distance = None  # 上一步距离，用于计算距离变化奖励
        self._last_rel_x = None  # 重置相对位置跟踪，用于“超过”惩罚
        self.current_yaw = np.pi / 2  # 重置偏航角


        # 每个回合采样一次观测噪声，模拟真实感知的随机偏差
        # 评估模式下禁用噪声，确保可复现性
        if self.evaluate:
            self.cube_obs_noise = {
                'pos': np.zeros(3, dtype=np.float32),
                'yaw': 0.0
            }
        else:
            self.cube_obs_noise = {
                'pos': np.random.normal(0.0, self.obs_noise_pos_std, size=3).astype(np.float32),
                # 均匀噪声的范围由配置给出（度），这里转换为弧度
                'yaw': float(np.deg2rad(np.random.uniform(-self.obs_noise_yaw_deg, self.obs_noise_yaw_deg)))
            }
        
        # 初始化训练分析指标（便于回合级记录）
        self.episode_metrics = {
            'reward_approach': 0.0,  # 累计靠近奖励
            'max_height': 0.0        # 本回合最大提升高度
        }
        
        # 初始化距离追踪（用于奖励计算）
        self._last_obj_pos = None
        self._last_xy_dist = None   # XY距离追踪
        self._last_z_dist = None    # Z距离追踪（阶段2）
        self._last_obj_z = None     # 物体Z高度追踪（阶段3）
        self._grasp_rewarded = False  # 抓取奖励是否已发放
        self.initial_object_height = None  # 物体初始高度（用于抬升目标计算）
        self._initial_tip_height = None  # 末端初始高度（用于高度保持惩罚）
        
        # 初始化tip位置和距离，确保第一步也有奖励信号
        tip_pos, _ = self._get_tip_position()
        self._last_tip_pos = np.array(tip_pos)
        self._initial_tip_height = tip_pos[2]  # 记录初始高度
        obj_pos, _ = self.base_env.physics_client.getBasePositionAndOrientation(self.object_id)
        rel_pos = np.array(obj_pos) - np.array(tip_pos)
        self._last_xy_dist = np.linalg.norm(rel_pos[:2])
        
        # Gym API (兼容 SB3 旧版 VecEnv): reset 返回 obs；如需新 API 可外部包 wrapper
        obs = self._get_observation()
        return obs
    
    def step(self, action):
        """执行一步强化学习动作。

        参数：
            action: 向量 [dx, dy, dz, gripper, approach_bias]（后两维可选/按配置使用）

        返回：
            obs, reward, done, info
        """
        # 动作裁剪到合法范围
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 保存当前动作用于奖励计算
        self._current_action = action.copy()
        

        # 1. 解析动作 (dx, dy, dz, dyaw, gripper)
        delta_pos = action[:3] * self.ee_action_scale
        delta_yaw = action[3] * self.ee_yaw_scale
        gripper_action = action[4]
        if getattr(self, 'invert_x_action', False):
            delta_pos[0] = -delta_pos[0]

        # [已移除] 渐进式下降限制
        # 之前的限制导致策略无法下降，现在完全依赖策略学习来控制

        # 可选控制噪声：模拟执行链路不确定性（仅训练时建议开启）
        if self.control_noise_std > 0:
            delta_pos = delta_pos + np.random.normal(0.0, self.control_noise_std, size=3)
            delta_yaw = delta_yaw + np.random.normal(0.0, self.control_noise_std) # 简单复用噪声std
        if self.gripper_noise_std > 0:
            gripper_action = gripper_action + float(np.random.normal(0.0, self.gripper_noise_std))

        # 夹爪逻辑：正值张开，负值闭合（与观测空间一致：+1=张开，-1=闭合）
        # 让策略自主学习何时闭合，不设置额外限制条件
        if gripper_action > 0.2:
            gripper_cmd = 1.0   # 张开
        elif gripper_action < -0.2:
            gripper_cmd = -1.0  # 闭合
        else:
            gripper_cmd = 0.0   # 中性不动，保持当前状态

        # 2. 获取指尖当前位姿
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        tip_pos, curr_orn = self._get_tip_position()
        # [方案A] 移除了 X 方向动态限制代码，该代码导致靠近时移动能力变弱

        # [Fix] 获取夹爪基座位置，用于 IK 计算
        gripper_link = self._get_gripper_base_link_index()
        try:
            gripper_base_pos, _ = self.base_env.models[0].links[gripper_link].get_pose()
        except Exception:
             st = pc.getLinkState(robot_id, gripper_link)
             gripper_base_pos = st[4]

        # 3. 计算目标位置和朝向
        # 先计算指尖的目标位置并进行空间限制
        target_tip_pos_unclipped = np.array(tip_pos) + delta_pos
        
        if self.ee_workspace_enabled:
            target_tip_pos = np.clip(target_tip_pos_unclipped, self.ee_pos_min, self.ee_pos_max)
            self._last_target_was_clipped = bool(np.any(np.abs(target_tip_pos - target_tip_pos_unclipped) > 1e-9))
        else:
            target_tip_pos = target_tip_pos_unclipped
            self._last_target_was_clipped = False

        # [简化方案] 观测点和 IK 控制点使用同一个参考系（gripper_base_link）
        # 这样可以避免复杂的坐标变换带来的误差
        # 
        # tip_pos 现在就是 gripper_base_link 的位置
        # target_tip_pos 就是 gripper_base_link 的目标位置
        # 直接用作 IK 目标
        target_pos = np.array(target_tip_pos)

        # 3.1 末端朝向：更新偏航角
        # 只有在非卡死状态下才更新偏航角
        if not self.is_stalled:
             self.current_yaw += delta_yaw
             # 限制在 [-pi, pi] 范围内 (可选，但欧拉角通常不限制也行，IK能解)
             self.current_yaw = (self.current_yaw + np.pi) % (2 * np.pi) - np.pi
        
        # [修复] 全程保持垂直向下姿态的 IK，确保机械臂以垂直姿态抓取物体
        # 使用 ee_down_orn 作为目标姿态约束，防止机械臂在下降过程中偏斜
        
        # 4. 逆解 IK：使用关节限位约束 + 姿态约束 提高解的稳定性
        lower_limits = [self.arm_joint_limits[i][0] for i in range(self.num_joints)]
        upper_limits = [self.arm_joint_limits[i][1] for i in range(self.num_joints)]
        joint_ranges = [upper_limits[i] - lower_limits[i] for i in range(self.num_joints)]
        rest_poses = list(self.desired_angles) if hasattr(self, 'desired_angles') and self.desired_angles is not None else [0.0] * self.num_joints

        gripper_link = self._get_gripper_base_link_index()
        joint_poses = pc.calculateInverseKinematics(
            robot_id,
            gripper_link,
            target_pos,
            targetOrientation=self.ee_down_orn,  # 添加垂直向下姿态约束
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=300,
            residualThreshold=1e-6
        )

        arm_target_angles = joint_poses[:6]
        self.desired_angles = np.array(arm_target_angles)
        
        # 5. Execute Action (Low-level Control)
        for _ in range(self.action_repeat):
            # Apply Arm Control
            # 提高位置增益以补偿较低的 action_repeat
            pc.setJointMotorControlArray(
                bodyIndex=robot_id,
                jointIndices=range(self.num_joints), # 0-5
                controlMode=pc.POSITION_CONTROL,
                targetPositions=self.desired_angles,
                forces=[500.0] * self.num_joints,  # 增加力矩确保快速响应
                positionGains=[0.8] * self.num_joints,  # 提高位置增益（从 0.4 到 0.8）
                velocityGains=[1.0] * self.num_joints
            )
            
            # Apply Gripper Control
            self._control_gripper(gripper_cmd)
            
            # Step Simulation
            for _ in range(self.sim_substeps):
                self.base_env.step_sim()
        
        self.current_step += 1
        
        # 每步更新卡住状态，供奖励与终止判断使用
        self.is_stalled = self._check_stalled()
        
        # 更新成功保持计数器
        if self.object_id is not None:
            obj_pos, _ = self.base_env.physics_client.getBasePositionAndOrientation(self.object_id)
            if obj_pos[2] > self.target_height:
                self.success_hold_counter += 1
            else:
                self.success_hold_counter = 0
        
        # 获取观测和奖励
        obs = self._get_observation()
        reward, reward_components = self._compute_reward()
        # 轻量级数值健壮性检查：防止 NaN/Inf 传播
        if np.isnan(obs).any() or np.isinf(obs).any():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        self.current_reward_components = reward_components
        
        # 累加奖励分量，便于回合级日志（只保留靠近奖励）
        if hasattr(self, 'episode_metrics'):
            self.episode_metrics['reward_approach'] = self.episode_metrics.get('reward_approach', 0.0) + reward_components.get('approach', 0.0)
            
        # 更新最大提升高度
        if self.object_id is not None:
             obj_pos, _ = self.base_env.physics_client.getBasePositionAndOrientation(self.object_id)
             self.episode_metrics['max_height'] = max(self.episode_metrics.get('max_height', 0.0), obj_pos[2])
        
        # Gym API 旧式 4-tuple（SB3 VecEnv 期望 done 标志）；若使用新 API，可在外部 wrapper 转换
        terminated = self._check_termination()
        truncated = self._check_truncation()
        done = terminated or truncated

        info = self._get_info()
        if truncated:
            info["TimeLimit.truncated"] = True

        # 更新上一帧相对位置，用于“超过”惩罚
        if self.object_id is not None:
            tip_pos, _ = self._get_tip_position()
            obj_pos, _ = self.base_env.physics_client.getBasePositionAndOrientation(self.object_id)
            rel_pos = np.array(obj_pos) - np.array(tip_pos)
            self._last_rel_x = rel_pos[0]
        else:
            self._last_rel_x = None

        return obs, reward, done, info
    
    def _get_observation(self):
        """构建归一化后的平坦观测向量 (9 维)。

        包含末端位姿、夹爪开合、物体相对末端的位姿差。
        注意：末端位置使用真正的指尖位置（gripper_base_link + 偏移）
        """
        pc = self.base_env.physics_client

        if self.object_id is None:
            return np.zeros(9, dtype=np.float32)

        # 1) 末端执行器位姿（世界系）——使用真正的指尖位置
        ee_pos = np.zeros(3, dtype=np.float32)
        ee_yaw = 0.0
        try:
            tip_pos, orn = self._get_tip_position()
            if tip_pos is not None:
                ee_pos = np.array(tip_pos, dtype=np.float32)
                # 末端主要关心平面对齐，取 yaw 并归一化到 [-1,1]
                ee_yaw = pc.getEulerFromQuaternion(orn)[2]
        except Exception:
            pass
        ee_pos_norm = np.clip(ee_pos / self.ee_pos_norm_scale, -1.0, 1.0)
        ee_yaw_norm = float(np.clip(ee_yaw / np.pi, -1.0, 1.0))

        # 2) 夹爪开合（-1=闭合，1=完全张开）
        gripper_pos = 0.0
        # 确保缓存已初始化
        if self.gripper_joints_cache is None:
            self._init_gripper_joints_cache()
        if self.gripper_joints_cache and 'wsg50_finger_left_joint' in self.gripper_joints_cache:
            try:
                gripper_pos = pc.getJointState(
                    self.base_env.robot_id, self.gripper_joints_cache['wsg50_finger_left_joint']
                )[0]
            except Exception:
                pass
        # WSG50 (50%缩放后): -0.005 (张开) ~ 0.025 (闭合)
        # 归一化到 [-1, 1]：-1=闭合(位置0.025), +1=张开(位置-0.005)
        gripper_open = -0.005
        gripper_closed = 0.025
        width_frac = (gripper_pos - gripper_closed) / (gripper_open - gripper_closed + 1e-6)
        width_frac = float(np.clip(width_frac, 0.0, 1.0))
        gripper_norm = width_frac * 2.0 - 1.0  # 0->-1(闭合), 1->+1(张开)

        # 3) 物体位姿（世界系）
        obj_pos = np.zeros(3, dtype=np.float32)
        obj_orn = np.array([0, 0, 0, 1], dtype=np.float32)
        if self.object_id is not None:
            try:
                pos, orn = pc.getBasePositionAndOrientation(self.object_id)
                obj_pos = np.array(pos, dtype=np.float32)
                obj_orn = np.array(orn, dtype=np.float32)
                obj_orn = obj_orn / max(np.linalg.norm(obj_orn), 1e-6)
            except Exception:
                pass

        # 在观测端注入姿态噪声：模拟真实传感器的位姿估计误差
        noisy_obj_pos = obj_pos + self.cube_obs_noise.get('pos', 0.0)

        # 4) 物体相对末端的位姿差
        #   使用相对位姿能让策略专注于“末端到物体”的误差，而非全局坐标，
        #   降低因为场景平移/随机重置带来的分布漂移，提高泛化性。
        rel_pos = noisy_obj_pos - ee_pos
        rel_pos_norm = np.clip(rel_pos / self.rel_pos_norm_scale, -1.0, 1.0)

        # 相对偏航角：忽略物体的翻滚/俯仰，仅关心抓取方向对齐
        yaw_obj = 0.0
        yaw_ee = ee_yaw
        try:
            yaw_obj = pc.getEulerFromQuaternion(obj_orn)[2]
        except Exception:
            pass
        # 对观测的偏航角注入均匀噪声（弧度），模拟感知方向误差
        noisy_yaw_obj = yaw_obj + self.cube_obs_noise.get('yaw', 0.0)

        rel_yaw = noisy_yaw_obj - yaw_ee
        rel_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap 到 [-pi, pi]
        rel_yaw_norm = float(np.clip(rel_yaw / np.pi, -1.0, 1.0))

        # 可选：将偏航观测维度置零（保持 shape=(9,) 不变）
        if self.zero_yaw_obs or self.zero_ee_yaw_obs:
            ee_yaw_norm = 0.0
        if self.zero_yaw_obs or self.zero_rel_yaw_obs:
            rel_yaw_norm = 0.0

        # 5) 机械臂关节角 (6维)
        joint_angles = self._get_joint_angles()
        joint_angles_norm = []
        for j, angle in enumerate(joint_angles):
            if j < self.num_joints:
                lower, upper = self.arm_joint_limits[j]
                # 归一化到 [-1, 1]
                norm_angle = 2.0 * (angle - lower) / (upper - lower) - 1.0
                joint_angles_norm.append(float(np.clip(norm_angle, -1.0, 1.0)))

        obs = np.concatenate([
            ee_pos_norm,           # [0:3] 末端位置（归一化）
            [ee_yaw_norm],         # [3] 末端偏航（归一化）
            [gripper_norm],        # [4] 夹爪开合幅度（归一化）
            rel_pos_norm,          # [5:8] 物体相对末端位置（归一化）
            [rel_yaw_norm],        # [8] 物体相对末端偏航角（归一化）
            joint_angles_norm      # [9:14] 关节角
        ]).astype(np.float32)

        # 防御式处理：去除 NaN/Inf
        if np.isnan(obs).any() or np.isinf(obs).any():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs if obs is not None else np.zeros(15, dtype=np.float32)
    
    def _get_joint_angles(self):
        """获取所有关节角度"""
        angles = []
        for j in range(self.num_joints):
            if j < len(self.base_env.models[0].joints):
                angle = self.base_env.models[0].joints[j].get_position()
                angles.append(angle)
        return angles
    
    def _get_joint_velocities(self):
        """获取所有关节速度"""
        vels = []
        for j in range(self.num_joints):
            if j < len(self.base_env.models[0].joints):
                joint_state = self.base_env.physics_client.getJointState(
                    self.base_env.robot_id, j
                )
                vels.append(joint_state[1])  # 索引1为速度
        return vels

    def _get_gripper_base_link_index(self):
        """获取夹爪基座 link 的索引（wsg50_base_link），未找到则退回末端 link。"""
        if self.gripper_base_link_idx is not None:
            return self.gripper_base_link_idx

        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        target_link_name = 'wsg50_base_link'
        candidate = None

        num_joints = pc.getNumJoints(robot_id)
        for i in range(num_joints):
            info = pc.getJointInfo(robot_id, i)
            link_name = info[12].decode('utf-8') if isinstance(info[12], (bytes, bytearray)) else str(info[12])
            if link_name == target_link_name:
                candidate = i
                break

        if candidate is None:
            # 若未找到，则退回模型最后一个 link，并输出一次警告
            candidate = len(self.base_env.models[0].links) - 1
            print("Warning: wsg50_base_link not found, fallback to last link.")

        self.gripper_base_link_idx = candidate
        return self.gripper_base_link_idx
    
    def _init_gripper_joints_cache(self):
        """初始化夹爪关节缓存 (WSG50 平行夹爪)"""
        if self.gripper_joints_cache is not None:
            return
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        self.gripper_joints_cache = {}
        # WSG50 夹爪的两个棱柱关节
        target_names = ['wsg50_finger_left_joint', 'wsg50_finger_right_joint']
        num_j = pc.getNumJoints(robot_id)
        for j in range(num_j):
            info = pc.getJointInfo(robot_id, j)
            name = info[1].decode('utf-8') if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            if name in target_names:
                self.gripper_joints_cache[name] = j
    
    def _init_gripper_link_indices(self):
        """初始化夹爪Link索引缓存 (WSG50 平行夹爪)"""
        if self.gripper_link_indices is not None:
            return
        
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        self.gripper_link_indices = []
        
        # WSG50 夹爪的指尖 Link 名称
        target_link_names = ['wsg50_finger_left', 'wsg50_finger_right']
        
        num_joints = pc.getNumJoints(robot_id)
        for i in range(num_joints):
            info = pc.getJointInfo(robot_id, i)
            link_name = info[12].decode('utf-8') if isinstance(info[12], (bytes, bytearray)) else str(info[12])
            if link_name in target_link_names:
                self.gripper_link_indices.append(i)
        
        # 提升手指摩擦系数，减少打滑
        try:
            for link_idx in self.gripper_link_indices:
                pc.changeDynamics(robot_id, link_idx, lateralFriction=2.5)
        except Exception:
            pass
    
    def _init_gripper_joint_limits(self):
        """为夹爪关节启用硬限位 (WSG50 平行夹爪)
        
        WSG50 夹爪（50%缩放后）：
        - 关节类型：棱柱关节（PRISMATIC）
        - 关节范围：[-0.005, 0.025] 米
        - 张开：两个关节都设为 -0.005
        - 闭合：两个关节都设为 0.025
        """
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        
        if self.gripper_joints_cache is None:
            self._init_gripper_joints_cache()
        
        # WSG50 夹爪关节限位（50%缩放后）
        wsg50_limits = (-0.005, 0.025)  # 米
        
        for joint_name in ['wsg50_finger_left_joint', 'wsg50_finger_right_joint']:
            if joint_name in self.gripper_joints_cache:
                joint_idx = self.gripper_joints_cache[joint_name]
                lower, upper = wsg50_limits
                # 启用关节限位强制执行
                pc.changeDynamics(
                    robot_id, 
                    joint_idx,
                    jointLowerLimit=lower,
                    jointUpperLimit=upper,
                    jointLimitForce=100.0  # 限位时的反作用力
                )
    
    def _get_tip_position(self):
        """获取末端参考点位置。

        直接返回 gripper_base_link 的位置和姿态。
        这样观测点和 IK 控制点完全一致，避免坐标变换误差。

        Returns:
            tip_pos: gripper_base_link 的世界坐标位置
            gripper_orn: gripper_base_link 的姿态四元数
        """
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        
        gripper_link = self._get_gripper_base_link_index()
        try:
            gripper_pos, gripper_orn = self.base_env.models[0].links[gripper_link].get_pose()
        except Exception:
            st = pc.getLinkState(robot_id, gripper_link)
            gripper_pos = st[4]
            gripper_orn = st[5]
        
        return np.array(gripper_pos, dtype=np.float32), gripper_orn
        
        # --- 以下为旧代码（指腹中心计算），已禁用 ---

        # 1) 先获取基座姿态（用于 IK/观测对齐）
        gripper_link = self._get_gripper_base_link_index()
        try:
            gripper_pos, gripper_orn = self.base_env.models[0].links[gripper_link].get_pose()
        except Exception:
            # 最差情况：直接用 PyBullet link state
            st = pc.getLinkState(robot_id, gripper_link)
            gripper_pos = st[4]
            gripper_orn = st[5]

        # 2) 尝试用两指“指腹接触面中心”的中点作为 tip
        #    这里按你的要求：以 wsg50_gripper_left / wsg50_gripper_right（两个指关节驱动的滑块 link）为基准，
        #    通过 URDF 固定关节变换得到 finger link 的姿态，再用 finger 的碰撞盒确定“指腹平面中心”。

        # finger 碰撞盒参数（rm65_wsg50.urdf）：
        # <collision origin xyz="0 0 0.021"/>  <box size="0.01 0.01 0.075"/>
        finger_box_center_local = np.array([0.0, 0.0, 0.021], dtype=np.float32)
        finger_box_half_extents = np.array([0.005, 0.005, 0.0375], dtype=np.float32)

        # gripper_(left/right) -> finger_(left/right) 的固定变换（rm65_wsg50.urdf）
        # wsg50_gripper_finger_left/right: origin xyz="-0.0035 0 0.0425" rpy about z = -pi/2
        gripper_to_finger_pos = (-0.0035, 0.0, 0.0425)
        gripper_to_finger_orn = pc.getQuaternionFromEuler((0.0, 0.0, -np.pi / 2.0))

        def _find_link_index_by_name(link_name: str):
            try:
                num = pc.getNumJoints(robot_id)
            except Exception:
                return None
            for j in range(num):
                info = pc.getJointInfo(robot_id, j)
                name = info[12].decode('utf-8') if isinstance(info[12], (bytes, bytearray)) else str(info[12])
                if name == link_name:
                    return j
            return None

        left_gripper_idx = _find_link_index_by_name('wsg50_gripper_left')
        right_gripper_idx = _find_link_index_by_name('wsg50_gripper_right')

        if left_gripper_idx is not None and right_gripper_idx is not None:
            try:
                stL = pc.getLinkState(robot_id, left_gripper_idx, computeForwardKinematics=True)
                stR = pc.getLinkState(robot_id, right_gripper_idx, computeForwardKinematics=True)
                pL_g = stL[4]
                qL_g = stL[5]
                pR_g = stR[4]
                qR_g = stR[5]

                # 推到 finger link 姿态
                pL_f, qL_f = pc.multiplyTransforms(pL_g, qL_g, gripper_to_finger_pos, gripper_to_finger_orn)
                pR_f, qR_f = pc.multiplyTransforms(pR_g, qR_g, gripper_to_finger_pos, gripper_to_finger_orn)
                pL_f = np.array(pL_f, dtype=np.float32)
                pR_f = np.array(pR_f, dtype=np.float32)

                vLR = (pR_f - pL_f).astype(np.float32)

                def _pad_center_world(p_f, q_f, v_to_other_world):
                    rot = np.array(pc.getMatrixFromQuaternion(q_f), dtype=np.float32).reshape(3, 3)
                    local_v = rot.T @ v_to_other_world
                    axis_idx = 0 if abs(float(local_v[0])) >= abs(float(local_v[1])) else 1
                    sign = 1.0 if float(local_v[axis_idx]) >= 0.0 else -1.0

                    local_offset = finger_box_center_local.copy()
                    local_offset[axis_idx] += sign * float(finger_box_half_extents[axis_idx])
                    return p_f + (rot @ local_offset)

                padL = _pad_center_world(pL_f, qL_f, vLR)
                padR = _pad_center_world(pR_f, qR_f, -vLR)
                tip_pos = (padL + padR) * 0.5
                return tip_pos, gripper_orn
            except Exception:
                pass

        # 3) 回退：直接使用夹爪基座位置（更安全；避免错误的固定偏移污染观测/奖励）
        print("Warning: fingertip pad center unavailable, fallback to wsg50_base_link position.")
        if gripper_pos is not None:
            tip_pos = np.array(gripper_pos, dtype=np.float32)
        else:
            tip_pos = np.zeros(3, dtype=np.float32)
        return tip_pos, gripper_orn
    
    def _clip_joint_angles(self, angles):
        """限制关节角在有效范围内"""
        # 使用机器人 URDF 定义的关节角限制进行裁剪
        clipped = []
        for j, angle in enumerate(angles):
            if j < self.num_joints:
                lower, upper = self.arm_joint_limits[j]
                angle = float(np.clip(angle, lower, upper))
            clipped.append(angle)
        return clipped
    
    def _control_gripper(self, cmd):
        """控制夹爪，使用位置控制 (WSG50 平行夹爪)

        WSG50 夹爪说明：
        - 两个棱柱关节：wsg50_finger_left_joint, wsg50_finger_right_joint
        - 由于右侧关节在 URDF 中有 rpy=π 旋转，所以对称运动时两个关节使用相同符号的目标位置
        - 张开：两个关节都设为 -0.005（最小值）
        - 闭合：两个关节都设为 0.025（最大值）

        参数：
            cmd: [-1, 1]，-1 闭合，1 张开，0 保持当前状态
        """
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id

        # 确保缓存已初始化
        if self.gripper_joints_cache is None:
            self._init_gripper_joints_cache()
        
        gripper_joints = self.gripper_joints_cache
        if not gripper_joints or 'wsg50_finger_left_joint' not in gripper_joints:
            return

        if cmd == 0:
            return  # 保持当前状态
        
        # WSG50 夹爪关节限位（50%缩放后）
        lower_limit = -0.005
        upper_limit = 0.025
        
        # WSG50 对称控制：两个关节使用相同的目标位置
        # 张开：-0.005（最小值），闭合：0.025（最大值）
        if cmd > 0:
            target = lower_limit  # 张开
            grip_force = 15.0
        else:
            target = upper_limit  # 闭合
            grip_force = 25.0  # 闭合时用更大力确保夹紧
        
        for joint_name in ['wsg50_finger_left_joint', 'wsg50_finger_right_joint']:
            if joint_name not in gripper_joints:
                continue
                
            joint_idx = gripper_joints[joint_name]
            
            # 获取当前位置
            current_pos = pc.getJointState(robot_id, joint_idx)[0]
            
            # 强制将当前位置限制在范围内
            if current_pos < lower_limit:
                final_target = lower_limit + 0.001
            elif current_pos > upper_limit:
                final_target = upper_limit - 0.001
            else:
                # 渐进式移动：每步最多移动 0.002 m
                step_size = 0.002
                if target > current_pos:
                    final_target = min(target, current_pos + step_size)
                else:
                    final_target = max(target, current_pos - step_size)
                
                # 确保目标在限位内
                final_target = max(lower_limit, min(upper_limit, final_target))
            
            pc.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=joint_idx,
                controlMode=pc.POSITION_CONTROL,
                targetPosition=final_target,
                force=grip_force,
                maxVelocity=0.5,
                positionGain=0.8,
                velocityGain=1.0
            )

    def open_gripper(self, settle_steps: int = 0, instant: bool = False):
        """打开夹爪（joint_b1 朝下限张开）。

        参数:
            settle_steps: 可选，在调用后跑若干仿真步以稳定动作。
            instant: 若为 True，使用 resetJointState 立即设置角度（用于 reset）
        """
        if instant:
            # 直接设置关节角度，无需等待物理模拟
            self._reset_gripper_to_open()
        else:
            self._control_gripper(1.0)
        for _ in range(int(max(0, settle_steps))):
            self.base_env.step_sim()
    
    def _reset_gripper_to_open(self):
        """使用 resetJointState 立即将夹爪设置为完全张开状态 (WSG50)"""
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        
        # 确保缓存已初始化
        if self.gripper_joints_cache is None:
            self._init_gripper_joints_cache()
        
        if not self.gripper_joints_cache:
            return
            
        # WSG50 完全张开时的关节位置（两个关节相同）
        open_position = -0.005  # 张开位置（最小值）
        
        for name in ['wsg50_finger_left_joint', 'wsg50_finger_right_joint']:
            if name in self.gripper_joints_cache:
                joint_idx = self.gripper_joints_cache[name]
                pc.resetJointState(robot_id, joint_idx, open_position, targetVelocity=0)
        
        # 重置后立即设置电机控制，防止关节漂移
        for name in ['wsg50_finger_left_joint', 'wsg50_finger_right_joint']:
            if name in self.gripper_joints_cache:
                pc.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=self.gripper_joints_cache[name],
                    controlMode=pc.POSITION_CONTROL,
                    targetPosition=open_position,
                    force=10.0,
                    maxVelocity=0.5
                )

    def close_gripper(self, settle_steps: int = 0):
        """闭合夹爪（joint_b1 朝上限闭合）。

        settle_steps: 可选，在调用后跑若干仿真步以稳定动作。
        """
        self._control_gripper(-1.0)
        for _ in range(int(max(0, settle_steps))):
            self.base_env.step_sim()
    
    def _check_collision(self):
        """检测碰撞状态：
        1. 机械臂与桌子的碰撞
        2. 机械臂末端是否过低（穿模）
        """
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        table_id = self.base_env.table_id
        
        # 1. 检测与桌子的接触
        if table_id is not None:
            # getContactPoints 返回列表，非空表示有接触
            contacts = pc.getContactPoints(bodyA=robot_id, bodyB=table_id)
            if len(contacts) > 0:
                return True
        
        return False

    def _check_finger_contact(self):
        """检测夹爪指腹是否接触到物体（有效抓取接触）
        
        基于 URDF 定义，检测以下 4 个 Link 与物体的接触：
        - gripper_link1, gripper_link3 (左指)
        - gripper_link2, gripper_link4 (右指)
        """
        if self.object_id is None:
            return False
            
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        
        # 确保缓存已初始化
        if self.gripper_link_indices is None:
            self._init_gripper_link_indices()

        # 2. 获取接触点
        # getContactPoints 返回列表，每个元素是一个 tuple
        contacts = pc.getContactPoints(bodyA=robot_id, bodyB=self.object_id)
        
        if len(contacts) == 0:
            return False
            
        # 3. 检查接触点是否属于夹爪的 4 个 Link 之一
        for pt in contacts:
            # pt[3] 是 linkIndexA (机器人的 Link 索引)
            link_index = pt[3] 
            if link_index in self.gripper_link_indices:
                return True
                
        return False

    def _check_dual_finger_contact(self):
        """检测是否双指都接触到物体（有效夹持）
        
        Returns:
            tuple: (left_contact, right_contact, both_contact)
                - left_contact: 左指是否接触物体
                - right_contact: 右指是否接触物体  
                - both_contact: 双指是否都接触物体
        """
        if self.object_id is None:
            return False, False, False
            
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        
        # 确保缓存已初始化
        if self.gripper_link_indices is None:
            self._init_gripper_link_indices()
        
        # 获取左右手指的 link 索引
        left_finger_links = set()
        right_finger_links = set()
        
        num_joints = pc.getNumJoints(robot_id)
        for i in range(num_joints):
            info = pc.getJointInfo(robot_id, i)
            link_name = info[12].decode('utf-8') if isinstance(info[12], (bytes, bytearray)) else str(info[12])
            if 'left' in link_name.lower() and ('finger' in link_name.lower() or 'gripper' in link_name.lower()):
                left_finger_links.add(i)
            elif 'right' in link_name.lower() and ('finger' in link_name.lower() or 'gripper' in link_name.lower()):
                right_finger_links.add(i)
        
        # 获取接触点
        contacts = pc.getContactPoints(bodyA=robot_id, bodyB=self.object_id)
        
        left_contact = False
        right_contact = False
        
        for pt in contacts:
            link_index = pt[3]  # linkIndexA
            if link_index in left_finger_links:
                left_contact = True
            if link_index in right_finger_links:
                right_contact = True
        
        both_contact = left_contact and right_contact
        return left_contact, right_contact, both_contact

    def _check_safety_violation(self):
        """检测严重安全违规（如撞桌、穿模），用于提前终止回合"""
        try:
            tip_pos, _ = self._get_tip_position()
            # 如果指尖低于桌面（z < 0.01），视为穿模/撞桌
            # 提高阈值：物体在 z≈0.017 附近，0.01 是桌面高度的安全边界
            if tip_pos[2] < 0.01:
                print(f"[DEBUG] 安全违规: 末端高度过低 z={tip_pos[2]:.4f} < 0.01", flush=True)
                return True
        except Exception:
            pass
        return False

    def _check_self_collision(self):
        """检测夹爪与机械臂自身的碰撞
        
        检测夹爪部分（gripper link）与机械臂本体（arm link）之间的接触。
        相邻 link 之间的接触（如手腕与夹爪底座）不计入自碰撞。
        
        Returns:
            bool: True 表示检测到自碰撞
        """
        pc = self.base_env.physics_client
        robot_id = self.base_env.robot_id
        
        # 确保夹爪 link 索引已初始化
        if self.gripper_link_indices is None:
            self._init_gripper_link_indices()
        
        # 获取机器人所有 link 数量
        num_joints = pc.getNumJoints(robot_id)
        
        # 定义夹爪相关 link（不应与机械臂本体碰撞）
        # 包括：夹爪手指、夹爪底座等
        gripper_link_names = [
            'wsg50_finger_left', 'wsg50_finger_right',
            'wsg50_gripper_left', 'wsg50_gripper_right',
            'wsg50_base_link', 'gripper_link1', 'gripper_link2',
            'gripper_link3', 'gripper_link4'
        ]
        gripper_links = set()
        
        # 机械臂本体 link（link 6 之前，即 base ~ link5）
        # 排除与夹爪直接相邻的 link（如 link6/手腕），避免相邻关节误报
        arm_body_links = set(range(6))  # link 0-5 为机械臂本体
        
        for i in range(num_joints):
            info = pc.getJointInfo(robot_id, i)
            link_name = info[12].decode('utf-8') if isinstance(info[12], (bytes, bytearray)) else str(info[12])
            if link_name in gripper_link_names:
                gripper_links.add(i)
        
        # 添加已缓存的夹爪手指 link
        if self.gripper_link_indices:
            gripper_links.update(self.gripper_link_indices)
        
        # 获取自碰撞接触点
        contacts = pc.getContactPoints(bodyA=robot_id, bodyB=robot_id)
        
        for pt in contacts:
            link_a = pt[3]  # linkIndexA
            link_b = pt[4]  # linkIndexB
            
            # 检测：夹爪 link 与机械臂本体 link 之间的碰撞
            if (link_a in gripper_links and link_b in arm_body_links) or \
               (link_b in gripper_links and link_a in arm_body_links):
                
                # 获取 Link 名称用于调试
                info_a = pc.getJointInfo(robot_id, link_a)
                name_a = info_a[12].decode('utf-8') if isinstance(info_a[12], (bytes, bytearray)) else str(info_a[12])
                
                info_b = pc.getJointInfo(robot_id, link_b)
                name_b = info_b[12].decode('utf-8') if isinstance(info_b[12], (bytes, bytearray)) else str(info_b[12])
                
                print(f"[DEBUG] 自碰撞检测: {name_a} (id={link_a}) <--> {name_b} (id={link_b})", flush=True)
                return True
        
        return False


    def _check_stalled(self):
        """检测机器人是否卡住（长时间未移动）"""
        # 获取当前末端位置（完整 3D 坐标）
        try:
            gripper_link = self._get_gripper_base_link_index()
            curr_pos, _ = self.base_env.models[0].links[gripper_link].get_pose()
            curr_pos = np.array(curr_pos)
        except Exception:
            return False

        # 若本步目标被工作空间裁剪，说明正在顶到边界/上限。
        # 此时“末端几乎不动”是可预期的饱和现象，不应累计 stalled。
        if getattr(self, '_last_target_was_clipped', False):
            if not hasattr(self, '_stall_counter'):
                self._stall_counter = 0
            else:
                self._stall_counter = 0
            self._last_ee_pos_stall = curr_pos.copy()
            return False

        # 初始化上一帧位置
        if not hasattr(self, '_last_ee_pos_stall'):
            self._last_ee_pos_stall = curr_pos.copy()
            self._stall_counter = 0
            return False

        # 计算位置变化（3D 距离，不只是高度）
        delta = np.linalg.norm(curr_pos - self._last_ee_pos_stall)
        self._last_ee_pos_stall = curr_pos.copy()

        # 如果变化小于 2mm (0.002m)，计数器+1
        if delta < 0.002:
            self._stall_counter += 1
        else:
            self._stall_counter = 0

        # 如果连续 50 步（约 4 秒）没有明显移动，判定为卡住
        # 排除以下情况：
        # 1. 已经接触物体（可能在夹紧）
        # 2. 物体已被抬起（成功抓取中）
        is_contact = self._check_finger_contact()
        obj_lifted = False
        if self.object_id is not None:
            try:
                obj_pos, _ = self.base_env.physics_client.getBasePositionAndOrientation(self.object_id)
                obj_lifted = obj_pos[2] > (self.initial_object_height or 0) + 0.02
            except Exception:
                pass
        
        # 如果在夹取或已抬起物体，不判定为卡住
        if is_contact or obj_lifted:
            self._stall_counter = 0  # 重置计数器
            return False
        
        # 训练初期策略可能输出小动作，放宽阈值
        # 连续 150 步（约 12 秒）没有明显移动才判定为卡住
        if self._stall_counter > 150:
            return True
            
        return False

    def _check_termination(self):
        """检查终止条件 (Terminated)

        简化版本 - 只训练第一阶段
        终止情形：
        1) 末端到达物体上方15cm（成功）
        2) 机械臂撞桌/安全违规（失败）
        3) 夹爪与机械臂自身碰撞（失败）
        
        设置 self._termination_reason 用于奖励计算
        """
        self._termination_reason = None
        
        # 1) 成功终止：末端保持在目标位置5步
        if self.success_hold_counter > 5:
            self._termination_reason = 'success'
            return True

        # 2) 安全违规终止：机械臂撞桌或穿模
        # 2) 安全违规终止：机械臂撞桌或穿模
        if self._check_safety_violation():
            self._termination_reason = 'safety_violation'
            print(f"[DEBUG] 回合终止: 安全违规 (Safety Violation)", flush=True)
            return True

        # 3) 自碰撞终止：夹爪与机械臂自身碰撞
        if self._check_self_collision():
            self._termination_reason = 'self_collision'
            print(f"[DEBUG] 回合终止: 自碰撞 (Self Collision)", flush=True)
            return True

        # 4) 桌面碰撞终止：机械臂与桌子发生接触
        if self._check_collision():
            self._termination_reason = 'table_collision'
            print(f"[DEBUG] 回合终止: 桌面碰撞 (Table Collision)", flush=True)
            return True

        return False

    def _check_truncation(self):
        """检查是否满足截断条件 (Truncated)
        
        Truncated 表示回合因时间限制等外部因素强制结束。
        """
        # 截止检测 (Max Steps) —— 只用于时间上限，不因接触提前截断
        if self.current_step >= self.max_episode_steps:
            return True
            
        return False

    def _update_curriculum_stage(self, ee_pos, obj_pos, is_contact: bool):
        """根据当前状态自动切换课程阶段。

        阶段定义：
        0: 末端到达物体上方 20cm（XY 误差 < 5cm 且 Z 高度差 >= 20cm）
        1: 与物体建立接触（检测指腹接触）
        2: 将物体抬升 20cm 并悬停 3s
        """
        if not self.curriculum_enabled or obj_pos is None:
            return

        # 确保有初始高度基准
        if self.initial_object_height is None:
            self.initial_object_height = obj_pos[2]

        ee_xy = np.array(ee_pos[:2])
        obj_xy = np.array(obj_pos[:2])
        xy_dist = float(np.linalg.norm(ee_xy - obj_xy))

        # 阶段 0 -> 1：对准并在上方 20cm
        if self.curriculum_stage == 0:
            above = ee_pos[2] >= obj_pos[2] + self.curriculum_approach_height
            aligned_xy = xy_dist <= self.curriculum_xy_thresh
            if above and aligned_xy:
                self.curriculum_stage = 1
                self.training_phase = 1
        # 阶段 1 -> 2：建立接触
        if self.curriculum_stage == 1:
            if is_contact:
                self.curriculum_stage = 2
                self.training_phase = 2
                self.curriculum_hover_counter = 0

        # 阶段 2 完成：抬升并悬停
        if self.curriculum_stage == 2:
            lift_target = self.initial_object_height + self.curriculum_lift_height
            if obj_pos[2] >= lift_target:
                self.curriculum_hover_counter += 1
            else:
                self.curriculum_hover_counter = 0

            if self.curriculum_hover_counter >= self.curriculum_hover_steps:
                self.curriculum_complete = True
    
    def _get_info(self):
        """返回调试信息"""
        info = {
            'step': self.current_step,
            'gripper_force': self.base_env.get_gripper_force() if self.object_id else 0.0,
        }

        # 课程阶段调试信息
        info['curriculum_stage'] = getattr(self, 'curriculum_stage', 0)
        info['curriculum_complete'] = bool(getattr(self, 'curriculum_complete', False))
        
        if self.object_id is not None:
            obj_pos, _ = self.base_env.physics_client.getBasePositionAndOrientation(
                self.object_id
            )
            info['object_height'] = obj_pos[2]
            info['object_pos'] = obj_pos
            # 成功标志：用于上层日志/评估，不依赖接触终止
            info['success'] = bool(self.success_hold_counter > 5)
            # 逐步奖励分量：便于直接绘图分析（轻量化字典，无需额外依赖）
            if hasattr(self, 'current_reward_components'):
                info.update({
                    'reward_approach': self.current_reward_components.get('approach', 0.0)
                })
            
        # 回合级指标：便于直接送入 Tensorboard/WandB
        if hasattr(self, 'episode_metrics'):
            info.update({
                'metric_reward_approach': self.episode_metrics.get('reward_approach', 0.0),
                'metric_max_height': self.episode_metrics.get('max_height', 0.0),
                'metric_success': float(self.success_hold_counter > 5),
                'metric_episode_length': self.current_step
            })
        
        # 添加终止原因（用于调试）
        if hasattr(self, '_termination_reason') and self._termination_reason:
            info['termination_reason'] = self._termination_reason
        
        return info
    
    def render(self, mode='human'):
        """渲染环境（PyBullet GUI 已实时显示）"""
        pass
    
    def close(self):
        """关闭环境"""
        self.base_env.close()
    
    def seed(self, seed=None):
        """设置随机种子"""
        self.base_env.seed(seed, evaluate=self.evaluate)
        return [seed]
