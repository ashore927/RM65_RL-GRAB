import pybullet as p


class Robot(object):
    def __init__(self, physics_client):
        """Model 包装，用于加载 URDF/SDF 并提供简单接口。

        参数:
            physics_client: BulletClient 或 pybullet 模块（用于调用 pybullet API）
        """
        self._physics_client = physics_client

    def load_model(self, path, start_pos=None, start_orn=None, scaling=1., static=False):
        """加载 URDF 或 SDF 模型并构造关节/链接包装对象。

        注意不要使用可变默认参数（list），因此在函数内创建默认值。
        返回加载得到的 model_id。
        """
        if start_pos is None:
            start_pos = [0, 0, 0]
        if start_orn is None:
            start_orn = [0, 0, 0, 1]

        if path.endswith('.sdf'):
            model_id = self._physics_client.loadSDF(path, globalScaling=scaling)[0]
            # SDF 可能需要显式设置基座位姿
            self._physics_client.resetBasePositionAndOrientation(model_id, start_pos, start_orn)
        else:
            model_id = self._physics_client.loadURDF(
                path, start_pos, start_orn,
                globalScaling=scaling, useFixedBase=static,
                flags=p.URDF_USE_SELF_COLLISION)
        self.model_id = model_id
        # self._get_limits(self.model_id)
        joints, links = {}, {}
        for i in range(self._physics_client.getNumJoints(self.model_id)):
            joint_info = self._physics_client.getJointInfo(self.model_id, i)
            # joint_info 字段参考 pybullet 文档，索引 8/9 为 lower/upper，10 为 max force/effort
            joint_limits = {'lower': joint_info[8], 'upper': joint_info[9],
                            'force': joint_info[10]}
            joints[i] = _Joint(self._physics_client, self.model_id, i, joint_limits)
            links[i] = _Link(self._physics_client, self.model_id, i)
        self.joints, self.links = joints, links

        return model_id

    def get_joints(self):
        """返回一个包含每个 jointInfo 的字典，而不修改 self.joints 的对象类型。"""
        infos = {}
        for i in range(self._physics_client.getNumJoints(self.model_id)):
            infos[i] = self._physics_client.getJointInfo(self.model_id, i)
        return infos

    def get_pose(self):
        """返回模型基座（base/link0）的位姿。

        使用 getBasePositionAndOrientation 更稳健（避免硬编码 link 索引）。
        """
        return self._physics_client.getBasePositionAndOrientation(self.model_id)
    
    def getBase(self):
        return self._physics_client.getBasePositionAndOrientation(self.model_id)

class _Link(object):
    def __init__(self, physics_client, model_id, link_id):
        self._physics_client = physics_client
        self.model_id = model_id
        self.lid = link_id

    def get_pose(self):
        """返回该 link 的位姿。

        优先返回 URDF frame（getLinkState 返回索引 4/5），如不可用则退回到索引 0/1。
        使用传入的 physics_client 确保与 BulletClient 一致。
        """
        link_state = self._physics_client.getLinkState(self.model_id, self.lid)
        # 当使用 BulletClient 时，返回的元组长度可能 >= 6
        if len(link_state) >= 6 and link_state[4] is not None:
            # worldLinkFramePosition, worldLinkFrameOrientation
            return link_state[4], link_state[5]
        # 回退到 linkWorldPosition, linkWorldOrientation
        return link_state[0], link_state[1]


class _Joint(object):
    def __init__(self, physics_client, model_id, joint_id, limits):
        self._physics_client = physics_client
        self.model_id = model_id
        self.jid = joint_id
        self.limits = limits

    def get_position(self):
        joint_state = self._physics_client.getJointState(
            self.model_id, self.jid)
        return joint_state[0]

    def set_position(self, position, max_force=100.):
        self._physics_client.setJointMotorControl2(
            self.model_id, self.jid,
            controlMode=p.POSITION_CONTROL,
            targetPosition=position,
            force=max_force)

    def disable_motor(self):
        self._physics_client.setJointMotorControl2(
            self.model_id, self.jid, controlMode=p.VELOCITY_CONTROL, force=0.)