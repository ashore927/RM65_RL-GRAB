import numpy as np
from robot_env import RobotGraspEnv

def test_env_config():
    config = {
        "simulation": {
            "visualize": False,
            "real_time": False
        },
        "scene": {}
    }
    env = RobotGraspEnv(config)
    print("动作空间:", env.action_space)
    print("观察空间:", env.observation_space)

    # 检查动作空间维度
    assert env.action_space.shape[0] == 4, f"动作空间维度应为4，实际为{env.action_space.shape[0]}"
    # 检查动作空间上下限
    assert np.all(env.action_space.low < env.action_space.high), "动作空间上下限配置异常"

    # 检查观测空间维度
    obs = env.reset()
    print("初始观测:", obs)
    assert env.observation_space.shape[0] == obs.shape[0], f"观测空间维度与实际观测不符: {env.observation_space.shape[0]} vs {obs.shape[0]}"
    assert not np.isnan(obs).any(), "观测中存在NaN"
    assert not np.isinf(obs).any(), "观测中存在Inf"

    # 检查step输出
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("step输出观测:", obs)
    print("step输出奖励:", reward)
    print("step输出done:", done)
    print("step输出info:", info)
    assert obs.shape[0] == env.observation_space.shape[0], "step输出观测维度异常"
    assert not np.isnan(obs).any(), "step输出观测存在NaN"
    assert not np.isinf(obs).any(), "step输出观测存在Inf"
    assert isinstance(reward, float), "奖励不是float类型"
    assert not np.isnan(reward), "奖励为NaN"
    assert not np.isinf(reward), "奖励为Inf"

    print("所有配置检查通过！")

if __name__ == "__main__":
    test_env_config()