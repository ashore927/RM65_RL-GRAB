"""简化训练脚本 - 不使用VecNormalize，直接诊断问题"""
import os
from robot_env import RobotGraspEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# 训练配置
train_config = {
    'scene': {'scene_type': 'OnTable'},
    'simulation': {
        'visualize': False,
        'real_time': False,
        'auto_init_test_env': True,
    }
}

class DebugCallback(BaseCallback):
    """打印训练过程中的关键信息"""
    def __init__(self, print_freq=5000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.current_episode_reward = 0
    
    def _on_step(self) -> bool:
        # 累积回合奖励
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward
        
        # 回合结束时记录
        done = self.locals.get('dones', [False])[0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        
        # 定期打印
        if self.n_calls % self.print_freq == 0:
            if len(self.episode_rewards) > 0:
                recent = self.episode_rewards[-10:]
                print(f"Step {self.n_calls}: Last 10 episodes avg reward = {np.mean(recent):.2f}")
        return True

def make_env():
    def _init():
        return RobotGraspEnv(train_config, evaluate=False, test=False, validate=False)
    return _init

def main():
    print("创建环境（不使用VecNormalize）...")
    env = DummyVecEnv([make_env()])
    
    os.makedirs("./simple_checkpoints", exist_ok=True)
    
    # 使用更保守的SAC配置
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cuda",
        learning_rate=1e-4,        # 降低学习率
        buffer_size=100000,
        batch_size=256,
        ent_coef='auto',           # 自动调整熵系数
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,     # 更多预热
        tau=0.005,                 # 更保守的软更新
        gamma=0.99,
        policy_kwargs=dict(
            net_arch=[256, 256]
        )
    )
    
    debug_cb = DebugCallback(print_freq=5000)
    
    print("开始简化训练...")
    model.learn(
        total_timesteps=100_000,
        callback=[debug_cb],
        progress_bar=True
    )
    
    model.save("simple_model")
    print("训练完成!")

if __name__ == "__main__":
    main()
