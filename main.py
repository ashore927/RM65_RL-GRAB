from encodings.punycode import T
import os
import time
from robot_env import RobotGraspEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed


class SyncNormalizationCallback(BaseCallback):
    """同步训练环境和评估环境的归一化统计量"""
    def __init__(self, eval_env, sync_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.sync_freq = sync_freq
    
    def _on_step(self) -> bool:
        if self.n_calls % self.sync_freq == 0:
            # 同步归一化统计量到评估环境
            sync_envs_normalization(self.training_env, self.eval_env)
        return True

# 训练配置：关闭 GUI 和 实时步进，进行全速后台训练
train_config = {
    'scene': {'scene_type': 'OnTable'},
    'simulation': {
        'visualize': False,  # 关闭 GUI 窗口（关键加速）
        'real_time': False,  # 关闭实时同步（关键加速，不再 sleep）
        'auto_init_test_env': True,
    }
}

# 评估配置：开启 GUI 便于观察
eval_config = {
    'scene': {'scene_type': 'OnTable'},
    'simulation': {
        'visualize': False,  # 评估时也关闭GUI加速
        'real_time': False,
        'auto_init_test_env': True,
    }
}

def make_env(evaluate=False):
    """创建环境工厂函数"""
    def _init():
        cfg = eval_config if evaluate else train_config
        return RobotGraspEnv(cfg, evaluate=evaluate, test=False, validate=False)
    return _init

def main():

    # 多进程并行训练配置
    # Windows 下建议 n_envs <= CPU核心数，且开启多进程启动较慢，请耐心等待
    n_envs = 14  # 使用4个并行环境提高采样效率
    print(f"正在启动 {n_envs} 个并行环境...")
    
    # 使用 SubprocVecEnv 进行多进程并行计算
    env = SubprocVecEnv([make_env(evaluate=False) for _ in range(n_envs)])
    
    # 添加观测归一化（不归一化奖励，让智能体看到真实奖励差异）
    # 注意：禁用 norm_obs，因为 robot_env 已经做了归一化
    env = VecNormalize(
        env, 
        norm_obs=False,     # 禁用！robot_env 已经归一化到 [-1, 1]
        norm_reward=False,  # 不归一化奖励！让智能体看到真实差异
        clip_obs=10.0,      # 裁剪观测范围
        clip_reward=10.0,   # 裁剪奖励范围
        gamma=0.99          # 用于reward归一化的折扣因子
    )

    # 创建评估环境 - 注意：需要在训练时同步统计量
    eval_env = SubprocVecEnv([make_env(evaluate=True)])
    # 评估环境使用相同的归一化参数，稍后会同步
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,     # 与训练环境一致
        norm_reward=False,  # 评估时不归一化奖励
        clip_obs=10.0,
        training=False,     # 评估模式，不更新统计量
        clip_reward=10.0
    )

    # 回调：定期保存 checkpoint 与最佳模型
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./best_model", exist_ok=True)
    os.makedirs("./runs", exist_ok=True)
    
    checkpoint_cb = CheckpointCallback(
        save_freq=10000 // n_envs,  # 按总步数计算
        save_path="./checkpoints", 
        name_prefix="sac_grasp"
    )
    
    # 同步归一化统计量回调
    sync_cb = SyncNormalizationCallback(eval_env, sync_freq=1000)
    
    # 评估回调：定期评估并保存最佳模型
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=50000 // n_envs,  # 每 50000 步评估一次
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )

    # SAC 算法配置 - 使用较低的固定熵系数
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./runs",
        device="cuda",
        learning_rate=3e-4,
        buffer_size=500000,
        batch_size=256,
        ent_coef=0.05,        # 较低的固定熵系数，减少随机性
        train_freq=1,
        gradient_steps=1,
        learning_starts=5000,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(
            net_arch=[256, 256]
        )
    )
    
    print("开始训练 SAC 模型...")
    model.learn(
        total_timesteps=100_000, 
        callback=[checkpoint_cb, sync_cb, eval_cb],
        progress_bar=True
    )
    
    # 保存最终模型和归一化统计量
    model.save("sac_grasp_final")
    # env.save("vec_normalize.pkl")  # 保存归一化参数，推理时需要加载
    print("训练完成，模型已保存。")

if __name__ == "__main__":
    main()