import os
import time
import glob
import datetime
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robot_env import RobotGraspEnv

# 可视化配置：开启 GUI 和 实时模式
viz_config = {
    'scene': {'scene_type': 'OnTable'},
    'simulation': {
        'visualize': True,   # 必须开启 GUI
        'real_time': True,   # 开启实时模式以便肉眼观察
        'auto_init_test_env': True,
    }
}

def make_viz_env():
    # evaluate=True 会固定随机种子，保证每次看到的场景一致（可选）
    # 注意：这里必须与训练时的环境参数保持一致（除了 visualize 和 real_time）
    return RobotGraspEnv(viz_config, evaluate=True, test=False, validate=False)

def get_latest_model():
    # 1. 优先找最终模型
    if os.path.exists("sac_grasp_final.zip"):
        return "sac_grasp_final.zip"
    
    # 2. 否则找 checkpoints 目录下最新的模型
    checkpoints = glob.glob("./checkpoints/*.zip")
    if not checkpoints:
        return None
    
    # 按修改时间排序，取最新的
    latest_model = max(checkpoints, key=os.path.getctime)
    return latest_model

def main():
    # 查找模型
    model_path = get_latest_model()
    if not model_path:
        print("错误：未找到模型文件！")
        print("请先运行 main.py 进行训练，或者确认 ./checkpoints/ 目录下有 .zip 文件。")
        return

    # 显示模型信息
    mod_time = os.path.getmtime(model_path)
    time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"正在加载模型: {model_path}")
    print(f"模型修改时间: {time_str}")

    # 创建环境
    # 注意：这里不需要 SubprocVecEnv，单进程即可
    env = DummyVecEnv([make_viz_env])
    # [Fix] 移除 VecTransposeImage，因为观测空间已改为一维向量
    # env = VecTransposeImage(env)

    # [修改] 训练时已禁用 norm_obs，这里不需要加载 VecNormalize
    # 如果将来重新启用 norm_obs，需要取消下面的注释
    norm_path = "vec_normalize.pkl"
    if os.path.exists(norm_path):
        print(f"发现 {norm_path}，但训练时已禁用 norm_obs，跳过加载。")
        # 如果需要使用 VecNormalize（即 norm_obs=True），取消以下注释：
        # from stable_baselines3.common.vec_env import VecNormalize
        # env = VecNormalize.load(norm_path, env)
        # env.training = False
        # env.norm_reward = False
    else:
        print("未找到 vec_normalize.pkl（已禁用 norm_obs，这是正常的）。")

    # 加载模型
    # custom_objects 用于解决不同 python 版本或环境可能导致的 pickle 问题
    try:
        model = SAC.load(model_path, env=env, device="cuda")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试使用 CPU 加载...")
        model = SAC.load(model_path, env=env, device="cpu")

    print("\n开始可视化回放...")
    print("按 Ctrl+C 停止")

    obs = env.reset()
    total_reward = 0
    steps = 0
    
    try:
        while True:
            # 预测动作 (deterministic=True 表示使用确定性策略，不加噪声)
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            total_reward += reward[0] # VecEnv 返回的是数组
            steps += 1
            
            # 输出末端和物体坐标用于观察对齐
            base_env = env.envs[0]
            try:
                tip_pos, _ = base_env._get_tip_position()
                obj_pos, _ = base_env.base_env.physics_client.getBasePositionAndOrientation(base_env.object_id)
                xy_dist = ((tip_pos[0]-obj_pos[0])**2 + (tip_pos[1]-obj_pos[1])**2)**0.5
                z_diff = tip_pos[2] - obj_pos[2]
                print(f"Step {steps}: 末端[{tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f}] "
                      f"物体[{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}] "
                      f"XY距离:{xy_dist*1000:.1f}mm Z差:{z_diff*1000:.1f}mm")
                
                # 可视化标记：在 PyBullet 中绘制末端点和物体的对齐情况
                pc = base_env.base_env.physics_client
                
                # 清除之前的调试线（使用固定的线段ID列表）
                if not hasattr(base_env, '_debug_line_ids'):
                    base_env._debug_line_ids = []
                for line_id in base_env._debug_line_ids:
                    pc.removeUserDebugItem(line_id)
                base_env._debug_line_ids = []
                
                # 1. 绿色竖线：从末端位置向下延伸到桌面，显示末端垂直投影
                line1 = pc.addUserDebugLine(
                    lineFromXYZ=[tip_pos[0], tip_pos[1], tip_pos[2]],
                    lineToXYZ=[tip_pos[0], tip_pos[1], 0.0],
                    lineColorRGB=[0, 1, 0],  # 绿色
                    lineWidth=2
                )
                base_env._debug_line_ids.append(line1)
                
                # 2. 红色竖线：从物体位置向上延伸，显示物体垂直位置
                line2 = pc.addUserDebugLine(
                    lineFromXYZ=[obj_pos[0], obj_pos[1], obj_pos[2]],
                    lineToXYZ=[obj_pos[0], obj_pos[1], tip_pos[2] + 0.1],
                    lineColorRGB=[1, 0, 0],  # 红色
                    lineWidth=2
                )
                base_env._debug_line_ids.append(line2)
                
                # 3. 黄色水平线：连接末端和物体在同一高度的投影，显示 XY 偏差
                line3 = pc.addUserDebugLine(
                    lineFromXYZ=[tip_pos[0], tip_pos[1], tip_pos[2]],
                    lineToXYZ=[obj_pos[0], obj_pos[1], tip_pos[2]],
                    lineColorRGB=[1, 1, 0],  # 黄色
                    lineWidth=3
                )
                base_env._debug_line_ids.append(line3)
                
                # 4. 青色小球标记末端位置（使用小球体）
                # 注意：PyBullet 没有直接画球的方法，用短粗线段十字交叉模拟
                cross_size = 0.01
                for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
                    line = pc.addUserDebugLine(
                        lineFromXYZ=[tip_pos[0]-dx*cross_size, tip_pos[1]-dy*cross_size, tip_pos[2]-dz*cross_size],
                        lineToXYZ=[tip_pos[0]+dx*cross_size, tip_pos[1]+dy*cross_size, tip_pos[2]+dz*cross_size],
                        lineColorRGB=[0, 1, 1],  # 青色
                        lineWidth=4
                    )
                    base_env._debug_line_ids.append(line)
                    
            except Exception as e:
                pass
            
            # 稍微延时，防止画面太快看不清 (虽然 real_time=True 已经限制了物理步长)
            # time.sleep(0.01) 

            if done[0]:
                print(f"Episode finished. Steps: {steps}, Total Reward: {total_reward:.2f}")
                
                # 打印一些 info 信息
                if isinstance(info[0], dict):
                    if 'metric_success' in info[0]:
                        success = "SUCCESS" if info[0]['metric_success'] > 0.5 else "FAIL"
                        print(f"Result: {success}, Max Height: {info[0].get('metric_max_height', 0):.3f}")
                
                obs = env.reset()
                total_reward = 0
                steps = 0
                print("-" * 30)
                
    except KeyboardInterrupt:
        print("\n可视化已停止。")
    finally:
        env.close()

if __name__ == "__main__":
    main()
