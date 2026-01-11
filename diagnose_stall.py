"""诊断脚本：分析策略为什么在35mm处停止

分析内容：
1. 策略输出的动作值
2. 高度保持惩罚是否过强
3. XY进展奖励信号强度
"""

import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from robot_env import RobotGraspEnv

viz_config = {
    'scene': {'scene_type': 'OnTable'},
    'simulation': {
        'visualize': True,
        'real_time': True,
        'auto_init_test_env': True,
    }
}

def make_viz_env():
    return RobotGraspEnv(viz_config, evaluate=True, test=False, validate=False)

def get_latest_model():
    if os.path.exists("sac_grasp_final.zip"):
        return "sac_grasp_final.zip"
    import glob
    checkpoints = glob.glob("./checkpoints/*.zip")
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def main():
    model_path = get_latest_model()
    if not model_path:
        print("错误：未找到模型文件！")
        return

    print(f"加载模型: {model_path}")
    
    env = DummyVecEnv([make_viz_env])
    
    try:
        model = SAC.load(model_path, env=env, device="cuda")
    except:
        model = SAC.load(model_path, env=env, device="cpu")

    base_env = env.envs[0]
    
    print("\n" + "="*80)
    print("策略行为分析")
    print("="*80)
    
    obs = env.reset()
    
    # 收集数据
    action_history = []
    reward_history = []
    xy_dist_history = []
    height_penalty_history = []
    xy_progress_history = []
    
    for step in range(50):
        # 获取动作
        action, _ = model.predict(obs, deterministic=True)
        raw_action = action[0]
        
        # 获取状态
        tip_pos, _ = base_env._get_tip_position()
        obj_pos, _ = base_env.base_env.physics_client.getBasePositionAndOrientation(base_env.object_id)
        
        rel_pos = np.array(obj_pos) - np.array(tip_pos)
        xy_dist = np.linalg.norm(rel_pos[:2])
        
        # 计算期望动作方向
        expected_dx = np.clip(rel_pos[0] / 0.05, -1, 1)
        expected_dy = np.clip(rel_pos[1] / 0.05, -1, 1)
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        # 提取奖励分量
        reward_info = info[0] if isinstance(info, list) else info
        height_penalty = reward_info.get('height_deviation', 0)
        xy_progress = reward_info.get('xy_progress', 0)
        
        # 记录
        action_history.append(raw_action.copy())
        reward_history.append(float(reward[0]))
        xy_dist_history.append(xy_dist)
        height_penalty_history.append(height_penalty)
        xy_progress_history.append(xy_progress)
        
        # 打印详细信息
        action_magnitude = np.sqrt(raw_action[0]**2 + raw_action[1]**2)
        
        # 打印绝对位置信息（用于诊断工作空间限制问题）
        if step < 5 or step >= 45:  # 只打印前5步和后5步的详细位置
            print(f"Step {step+1:2d}: "
                  f"Action[dx={raw_action[0]:+.3f}, dy={raw_action[1]:+.3f}, dz={raw_action[2]:+.3f}] "
                  f"期望[dx={expected_dx:+.3f}, dy={expected_dy:+.3f}] "
                  f"|a|={action_magnitude:.3f} "
                  f"XY={xy_dist*1000:.1f}mm\n"
                  f"         末端位置: [{tip_pos[0]:.4f}, {tip_pos[1]:.4f}, {tip_pos[2]:.4f}] "
                  f"物体位置: [{obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f}]")
        else:
            print(f"Step {step+1:2d}: "
                  f"Action[dx={raw_action[0]:+.3f}, dy={raw_action[1]:+.3f}, dz={raw_action[2]:+.3f}] "
                  f"期望[dx={expected_dx:+.3f}, dy={expected_dy:+.3f}] "
                  f"|a|={action_magnitude:.3f} "
                  f"XY={xy_dist*1000:.1f}mm")
        
        if done[0]:
            print("Episode 结束")
            break
    
    print("\n" + "="*80)
    print("统计分析")
    print("="*80)
    
    actions = np.array(action_history)
    
    print(f"\n动作统计:")
    print(f"  dx: 均值={np.mean(actions[:,0]):.4f}, 范围=[{np.min(actions[:,0]):.3f}, {np.max(actions[:,0]):.3f}]")
    print(f"  dy: 均值={np.mean(actions[:,1]):.4f}, 范围=[{np.min(actions[:,1]):.3f}, {np.max(actions[:,1]):.3f}]")
    print(f"  dz: 均值={np.mean(actions[:,2]):.4f}, 范围=[{np.min(actions[:,2]):.3f}, {np.max(actions[:,2]):.3f}]")
    
    action_magnitudes = np.sqrt(actions[:,0]**2 + actions[:,1]**2)
    print(f"\n动作幅度 (XY平面):")
    print(f"  均值: {np.mean(action_magnitudes):.4f}")
    print(f"  最小: {np.min(action_magnitudes):.4f}")
    print(f"  最大: {np.max(action_magnitudes):.4f}")
    
    # 分析前10步和后10步
    early_mag = np.mean(np.sqrt(actions[:10,0]**2 + actions[:10,1]**2))
    late_mag = np.mean(np.sqrt(actions[-10:,0]**2 + actions[-10:,1]**2))
    
    print(f"\n动作趋势:")
    print(f"  前10步平均幅度: {early_mag:.4f}")
    print(f"  后10步平均幅度: {late_mag:.4f}")
    print(f"  衰减比例: {late_mag/early_mag*100:.1f}%")
    
    print(f"\nXY距离变化:")
    print(f"  初始: {xy_dist_history[0]*1000:.1f}mm")
    print(f"  最终: {xy_dist_history[-1]*1000:.1f}mm")
    print(f"  减少: {(xy_dist_history[0] - xy_dist_history[-1])*1000:.1f}mm")
    
    # 分析奖励信号
    if any(h != 0 for h in height_penalty_history):
        print(f"\n高度惩罚:")
        print(f"  总惩罚: {sum(height_penalty_history):.4f}")
        print(f"  平均每步: {np.mean(height_penalty_history):.4f}")
    
    print("\n" + "="*80)
    print("问题诊断")
    print("="*80)
    
    if late_mag < 0.05:
        print("\n⚠️ 问题：策略后期动作幅度接近0！")
        print("   可能原因：")
        print("   1. 高度保持惩罚过强，策略害怕移动")
        print("   2. XY进展奖励信号太弱")
        print("   3. 策略陷入局部最优（不动=不被惩罚）")
        
    if late_mag / early_mag < 0.3:
        print("\n⚠️ 问题：动作幅度衰减过快！")
        print("   建议：增加探索噪声或降低高度惩罚")
    
    # 检查方向是否正确
    tip_pos, _ = base_env._get_tip_position()
    obj_pos, _ = base_env.base_env.physics_client.getBasePositionAndOrientation(base_env.object_id)
    final_rel = np.array(obj_pos[:2]) - np.array(tip_pos[:2])
    avg_action = np.mean(actions[-5:, :2], axis=0)
    
    dot_product = np.dot(final_rel / np.linalg.norm(final_rel), avg_action / (np.linalg.norm(avg_action) + 1e-8))
    
    print(f"\n方向分析:")
    print(f"  物体在末端的方向: [{final_rel[0]:.3f}, {final_rel[1]:.3f}]")
    print(f"  策略最后5步平均动作: [{avg_action[0]:.3f}, {avg_action[1]:.3f}]")
    print(f"  方向一致性 (点积): {dot_product:.3f}")
    
    if dot_product < 0.5:
        print("   ⚠️ 策略动作方向与目标方向不一致！")
    
    print("\n" + "="*80)
    print("建议修改")
    print("="*80)
    print("""
1. 降低高度保持惩罚系数（从2.0降到0.5）
2. 增加XY进展奖励系数（从5.0增到10.0）
3. 或者考虑：移除高度保持惩罚，改用更温和的方式
4. 增加训练步数到50万步
    """)
    
    env.close()

if __name__ == "__main__":
    main()
