import cv2
from env import Env


# minimal config for Env: scene and simulation sections are required by Env
config = {
    'scene': {'scene_type': 'OnTable'},
    'simulation': {
        'visualize': True,
        'real_time': True,
        'auto_init_test_env': True
    }
}


if __name__ == "__main__":
    # 创建环境实例：(config, evaluate, test, validate)
    env = Env(config, evaluate=False, test=False, validate=False)

    try:
        while True:
            # 先读取并应用 UI 中的关节目标（若已创建滑块）
            try:
                env.update_joints_from_ui()
            except Exception:
                pass

            # 使用 Env 提供的步进接口（会自动处理 real-time 同步）
            env.step_sim()
            # 可视化相机视锥（调试用）
            # env.sensor.draw_camera_frustum(depth=1.0)
            # 获取相机数据（如果可用）
            rgb, depth, mask = env.get_sensor_data()

            # 可选：在这里显示或处理图片，例如使用 cv2.imshow
            # if rgb is not None:
            #     cv2.imshow('rgb', rgb)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break

    except KeyboardInterrupt:
        print("模拟已终止。")

    env.close()
    cv2.destroyAllWindows()