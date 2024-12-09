import os
import numpy as np
import metaworld
import mediapy
import cv2

from config import *

os.environ["MUJOCO_GL"] = "egl"

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
CAMERA_VIEW = 'corner2'
CROP_RATIO = 0.2

# Need to use MT (multi-task) benchmark so that the goal position is included in obs for the scripted policy
# benchmark = metaworld.MT1('assembly-v2', seed=0)
benchmark = metaworld.MT50(seed=0)

for name, env_cls in benchmark.train_classes.items():
    print (name)
    # https://github.com/Farama-Foundation/Metaworld/blob/master/docs/rendering/rendering.md
    env = env_cls(render_mode='rgb_array', camera_name=CAMERA_VIEW)
    tasks = [task for task in benchmark.train_tasks if task.env_name == name]
    policy = policies[name][0]()
    language_instruction = policies[name][1]
    folder_name = language_instruction.replace(' ', '_')

    demo_save_path = f'data/metaworld_single_demos/{name}'
    os.makedirs(demo_save_path, exist_ok=True)

    for i, task in enumerate(tasks):
        env.set_task(task)
        obs, info = env.reset()  # Reset environment

        # simulate for a while to get a steady state
        dummy_action = np.zeros(4)
        for _ in range(20):
            obs, reward, done, truncated, info = env.step(dummy_action)

        demo = []
        for _ in range(400):
            action = policy.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            image = env.render()
            height, width = image.shape[0], image.shape[1]
            image = image[int(height * CROP_RATIO):int(height * (1 - CROP_RATIO)), int(width * CROP_RATIO):int(width * (1 - CROP_RATIO))]
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            demo.append({'obs': image[::-1].astype(np.uint8), 'action': action, 'instruction': language_instruction}) # the rendering image is up side down for some camera view
            if int(info["success"]) == 1:
                print (f'{i + 1} / 50 success, length = {len(demo)}')
                np.save(f'{demo_save_path}/episode_{i}.npy', demo)
                break
        if i == 0:
            mediapy.write_video(f'figures/{name}.mp4', [step['obs'] for step in demo], fps=10)

    env.close()
