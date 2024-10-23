import os
import h5py
import mediapy


task_suite = 'libero_90'
source_folder = f'data/libero_origin/{task_suite}'

episode_id = 0
for task in os.listdir(source_folder):
    demo_file = source_folder + '/' + task
    with h5py.File(demo_file, "r") as f:
        for i in range(50):
            obs = f[f"data/demo_{i}/obs/agentview_rgb"][()]
            images = [obs[i, ::-1] for i in range(obs.shape[0])]
            # breakpoint()
            mediapy.write_video('test.mp4', images, fps=10)
            breakpoint()
            break
    break