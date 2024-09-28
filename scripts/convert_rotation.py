import numpy as np
import os
from transforms3d.euler import axangle2euler


def convert_axangle_to_rpy(axangle):
    # RT-1 uses axis angle for rotation delta in google robot tasks
    # need to convert to roll, pitch, yaw representation used by octo
    action_rotation_delta = axangle.astype(np.float64)
    action_rotation_angle = np.linalg.norm(action_rotation_delta)
    action_rotation_ax = (
        action_rotation_delta / action_rotation_angle
        if action_rotation_angle > 1e-6
        else np.array([0.0, 1.0, 0.0])
    )
    roll, pitch, yaw = axangle2euler(action_rotation_ax, action_rotation_angle)
    return np.array([roll, pitch, yaw], dtype=np.float32)


folder = 'data/self_generated_demo/rt_1_x'
for task in os.listdir(folder):
    print (task)
    demos = os.listdir(f'{folder}/{task}/train')
    for demo in demos:
        traj = np.load(f'{folder}/{task}/train/{demo}', allow_pickle=True)
        for step in traj:
            step['action']['rotation_delta'] = convert_axangle_to_rpy(step['action']['rotation_delta'])
        np.save(f'{folder}/{task}/train/{demo}', demo)
