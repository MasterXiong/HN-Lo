import json
import os
import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mediapy

from octo.domains.utils.action_space import convert_axangle_to_rpy


def convert_action_space(actions):
    # convert rotation from axangle to euler
    for i in range(actions.shape[0]):
        actions[i, 3:6] = convert_axangle_to_rpy(actions[i, 3:6])
    # convert gripper open/close from (1, -1) to (1, 0)
    actions[:, -1] = (actions[:, -1] + 1) / 2
    return actions


def convert_demo_data(task_suite='libero_90', dataset_name=None, left_right_flip=False, demo_num=None):
    source_folder = f'data/libero_origin/{task_suite}'
    if dataset_name is not None:
        target_folder = f'data/libero_separate_demos/{dataset_name}'
    else:
        target_folder = f'data/libero_separate_demos/{task_suite}'
    os.makedirs(target_folder, exist_ok=True)

    if 'libero_90' in task_suite:
        with open('octo/domains/LIBERO/task_split.pkl', 'rb') as f:
            train_tasks, _ = pickle.load(f)
    else:
        train_tasks = [x for x in os.listdir(source_folder) if x.endswith('hdf5')]

    episode_id = 0
    for task in train_tasks:
        demo_file = source_folder + '/' + task
        with h5py.File(demo_file, "r") as f:
            # problem_info = json.loads(f["data"].attrs["problem_info"])
            # language_instruction = "".join(problem_info["language_instruction"])
            if 'SCENE' in task:
                language_instruction = ' '.join(task.split('SCENE')[-1].split('_')[1:-1])
            else:
                language_instruction = ' '.join(task.split('_')[:-1])
            print (language_instruction)
            demos = list(f["data"].keys())
            if demo_num is not None:
                demos = demos[:demo_num]
            for demo in demos:
                obs = f[f"data/{demo}/obs/agentview_rgb"][()]
                if left_right_flip:
                    obs = obs[:, ::-1, ::-1]
                else:
                    obs = obs[:, ::-1]
                actions = f[f"data/{demo}/actions"][()]
                actions = convert_action_space(actions)
                episode_data = {
                    'obs': obs, # the demo images are up side down
                    'actions': actions, 
                    'language_instruction': language_instruction, 
                }
                np.save(f"{target_folder}/episode_{episode_id}.npy", episode_data)
                episode_id += 1


# generate a smaller training set
def subsample_demo_data(source_folder, target_folder, compress_ratio=5):
    source_folder = f'data/libero_separate_demos/{source_folder}'
    target_folder = f'data/libero_separate_demos/{target_folder}'
    os.makedirs(target_folder, exist_ok=True)
    sample_idx = list(range(len(os.listdir(source_folder))))[::compress_ratio]
    for idx in sample_idx:
        os.system(f'cp {source_folder}/episode_{idx}.npy {target_folder}/')


# generate single-task demos
def generate_single_task_demos(task_suite='libero_90_preprocessed'):
    source_folder = f'data/libero_origin/{task_suite}'
    with open('octo/domains/LIBERO/task_split.pkl', 'rb') as f:
        train_tasks, _ = pickle.load(f)
    
    # randomly sample a task from each scene
    # sampled_scenes = []
    # tasks = []
    # for task in train_tasks:
    #     if task.startswith('LIVING_ROOM'):
    #         scene = '_'.join(task.split('_')[:3])
    #     else:
    #         scene = '_'.join(task.split('_')[:2])
    #     if scene not in sampled_scenes:
    #         sampled_scenes.append(scene)
    #         tasks.append(task[:-10])
    tasks = [x[:-10] for x in train_tasks]

    for task in tasks:
        target_folder = f'data/libero_separate_demos/{task}'
        if os.path.exists(target_folder):
            continue
        os.makedirs(target_folder, exist_ok=True)
        demo_file = f'{source_folder}/{task}_demo.hdf5'
        with h5py.File(demo_file, "r") as f:
            # problem_info = json.loads(f["data"].attrs["problem_info"])
            # language_instruction = "".join(problem_info["language_instruction"])
            language_instruction = ' '.join(task.split('SCENE')[-1].split('_')[1:])
            print (language_instruction)
            episode_id = 0
            for demo in f["data"].keys():
                obs = f[f"data/{demo}/obs/agentview_rgb"][()]
                obs = obs[:, ::-1] # the demo images are up side down
                actions = f[f"data/{demo}/actions"][()]
                actions = convert_action_space(actions)
                episode_data = {
                    'obs': obs, 
                    'actions': actions, 
                    'language_instruction': language_instruction, 
                }
                np.save(f"{target_folder}/episode_{episode_id}.npy", episode_data)
                episode_id += 1


# convert_demo_data('libero_90_preprocessed', 'libero_90_preprocessed_compress_50', False, demo_num=1)
# subsample_demo_data('libero_90_preprocessed', 'libero_90_preprocessed_compress_10', compress_ratio=10)
generate_single_task_demos()

# check image
# file_path = '/user/octo/data/libero_separate_demos/10_demo_per_task/episode_0.npy'
# data = np.load(file_path, allow_pickle=True).item()     # this is a list of dicts in our case
# obs = data['obs']
# plt.figure()
# plt.imshow(obs[0, :, :64])
# plt.savefig('test.png')
# plt.close()
