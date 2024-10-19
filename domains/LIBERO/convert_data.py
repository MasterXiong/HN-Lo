import json
import os
import h5py
import numpy as np
import pickle


# task_suite = 'libero_90'
# source_folder = f'data/libero_origin/{task_suite}'
# target_folder = f'data/libero_separate_demos/{task_suite}'
# os.makedirs(target_folder, exist_ok=True)

# with open('domains/LIBERO/task_split.pkl', 'rb') as f:
#     train_tasks, _ = pickle.load(f)

# episode_id = 0
# for task in train_tasks:
#     print (task)
#     demo_file = source_folder + '/' + task
#     with h5py.File(demo_file, "r") as f:
#         problem_info = json.loads(f["data"].attrs["problem_info"])
#         language_instruction = "".join(problem_info["language_instruction"])
#         for i in range(50):
#             obs = f[f"data/demo_{i}/obs/agentview_rgb"][()]
#             actions = f[f"data/demo_{i}/actions"][()]
#             episode_data = {
#                 'obs': obs, 
#                 'actions': actions, 
#                 'language_instruction': language_instruction, 
#             }
#             np.save(f"{target_folder}/episode_{episode_id}.npy", episode_data)
#             episode_id += 1


# generate a smaller training set
source_folder = 'data/libero_separate_demos/libero_90'
compress_ratio = 5
num_demo_per_task = 50 // compress_ratio
target_folder = f'data/libero_separate_demos/{num_demo_per_task}_demo_per_task'
os.makedirs(target_folder, exist_ok=True)

sample_idx = list(range(len(os.listdir(source_folder))))[::compress_ratio]
for idx in sample_idx:
    os.system(f'cp {source_folder}/episode_{idx}.npy {target_folder}/')
