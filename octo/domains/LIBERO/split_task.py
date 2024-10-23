import os
from collections import defaultdict
import pickle


task_suite = 'libero_90'
folder = f'data/libero_origin/{task_suite}'

tasks_of_scene = defaultdict(list)
for task in os.listdir(folder):
    if task.startswith('LIVING_ROOM'):
        scene_name = '_'.join(task.split('_')[:3])
    else:
        scene_name = '_'.join(task.split('_')[:2])
    tasks_of_scene[scene_name].append(task)

train_tasks, test_tasks = [], []
for scene in tasks_of_scene:
    train_tasks.extend(tasks_of_scene[scene][:-1])
    test_tasks.append(tasks_of_scene[scene][-1])
breakpoint()
with open('domains/LIBERO/task_split.pkl', 'wb') as f:
    pickle.dump([train_tasks, test_tasks], f)
