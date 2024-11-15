import os
import h5py
import pickle


task_suite = 'libero_90'

source_folder = f'data/libero_origin/{task_suite}'
tasks = [x for x in os.listdir(source_folder) if x.endswith('hdf5')]

task_demo_length = dict()
for task in tasks:
    demo_file = source_folder + '/' + task
    with h5py.File(demo_file, "r") as f:
        max_length = max([f[f"data/demo_{i}/dones"].shape[0] for i in range(50)])
    task_demo_length[task[:-10]] = max_length
    print (task[:-5], max_length)

with open('octo/domains/LIBERO/task_demo_length.pkl', 'wb') as f:
    pickle.dump(task_demo_length, f)
