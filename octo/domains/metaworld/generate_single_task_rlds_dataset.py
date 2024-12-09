import os

from config import *

RLDS_PATH = '/user/hypervla/data/rlds_converter/meta_world_dataset'


def generate_demos(task, demo_num):

    source_path = '/user/hypervla/dataset/metaworld_single_demos'
    target_path = f'/user/hypervla/dataset/metaworld_single_task_{demo_num}_demos/{task}'
    os.system(f'rm -r {target_path}')

    # clear space
    os.system(f'rm -r {RLDS_PATH}/dataset')
    os.makedirs(f'{RLDS_PATH}/dataset', exist_ok=True)

    count = 0
    for i in range(50):
        if os.path.exists(f'{source_path}/{task}/episode_{i}.npy'):
            os.system(f'cp {source_path}/{task}/episode_{i}.npy {RLDS_PATH}/dataset/episode_{count}.npy')
            count += 1
            if count == demo_num:
                break

    # generate dataset
    os.chdir(RLDS_PATH)
    os.system(f'tfds build --overwrite')
    # move dataset
    os.system(f'mv /user/tensorflow_datasets/meta_world_dataset {target_path}')

    # clear space
    os.system(f'rm -r {RLDS_PATH}/dataset')



if __name__ == '__main__':

    tasks = sorted(os.listdir('dataset/metaworld_single_demos'))
    for task in ML45_test_tasks:
        tasks.remove(task)

    demo_num = 1
    for task in tasks:
        os.makedirs(f'/user/hypervla/dataset/metaworld_single_task_{demo_num}_demos', exist_ok=True)
        generate_demos(task, demo_num)
