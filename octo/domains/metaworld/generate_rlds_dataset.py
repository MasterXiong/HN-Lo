import argparse
import os

from config import *

RLDS_PATH = '/user/octo/octo/domains/rlds_converter/meta_world_dataset'


def generate_demos(args):

    demo_num = args.demo_num
    source_path = args.source_path

    # only include ML45 tasks in the training data
    tasks = sorted(os.listdir(source_path))
    for task in ML45_test_tasks:
        tasks.remove(task)
    print (len(tasks))

    target_path = f'/user/octo/data/metaworld_rlds/metaworld_{demo_num}_demos_per_task'
    os.system(f'rm -r {target_path}')

    # clear space
    os.system(f'rm -r {RLDS_PATH}/dataset')
    os.makedirs(f'{RLDS_PATH}/dataset', exist_ok=True)

    count = 0
    for task in tasks:
        actual_demo_num = min(len(os.listdir(f'{source_path}/{task}')), demo_num)
        print (task, actual_demo_num)
        task_demo_num = 0
        for i in range(50):
            if os.path.exists(f'{source_path}/{task}/episode_{i}.npy'):
                os.system(f'cp {source_path}/{task}/episode_{i}.npy {RLDS_PATH}/dataset/episode_{count}.npy')
                count += 1
                task_demo_num += 1
                if task_demo_num == demo_num:
                    break

    # generate dataset
    os.chdir(RLDS_PATH)
    os.system(f'tfds build --overwrite')
    # move dataset
    os.system(f'mv /user/tensorflow_datasets/meta_world_dataset {target_path}')

    # clear space
    os.system(f'rm -r {RLDS_PATH}/dataset')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_num', type=int, default=50, help='the number of demos for each task')
    parser.add_argument('--source_path', type=str, default='/user/octo/octo/metaworld_single_demos', help='the path to load npy demo files')
    args = parser.parse_args()

    generate_demos(args)
