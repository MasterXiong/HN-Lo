import argparse
import numpy as np
import os
import tensorflow as tf
import json
import pickle
import h5py

from libero.libero import get_libero_path
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from octo.domains.utils.multi_env_interface import OctoInference
from octo.domains.utils.venv import SubprocVectorEnv
from octo.utils.attention import *

import mediapy


# prevent a single jax process from taking up all the GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3072)],
    )

def load_model(model_name, model_path, input_rng=0, step=None, action_ensemble=False, crop=False, image_horizon=2):
    if 'hypernet' in model_path or 'vanilla_lora' in model_path:
        from octo.model_lora.octo_model import OctoModel
    else:
        from octo.model.octo_model import OctoModel
    tempmodel = OctoModel.load_pretrained(model_path, step=step)
    model = OctoInference(
        model=tempmodel, 
        policy_setup='libero', 
        init_rng=input_rng, 
        action_ensemble=action_ensemble, 
        horizon=image_horizon, 
        crop=crop)
    return model


def evaluate(model_name, model_path, task_suite_name, seed=0, checkpoint_step=None, split='train', save_video=False, env_num=20, action_ensemble=False, flipping=False, crop=False, image_horizon=2, recompute=False):

    if model_path == 'hf://rail-berkeley/octo-base-1.5':
        eval_path = f'eval_results/libero/octo-base/{seed}'
    else:
        save_dir = 'eval_results/libero/' + '/'.join(model_path.split('/')[1:])
        eval_path = f'{save_dir}/eval_step_{checkpoint_step}/{seed}'
    os.makedirs(eval_path, exist_ok=True)

    save_file_name = f'success_rate_{split}'
    if action_ensemble:
        save_file_name += '_action_ensemble'
    if crop:
        save_file_name += '_crop'
    if image_horizon != 2:
        save_file_name += f'_horizon_{image_horizon}'
    if os.path.exists(f'{eval_path}/{save_file_name}.json'):
        with open(f'{eval_path}/{save_file_name}.json', 'r') as f:
            all_tasks_success_rate = json.load(f)
    else:
        all_tasks_success_rate = dict()

    model = load_model(model_name, model_path, seed, step=checkpoint_step, action_ensemble=action_ensemble, crop=crop, image_horizon=image_horizon)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    if task_suite_name == 'libero_90':
        all_task_names = [task.name for task in task_suite.tasks]
        with open('octo/domains/LIBERO/task_split.pkl', 'rb') as f:
            train_tasks, test_tasks = pickle.load(f)
        if 'train' in split:
            tasks = train_tasks
        else:
            tasks = test_tasks
        tasks = [all_task_names.index(task_name[:-10]) for task_name in tasks]
    else:
        tasks = list(range(task_suite.get_num_tasks()))

    with open('octo/domains/LIBERO/task_demo_length.pkl', 'rb') as f:
        task_demo_length = pickle.load(f)

    for task_id in tasks:

        # retrieve a specific task
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        if task_name in all_tasks_success_rate and not recompute:
            continue

        video_path = f"{eval_path}/video/{split}/{task_name}"
        os.makedirs(video_path, exist_ok=True)

        # approach 1: single process
        # env_args = {
        #     "bddl_file_name": task_bddl_file,
        #     "camera_heights": 256,
        #     "camera_widths": 256
        # }
        # env = OffScreenRenderEnv(**env_args)
        # env.seed(0)
        # env.reset()
        # init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix a set of initial states

        # model.reset(task_description)

        # print (f'===== {task_name} =====')
        # success_count = 0
        # episode_results = []
        # total_runs = 1
        # for run in range(total_runs):
        #     env.reset()
        #     init_state_id = run
        #     init_state = env.set_init_state(init_states[init_state_id])

        #     image = init_state['agentview_image'][::-1]  # the simulation image is up side down, need to flip manually
        #     images = [image]
        #     done = False
        #     for t in range(600):
        #         # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        #         raw_action, action, _, _ = model.step(image)
        #         obs, reward, done, info = env.step(
        #             np.concatenate([action["world_vector"], action["rotation_delta"], action["gripper"]])
        #         )
        #         # update image observation
        #         image = obs['agentview_image'][::-1]
        #         images.append(image)
        #         if done:
        #             break
        #     success = (reward > 0)
        #     if success:
        #         success_count += 1
        #     episode_results.append(success)
        #     print(run+1, success_count, success_count/(run+1)*100)
        #     if save_video:
        #         result = 'success' if success else 'fail'
        #         mediapy.write_video(f'{video_path}/{run + 1}_{result}.mp4', images, fps=10)
        # env.close()
        # all_tasks_success_rate[task_name] = [success_count / total_runs, episode_results]
        # print ({key: all_tasks_success_rate[key][0] for key in all_tasks_success_rate})
        # with open(f'{eval_path}/{save_file_name}.json', 'w') as f:
        #     json.dump(all_tasks_success_rate, f)

        # approach 2: multi-processing
        # reset the model with the task instruction
        model.reset(task_description)

        # initialize the envs
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256
        }
        env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
        env.seed(0)
        env.reset()

        # set the initial states
        if split == 'train_demo':
            init_states = []
            demo_file = f'data/libero_origin/libero_90/{task_name}_demo.hdf5'
            with h5py.File(demo_file, "r") as f:
                init_states = np.stack([f[f"data/demo_{i}/states"][0] for i in range(50)])
        else:
            init_states = task_suite.get_task_init_states(task_id)
        indices = np.arange(env_num) % init_states.shape[0]
        obs = env.set_init_state(init_states[indices])

        for _ in range(10):  # simulate the physics without any actions
            obs, _, _, _ = env.step(np.zeros((env_num, 7)))

        if flipping:
            images = np.stack([obs[i]['agentview_image'][::-1, ::-1] for i in range(len(obs))])
        else:
            images = np.stack([obs[i]['agentview_image'][::-1] for i in range(len(obs))])
        images_history = [images]
        images_with_attention_weights = []
        attention_history = []

        print (f'===== {task_name} =====')
        finished_tasks = [False] * env_num
        max_step = task_demo_length[task_name] + 30
        episode_length = [max_step] * env_num
        for step in range(max_step):
            raw_actions, actions, action_attention_weights, _ = model.step(images)
            # heatmaps = generate_attention_map(action_attention_weights['mean'][-1])
            # masked_images = combine_image_and_heatmap(images, heatmaps)
            # images_with_attention_weights.append(masked_images)
            # attention_history.append(action_attention_weights)
            actions = np.concatenate([actions['world_vector'], actions['rot_axangle'], actions['gripper'].reshape(-1, 1)], axis=1)
            obs, rewards, dones, infos = env.step(actions)
            # check whether succeed
            for k in range(env_num):
                if dones[k]:
                    finished_tasks[k] = True
                    episode_length[k] = min(step + 1, episode_length[k])
            if all(finished_tasks):
                break
            if flipping:
                images = np.stack([obs[i]['agentview_image'][::-1, ::-1] for i in range(len(obs))])
            else:
                images = np.stack([obs[i]['agentview_image'][::-1] for i in range(len(obs))])
            images_history.append(images)

        success_rate = sum(finished_tasks) / env_num
        # env.close()

        if save_video:
            os.system(f'rm {video_path}/*.mp4')
            for i in range(env_num):
                result = 'success' if finished_tasks[i] else 'fail'
                # images = [x[i] for x in images_with_attention_weights[:episode_length[i]]]
                images = [x[i] for x in images_history[:episode_length[i]]]
                mediapy.write_video(f'{video_path}/{i + 1}_{result}.mp4', images, fps=10)
            # with open(f'{video_path}/record.pkl', 'wb') as f:
            #     pickle.dump([images_history, attention_history, episode_length], f)

        all_tasks_success_rate[task_name] = success_rate
        sorted_scores = sorted(all_tasks_success_rate.items(), key=lambda x: x[1])
        for x in sorted_scores:
            print (x)
        with open(f'{eval_path}/{save_file_name}.json', 'w') as f:
            json.dump(all_tasks_success_rate, f)



if __name__ == '__main__':

    # Add arguments
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("--model", choices=["octo-small", "octo-base", "octo-custom", "rt_1_x", "rt_1_400k"], default="octo-custom", help="The model used for evaluation")
    parser.add_argument("--model_path", type=str, default='', help="The path of the custom model (only useful for octo-custom?)")
    parser.add_argument("--task_suite_name", type=str, default='libero_90', help="the task suite to evaluate on")
    parser.add_argument("--seeds", type=str, default='0+1+2+3', help="seeds for policy and env")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step to evaluate")
    parser.add_argument("--split", type=str, default='train', help="evaluate on the train or test split")
    parser.add_argument("--save_video", action='store_true', help="save evaluation video or not")
    parser.add_argument("--flipping", action='store_true', help="flip the left and right side of the image or not")
    parser.add_argument("--action_ensemble", action='store_true', help="Use action ensemble or not")
    parser.add_argument("--crop", action='store_true', help="Whether to crop the image or not")
    parser.add_argument("--image_horizon", type=int, default=2, help="The horizon of image history")
    parser.add_argument("--recompute", action='store_true', help="Whether to recompute for existing results")
    # Parse the arguments
    args = parser.parse_args()

    seeds = [eval(seed) for seed in args.seeds.split('+')]
    for seed in seeds:
        evaluate(args.model, args.model_path, args.task_suite_name, seed=seed, checkpoint_step=args.step, split=args.split, save_video=args.save_video, flipping=args.flipping, action_ensemble=args.action_ensemble, crop=args.crop, image_horizon=args.image_horizon, recompute=args.recompute)
