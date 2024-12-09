import argparse
import numpy as np
import os
import tensorflow as tf
import json
import pickle
import cv2
import matplotlib.pyplot as plt

import metaworld
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from octo.domains.metaworld.config import *

from octo.domains.utils.multi_env_interface import OctoInference
from octo.utils.attention import *

import mediapy

os.environ["MUJOCO_GL"] = "egl"

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
CAMERA_VIEW = 'corner2'
CROP_RATIO = 0.2

# prevent a single jax process from taking up all the GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3072)],
    )


def make_env(env_cls, initial_state):
    def _init():
        env = env_cls(render_mode='rgb_array', camera_name='corner2')
        env.set_task(initial_state)
        return env
    return _init


def process_image(image):
    height, width = image.shape[0], image.shape[1]
    image = image[int(height * CROP_RATIO):int(height * (1 - CROP_RATIO)), int(width * CROP_RATIO):int(width * (1 - CROP_RATIO))]
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return image[::-1].astype(np.uint8)


def load_model(model_path, mode, input_rng=0, step=None, action_ensemble=False, crop=False, image_horizon=2):
    if mode == 'hypernet_v2':
        from octo.model_lora_v2.octo_model import OctoModel
    elif mode == 'hypernet':
        from octo.model_lora.octo_model import OctoModel
    else:
        from octo.model.octo_model import OctoModel
    tempmodel = OctoModel.load_pretrained(model_path, step=step)
    model = OctoInference(
        model=tempmodel, 
        policy_setup='metaworld', 
        init_rng=input_rng, 
        action_ensemble=action_ensemble, 
        horizon=image_horizon, 
        crop=crop)
    return model


def evaluate(model_path, seed=0, checkpoint_step=None, split='train', save_video=False, env_num=5, action_ensemble=False, image_horizon=2, recompute=False):

    if model_path == 'hf://rail-berkeley/octo-base-1.5':
        eval_path = f'eval_results/metaworld/octo-base/{seed}'
    else:
        save_dir = 'eval_results/metaworld/' + '/'.join(model_path.split('/')[2:])
        eval_path = f'{save_dir}/eval_step_{checkpoint_step}/{seed}'
    os.makedirs(eval_path, exist_ok=True)

    save_file_name = f'success_rate_{split}'
    if action_ensemble:
        save_file_name += '_action_ensemble'
    if image_horizon != 2:
        save_file_name += f'_horizon_{image_horizon}'
    if os.path.exists(f'{eval_path}/{save_file_name}.json'):
        with open(f'{eval_path}/{save_file_name}.json', 'r') as f:
            all_tasks_success_rate = json.load(f)
    else:
        all_tasks_success_rate = dict()

    with open(f'{model_path}/finetune_config.json', 'r') as f:
        finetune_config = json.load(f)

    model = load_model(model_path, finetune_config['finetuning_mode'], seed, step=checkpoint_step, action_ensemble=action_ensemble, image_horizon=image_horizon)

    benchmark = metaworld.ML45(seed=seed)
    if split == 'train':
        tasks = benchmark.train_classes
        initial_states = benchmark.train_tasks
    else:
        tasks = benchmark.test_classes
        initial_states = benchmark.test_tasks

    for name, env_cls in tasks.items():

        if name in all_tasks_success_rate and not recompute:
            continue

        video_path = f"{eval_path}/video/{split}/{name}"
        os.makedirs(video_path, exist_ok=True)

        init_states = [s for s in initial_states if s.env_name == name]
        envs = AsyncVectorEnv([make_env(env_cls, init_states[i]) for i in range(env_num)])

        language_instruction = policies[name][1]
        print (f'===== {name}: {language_instruction} =====')

        # reset the model with the task instruction
        model.reset(language_instruction, remove_useless_token=finetune_config.get('remove_useless_token', False), env_num=env_num)

        obs, _ = envs.reset()  # Reset environment
        # simulate for a while to get a steady state
        dummy_action = np.zeros((env_num, 4))
        for _ in range(20):
            obs, reward, done, truncated, info = envs.step(dummy_action)

        images = envs.render()
        images = np.stack(process_image(image) for image in images)
        images_history = []
        images_with_attention_weights = []
        attention_history = []

        finished_tasks = [False] * env_num
        # max_step = task_demo_length[task_name] + 30
        # TODO: how to set max_step?
        max_step = 300
        episode_length = [max_step] * env_num
        for step in range(max_step):
            raw_actions, actions, attention_weights, _, (language_instruction, tokens) = model.step(images)

            if step == 0 and 'hypernet' in attention_weights:
                task_attention_weights = []
                for i in range(3):
                    # shape: head_num * token_num * token_num
                    w = attention_weights['hypernet']['Transformer_0'][f'encoderblock_{i}']['MultiHeadDotProductAttention_0']['attention_weights'][0][0]
                    task_attention_weights.append(w)
                os.makedirs(f'{eval_path}/task_attention_weights/{split}', exist_ok=True)
                with open(f'{eval_path}/task_attention_weights/{split}/{name}.pkl', 'wb') as f:
                    pickle.dump([task_attention_weights, language_instruction, tokens], f)

            # analyze action attention mask
            # action_attention_weights = {'mean': [], 'max': []}
            # for i in range(12):
            #     # attention map shape: batch_size * head_num * token_num * token_num
            #     # shape of w: batch_size * head_num * image_token_num (256)
            #     w = attention_weights['Transformer_0'][f'encoderblock_{i}']['MultiHeadDotProductAttention_0']['attention_weights'][0][:, :, -1, -(1 + 16 + 256):-(1 + 16)]
            #     w_mean = w.mean(axis=1)
            #     w_max = w.max(axis=1)
            #     action_attention_weights['mean'].append(w_mean)
            #     action_attention_weights['max'].append(w_max)
            # heatmaps = generate_attention_map(action_attention_weights['mean'][-1])
            # masked_images = combine_image_and_heatmap(images, heatmaps)
            # images_with_attention_weights.append(masked_images)
            # attention_history.append(action_attention_weights)

            actions = np.concatenate([actions['world_vector'], actions['gripper'].reshape(-1, 1)], axis=1)
            obs, rewards, dones, truncated, infos = envs.step(actions)
            # check whether succeed
            for k in range(env_num):
                if int(infos["success"][k]) == 1:
                    finished_tasks[k] = True
                    episode_length[k] = min(step + 1, episode_length[k])
            if all(finished_tasks):
                break
            images = envs.render()
            images = np.stack(process_image(image) for image in images)
            images_history.append(images)

        success_rate = sum(finished_tasks) / env_num
        envs.close()

        if save_video:
            os.system(f'rm {video_path}/*.mp4')
            for i in range(env_num):
                result = 'success' if finished_tasks[i] else 'fail'
                # images = [x[i] for x in images_with_attention_weights[:episode_length[i]]]
                images = [x[i] for x in images_history[:episode_length[i]]]
                mediapy.write_video(f'{video_path}/{i + 1}_{result}.mp4', images, fps=10)
            # with open(f'{video_path}/record.pkl', 'wb') as f:
            #     pickle.dump([images_history, attention_history, episode_length], f)

        all_tasks_success_rate[name] = success_rate
        sorted_scores = sorted(all_tasks_success_rate.items(), key=lambda x: x[1])
        for x in sorted_scores:
            print (x)
        with open(f'{eval_path}/{save_file_name}.json', 'w') as f:
            json.dump(all_tasks_success_rate, f)

    full_results = sorted(all_tasks_success_rate.items(), key=lambda x: x[0])
    for x in full_results:
        print (x)
    print (np.mean([x[1] for x in full_results]))



if __name__ == '__main__':

    # Add arguments
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("--model_path", type=str, default='', help="The path of the custom model (only useful for octo-custom?)")
    parser.add_argument("--seeds", type=str, default='0+1+2+3', help="seeds for policy and env")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step to evaluate")
    parser.add_argument("--split", type=str, default='train', help="evaluate on the train or test split")
    parser.add_argument("--save_video", action='store_true', help="save evaluation video or not")
    parser.add_argument("--action_ensemble", action='store_true', help="Use action ensemble or not")
    parser.add_argument("--image_horizon", type=int, default=2, help="The horizon of image history")
    parser.add_argument("--recompute", action='store_true', help="Whether to recompute for existing results")
    # Parse the arguments
    args = parser.parse_args()

    seeds = [eval(seed) for seed in args.seeds.split('+')]
    for seed in seeds:
        evaluate(args.model_path, seed=seed, checkpoint_step=args.step, split=args.split, save_video=args.save_video, action_ensemble=args.action_ensemble, image_horizon=args.image_horizon, recompute=args.recompute)
