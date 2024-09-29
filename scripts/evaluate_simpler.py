import argparse
import numpy as np
import os
import tensorflow as tf
import json

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien

import mediapy
import gymnasium as gym

import octo.simpler_new


# prevent a single jax process from taking up all the GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3072)],
    )

def load_model(model_name, model_path, policy_setup, input_rng=0, step=None):
    if "rt_1" in model_name:
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        ckpt_path = get_rt_1_checkpoint(model_name)
        model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
    elif "octo" in model_name:
        from octo.simpler_new.octo_model import OctoInference
        if 'hypernet' in model_path or 'vanilla_lora' in model_path:
            from octo.model_lora.octo_model import OctoModel
        else:
            from octo.model.octo_model import OctoModel
        tempmodel = OctoModel.load_pretrained(model_path, step=step)
        model = OctoInference(model=tempmodel, policy_setup=policy_setup, init_rng=input_rng)
    else:
        raise ValueError(model_name)
    return model


def evaluate(model_name, model_path, tasks, seed=0, checkpoint_step=None, split='train', save_video=False):

    previous_policy_setup = ''
    if model_path == 'hf://rail-berkeley/octo-base-1.5':
        eval_path = f'eval_results/octo-base/{seed}'
    else:
        save_dir = 'eval_results/' + '/'.join(model_path.split('/')[1:])
        eval_path = f'{save_dir}/eval_step_{checkpoint_step}/{seed}'
    os.makedirs(eval_path, exist_ok=True)

    save_file_name = f'success_rate_simpler_{split}'
    if os.path.exists(f'{eval_path}/{save_file_name}.json'):
        with open(f'{eval_path}/{save_file_name}.json', 'r') as f:
            all_tasks_success_rate = json.load(f)
    else:
        all_tasks_success_rate = dict()

    for task_name in tasks:

        if task_name in all_tasks_success_rate:
            continue

        video_path = f"{eval_path}/video/{task_name}"
        os.makedirs(video_path, exist_ok=True)

        if "google" in task_name:
            policy_setup = "google_robot"
        else:
            policy_setup = "widowx_bridge"

        # reduce the number of model loading
        if policy_setup != previous_policy_setup:
            model = load_model(model_name, model_path, policy_setup, seed, step=checkpoint_step)
        previous_policy_setup = policy_setup

        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env

        env_name, total_runs, options = tasks[task_name]
        if env_name is not None:
            kwargs = dict()
            kwargs["prepackaged_config"] = True
            env = gym.make(env_name, obs_mode="rgbd", **kwargs)
        else:
            env = simpler_env.make(task_name)

        # turned off the denoiser as the colab kernel will crash if it's turned on
        sapien.render_config.rt_use_denoiser = False

        print (f'===== {task_name} =====')
        obs, reset_info = env.reset(seed=seed)
        instruction = env.get_language_instruction()

        success_count = 0
        episode_results = []
        for run in range(total_runs):
            if options is not None:
                obs, reset_info = env.reset(options=options[run])
            else:
                obs, reset_info = env.reset()

            instruction = env.get_language_instruction()
            is_final_subtask = env.is_final_subtask() 

            model.reset(instruction)
            print (instruction)

            image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
            images = [image]
            predicted_terminated, success, truncated = False, False, False
            while not (truncated or success):
                # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
                raw_action, action = model.step(image, instruction)
                predicted_terminated = bool(action["terminate_episode"][0] > 0)
                if predicted_terminated:
                    if not is_final_subtask:
                        # advance the environment to the next subtask
                        predicted_terminated = False
                        env.advance_to_next_subtask()

                obs, reward, success, truncated, info = env.step(
                    np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                )

                new_instruction = env.get_language_instruction()
                if new_instruction != instruction:
                    # update instruction for long horizon tasks
                    instruction = new_instruction
                    print (instruction)
                is_final_subtask = env.is_final_subtask() 
                # update image observation
                image = get_image_from_maniskill2_obs_dict(env, obs)
                images.append(image)
            if success:
                success_count += 1
            episode_results.append(success)
            print(run+1, success_count, success_count/(run+1)*100)
            if save_video:
                result = 'success' if success else 'fail'
                mediapy.write_video(f'{video_path}/{run + 1}_{result}.mp4', images, fps=10)
        env.close()
        all_tasks_success_rate[task_name] = [success_count / total_runs, episode_results]
        print ({key: all_tasks_success_rate[key][0] for key in all_tasks_success_rate})
        with open(f'{eval_path}/{save_file_name}.json', 'w') as f:
            json.dump(all_tasks_success_rate, f)



if __name__ == '__main__':

    # Add arguments
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("--model", choices=["octo-small", "octo-base", "octo-custom", "rt_1_x", "rt_1_400k"], default="octo-custom", help="The model used for evaluation")
    parser.add_argument("--model_path", type=str, default='', help="The path of the custom model (only useful for octo-custom?)")
    parser.add_argument("--seeds", type=str, default='0+1+2+3', help="seeds for policy and env")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step to evaluate")
    parser.add_argument("--split", type=str, default='train', help="evaluate on the train or test split")
    parser.add_argument("--save_video", action='store_true', help="save evaluation video or not")
    # Parse the arguments
    args = parser.parse_args()

    if args.split == 'test':
        num_eval_per_setting_for_move = 5
        source_target_id = np.array([4, 5, 10, 11])
        episode_ids = np.concatenate([source_target_id + i * 12 for i in range(4)])
        episode_ids = np.concatenate([episode_ids, np.arange(48, 60)])
        move_task_options = []
        for i in episode_ids:
            move_task_options.extend([{"obj_init_options": {"episode_id": i}}] * num_eval_per_setting_for_move)
        # for each task, the three args corresponds to the env register name (None for using the task name), 
        # total number of evaluation, and options for each evaluation episode
        tasks = {
            "google_robot_pick_coke_can": (None, 20, None), 
            "google_robot_pick_apple": ("GraspSingleAppleInScene-v0", 20, None), 
            "google_robot_pick_spoon": ("GraspSingleBridgeSpoonInScene-v0", 20, None), 
            "google_robot_pick_cube": ("GraspSingleGreenCubeInScene-v0", 20, None), 
            "google_robot_move_near": (None, len(move_task_options), move_task_options),
            "google_robot_close_middle_drawer": (None, 50, None),
            # "google_robot_open_bottom_drawer": (None, 50),
        }
    elif args.split == 'train':
        # options for the move task
        num_eval_per_setting_for_move = 5
        source_target_id = np.array([0, 1, 2, 3, 6, 7, 8, 9])
        episode_ids = np.concatenate([source_target_id + i * 12 for i in range(4)])
        move_task_options = []
        for i in episode_ids:
            move_task_options.extend([{"obj_init_options": {"episode_id": i}}] * num_eval_per_setting_for_move)
        # options for the pick task
        # reset_options = [
        #     "opened_pepsi_can",
        #     # "opened_coke_can",
        #     "opened_sprite_can",
        #     "opened_fanta_can",
        #     "opened_redbull_can",
        #     "blue_plastic_bottle",
        #     # "apple",
        #     "orange",
        #     "sponge",
        #     # "bridge_spoon_generated_modified",
        #     "bridge_carrot_generated_modified",
        #     # "green_cube_3cm",
        #     "yellow_cube_3cm",
        #     "eggplant",
        # ]
        # num_demo_per_option = 10
        # define tasks
        tasks = {
            "google_robot_pick_object": ("GraspSingleRandomTrainObjectInScene-v0", 50, None), 
            "google_robot_move_near": (None, len(move_task_options), move_task_options),
            "google_robot_close_top_drawer": (None, 25, None),
            "google_robot_close_bottom_drawer": (None, 25, None),
            # "google_robot_open_top_drawer": (None, 25),
            # "google_robot_open_middle_drawer": (None, 25),
        }

    seeds = [eval(seed) for seed in args.seeds.split('+')]
    for seed in seeds:
        evaluate(args.model, args.model_path, tasks, seed=seed, checkpoint_step=args.step, split=args.split, save_video=args.save_video)
