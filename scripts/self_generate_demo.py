import argparse
import numpy as np
import os
import shutil
import tensorflow as tf

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien

import gymnasium as gym

# prevent a single jax process from taking up all the GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    # prevent a single tf process from taking up all the GPU memory
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3072)],
    )


def load_model(model_name, model_path, policy_setup, seed=0):
    if "rt_1" in model_name:
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        # ckpt_path = get_rt_1_checkpoint(model_name)
        model = RT1Inference(saved_model_path=model_path, policy_setup=policy_setup)
    elif "octo" in model_name:
        from octo.simpler_new.octo_model import OctoInference
        if 'hypernet' in model_path or 'vanilla_lora' in model_path:
            from octo.model_lora.octo_model import OctoModel
        else:
            from octo.model.octo_model import OctoModel
        tempmodel = OctoModel.load_pretrained(model_path)
        model = OctoInference(model=tempmodel, policy_setup=policy_setup, init_rng=seed)
    else:
        raise ValueError(model_name)
    return model

def generate_demos(model_name, model_path, tasks, split='train', seed=0, num_of_successes=100, total_attempts=200):

    previous_policy_setup = ''
    for task_name in tasks:

        if task_name == 'google_robot_move_near':
            source_target_id = np.array([0, 1, 2, 3, 6, 7, 8, 9])
            episode_ids = np.concatenate([source_target_id + i * 12 for i in range(4)])
            reset_options = episode_ids
            num_demo_per_option = 5
            max_trial_per_option = 150
        elif task_name == 'google_robot_pick_object':
            reset_options = [
                "opened_pepsi_can",
                # "opened_coke_can",
                "opened_sprite_can",
                "opened_fanta_can",
                "opened_redbull_can",
                "blue_plastic_bottle",
                # "apple",
                "orange",
                "sponge",
                # "bridge_spoon_generated_modified",
                "bridge_carrot_generated_modified",
                # "green_cube_3cm",
                "yellow_cube_3cm",
                "eggplant",
            ]
            num_demo_per_option = 10
            max_trial_per_option = 400

        os.makedirs(f"data/self_generated_demo/{model_name}/{task_name}/{split}", exist_ok=True)

        if "google" in task_name:
            policy_setup = "google_robot"
        else:
            policy_setup = "widowx_bridge"

        # reduce the number of model loading
        if policy_setup != previous_policy_setup:
            model = load_model(model_name, model_path, policy_setup, seed=seed)
        previous_policy_setup = policy_setup

        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env

        if task_name == 'google_robot_pick_object':
            kwargs = dict()
            kwargs["prepackaged_config"] = True
            env = gym.make("GraspSingleCustomOrientationInScene-v0", obs_mode="rgbd", **kwargs)
        else:
            env = simpler_env.make(task_name)

        # turned off the denoiser as the colab kernel will crash if it's turned on
        sapien.render_config.rt_use_denoiser = False

        print (f'===== {task_name} =====')
        seed = int(seed + 1e8) # add an offset to the seed to avoid overlapping with the evaluation seed
        obs, reset_info = env.reset(seed=seed)
        instruction = env.get_language_instruction()

        success_count = 0
        total_count = 0
        current_option = 0
        current_option_trial_num = 0
        current_option_success_num = 0
        while True:
            if task_name == 'google_robot_move_near':
                episode_id = reset_options[current_option]
                options = {"obj_init_options": {"episode_id": episode_id}}
                obs, reset_info = env.reset(options=options)
            elif task_name == 'google_robot_pick_object':
                object_id = reset_options[current_option]
                options = {"model_id": object_id}
                obs, reset_info = env.reset(options=options)
            else:
                obs, reset_info = env.reset()
            instruction = env.get_language_instruction()
            is_final_subtask = env.is_final_subtask()

            # if task_name == 'google_robot_pick_object':
            #     # do not rollout on hold-out test objects
            #     test_objects = ['opened_coke_can', 'apple', 'bridge_spoon_generated_modified', 'green_cube_3cm']
            #     if reset_info['model_id'] in test_objects:
            #         continue

            model.reset(instruction)
            print (instruction)

            image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
            images = [image]
            predicted_terminated, success, truncated = False, False, False
            timestep = 0
            episode = []

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

                if "rt_1" in model_name:
                    singlestep = {'image': np.asarray(image, dtype=np.uint8), 'action': raw_action, 'language_instruction': instruction}
                else:
                    tempact = np.concatenate([raw_action['world_vector'], raw_action["rotation_delta"], raw_action['open_gripper'], [int((predicted_terminated or truncated or success))]])
                    singlestep = {'image': np.asarray(image, dtype=np.uint8), 'action': np.asarray(tempact, dtype=np.float32), 'language_instruction': instruction}
                episode.append(singlestep)
                
                image = get_image_from_maniskill2_obs_dict(env, obs)
                images.append(image)
                timestep += 1

            if success:
                if task_name == 'google_robot_move_near' or task_name == 'google_robot_pick_object':
                    episode_id = current_option * num_demo_per_option + current_option_success_num
                    current_option_success_num += 1
                else:
                    episode_id = success_count
                    success_count += 1
                np.save(f"data/self_generated_demo/{model_name}/{task_name}/{split}/episode_{episode_id}.npy", episode)

            if task_name == 'google_robot_move_near' or task_name == 'google_robot_pick_object':
                current_option_trial_num += 1
                print(f'{current_option_success_num} of {current_option_trial_num} episodes success for the current option')
                if current_option_success_num == num_demo_per_option or current_option_trial_num == max_trial_per_option:
                    current_option += 1
                    current_option_trial_num = 0
                    current_option_success_num = 0
                if current_option == len(reset_options):
                    break
            else:
                total_count += 1
                print(f'{success_count} of {total_count} episodes success')
                if success_count == num_of_successes or total_count == total_attempts:
                    # temparr = [totalruns, successes, successes/totalruns]
                    # shutil.make_archive(f'{base_path}/{model_name}_episodes/{task_name}', 'zip', f'{base_path}/{model_name}_episodes/{task_name}')
                    # np.savetxt(f"{base_path}/{model_name}_episodes/{task_name}.txt", temparr, newline=", ")
                    break
            
        env.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="A simple example of argparse")
    parser.add_argument("--model", choices=["octo-small", "octo-base", "octo-custom", "rt_1_x", "rt_1_400k"], default="octo-custom", help="The model used for evaluation")
    parser.add_argument("--model_path", type=str, default='', help="The path of the custom model (only useful for octo-custom?)")
    parser.add_argument("--tasks", type=str, default='google_robot_pick_coke_can', help="tasks to run, connected by +")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the environment and policy")
    parser.add_argument("--split", type=str, default='train', help="train or validation split to generate")
    parser.add_argument("--num_of_successes", type=int, default=200, help="Number of successful episodes to collect")
    parser.add_argument("--total_attempts", type=int, default=4000, help="Total number of attempts to make")

    args = parser.parse_args()
    tasks = args.tasks.split('+')
    generate_demos(args.model, args.model_path, tasks, seed=args.seed, split=args.split, num_of_successes=args.num_of_successes, total_attempts=args.total_attempts)
