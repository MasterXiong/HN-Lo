import os
import multiprocessing

finished_tasks = [
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet", 
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate", 
    "KITCHEN_SCENE3_turn_on_the_stove", 
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack", 
    "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket", 
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate", 
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy", 
    "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy", 
    "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet", 
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug", 
    "KITCHEN_SCENE7_open_the_microwave", 
    "KITCHEN_SCENE8_turn_off_the_stove", 
    "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket", 
    "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray", 
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy", 
    "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf", 
]


def run_task_on_gpu(gpu_id, tasks, setup):
    for task in tasks:
        command = f'CUDA_VISIBLE_DEVICES={gpu_id} python ./scripts/finetune.py \
            --config=scripts/configs/finetune_config.py:head_only,language_conditioned \
            --config.pretrained_path=hf://rail-berkeley/octo-base-1.5 \
            --config.dataset_kwargs.name={setup}/{task} \
            --config.save_dir=/user/octo/finetune_saves/{setup}/{task} \
            --name=vanilla_lora_48_rank_100k_steps_32_batch \
            --config.finetuning_mode=hypernet --config.hypernet_kwargs.lora_type=vanilla \
            --config.hypernet_kwargs.lora_rank=48 --config.batch_size=32 \
            --config.num_steps=100000 --config.save_interval=50000 \
            --config.dataset_kwargs.standardize_fn octo.data.oxe.oxe_standardization_transforms:libero_dataset_transform \
            --debug'
        os.system(command)


def distribute_and_run_tasks(setup, num_gpus=8):
    tasks = sorted(os.listdir(f'data/{setup}'))
    for task in finished_tasks:
        tasks.remove(task)

    # Use multiprocessing to execute tasks on all GPUs simultaneously
    processes = []
    for gpu_id in range(num_gpus):
        tasks_to_run = tasks[gpu_id::num_gpus]
        p = multiprocessing.Process(target=run_task_on_gpu, args=(gpu_id, tasks_to_run, setup))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()



if __name__ == "__main__":

    distribute_and_run_tasks('libero_single_task_1_demos')
