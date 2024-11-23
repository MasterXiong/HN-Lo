#!/bin/bash

# List of tasks to train on
tasks=(
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet" 
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate" 
    "KITCHEN_SCENE3_turn_on_the_stove" 
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack" 
    "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket" 
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate" 
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy" 
    "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy"
)

# Run tasks in parallel
for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  CUDA_VISIBLE_DEVICES=$i python ./scripts/finetune.py \
    --config=scripts/configs/finetune_config.py:head_only,language_conditioned \
    --config.pretrained_path=hf://rail-berkeley/octo-base-1.5 \
    --config.dataset_kwargs.name=libero_single_task_10_demos/$task \
    --config.save_dir=/user/octo/finetune_saves/libero_single_task_10_demos/$task \
    --name='vanilla_lora_48_rank_100k_steps_32_batch' \
    --config.finetuning_mode='hypernet' --config.hypernet_kwargs.lora_type='vanilla' \
    --config.hypernet_kwargs.lora_rank=48 --config.batch_size=32 \
    --config.num_steps=100000 --config.save_interval=10000 \
    --config.dataset_kwargs.standardize_fn octo.data.oxe.oxe_standardization_transforms:libero_dataset_transform \
    --debug &
done

# Wait for all background tasks to complete
wait

echo "All tasks completed."
