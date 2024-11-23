import os


# List of tasks to train on
tasks=[
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet", 
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
    "KITCHEN_SCENE3_turn_on_the_stove",
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
    "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy"
]

step_num = 100000
# num_parallel = 4
setup = 'libero_single_task_10_demos'

for task in tasks:
    model_path = os.listdir(f'finetune_saves/{setup}/{task}/octo_finetune')[0]
    model_path = f'finetune_saves/{setup}/{task}/octo_finetune/' + model_path
    command = f'python octo/domains/LIBERO/evaluate.py \
        --model octo-custom --model_path {model_path} \
        --step {step_num} --split single_task --seed 0 --action_ensemble --image_horizon 1'
    os.system(command)
