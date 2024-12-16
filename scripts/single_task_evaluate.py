import os
import argparse


def evaluate(setup, step_num=100000):
    for task in sorted(os.listdir(f'finetune_saves/{setup}')):
        if not os.path.exists(f'finetune_saves/{setup}/{task}/octo_finetune'):
            continue
        model_paths = sorted(os.listdir(f'finetune_saves/{setup}/{task}/octo_finetune'))
        for model_path in model_paths:
            path = f'finetune_saves/{setup}/{task}/octo_finetune/' + model_path
            if not os.path.exists(f'{path}/{step_num}'):
                continue
            command = f'python octo/domains/LIBERO/evaluate.py \
                --model octo-custom --model_path {path} \
                --step {step_num} --split single_task --seed 0 --action_ensemble --image_horizon 1'
            os.system(command)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=str, default="libero_single_task")
    parser.add_argument('--step_num', type=int, default=100000)
    args = parser.parse_args()

    evaluate(args.setup, args.step_num)
