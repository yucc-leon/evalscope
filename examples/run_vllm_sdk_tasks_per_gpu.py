"""Run each eval task on a separate GPU using vLLM SDK (eval_type='vllm_openai').

This example maps TaskConfigs to GPUs one-to-one. If you have more tasks than
GPUs, tasks are executed in waves. Modify the `tasks` list to fit your needs.
"""

import argparse
import multiprocessing as mp
import os
from typing import List

from evalscope import TaskConfig, run_task


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', type=int, default=2, help='Number of GPUs to use (0..gpus-1)')
    p.add_argument('--max_tokens', type=int, default=2048)
    p.add_argument('--temperature', type=float, default=0.75)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--eval_type', type=str, default='vllm_openai')
    p.add_argument('--top_k', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--model', type=str, 
        default="/sharedata/zimoliu/ckpts/jamba_60b_aws_oh_pp8_ep4_efa_512k_sft_v1_16node_ckpt75000/hf")
    p.add_argument('--model_alias', type=str, default='zm60b', help="Name of the model version evaluated. Set to be required for reusing cache and saving time.")

    return p.parse_args()


def worker(gpu_id: int, cfg: TaskConfig):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # configure tqdm line position per GPU to avoid overlapping bars
    os.environ['EVALSCOPE_TQDM_POSITION'] = str(gpu_id)
    # isolate outputs per task/gpu to avoid collisions
    # cfg.work_dir = os.path.join(cfg.work_dir, f'gpu{gpu_id}')
    if cfg.model_alias:
        cfg.use_cache = os.path.join(cfg.work_dir, cfg.model_alias)
    
    run_task(cfg)


def main():
    args = parse_args()

    # Define your tasks here: each TaskConfig will run on its own GPU
    # Example with two tasks; add more entries as needed.
    tasks: List[TaskConfig] = [
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            eval_type=args.eval_type,
            datasets=['cmmlu'],
            eval_batch_size=args.batch_size,
            dataset_args={
                # 'gsm8k': {'few_shot_num': 0},
                # 'competition_math': {'few_shot_num': 0},
                'cmmlu': {'few_shot_num': 0, 'subset_list': ['college_mathematics', 'high_school_mathematics']},
                # 'ceval': {'few_shot_num': 0, 'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']}
            },
            generation_config={
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
            repeats=4
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            eval_type=args.eval_type,
            datasets=['ceval'],
            eval_batch_size=args.batch_size,
            dataset_args={'ceval': {'few_shot_num': 0, 'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']}},
            generation_config={
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p
            },
            repeats=4
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            eval_type=args.eval_type,
            datasets=['gsm8k'],
            dataset_args={
                'gsm8k': {'few_shot_num': 0},
                # 'competition_math': {'few_shot_num': 0},
                # 'cmmlu': {'few_shot_num': 0, 'subset_list': ['college_mathematics', 'high_school_mathematics']},
                # 'ceval': {'few_shot_num': 0, 'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']}
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            eval_type=args.eval_type,
            datasets=['competition_math'],
            dataset_args={
                'competition_math': {'few_shot_num': 0},
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens*4,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            eval_type=args.eval_type,
            datasets=['math_500'],
            dataset_args={
                'math_500': {'few_shot_num': 0},
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens*4,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
            repeats=4
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            eval_type=args.eval_type,
            datasets=['aime24'],
            dataset_args={
                'aime24': {'few_shot_num': 0},
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens*4,
                'temperature': args.temperature,
                'top_p': args.top_p
            },
            repeats=8
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            eval_type=args.eval_type,
            datasets=['aime25'],
            dataset_args={
                'aime25': {'few_shot_num': 0},
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens*4,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
            repeats=8
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            eval_type=args.eval_type,
            datasets=['amc'],
            dataset_args={
                'amc': {'few_shot_num': 0},
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens*4,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
            repeats=8
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            eval_type=args.eval_type,
            datasets=['minerva_math'],
            dataset_args={
                'minerva_math': {'few_shot_num': 0},
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens*4,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
            repeats=8
        ),
    ]

    world_size = max(1, int(args.gpus))
    idx = 0
    while idx < len(tasks):
        procs = []
        # launch up to `world_size` tasks in parallel
        for local_rank in range(world_size):
            if idx >= len(tasks):
                break
            gpu_id = local_rank  # simple mapping: one task per GPU id
            p = mp.get_context('spawn').Process(target=worker, args=(gpu_id, tasks[idx]))
            p.start()
            procs.append(p)
            idx += 1
        for p in procs:
            p.join()


if __name__ == '__main__':
    main()
