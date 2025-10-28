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
    p.add_argument('--max_tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, default=0.75)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--top_k', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--model', type=str, 
        default="/sharedata/zimoliu/ckpts/jamba_60b_aws_oh_pp8_ep4_efa_512k_sft_v1_16node_ckpt75000/hf")
    p.add_argument('--model_alias', type=str, default='zm60b', help="Name of the model version evaluated. Set to be required for reusing cache and saving time.")
    p.add_argument('--balance', action='store_true', help='Greedy balance tasks across GPUs by estimated dataset size')

    return p.parse_args()


def worker(gpu_id: int, task_list: List[TaskConfig]):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['EVALSCOPE_TQDM_POSITION'] = str(gpu_id)

    # Initialize a single vLLM engine once per GPU and reuse it across tasks
    from evalscope.models.vllm_openai import VllmOpenAIAPI
    base_model = task_list[0].model if isinstance(task_list[0].model, str) else task_list[0].model_id
    base_args = getattr(task_list[0], 'model_args', {}) or {}
    # Force a consistent dtype policy to avoid mixed bf16/fp16 issues
    base_args = dict(base_args)
    base_args.setdefault('precision', 'auto')  # mapped to vLLM dtype='auto'
    api = VllmOpenAIAPI(model_name=str(base_model), **base_args)

    for idx, cfg in enumerate(task_list):
        # use alias as key for cache reuse
        if cfg.model_alias:
            cfg.use_cache = os.path.join(cfg.work_dir, cfg.model_alias)
        # force SDK path and reuse the initialized API
        cfg.eval_type = 'vllm_openai'
        cfg.model = api
        run_task(cfg)


def main():
    args = parse_args()

    # Define your tasks here: each TaskConfig will run on its own GPU
    # Example with two tasks; add more entries as needed.
    tasks: List[TaskConfig] = [
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            datasets=['cmmlu'],
            eval_batch_size=args.batch_size,
            dataset_args={
                'cmmlu': {'few_shot_num': 0, 'subset_list': ['college_mathematics', 'high_school_mathematics']},
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
            datasets=['math_500'],
            dataset_args={
                'math_500': {'few_shot_num': 0},
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens*16,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
            repeats=4
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            datasets=['aime24'],
            dataset_args={
                'aime24': {'few_shot_num': 0},
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens*16,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
            repeats=8
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            datasets=['aime25'],
            dataset_args={
                'aime25': {'few_shot_num': 0},
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens*16,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
            repeats=8
        ),
        TaskConfig(
            model=args.model,
            model_alias=args.model_alias,
            datasets=['amc'],
            dataset_args={
                'amc': {'few_shot_num': 0},
            },
            eval_batch_size=args.batch_size,
            generation_config={
                'max_tokens': args.max_tokens*16,
                'temperature': args.temperature,
                'top_p': args.top_p,
            },
            repeats=8
        ),
    ]

    world_size = max(1, int(args.gpus))
    # Distribute tasks across GPUs
    per_gpu_tasks: List[List[TaskConfig]] = [[] for _ in range(world_size)]

    if args.balance:
        # Greedy bin-packing by estimated cost (dataset size * repeats)
        def estimate_task_cost(cfg: TaskConfig) -> int:
            try:
                from evalscope.api.registry import get_benchmark
                # sum lengths across all subsets for the first dataset name
                total = 0
                for name in cfg.datasets:
                    adapter = get_benchmark(name, cfg)
                    ds = adapter.load_dataset()
                    total += sum(len(subset) for subset in ds.values())
                # factor in repeats if set
                r = int(getattr(cfg, 'repeats', 1) or 1)
                return total * max(1, r)
            except Exception:
                return 1

        tasks_by_cost = sorted(tasks, key=estimate_task_cost, reverse=True)
        loads = [0 for _ in range(world_size)]
        for t in tasks_by_cost:
            # assign to GPU with minimal current load
            gpu_id = min(range(world_size), key=lambda i: loads[i])
            per_gpu_tasks[gpu_id].append(t)
            loads[gpu_id] += estimate_task_cost(t)
    else:
        # round-robin
        for i, t in enumerate(tasks):
            per_gpu_tasks[i % world_size].append(t)

    procs = []
    for gpu_id, task_list in enumerate(per_gpu_tasks):
        if not task_list:
            continue
        p = mp.get_context('spawn').Process(target=worker, args=(gpu_id, task_list))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == '__main__':
    main()
