"""Data-parallel evaluation runner for vLLM SDK.

Spawns N processes, each bound to one GPU via CUDA_VISIBLE_DEVICES,
and runs EvalScope `run_task` with the vLLM OpenAI SDK adapter.

Usage:
    python examples/run_dp_eval.py --dp 8 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --datasets cmmlu ceval \
        --max_tokens 1024 --temperature 0.0 --top_p 0.9
"""

import argparse
import multiprocessing as mp
import os
from typing import Dict

from evalscope import TaskConfig, run_task


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dp', type=int, default=2, help='Number of GPUs/processes')
    p.add_argument('--model', type=str, required=True, help='Local model path or model ID')
    p.add_argument('--datasets', type=str, nargs='+', required=True, help='Dataset names')
    p.add_argument('--max_tokens', type=int, default=1024)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top_p', type=float, default=None)
    p.add_argument('--top_k', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=1)
    return p.parse_args()


def worker(rank: int, cfg_dict: Dict):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    # Each rank can optionally use a unique work_dir
    cfg_dict['work_dir'] = os.path.join(cfg_dict.get('work_dir', './outputs'), f'rank{rank}')
    run_task(TaskConfig(**cfg_dict))


def main():
    args = parse_args()
    cfg = TaskConfig(
        model=args.model,
        eval_type='vllm_openai',
        datasets=args.datasets,
        eval_batch_size=args.batch_size,
        generation_config={
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
        },
    )

    cfg_dict = cfg.to_dict()
    with mp.get_context('spawn').Pool(processes=args.dp) as pool:
        for rank in range(args.dp):
            pool.apply_async(worker, args=(rank, dict(cfg_dict)))
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()

