"""Data-parallel runner for vLLM SDK (eval_type='vllm_openai').

Each process sets CUDA_VISIBLE_DEVICES to its rank, initializes vLLM LLM
like examples/test_chat.py, and runs EvalScope's run_task.

Note: This script does not auto-shard datasets; for large runs, pre-split
datasets externally or customize per-rank selection.
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

    # Build distinct cfg dicts per rank (separate work_dir, optional hints)
    base = cfg.to_dict()
    cfg_list = []
    for rank in range(args.dp):
        c = dict(base)
        c['work_dir'] = os.path.join(base.get('work_dir', './outputs'), f'rank{rank}')
        # optional: pass rank/world_size for custom dataset sharding in adapters if supported
        c.setdefault('eval_config', {})
        if isinstance(c['eval_config'], dict):
            c['eval_config'].update({'dp_world_size': args.dp, 'dp_rank': rank})
        cfg_list.append(c)

    with mp.get_context('spawn').Pool(processes=args.dp) as pool:
        for rank, cdict in enumerate(cfg_list):
            pool.apply_async(worker, args=(rank, cdict))
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
