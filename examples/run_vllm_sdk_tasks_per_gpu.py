"""Run each eval task on a separate GPU using vLLM SDK (eval_type='vllm_openai').

This example maps TaskConfigs to GPUs one-to-one. If you have more tasks than
GPUs, tasks are executed in waves. Modify the `tasks` list to fit your needs.
"""

import argparse
import atexit
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from typing import List

from evalscope import TaskConfig, run_task
from evalscope.utils.logger import get_logger


logger = get_logger()


MIN_FREE_MEM_MB_DEFAULT = 112640  # ~110 GB
MAX_UTIL_PERCENT_DEFAULT = 10     # <=10% util considered idle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max_use_gpu', type=int, default=0, help='Maximum number of GPUs to use (defaults to total GPUs)')
    p.add_argument('--max_tokens', type=int, default=2048)
    p.add_argument('--temperature', type=float, default=0.75)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--top_k', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--model', type=str, 
        default="/sharedata/zimoliu/ckpts/jamba_60b_aws_oh_pp8_ep4_efa_512k_sft_v1_16node_ckpt75000/hf")
    p.add_argument('--model_alias', type=str, default='zm60b', help="Name of the model version evaluated. Set to be required for reusing cache and saving time.")
    p.add_argument('--balance', action='store_true', default=True,
                   help='Greedy balance tasks across GPUs by estimated dataset size (default: on)')
    p.add_argument('--auto', action='store_true', default=True,
                   help='Auto-detect available GPUs and schedule tasks dynamically (default: on)')
    p.add_argument('--task_set', type=str, choices=['quick', 'easy', 'difficult', 'all'], default='all',
                   help='Select which task set to run')
    p.add_argument('--enforce_eager', action='store_true', default=False,
                   help='Enable vLLM engine enforce_eager for eager execution')
    p.add_argument('--debug', action='store_true', default=False,
                   help='Enable debug logging in EvalScope pipeline')
    return p.parse_args()


def worker(gpu_id: int, task_list: List[TaskConfig], rank: int = 0):
    # Put worker in its own process group so we can
    # terminate the entire subtree without touching others.
    try:
        os.setpgrp()
    except Exception:
        pass
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Use per-process rank for tqdm position to keep small indices (0..N-1)
    os.environ['EVALSCOPE_TQDM_POSITION'] = str(rank)

    # Initialize models per GPU once, then dynamically dispatch samples via a shared queue
    from evalscope.models.vllm_openai import VllmOpenAIAPI
    # Build a unique set of models to initialize (by model string or id)
    unique_models = []
    for t in task_list:
        m = t.model if isinstance(t.model, str) else t.model_id
        if m not in unique_models:
            unique_models.append(m)
    apis = {}
    for m in unique_models:
        # Extract args from the first task using this model
        first = next(cfg for cfg in task_list if (cfg.model if isinstance(cfg.model, str) else cfg.model_id) == m)
        base_args = dict(getattr(first, 'model_args', {}) or {})
        base_args.setdefault('gpu_memory_utilization', 0.9)
        logger.info(f"[GPU {gpu_id}] Initializing VllmOpenAIAPI for model: {m}")
        apis[m] = VllmOpenAIAPI(model_name=str(m), **base_args)

    # Dispatch tasks sequentially but reuse the API per model
    for idx, cfg in enumerate(task_list):
        if cfg.model_alias:
            cfg.use_cache = os.path.join(cfg.work_dir, cfg.model_alias)
        mkey = cfg.model if isinstance(cfg.model, str) else cfg.model_id
        cfg.model = apis[mkey]
        run_task(cfg)


# Track processes we start so Ctrl-C can cleanly terminate only our children
_PROCS: List[mp.Process] = []


def _signal_proc_tree(pid: int, sig: int) -> None:
    """Signal a process group if available; otherwise the single process."""
    try:
        os.killpg(pid, sig)
    except Exception:
        try:
            os.kill(pid, sig)
        except Exception:
            pass


def _terminate_children(timeout: float = 15.0) -> None:
    """Gracefully terminate only processes started by this script."""
    if not _PROCS:
        return
    # Politely ask children to exit
    for p in list(_PROCS):
        try:
            if p.is_alive():
                _signal_proc_tree(p.pid, signal.SIGTERM)
        except Exception:
            pass
    # Wait up to timeout
    end = time.time() + max(0.0, timeout)
    for p in list(_PROCS):
        try:
            remaining = max(0.0, end - time.time())
            if p.is_alive():
                p.join(timeout=remaining)
        except Exception:
            pass
    # Force kill any stubborn children
    for p in list(_PROCS):
        try:
            if p.is_alive():
                _signal_proc_tree(p.pid, signal.SIGKILL)
                p.join(timeout=2.0)
        except Exception:
            pass


def _install_signal_handlers() -> None:
    def _handler(signum, frame):
        logger.warning(f"Received signal {signum}; terminating child processes...")
        _terminate_children(timeout=10.0)
        if signum == signal.SIGINT:
            sys.exit(130)
        sys.exit(128 + signum)

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def _estimate_task_cost(cfg: TaskConfig) -> int:
    """Estimate task cost by dataset size * repeats."""
    try:
        from evalscope.api.registry import get_benchmark
        total = 0
        for name in cfg.datasets:
            adapter = get_benchmark(name, cfg)
            ds = adapter.load_dataset()
            total += sum(len(subset) for subset in ds.values())
        r = int(getattr(cfg, 'repeats', 1) or 1)
        return total * max(1, r)
    except Exception:
        return 1


def _assign_tasks(tasks: List[TaskConfig], world_size: int, balance: bool) -> List[List[TaskConfig]]:
    """Return per-GPU task lists using greedy balance or round-robin."""
    per_gpu: List[List[TaskConfig]] = [[] for _ in range(world_size)]
    if balance:
        tasks_by_cost = sorted(tasks, key=_estimate_task_cost, reverse=True)
        loads = [0 for _ in range(world_size)]
        for t in tasks_by_cost:
            idx = min(range(world_size), key=lambda i: loads[i])
            per_gpu[idx].append(t)
            loads[idx] += _estimate_task_cost(t)
    else:
        for i, t in enumerate(tasks):
            per_gpu[i % world_size].append(t)
    return per_gpu


def _query_gpus() -> List[dict]:
    """Query GPU status via nvidia-smi."""
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=index,memory.free,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], encoding='utf-8')
        gpus = []
        for line in out.strip().splitlines():
            idx, mem_free, mem_total, util = [s.strip() for s in line.split(',')]
            gpus.append({
                'id': int(idx),
                'mem_free_mb': int(mem_free),
                'mem_total_mb': int(mem_total),
                'util_percent': int(util),
            })
        return gpus
    except Exception:
        # Fallback: use torch to count devices
        try:
            import torch
            return [{'id': i, 'mem_free_mb': 0, 'mem_total_mb': 0, 'util_percent': 0} for i in range(torch.cuda.device_count())]
        except Exception:
            return []


def _select_idle_gpus(count: int) -> List[int]:
    """Select `count` idle GPUs based on thresholds.

    Returns a list of GPU ids sorted by free memory (desc). Raises AssertionError if
    there aren't enough idle GPUs.
    """
    gpus = _query_gpus()
    candidates = [
        g for g in gpus if g['mem_free_mb'] >= MIN_FREE_MEM_MB_DEFAULT and g['util_percent'] <= MAX_UTIL_PERCENT_DEFAULT
    ]
    assert len(candidates) >= count, (
        f'Not enough idle GPUs: requested {count}, available {len(candidates)} with '
        f'free_mem>={MIN_FREE_MEM_MB_DEFAULT}MB and util<={MAX_UTIL_PERCENT_DEFAULT}%.'
    )
    candidates.sort(key=lambda x: x['mem_free_mb'], reverse=True)
    return [g['id'] for g in candidates[:count]]


def _build_tasks(args) -> List[TaskConfig]:
    """Build fixed TaskConfigs and filter by args.task_set (quick/easy/difficult/all)."""
    base = dict(
        model=args.model,
        model_alias=args.model_alias,
        eval_type='vllm_openai',
        eval_batch_size=args.batch_size,
        model_args={'enforce_eager': args.enforce_eager},
    )

    defs = [
        dict(id='cmmlu', diff='quick', 
             cfg=TaskConfig(**base,
                            datasets=['cmmlu'],
                            dataset_args={'cmmlu': {'few_shot_num': 0, 'subset_list': ['college_mathematics', 'high_school_mathematics']}},
                            generation_config={'max_tokens': args.max_tokens, 'temperature': args.temperature, 'top_p': args.top_p},
                            debug=args.debug,
                            )),
                            # repeats=4)),
        dict(id='ceval', diff='quick', 
             cfg=TaskConfig(**base,
                            datasets=['ceval'],
                            dataset_args={'ceval': {'few_shot_num': 0, 'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']}},
                            generation_config={'max_tokens': args.max_tokens, 'temperature': args.temperature, 'top_p': args.top_p},
                            debug=args.debug,
                            )),
                            # repeats=4)),
        dict(id='gsm8k', diff='easy', 
             cfg=TaskConfig(**base,
                            datasets=['gsm8k'],
                            dataset_args={'gsm8k': {'few_shot_num': 0}},
                            generation_config={'max_tokens': args.max_tokens, 'temperature': args.temperature, 'top_p': args.top_p},
                            debug=args.debug,
                            )),
        dict(id='amc', diff='difficult', 
             cfg=TaskConfig(**base,
                            datasets=['amc'],
                            dataset_args={'amc': {'few_shot_num': 0}},
                            generation_config={'max_tokens': max(4096, args.max_tokens), 'temperature': args.temperature, 'top_p': args.top_p},
                            debug=args.debug,
                            )),
        dict(id='competition_math', diff='difficult', 
             cfg=TaskConfig(**base,
                            datasets=['competition_math'],
                            dataset_args={'competition_math': {'few_shot_num': 0}},
                            generation_config={'max_tokens': max(4096, args.max_tokens), 'temperature': max(0.6, args.temperature), 'top_p': args.top_p},
                            debug=args.debug,
                            )),
        dict(id='math_500', diff='difficult', 
             cfg=TaskConfig(**base,
                            datasets=['math_500'],
                            dataset_args={'math_500': {'few_shot_num': 0}},
                            generation_config={'max_tokens': max(8192, args.max_tokens), 'temperature': max(0.6, args.temperature), 'top_p': args.top_p},
                            debug=args.debug,
                            repeats=4)),
        dict(id='aime24', diff='difficult', 
             cfg=TaskConfig(**base,
                            datasets=['aime24'],
                            dataset_args={'aime24': {'few_shot_num': 0}},
                            generation_config={'max_tokens': max(8192, args.max_tokens), 'temperature': max(0.6, args.temperature), 'top_p': args.top_p},
                            debug=args.debug,
                            repeats=8)),
        dict(id='aime25', diff='difficult', 
             cfg=TaskConfig(**base,
                            datasets=['aime25'],
                            dataset_args={'aime25': {'few_shot_num': 0}},
                            generation_config={'max_tokens': max(8192, args.max_tokens), 'temperature': max(0.6, args.temperature), 'top_p': args.top_p},
                            debug=args.debug,
                            repeats=8)),
    ]
    if args.task_set == 'quick':
        return [d['cfg'] for d in defs if d['diff'] == 'quick']
    if args.task_set == 'easy':
        return [d['cfg'] for d in defs if d['diff'] == 'easy']
    if args.task_set == 'difficult':
        return [d['cfg'] for d in defs if d['diff'] == 'difficult']
    return [d['cfg'] for d in defs]


def main():
    args = parse_args()
    _install_signal_handlers()
    atexit.register(lambda: _terminate_children(timeout=5.0))

    # Build tasks by set selection
    tasks: List[TaskConfig] = _build_tasks(args)

    # Determine default max_use_gpu if not set
    if args.max_use_gpu <= 0:
        try:
            import torch
            args.max_use_gpu = max(1, torch.cuda.device_count())
        except Exception:
            args.max_use_gpu = 1
    world_size = max(1, int(args.max_use_gpu))
    # Distribute tasks across GPUs (auto selects GPUs, balance controls assignment)
    if args.auto:
        world_size = max(1, int(args.max_use_gpu))
        # Soft behavior: if fewer idle GPUs than requested, use available and warn
        try:
            selected_ids = _select_idle_gpus(world_size)
        except AssertionError as e:
            gpus = _query_gpus()
            candidates = [
                g for g in gpus if g['mem_free_mb'] >= MIN_FREE_MEM_MB_DEFAULT and g['util_percent'] <= MAX_UTIL_PERCENT_DEFAULT
            ]
            available = len(candidates)
            print(f"Warning: {e}. Using {available} GPUs instead.")
            if available == 0:
                print("No idle GPUs meet the thresholds; exiting.")
                return
            selected_ids = [g['id'] for g in sorted(candidates, key=lambda x: x['mem_free_mb'], reverse=True)[:available]]
            world_size = available
        per_gpu_tasks = _assign_tasks(tasks, world_size, balance=args.balance)
        logger.info(f"Selected GPUs: {selected_ids}; per-GPU tasks: {[len(t) for t in per_gpu_tasks]}")
        ctx = mp.get_context('spawn')
        procs = []
        for i, task_list in enumerate(per_gpu_tasks):
            if not task_list:
                continue
            gid = selected_ids[i]
            p = ctx.Process(target=worker, args=(gid, task_list, i))
            p.start()
            procs.append(p)
            _PROCS.append(p)
        try:
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            logger.warning('Interrupted; terminating child processes...')
            _terminate_children()
        return

    # Static assignment without auto GPU selection (use 0..world_size-1)
    per_gpu_tasks = _assign_tasks(tasks, world_size, balance=args.balance)

    logger.info(f"Static GPUs: {list(range(world_size))}; per-GPU tasks: {[len(t) for t in per_gpu_tasks]}")
    ctx = mp.get_context('spawn')
    procs = []
    for i, task_list in enumerate(per_gpu_tasks):
        if not task_list:
            continue
        p = ctx.Process(target=worker, args=(i, task_list, i))
        p.start()
        procs.append(p)
        _PROCS.append(p)
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        logger.warning('Interrupted; terminating child processes...')
        _terminate_children()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning('Interrupted by user; terminating child processes...')
        _terminate_children()
        sys.exit(130)
