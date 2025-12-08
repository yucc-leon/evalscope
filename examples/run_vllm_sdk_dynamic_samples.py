"""Dynamic sample-level dispatcher for multi-GPU vLLM SDK runs (eval_type='vllm_openai').

This script decouples model engines (interfaces) from data processing by:
- Initializing one vLLM engine per GPU (and per model used)
- Filling a shared queue with sample-level work items across tasks/datasets
- Workers pull sample batches, run batched chat generation, and save caches
- After predictions are complete, run standard evaluators to compute reviews/reports

Usage mirrors run_vllm_sdk_tasks_per_gpu.py but schedules at sample granularity.
"""

from __future__ import annotations

import argparse
import atexit
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from evalscope import TaskConfig
from evalscope.utils.logger import configure_logging, get_logger

logger = get_logger()


MIN_FREE_MEM_MB_DEFAULT = 112640  # ~110 GB
MAX_UTIL_PERCENT_DEFAULT = 10     # <=10% util considered idle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max_use_gpu', type=int, default=0, help='Maximum number of GPUs to use (defaults to total GPUs)')
    p.add_argument('--batch_size', type=int, default=1, help='Evaluation batch size per worker')
    p.add_argument('--max_tokens', type=int, default=2048)
    p.add_argument('--temperature', type=float, default=0.75)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--top_k', type=int, default=None)
    p.add_argument('--model', type=str,
                  default="/sharedata/zimoliu/ckpts/jamba_60b_aws_oh_pp8_ep4_efa_512k_sft_v1_16node_ckpt75000/hf")
    p.add_argument('--model_alias', type=str, default='test',
                   help="Name of the evaluated model version for cache reuse.")
    p.add_argument('--task_set', type=str, choices=['quick', 'easy', 'difficult', 'all'], default='all',
                   help='Select which task set to run')
    p.add_argument('--enforce_eager', action='store_true', default=False,
                   help='Enable vLLM engine enforce_eager for eager execution')
    p.add_argument('--debug', action='store_true', default=False, help='Enable debug logging')
    p.add_argument('--enable_batch', action='store_true', default=False,
                   help='Enable request-level chat batching (may reduce throughput).')
    p.add_argument('--dry_run', action='store_true', default=False,
                   help='Dry-run mode: only load datasets without starting vLLM or using GPU.')
    return p.parse_args()


def _obj_to_dict(obj):
    """Best-effort conversion to a JSON-serializable dict for debug printing."""
    try:
        from dataclasses import is_dataclass

        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    for attr in ('model_dump', 'dict', 'to_dict'):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if hasattr(obj, '__dict__'):
        try:
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        except Exception:
            pass
    return str(obj)


def _debug_log_input_preview(item: dict, msg_list: List) -> None:
    if not item.get('debug') or item.get('global_idx') not in (0, 1):
        return
    msg_repr = [_obj_to_dict(m) for m in msg_list]
    sample_repr = _obj_to_dict(item.get('sample'))
    logger.info(
        (
            f"\n==== DEBUG INPUT PREVIEW SAMPLE #{item.get('global_idx')} ===="
            f"\nDataset: {item.get('dataset_name')} / {item.get('subset_name')}"
            f"\nModel: {item.get('model_name')}"
            f"\n-- Original Sample --\n{json.dumps(sample_repr, ensure_ascii=False, indent=2, default=str)}"
            f"\n-- Input Messages --\n{json.dumps(msg_repr, ensure_ascii=False, indent=2, default=str)}"
            f"\n===============================================\n"
        )
    )


def _debug_log_output_preview(item: dict, msg_list: List, out) -> None:
    if not item.get('debug') or item.get('global_idx') not in (0, 1):
        return
    msg_repr = [_obj_to_dict(m) for m in msg_list]
    sample_repr = _obj_to_dict(item.get('sample'))
    out_repr = {
        'completion': getattr(out, 'completion', None),
        'usage': getattr(out, 'usage', None),
        'raw': _obj_to_dict(out),
    }
    logger.info(
        (
            f"\n==== DEBUG PREVIEW SAMPLE #{item.get('global_idx')} ===="
            f"\nDataset: {item.get('dataset_name')} / {item.get('subset_name')}"
            f"\nModel: {item.get('model_name')}"
            f"\n-- Original Sample --\n{json.dumps(sample_repr, ensure_ascii=False, indent=2, default=str)}"
            f"\n-- Input Messages --\n{json.dumps(msg_repr, ensure_ascii=False, indent=2, default=str)}"
            f"\n-- Output --\n{json.dumps(out_repr, ensure_ascii=False, indent=2, default=str)}"
            f"\n===============================\n"
        )
    )


def _signal_proc_tree(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except Exception:
        try:
            os.kill(pid, sig)
        except Exception:
            pass


_PROCS: List[mp.Process] = []


def _terminate_children(timeout: float = 15.0) -> None:
    if not _PROCS:
        return
    for p in list(_PROCS):
        try:
            if p.is_alive():
                _signal_proc_tree(p.pid, signal.SIGTERM)
        except Exception:
            pass
    end = time.time() + max(0.0, timeout)
    for p in list(_PROCS):
        try:
            remaining = max(0.0, end - time.time())
            if p.is_alive():
                p.join(timeout=remaining)
        except Exception:
            pass
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


def _query_gpus() -> List[dict]:
    import subprocess
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
        try:
            import torch
            return [{'id': i, 'mem_free_mb': 0, 'mem_total_mb': 0, 'util_percent': 0} for i in range(torch.cuda.device_count())]
        except Exception:
            return []


def _select_idle_gpus(count: int) -> List[int]:
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
    base = dict(
        model=args.model,
        model_alias=args.model_alias,
        eval_type='vllm_openai',
        eval_batch_size=args.batch_size,
        model_args={'enforce_eager': args.enforce_eager},
        debug=args.debug,
    )
    defs = [
        dict(id='cmmlu', diff='quick',
             cfg=TaskConfig(**base,
                            datasets=['cmmlu'],
                            dataset_args={'cmmlu': {'few_shot_num': 0, 'tags': ['zh'], 'task_type': 'exam', 'subset_list': ['college_mathematics', 'high_school_mathematics']}},
                            generation_config={'max_tokens': args.max_tokens, 'temperature': args.temperature, 'top_p': args.top_p})),
        dict(id='ceval', diff='quick',
             cfg=TaskConfig(**base,
                            datasets=['ceval'],
                            dataset_args={'ceval': {'few_shot_num': 0, 'tags': ['zh'], 'task_type': 'exam', 'subset_list': ['advanced_mathematics', 'high_school_mathematics', 'discrete_mathematics', 'middle_school_mathematics']}},
                            generation_config={'max_tokens': args.max_tokens, 'temperature': args.temperature, 'top_p': args.top_p})),

        dict(id='gsm8k', diff='easy',
             cfg=TaskConfig(**base,
                            datasets=['gsm8k'],
                            dataset_args={'gsm8k': {'few_shot_num': 0, 'task_type': 'math', 'tags': ['en']}},
                            generation_config={'max_tokens': args.max_tokens, 'temperature': args.temperature, 'top_p': args.top_p})),

        dict(id='amc', diff='difficult',
             cfg=TaskConfig(**base,
                            datasets=['amc'],
                            dataset_args={'amc': {'few_shot_num': 0, 'task_type': 'math', 'tags': ['en']}},
                            generation_config={'max_tokens': max(4096, args.max_tokens), 'temperature': args.temperature, 'top_p': args.top_p})),
        dict(id='competition_math', diff='difficult',
             cfg=TaskConfig(**base,
                            datasets=['competition_math'],
                            dataset_args={'competition_math': {'few_shot_num': 0, 'task_type': 'math', 'tags': ['en']}},
                            generation_config={'max_tokens': max(2048, args.max_tokens), 'temperature': max(0.6, args.temperature), 'top_p': args.top_p})),
        dict(id='math_500', diff='difficult',
             cfg=TaskConfig(**base,
                            datasets=['math_500'],
                            dataset_args={'math_500': {'few_shot_num': 0, 'task_type': 'math', 'tags': ['en']}},
                            generation_config={'max_tokens': max(4096, args.max_tokens), 'temperature': max(0.6, args.temperature), 'top_p': args.top_p},
                            repeats=4)),
        dict(id='aime24', diff='difficult',
             cfg=TaskConfig(**base,
                            datasets=['aime24'],
                            dataset_args={'aime24': {'few_shot_num': 0, 'task_type': 'math', 'tags': ['en']}},
                            generation_config={'max_tokens': max(4096, args.max_tokens), 'temperature': max(0.6, args.temperature), 'top_p': args.top_p},
                            repeats=8)),
        dict(id='aime25', diff='difficult',
             cfg=TaskConfig(**base,
                            datasets=['aime25'],
                            dataset_args={'aime25': {'few_shot_num': 0, 'task_type': 'math', 'tags': ['en']}},
                            generation_config={'max_tokens': max(4096, args.max_tokens), 'temperature': max(0.6, args.temperature), 'top_p': args.top_p},
                            repeats=8)),
    ]
    if args.task_set == 'quick':
        return [d['cfg'] for d in defs if d['diff'] == 'quick']
        
    if args.task_set == 'easy':
        return [d['cfg'] for d in defs if d['diff'] == 'easy']
    if args.task_set == 'difficult':
        return [d['cfg'] for d in defs if d['diff'] == 'difficult']
    return [d['cfg'] for d in defs]


def _prepare_work_items(tasks: List[TaskConfig], return_stats: bool = False):
    """Load datasets, drop cached samples, and expand the remaining ones into work items.
    
    Args:
        tasks: List of TaskConfig to process
        return_stats: If True, return (items, stats) tuple; otherwise return items only
        
    Returns:
        If return_stats is False: List[dict] of work items
        If return_stats is True: Tuple[List[dict], dict] where stats contains dataset statistics
    """
    from evalscope.api.registry import get_benchmark
    from evalscope.api.evaluator.cache import CacheManager
    from evalscope.utils.io_utils import OutputsStructure
    from collections import defaultdict

    items: List[dict] = []
    global_idx = 0
    reused = 0
    stats = defaultdict(lambda: {'total': 0, 'cached': 0, 'remaining': 0}) if return_stats else None
    
    for t in tasks:
        if t.model_alias and not t.use_cache:
            t.use_cache = os.path.join(t.work_dir, t.model_alias)

        work_dir = t.use_cache or t.work_dir
        outputs = OutputsStructure(outputs_dir=work_dir, is_make=False)

        for ds_name in t.datasets:
            adapter = get_benchmark(ds_name, t)
            ds_dict = adapter.load_dataset()
            cache_mgr = CacheManager(outputs=outputs, model_name=t.model_id, benchmark_name=adapter.name)
            for subset_name, dataset in ds_dict.items():
                total_cnt = len(dataset)
                # Build a mapping from sample.id to sample for efficient lookup
                sample_id_to_sample = {sample.id: sample for sample in dataset if sample.id is not None}
                
                # Try to filter cache, but handle index mismatches gracefully
                cached_states = []
                cached_sample_ids = set()
                cache_file = cache_mgr.get_prediction_cache_path(subset_name)
                
                if os.path.exists(cache_file):
                    from evalscope.utils.io_utils import jsonl_to_list
                    from evalscope.api.evaluator.cache import ModelResult
                    from evalscope.api.evaluator.state import TaskState
                    from evalscope.api.model import ModelOutput
                    
                    cache_items = jsonl_to_list(cache_file)
                    skipped_invalid = 0
                    
                    for cache_item in cache_items:
                        try:
                            cached_model_result = ModelResult.model_validate(cache_item)
                            # Try to find sample by sample_id instead of position index
                            sample_id = cached_model_result.index  # This is actually sample_id
                            
                            if sample_id in sample_id_to_sample:
                                sample = sample_id_to_sample[sample_id]
                                # Update metadata if exists
                                if cached_model_result.metadata:
                                    sample.metadata.update(cached_model_result.metadata)
                                
                                cached_state = TaskState(
                                    model=cached_model_result.model,
                                    sample=sample,
                                    messages=cached_model_result.messages,
                                    output=ModelOutput.model_validate(cached_model_result.model_output),
                                    completed=True,
                                )
                                cached_states.append(cached_state)
                                cached_sample_ids.add(sample_id)
                            else:
                                skipped_invalid += 1
                        except Exception as e:
                            skipped_invalid += 1
                            continue
                    
                    if skipped_invalid > 0:
                        logger.warning(
                            f"[CACHE] {adapter.name}/{subset_name}: skipped {skipped_invalid} invalid cache entries "
                            f"(sample_id not found in current dataset)")
                
                # Filter out cached samples from dataset
                remaining = [sample for sample in dataset if sample.id not in cached_sample_ids]
                cached_cnt = len(cached_states)
                remaining_cnt = len(remaining)
                
                if cached_cnt > 0:
                    logger.info(
                        f"[CACHE] {adapter.name}/{subset_name}: reused {cached_cnt} prediction(s), "
                        f"left {remaining_cnt} samples")
                
                if return_stats:
                    key = f'{adapter.name}/{subset_name}'
                    stats[key]['total'] = total_cnt
                    stats[key]['cached'] = cached_cnt
                    stats[key]['remaining'] = remaining_cnt
                
                if cached_cnt > 0:
                    reused += cached_cnt
                    logger.info(
                        f"[CACHE] {adapter.name}/{subset_name}: reused {cached_cnt} prediction(s), left {remaining_cnt}")
                if remaining_cnt == 0:
                    continue
                for sample in remaining:
                    items.append({
                        'model_id': t.model_id,
                        'model_name': (t.model if isinstance(t.model, str) else t.model_id),
                        'work_dir': work_dir,
                        'dataset_name': adapter.name,
                        'subset_name': subset_name,
                        'sample': sample,
                        'generation_config': t.generation_config,
                        'model_args': t.model_args,
                        'debug': bool(getattr(t, 'debug', False)),
                        'global_idx': global_idx,
                    })
                    global_idx += 1

    if reused > 0:
        logger.info(f'Reused {reused} cached prediction(s) before dispatching new work items.')
    else:
        logger.info('No cached predictions found; dispatching all samples for generation.')

    if return_stats:
        return items, dict(stats)
    return items


def _group_key(gen_cfg) -> Tuple:
    return (
        gen_cfg.max_tokens,
        gen_cfg.temperature,
        gen_cfg.top_p,
        gen_cfg.top_k,
        tuple(gen_cfg.stop_seqs or []),
        gen_cfg.n,
        gen_cfg.best_of,
        gen_cfg.logprobs,
        gen_cfg.top_logprobs,
        gen_cfg.seed,
    )


def worker(gpu_id: int, rank: int, queue: mp.JoinableQueue, enable_batch: bool, max_batch_size: int,
           progress_q: mp.Queue):
    try:
        os.setpgrp()
    except Exception:
        pass
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['EVALSCOPE_TQDM_POSITION'] = str(rank)

    from evalscope.models.vllm_openai import VllmOpenAIAPI
    from evalscope.api.messages import ChatMessageUser
    from evalscope.api.evaluator.cache import CacheManager
    from evalscope.utils.io_utils import OutputsStructure
    from evalscope.api.evaluator.state import TaskState

    apis: Dict[str, VllmOpenAIAPI] = {}
    caches: Dict[Tuple[str, str, str], CacheManager] = {}
    # Notify parent that worker is alive so progress bar appears immediately
    try:
        progress_q.put(('init', rank))
    except Exception:
        pass

    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break

        # Collect a batch including items compatible by sampling params and model
        batch: List[dict] = [item]
        k_model = item['model_id']
        k_cfg = _group_key(item['generation_config'])
        if enable_batch and max_batch_size > 1:
            try:
                # Non-blocking drain to fill up to max_batch_size
                for _ in range(max_batch_size - 1):
                    nxt = queue.get_nowait()
                    if nxt is None:
                        queue.task_done()
                        continue
                    if nxt['model_id'] == k_model and _group_key(nxt['generation_config']) == k_cfg:
                        batch.append(nxt)
                    else:
                        # put back if incompatible for this batch
                        queue.put(nxt)
                        break
            except Exception:
                pass

        # Ensure API for this model
        api = apis.get(k_model)
        if api is None:
            base_args = dict(item.get('model_args', {}) or {})
            base_args.setdefault('gpu_memory_utilization', 0.9)
            model_name = item.get('model_name', k_model)
            logger.info(f"[GPU {gpu_id}] Initializing VllmOpenAIAPI for model: {model_name}")
            apis[k_model] = VllmOpenAIAPI(model_name=str(model_name), **base_args)
            api = apis[k_model]
            try:
                progress_q.put(('init', rank))
            except Exception:
                pass

        # Build conversations and configs
        inputs = []
        tools = []
        tool_choices = []
        configs = []
        messages_per_item: List[List] = []
        for it in batch:
            sample = it['sample']
            if isinstance(sample.input, str):
                msg_list = [ChatMessageUser(content=sample.input)]
            else:
                msg_list = sample.input
            inputs.append(msg_list)
            messages_per_item.append(msg_list)
            tools.append(list(sample.tools) if sample.tools else [])
            tool_choices.append('none')
            configs.append(it['generation_config'])

        # Optional pre-generation debug preview for the first two samples
        try:
            for idx_preview, it_preview in enumerate(batch):
                _debug_log_input_preview(it_preview, messages_per_item[idx_preview])
        except Exception as _dbg_exc:
            logger.warning(f"[GPU {gpu_id}] Debug pre-generate preview failed: {_dbg_exc}")

        # Run batch chat generation with timing
        try:
            _t0 = time.perf_counter()
            outs = api.batch_generate(inputs=inputs, tools=tools, tool_choices=tool_choices, configs=configs)
            _t1 = time.perf_counter()
        except Exception as exc:
            logger.warning(f'[GPU {gpu_id}] batch_generate failed: {exc}')
            outs = []
            _t0 = _t1 = time.perf_counter()

        # Save caches per item
        # Report progress early based on number of completed generations
        try:
            progress_q.put(('prog', rank, (len(outs) if outs else len(batch))))
        except Exception:
            pass

        per_item_time = None
        try:
            if outs:
                per_item_time = (_t1 - _t0) / max(1, len(outs))
        except Exception:
            pass

        for idx, (it, out) in enumerate(zip(batch, outs)):
            key = (it['work_dir'], it['model_id'], it['dataset_name'])
            cm = caches.get(key)
            if cm is None:
                outputs = OutputsStructure(outputs_dir=it['work_dir'])
                cm = CacheManager(outputs=outputs, model_name=it['model_id'], benchmark_name=it['dataset_name'])
                caches[key] = cm

            # Pretty-print the first two samples' inputs and outputs when debug is enabled
            try:
                _debug_log_output_preview(it, messages_per_item[idx], out)
            except Exception as _dbg_exc:
                logger.warning(f"[GPU {gpu_id}] Debug pretty-print failed: {_dbg_exc}")
            try:
                state = TaskState(
                    model=it['model_id'],
                    sample=it['sample'],
                    messages=messages_per_item[idx],
                    output=out,
                    completed=True,
                )
                # Attach perf metadata similar to evaluator
                try:
                    output_text = state.output.completion if state and state.output else ''
                    output_chars = len(output_text) if isinstance(output_text, str) else 0
                    output_tokens = None
                    if state and state.output and state.output.usage:
                        output_tokens = state.output.usage.output_tokens or None
                    gen_time = float(per_item_time) if isinstance(per_item_time, (int, float)) else None
                    tok_per_s = (float(output_tokens) / gen_time) if (output_tokens is not None and gen_time and gen_time > 0) else None
                    chars_per_s = (float(output_chars) / gen_time) if (output_chars and gen_time and gen_time > 0) else None
                    perf = {
                        'time_s': gen_time,
                        'output_tokens': output_tokens,
                        'output_chars': output_chars,
                        'tok_per_s': tok_per_s,
                        'chars_per_s': chars_per_s,
                    }
                    meta = state.metadata or {}
                    meta['perf'] = perf
                    state.metadata = meta
                except Exception:
                    pass
                cm.save_prediction_cache(it['subset_name'], state, save_metadata=True)
            except Exception as e:
                logger.warning(f'[GPU {gpu_id}] Failed to save cache: {e}')

        # Mark processed items done
        for _ in batch:
            queue.task_done()


def main():
    args = parse_args()
    # Enable verbose logging in debug mode
    try:
        configure_logging(debug=bool(getattr(args, 'debug', False)))
    except Exception:
        pass
    _install_signal_handlers()
    atexit.register(lambda: _terminate_children(timeout=5.0))

    # Build tasks and work items
    tasks = _build_tasks(args)
    
    # Dry-run mode: only load datasets and print statistics, then exit
    if args.dry_run:
        items, dataset_stats = _prepare_work_items(tasks, return_stats=True)
        
        logger.info('=' * 80)
        logger.info('DRY-RUN MODE: Dataset loading complete (no vLLM/GPU used)')
        logger.info('=' * 80)
        
        # Print task summary
        logger.info(f'\nTask Summary:')
        logger.info(f'  Total tasks: {len(tasks)}')
        for i, t in enumerate(tasks, 1):
            logger.info(f'  Task {i}: {t.model_id} on {t.datasets}')
        
        # Print dataset statistics
        logger.info(f'\nDataset Statistics:')
        total_samples = 0
        total_cached = 0
        total_remaining = 0
        for key in sorted(dataset_stats.keys()):
            stats = dataset_stats[key]
            logger.info(f'  {key}:')
            logger.info(f'    Total: {stats["total"]}')
            logger.info(f'    Cached: {stats["cached"]}')
            logger.info(f'    Remaining: {stats["remaining"]}')
            total_samples += stats['total']
            total_cached += stats['cached']
            total_remaining += stats['remaining']
        
        logger.info(f'\nOverall Statistics:')
        logger.info(f'  Total samples: {total_samples}')
        logger.info(f'  Cached samples: {total_cached}')
        logger.info(f'  Remaining samples: {total_remaining}')
        logger.info(f'  Work items to process: {len(items)}')
        logger.info('=' * 80)
        logger.info('Dry-run complete. Exiting without starting vLLM or using GPU.')
        return
    
    items = _prepare_work_items(tasks)

    # Determine number of GPUs
    if args.max_use_gpu <= 0:
        try:
            import torch
            args.max_use_gpu = max(1, torch.cuda.device_count())
        except Exception:
            args.max_use_gpu = 1
    world_size = max(1, int(args.max_use_gpu))

    # Select idle GPUs
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

    if args.debug:
        # Only run and print first two samples in debug mode
        items = items[:2]
        logger.info(f'DEBUG mode: limiting to first {len(items)} sample(s) and exiting after inference.')
    logger.info(f'Total work items: {len(items)}')

    # Shared queue across workers
    ctx = mp.get_context('spawn')
    queue: mp.JoinableQueue = ctx.JoinableQueue(maxsize=world_size * max(1, args.batch_size) * 4)
    progress_q: mp.Queue = ctx.Queue()

    # Start workers (keep GPU busy by feeding queue until empty)
    # In debug mode, limit workers to at most two for quick turnaround
    if args.debug:
        selected_ids = selected_ids[:max(1, min(2, len(selected_ids)))]
    procs = []
    for i, gid in enumerate(selected_ids):
        p = ctx.Process(target=worker, args=(gid, i, queue, args.enable_batch, args.batch_size, progress_q))
        p.start()
        procs.append(p)
        _PROCS.append(p)

    # Feed queue with items
    for it in items:
        queue.put(it)
    # Signal termination to workers
    for _ in selected_ids:
        queue.put(None)

    # Multi-bar progress: one per GPU rank + a global bar
    start_t = time.perf_counter()
    per_rank_completed = {i: 0 for i in range(len(selected_ids))}
    total_completed = 0
    # Create individual bars
    bars = [tqdm(total=0, position=i, desc=f'GPU[{gid}] init...', leave=False, dynamic_ncols=True) for i, gid in enumerate(selected_ids)]
    # Global bar at the bottom
    global_bar = tqdm(total=len(items), desc='Predicting[dynamic]: ', position=len(selected_ids), leave=True, dynamic_ncols=True)
    # Process progress events
    while total_completed < len(items):
        drained = 0
        try:
            while drained < 32:
                msg = progress_q.get_nowait()
                if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] == 'init':
                    rank = int(msg[1])
                    if 0 <= rank < len(bars):
                        bars[rank].total = 0
                        bars[rank].set_description(f'GPU[{selected_ids[rank]}] ready')
                        bars[rank].refresh()
                elif isinstance(msg, tuple) and len(msg) == 3 and msg[0] == 'prog':
                    rank = int(msg[1])
                    inc = int(msg[2])
                    if 0 <= rank < len(bars):
                        per_rank_completed[rank] += inc
                        bars[rank].total = None  # indeterminate per-rank total
                        bars[rank].set_description(f'GPU[{selected_ids[rank]}]')
                        bars[rank].update(inc)
                    total_completed += inc
                    global_bar.update(inc)
                drained += 1
        except Exception:
            pass
        elapsed = time.perf_counter() - start_t
        global_bar.set_postfix({'elapsed_s': f'{elapsed:.1f}', 'done': f'{total_completed}/{len(items)}'}, refresh=False)
        alive = any(p.is_alive() for p in procs)
        if not alive and total_completed >= len(items):
            break
    # Close bars
    for b in bars:
        try:
            b.close()
        except Exception:
            pass
    try:
        global_bar.close()
    except Exception:
        pass

    # Wait until all items are processed (safety)
    queue.join()
    for p in procs:
        p.join()

    # After predictions, run evaluators to compute reviews and reports
    if args.debug:
        logger.info('DEBUG mode complete: skipping evaluator and exiting.')
        return

    from evalscope.run import run_task
    for t in tasks:
        try:
            # Use original TaskConfig semantics (model_alias/use_cache) to resolve work_dir consistently
            run_task(t)
        except Exception as e:
            logger.warning(f'Evaluator run failed for {t.model_id}@{t.datasets}: {e}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning('Interrupted by user; terminating child processes...')
        _terminate_children()
        sys.exit(130)
