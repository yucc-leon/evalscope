# Data-Parallel Evaluation (DP)

This guide shows how to run EvalScope with multiple GPUs in parallel using the vLLM SDK, and contrasts it with the OpenAI-compatible server route.

## Overview
- SDK adapter: `eval_type='vllm_openai'` initializes vLLM (`LLM.chat`) locally per process, matching `examples/test_chat.py`.
- Service adapter: `eval_type=EvalType.SERVICE` talks to an HTTP server (e.g., vLLM OpenAI server). Use this if you prefer centralized serving.

## Prerequisites
- Install vLLM and ensure GPUs are visible via `CUDA_VISIBLE_DEVICES`.
- Use a local model path or a model ID resolvable by vLLM.

## SDK DP Runner
- Run one process per GPU; each process binds to a device and initializes its own `LLM`.
- Command:
  - `python examples/run_dp_eval.py --dp 8 --model Qwen/Qwen2.5-0.5B-Instruct --datasets cmmlu ceval --max_tokens 1024 --temperature 0.0 --top_p 0.9`
- Internals:
  - Each rank sets `CUDA_VISIBLE_DEVICES=<rank>` and calls `run_task(TaskConfig(..., eval_type='vllm_openai'))`.
  - Engine initialization mirrors `examples/test_chat.py` for correctness.

## Service Route (Optional)
- Start a vLLM OpenAI server and use `EvalType.SERVICE`:
  - `python -m vllm.entrypoints.openai.api_server --model /path/to/model --host 127.0.0.1 --port 8801`
  - Configure `TaskConfig(api_url='http://127.0.0.1:8801/v1', api_key='EMPTY', eval_type=EvalType.SERVICE)`.

## Tensor Parallel vs Data Parallel
- Tensor Parallel (TP): Single process, shard one model across GPUs via `model_args` (e.g., `tensor_parallel_size=8`).
- Data Parallel (DP): Multiple processes, each with its own `LLM` instance on a different GPU (see runner above).

## Sharding & Caching Notes
- The example runner does not auto-shard datasets; all ranks run the same task. For large runs, pre-split datasets externally or customize the script per rank.
- Use unique `work_dir` per rank to avoid cache file contention; merging reports can be done post-run via `evalscope.report.gen_table` over multiple output directories.

## Tips
- Validate small runs first (dp=1), then scale.
- Pass vLLM engine options via `TaskConfig.model_args` (e.g., `gpu_memory_utilization`, `trust_remote_code`).
