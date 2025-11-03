# Repository Guidelines

## Project Structure & Module Organization
- Core package: `evalscope/` (APIs, evaluators, models, metrics, CLI).
- Benchmarks and adapters: `evalscope/benchmarks/`.
- Examples and scripts: `examples/`.
- Tests: `tests/` (Python `unittest`-based suites).
- Docs: `docs/en/`, `docs/zh/` (Sphinx).
- Config: `pyproject.toml`, `setup.cfg`, `requirements/`, `.pre-commit-config.yaml`.

## Build, Test, and Development Commands
- Install (editable): `pip install -e .` or `make install`.
- Dev setup (extras + hooks): `make dev`.
- Lint/format (pre-commit): `make lint` (runs all hooks on entire repo).
- Run tests (unittest): `python -m unittest discover tests`.
- Build docs: `make docs` (or `make docs-en`, `make docs-zh`).
- CLI entry: `evalscope ...` (after install) or `python -m evalscope.run ...`.

## Coding Style & Naming Conventions
- Python 3.10+; PEP 8 with 4-space indentation; max line length 120.
- Imports sorted by `isort` (see `setup.cfg`); formatting via `yapf`.
- Linting via `flake8` with project ignores (see `setup.cfg`).
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Use type hints and keep public APIs stable; prefer small, focused modules.

## Testing Guidelines
- Framework: `unittest`; organize by feature area under `tests/`.
- Naming: files `test_*.py`, classes `Test*`, methods `test_*`.
- Run all: `python -m unittest discover tests`.
- Add tests for new behavior and edge cases; keep tests deterministic and isolated (no network by default).

## Commit & Pull Request Guidelines
- Commit messages: concise imperative; common prefixes observed: `[Feature]`, `[Fix]`, `[Doc]`, `[Benchmark]`.
- Scope small, self-contained commits; reference issues: `Fix #123`.
- PRs must include:
  - Clear description of motivation and approach.
  - Usage notes or screenshots/logs if user-facing.
  - Links to issues; checklist for tests, docs, and backward compatibility.

## Security & Configuration Tips
- Do not commit secrets or large datasets; prefer env vars (e.g., `EVALSCOPE_API_KEY`, `EVALSCOPE_BASE_URL`).
- Cache/artifacts: keep under `outputs/` (git-ignored) and dataset/model caches in default locations.
- For optional backends (e.g., vLLM, diffusers), guard imports and document extras (see `requirements/*`).

## Work Summary & Open Items
- vLLM SDK API: engine-arg sanitization and init logging in `evalscope/models/vllm_openai.py`.
- API registry: added `vllm_openai` entry in `evalscope/models/model_apis.py`.
- Evaluator UX: per-sample perf timing + stable avg tok/s postfix in `evalscope/evaluator/evaluator.py`.
- DP runner: auto GPU pick, greedy balance, logging, reuse-one-engine-per-GPU in `examples/run_vllm_sdk_tasks_per_gpu.py`.
- Cache cleanup: `scripts/cleanup_enforce_eager.py` filters `.jsonl` by `metadata.perf.tok_per_s` threshold.

Remaining (nice-to-have)
- Batch generation: implement prompt-based `LLM.generate` in `evalscope/models/vllm_openai.py` and wire evaluator to use it when `eval_batch_size > 1`.
- Dataset sharding per GPU rank (avoid overlap) and optional YAML-driven task configs for runners.
- Performance guards: safe defaults for `max_model_len`; avoid forcing `max_num_seqs` (let vLLM profile per-model).

## Conversation Snapshot
- Added `VllmOpenAIAPI` and registry hook; sanitized engine args (drop `precision/torch_dtype`, map `tokenizer_path→tokenizer`, default `trust_remote_code=True`).
- Fixed logging misuse (placeholder formatting) and EngineArgs mismatch causing `precision` errors.
- Built DP runner with one-engine-per-GPU reuse, auto GPU selection, greedy balancing, and clear GPU/task logs.
- Improved evaluator UX: per‑sample timing and stable avg tok/s postfix; env flags for multi‑process tqdm placement/disable.
- Investigated performance: `enforce_eager` affects speed; resolved confusion by aligning init with examples and measuring correctly.
- Cleaned mixed results: created `scripts/cleanup_enforce_eager.py` to filter `.jsonl` caches by `metadata.perf.tok_per_s`.
- Enabled batch path: prompt‑based `LLM.generate` batching in `vllm_openai`; evaluator now calls `model.batch_generate` when supported.
- Debugged batch empties: added prompt preview logs, per‑group empty/short summaries, and single‑sample retry on empty batch results.
- Avoided fragile settings for this model: do not force `max_num_seqs`; keep `gpu_memory_utilization≈0.9`.
- Simplified examples: easy/difficult task set switch in `examples/run_vllm_sdk_tasks_per_gpu.py` to streamline runs.

---

# Agent Operating Notes

## Scope & Precedence
- This AGENTS.md applies to the entire repository rooted here.
- If a more-deeply-nested `AGENTS.md` exists, its instructions take precedence for files in its scope.
- Direct user/developer instructions in a session override anything here when they conflict.

## Agent Workflow
1. Install in editable mode: `pip install -e .` or `make install`.
2. Dev setup with extras and hooks: `make dev`.
3. Before edits, skim impacted module and relevant tests under `tests/`.
4. Make focused changes only; keep style consistent (PEP 8, yapf, isort, flake8 cfg).
5. Run unit tests for the affected area: `python -m unittest discover tests` (no network).
6. Lint/format: `make lint`.
7. Update docs/examples only when user-facing behavior changes.

## Critical Files & Modules
- Models API: `evalscope/models/vllm_openai.py`, `evalscope/models/model_apis.py`.
- Evaluator: `evalscope/evaluator/evaluator.py`.
- Benchmarks/adapters: `evalscope/benchmarks/`.
- Runners/examples: `examples/run_vllm_sdk_tasks_per_gpu.py`.
- Cleanup scripts: `scripts/cleanup_enforce_eager.py`.
- Config: `pyproject.toml`, `setup.cfg`, `requirements/`.

## vLLM OpenAI API Rules
- Do not pass `precision`/`torch_dtype` into vLLM EngineArgs; drop/sanitize these.
- Map `tokenizer_path` → `tokenizer`; set `trust_remote_code=True` by default.
- Do not hard-force `max_num_seqs`; let vLLM profile per-model.
- Prefer `gpu_memory_utilization≈0.9` unless user specifies otherwise.
- Keep safe defaults for `max_model_len` and document any overrides in examples.
- Log sanitized engine args during init for reproducibility.

## Evaluator & Batching
- Per-sample timing is required; keep stable avg tok/s postfix in logs.
- When `eval_batch_size > 1` and model supports it, prefer `model.batch_generate`.
- For batch empties, log prompt previews and group summaries; retry individually for empty results.
- Keep evaluator resilient to optional backends via guarded imports.

## DP Runner & GPUs
- Reuse one engine per GPU; auto-pick GPUs and greedily balance tasks.
- Provide clear per-GPU and per-task logs.
- Respect environment flags for tqdm/multiprocess placement/disable.
- Avoid fragile engine settings; do not degrade defaults to “optimize” unless measured.

## Caching & Cleanup
- Write artifacts under `outputs/` (git-ignored) and keep dataset/model caches in defaults.
- To filter noisy runs impacted by `enforce_eager`, use `scripts/cleanup_enforce_eager.py` with a `metadata.perf.tok_per_s` threshold.

## Testing & Linting
- Run all tests locally: `python -m unittest discover tests`.
- Prefer targeted tests for changed modules to start, then broader suite.
- Lint/format the whole repo via pre-commit: `make lint`.
- Keep tests deterministic and offline; guard any optional backend/network use.

## Common Commands
- Install editable: `pip install -e .`
- Dev setup: `make dev`
- Lint/format: `make lint`
- Run tests: `python -m unittest discover tests`
- Build docs: `make docs` | `make docs-en` | `make docs-zh`
- CLI entry: `evalscope ...` (after install) or `python -m evalscope.run ...`

## Do / Don’t
- Do keep changes minimal, focused, and style-consistent.
- Do update usage notes when changing user-facing behavior.
- Do guard optional backends and document extras in `requirements/*`.
- Don’t commit secrets or large datasets; use env vars (e.g., `EVALSCOPE_API_KEY`).
- Don’t “optimize” by forcing model/engine limits (`max_num_seqs`, unsafe dtype) without data.
- Don’t refactor unrelated code when fixing a targeted issue.

## Open TODOs for Agents
- Implement prompt-based `LLM.generate` batch path in `vllm_openai` and ensure evaluator uses it.
- Add dataset sharding per GPU rank to avoid overlap.
- Support optional YAML-driven task configs for runners.
- Add safe defaults/guards for `max_model_len` and document behavior.

## Troubleshooting
- EngineArgs precision error: ensure `precision`/`torch_dtype` are not passed; sanitize in API.
- Empty batch results: log previews, summarize groups, then retry per-sample.
- Slowdowns with `enforce_eager`: measure accurately; filter caches using the cleanup script.
- Tokenizer mapping: use `tokenizer` (mapped from `tokenizer_path`) and set `trust_remote_code=True`.
