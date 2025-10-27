# Repository Guidelines

## Project Structure & Module Organization
- Core package: `evalscope/` (CLI entry: `evalscope.cli.cli:run_cmd`).
- Tests: `tests/` (organized by domain: `benchmark/`, `rag/`, `swift/`, `vlm/`, etc.).
- Docs: `docs/en` and `docs/zh` (Sphinx builds via Makefile).
- Examples and customizations: `examples/`, `custom_eval/`.
- Packaging/config: `pyproject.toml`, `setup.cfg`, `requirements/`.

## Build, Test, and Development Commands
- Install (editable): `pip install -e .` (default) or `make install`.
- Dev setup: `make dev` (installs extras `[dev,perf,docs]` and pre-commit).
- Lint & formatting: `make lint` (runs pre-commit across files).
- Docs: `make docs`, or language-specific `make docs-en`, `make docs-zh`.
- Run CLI locally: `python -m evalscope.cli.cli ...` or `evalscope ...` after install.

## Coding Style & Naming Conventions
- Python 3.10+ required. Follow PEP 8 with 4-space indentation and 120-char lines.
- Imports: `isort` settings in `setup.cfg` (first-party `evalscope`).
- Linting: `flake8` with project ignores; run via pre-commit.
- Formatting: `yapf` (PEP 8 base, 120 columns). Keep module and function names snake_case; classes in PascalCase.

## Testing Guidelines
- Framework: `unittest` (tests in `tests/`), with optional `pytest` available in `requirements/dev.txt`.
- Run all: `python -m unittest discover tests` (example: `TEST_LEVEL_LIST=0,1 python -m unittest discover tests`).
- Naming: test files start with `test_*.py`; group by feature (e.g., `tests/benchmark/test_eval.py`).
- Aim for meaningful coverage on new/changed code; add fixtures to `tests/common.py` where appropriate.

## Commit & Pull Request Guidelines
- Commit style: concise, present tense; common prefixes observed: `[Feature]`, `[Fix]`, `[Doc]`, `[Benchmark]`.
- Branching: `feature/<name>`, `fix/<issue>`, or similar descriptive names.
- Pre-commit: run `pre-commit run --all-files` before pushing.
- PRs: include summary, rationale, usage notes, linked issues, and any screenshots/logs. Ensure tests pass and docs updated when relevant.

## Security & Configuration Tips
- Avoid committing secrets; prefer environment variables for keys/tokens.
- Large/model datasets are referenced externally—do not add them to the repo. Use `.gitignore` patterns as provided.

